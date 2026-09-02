"""The training entrypoint for the crowned recipe (build plan's Phase P7/P8): a LEAN driver over
the fixed, already-selected architecture/hyperparameters -- encoder=conv_trunk, predictor 4-layer
transformer, cyclic harmonic operator k0=4/n_j=1..10/k_j=(24,24,20,16,12,10,8,6,4,2), lambda_sig=
0.15/lambda_pred=1.0/lambda_trans in {1.0, 0.0}, the full V5 augmentation stack at prob 0.5 -- not
the reference repo's `scripts/s2_pretrain_jepa.py`, which accreted ~30+ flags across the rejected/
exploratory arms of a ~60-variant campaign this build leaves behind. Every CLI flag below exists
because Phase P8's four launch invocations (`src/winder/ablations.py`'s `ABLATION_ARMS`) actually
pass it; nothing here is exposed "for completeness".

Real-data mode only -- no `--synthetic` twin, since this driver has exactly one recipe and it
trains on real PTB-XL. The transport operator is ALWAYS constructed (`--transport-arm` has no
`"none"` choice): the control arm is `--lambda-trans 0.0` with the SAME operator construction as
the signal arm, never a structurally different, operator-less run (build plan's "all else equal"
requirement, verified in Phase P6 Tier 0 against the reference repo's own `FIN_LAM0_seed0`
checkpoint). This also means `train_step` always receives `theta`/`operator`, so there is no
"real batches, no theta" code path to maintain here (contrast the reference driver's
`_real_batches` vs. `_real_batches_with_theta` split) -- one dataset wrapper, one batch iterator.

THE CONFIG-DIFF DRIFT GUARD (`_expected_config_diff` + `winder.config.assert_expected_config_diff`,
called in `main()` right after `resolved_config_yaml` and before the dataset build, so a wiring bug
fails in seconds, not after a metadata/manifest load): every CLI flag or hardcoded construction
choice this driver believes changes the resolved `config.yaml` relative to
`--reference-config-path` (default the copied-in `FIN_seed0/checkpoint_step5000/config.yaml`) is
named explicitly in `_expected_config_diff`. The check then fails loud in exactly the two ways that
matter before a real GPU run: an EXPECTED change that never reached the resolved config (an unwired
flag -- e.g. `--lambda-trans 0.0` silently not reaching `TrainConfig` would train a "control" as a
second "signal") and an OBSERVED change nothing here predicts (drift -- e.g. a hardcoded default
moving without a CLI flag to explain it). See `winder.config.assert_expected_config_diff`'s own
docstring for the exact semantics, and this module's `_expected_config_diff` for the field-by-field
reasoning.

The empirically-verified allowed-diff set against `FIN_seed0/checkpoint_step5000/config.yaml`
(read directly, not assumed from the build plan's prose) is exactly `{train.lambda_trans,
train.seed_pretrain, arm.seed, arm.name}` -- FOUR fields, not the plan's claimed five. Reading
`FIN_seed1/checkpoint_step5000/config.yaml` directly shows `jepa.seed_pretrain: 0`, identical to
FIN_seed0's, even though `train.seed_pretrain`/`arm.seed`/`arm.name` all correctly carry seed 1 --
the reference driver's own `main()` never assigns `args.seed` into the `JepaConfig` it builds
(only `TrainConfig(seed_pretrain=...)` and `ArmConfig(seed=..., name=f"..._seed{seed}")` read it),
so `jepa.seed_pretrain` stays at its dataclass default (0) regardless of `--seed`. This driver
reproduces that exact behaviour (bug-for-bug, per the "reproduce the crowned recipe's own
observable surface, don't silently fix it" mandate) rather than wiring `--seed` into
`JepaConfig.seed_pretrain` to make the diff set "cleaner" -- `_expected_config_diff` does not name
`jepa.seed_pretrain` at all, so the guard would itself fail loud if a future edit accidentally
wired it up (a now-unexplained "drift").

Outputs (under --artifacts-dir, gitignored, matching the reference driver's own filenames so any
downstream eval tooling that expects them needs no changes):
  s2_history.jsonl   one StepMetrics record per step, JSON Lines, truncated at the start of the
                     run and appended-and-flushed every step (survives a `kill -9` mid-run)
  s2_summary.json    final losses, embedding diagnostics, provenance (git SHA, device, config,
                     the integrity report, the FLAT_SIGNAL count, and the config-diff guard's own
                     verified diff, for a durable record of what actually differed from the
                     reference recipe on this specific run)
  checkpoint/        state.pt/config.yaml/meta.json (winder.jepa.checkpoint), plus
                     operator_state_dict/an "arm:" config.yaml section -- the operator is always
                     built, so this is always present, including for the control arm
  checkpoint_step{N}/  one full bundle per --checkpoint-at entry
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from winder.config import (
    ArmConfig,
    assert_expected_config_diff,
    flatten_yaml,
    resolve_operator_config,
)
from winder.data.folds import FoldConfig, folds
from winder.data.integrity import assemble_integrity_report, git_sha, sha256_file
from winder.data.norm_stats import LeadStats
from winder.data.ptbxl import load_metadata
from winder.determinism import generator, init_parameters
from winder.jepa import checkpoint
from winder.jepa.dataset import EcgWindowDataset, EcgWindowItem
from winder.jepa.diagnostics import spectrum_report
from winder.jepa.model import JepaConfig, JepaModel, build_jepa
from winder.jepa.registry import ENCODER_REGISTRY
from winder.jepa.train import StepMetrics, TrainConfig, fit
from winder.operators.harmonic import HarmonicTransport
from winder.operators.registry import OPERATOR_REGISTRY
from winder.paths import default_data_root
from winder.transport.dataset import PhaseTaggedDataset, PhaseTaggedItem, load_theta_tokens

#: CON-04's integrity set E: manifest reason codes meaning the waveform itself is unreadable or
#: non-finite -- ported verbatim from the reference driver's own `_INTEGRITY_EXCLUDE_CODES`. The
#: seven phase-clock codes (including FLAT_SIGNAL) are deliberately absent: phase-clock QC has
#: nothing to do with a JEPA that has no phase clock over the WAVEFORM path (the transport arm's
#: own theta lookup handles ITS eligibility separately, via an all-NaN row -- see
#: `winder.transport.dataset.PhaseTaggedDataset`'s own docstring).
_INTEGRITY_EXCLUDE_CODES: tuple[str, ...] = ("READ_ERROR", "WRONG_SHAPE", "NAN")

_DEFAULT_DATA_ROOT = default_data_root()
#: Tracked in git (unlike artifacts/reference/, a 424 MB copy-in from the predecessor repo
#: excluded by .gitignore) -- a bare clone must be able to run this guard, so the file it
#: diffs against cannot live only in an untracked artifacts tree. Contains no machine- or
#: user-specific paths; it is the crowned recipe's resolved config.yaml, verbatim.
_DEFAULT_REFERENCE_CONFIG = "configs/crowned_recipe_config.yaml"

#: Hardcoded, not CLI flags: the crowned recipe's own architecture family and the "free" arm's
#: (never launched by Phase P8, but a registered OPERATOR_REGISTRY choice) own optimizer knobs --
#: matching the reference driver's own pre-existing defaults exactly. Absent from
#: _expected_config_diff below by construction: no CLI flag can move these, so a resolved diff
#: touching them is unexplained DRIFT the guard should catch, not something to predict here.
_N_TOKENS = 125
_PROJECTOR_NAME = "mlp"
_MASK_SAMPLER_NAME = "causal_block"
_PREDICTION_LOSS_NAME = "mse"
_REGULARIZER_NAME = "sigreg"
_OPERATOR_LR = 1e-2
_OPERATOR_WEIGHT_DECAY = 0.0


def _parse_int_csv(raw: str, flag_name: str) -> list[int]:
    """`"1,2,3"` -> `[1, 2, 3]`, `ValueError` naming `flag_name` on a malformed token."""
    try:
        return [int(tok) for tok in raw.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{flag_name} must be a comma-separated list of ints, got {raw!r}"
        ) from exc


def _parse_json_dict(raw: str, flag_name: str) -> dict[str, Any]:
    """`'{"n_layers":4}'` -> `{"n_layers": 4}`, `ValueError` naming `flag_name` on malformed or
    non-object JSON."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} must be a JSON object, got {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must be a JSON object, got {raw!r}")
    return parsed


def _restrict_to_integrity_set(metadata: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """CON-04's integrity set E: drop only rows whose manifest `reason_code` is one of
    `_INTEGRITY_EXCLUDE_CODES`. Every `metadata['ecg_id']` must have a matching manifest row -- a
    record with no manifest row is an error, never an implicit include or exclude."""
    meta_ids = set(metadata["ecg_id"].tolist())
    man_ids = set(manifest_df["ecg_id"].tolist())
    missing = meta_ids - man_ids
    if missing:
        raise ValueError(
            f"{len(missing)} metadata ecg_id(s) have no row in the manifest, e.g. "
            f"{sorted(missing)[:10]} -- a metadata record absent from the manifest is an error, "
            f"never an implicit include or exclude."
        )
    excluded_ids = set(
        manifest_df.loc[manifest_df["reason_code"].isin(_INTEGRITY_EXCLUDE_CODES), "ecg_id"]
    )
    return metadata.loc[~metadata["ecg_id"].isin(excluded_ids)]


def _flat_signal_count(manifest_df: pd.DataFrame, ecg_ids: pd.Series) -> int:
    """Count of `ecg_ids` whose manifest row carries reason_code FLAT_SIGNAL -- a report, never a
    filter (CON-04 keeps these records eligible)."""
    in_pool = manifest_df["ecg_id"].isin(set(ecg_ids.tolist()))
    return int((in_pool & (manifest_df["reason_code"] == "FLAT_SIGNAL")).sum())


def _collate_waveforms(batch: list[EcgWindowItem]) -> torch.Tensor:
    """Extracts and stacks only the waveform tensor (the frozen-encoder diagnostics batch)."""
    return torch.stack([item["waveform"] for item in batch])


def _collate_phase_tagged(batch: list[PhaseTaggedItem]) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts and stacks only the waveform and theta tensors -- `train_step` never reads labels/
    ecg_id/patient_id (pretraining must not read labels)."""
    waveform = torch.stack([item["waveform"] for item in batch])
    theta = torch.stack([item["theta"] for item in batch])
    return waveform, theta


def _phase_tagged_batches(
    dataset: PhaseTaggedDataset,
    batch_size: int,
    steps: int,
    gen_data: torch.Generator,
    device: torch.device,
    *,
    num_workers: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Cycles `dataset` in a seeded-shuffle order, yielding exactly `steps`
    `((B, 12, 1000), (B, n_tokens))` waveform/theta pairs, re-shuffling every full pass.
    `num_workers=0` (matching the reference driver's own default, and every existing checkpoint
    this recipe has ever produced) is single-process and the only value proven not to change
    which samples land in which batch -- `shuffle=True`'s per-epoch permutation is drawn from
    `gen_data` either way, but a `num_workers>0` DataLoader distributes work across persistent
    subprocesses, and nothing here has verified that changes no other draw order this stream
    depends on. Raising it is a real lever for throughput on a slower data-loading node (e.g. a
    network filesystem on a cluster), traded explicitly against that unverified risk -- exposed
    as a flag rather than a silently different default."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen_data,
        collate_fn=_collate_phase_tagged,
        num_workers=num_workers,
    )
    yielded = 0
    while yielded < steps:
        for waveform, theta in loader:
            if yielded >= steps:
                return
            yield waveform.to(device), theta.to(device)
            yielded += 1


def _expected_config_diff(
    args: argparse.Namespace,
    arm_name: str,
    n_j: list[int],
    k_j: list[int],
    predictor_overrides: dict[str, Any],
    reference_flat: dict[str, Any],
) -> dict[str, Any]:
    """Every config.yaml leaf THIS driver's CLI flags (or its own hardcoded construction, e.g.
    `jepa.encoder_name` always being set explicitly) control, mapped to its resolved value on
    THIS run -- filtered to only the entries that actually differ from `reference_flat`
    (`winder.config.flatten_yaml` of `--reference-config-path`). This is what
    `winder.config.assert_expected_config_diff` checks the observed diff against; see this
    module's own docstring for why the four-field result differs from the build plan's claimed
    five, and `winder.config.assert_expected_config_diff`'s docstring for what each mismatch
    category means."""
    candidate: dict[str, Any] = {
        "train.n_steps": args.steps,
        "train.lambda_sig": args.lambda_sig,
        "train.lambda_trans": args.lambda_trans,
        "train.seed_pretrain": args.seed,
        "train.augment": args.augment,
        "train.augment_prob": args.augment_prob,
        "jepa.encoder_name": args.encoder_name,
        "arm.name": arm_name,
        "arm.seed": args.seed,
        "arm.operator_name": args.transport_arm,
        "arm.operator.k0": args.k0,
        "arm.operator.n_j": list(n_j),
        "arm.operator.k_j": list(k_j),
    }
    for key, value in predictor_overrides.items():
        candidate[f"jepa.predictor.{key}"] = value
    return {key: value for key, value in candidate.items() if reference_flat.get(key) != value}


def _build_model(config: JepaConfig, seed: int, device: torch.device) -> JepaModel:
    model = build_jepa(config, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    model.to(device)
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader subprocess count -- 0 (default) matches every checkpoint this recipe "
        "has produced so far; see _phase_tagged_batches's own docstring before raising it",
    )
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--data-root", default=_DEFAULT_DATA_ROOT, help="PTB-XL root (records500/)")
    ap.add_argument(
        "--train-folds",
        required=True,
        help="comma-separated PTB-XL strat_fold ints, e.g. '1,2,3,4,5,6,7,8,9'",
    )
    ap.add_argument("--lead-stats-path", required=True, help="fitted LeadStats JSON (never refit)")
    ap.add_argument("--manifest-path", required=True, help="per-record manifest parquet")
    ap.add_argument("--theta-tokens-path", required=True, help="theta_tokens.npz")
    ap.add_argument(
        "--lambda-sig",
        type=float,
        required=True,
        help="TrainConfig.lambda_sig -- NOT JepaConfig.lambda_sig, a distinct, unwired field of "
        "the same short name the reference recipe's own config.yaml also carries at a different "
        "value (0.1); see this module's own docstring",
    )
    ap.add_argument(
        "--checkpoint-at",
        default="",
        help="comma-separated step counts; after each COMPLETES, save a full checkpoint bundle "
        "to <artifacts-dir>/checkpoint_step{N}/",
    )
    ap.add_argument("--transport-arm", required=True, choices=list(OPERATOR_REGISTRY))
    ap.add_argument(
        "--lambda-trans",
        type=float,
        default=0.0,
        help="transport loss weight -- 0.0 is the control point (operator still built, just "
        "zero-weighted; this driver has no operator-less 'none' arm)",
    )
    ap.add_argument(
        "--k0", type=int, required=True, help="transport operator invariant-block width"
    )
    ap.add_argument("--n-j", required=True, help="comma-separated contiguous harmonic indices")
    ap.add_argument("--k-j", required=True, help="comma-separated per-harmonic multiplicities")
    ap.add_argument("--encoder-name", required=True, choices=list(ENCODER_REGISTRY))
    ap.add_argument(
        "--predictor-json",
        default="",
        help="JSON object merged into config.predictor, e.g. '{\"n_layers\":4}'; empty leaves "
        "the predictor registry's own schema default",
    )
    ap.add_argument(
        "--augment",
        default="",
        help="comma-separated subset of winder.jepa.train.AUGMENT_VOCABULARY; empty is off",
    )
    ap.add_argument("--augment-prob", type=float, default=0.5)
    ap.add_argument(
        "--reference-config-path",
        default=_DEFAULT_REFERENCE_CONFIG,
        help="the crowned recipe's own resolved config.yaml -- the config-diff drift guard "
        "(module docstring) diffs THIS run's resolved config against it at startup",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="a checkpoint_step{N}/ or checkpoint/ bundle written by a PRIOR invocation of this "
        "exact recipe (same flags) -- restores model/optimizer/operator weights and all five "
        "generator streams, and continues the SAME cosine schedule from the saved step rather "
        "than starting a new one. --steps is still the schedule's total length, not the number "
        "of additional steps to run (see fit()'s own docstring on start_step vs. n_steps).",
    )
    args = ap.parse_args(argv)

    if not 0.0 <= args.augment_prob <= 1.0:
        ap.error(f"--augment-prob must be in [0, 1], got {args.augment_prob}")
    if args.augment == "" and args.augment_prob != 0.5:
        ap.error(
            f"--augment-prob={args.augment_prob} with --augment unset would be silently inert "
            "-- pass --augment, or leave --augment-prob at its 0.5 default"
        )
    try:
        n_j = _parse_int_csv(args.n_j, "--n-j")
        k_j = _parse_int_csv(args.k_j, "--k-j")
        train_folds = tuple(sorted(set(_parse_int_csv(args.train_folds, "--train-folds"))))
        predictor_overrides = (
            _parse_json_dict(args.predictor_json, "--predictor-json") if args.predictor_json else {}
        )
        checkpoint_at_steps = (
            sorted(set(_parse_int_csv(args.checkpoint_at, "--checkpoint-at")))
            if args.checkpoint_at
            else []
        )
    except ValueError as exc:
        ap.error(str(exc))
    if any(n <= 0 for n in checkpoint_at_steps):
        ap.error(f"--checkpoint-at steps must be positive, got {args.checkpoint_at!r}")

    t0 = time.time()
    winder_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)
    print(f"[pretrain] device={device}", flush=True)

    config = JepaConfig(
        n_leads=12,
        n_samples=1000,
        n_tokens=_N_TOKENS,
        encoder_name=args.encoder_name,
        encoder={},
        projector_name=_PROJECTOR_NAME,
        projector={},
        predictor_name="transformer",
        predictor=predictor_overrides,
        mask_sampler_name=_MASK_SAMPLER_NAME,
        mask_sampler={},
        prediction_loss_name=_PREDICTION_LOSS_NAME,
        prediction_loss={},
        regularizer_name=_REGULARIZER_NAME,
        regularizer={},
    )
    model = _build_model(config, args.seed, device)

    train_cfg = TrainConfig(
        n_steps=args.steps,
        lambda_sig=args.lambda_sig,
        lambda_trans=args.lambda_trans,
        augment=args.augment,
        augment_prob=args.augment_prob,
        seed_pretrain=args.seed,
    )

    arm_name = f"{args.transport_arm}_seed{args.seed}"
    arm_config = ArmConfig(
        name=arm_name,
        operator_name=args.transport_arm,
        seed=args.seed,
        operator={"k0": args.k0, "n_j": n_j, "k_j": k_j},
    )
    resolved_operator_cfg = resolve_operator_config(arm_config)
    _schema_cls, operator_ctor = OPERATOR_REGISTRY[args.transport_arm]
    built_operator = operator_ctor(resolved_operator_cfg)
    assert isinstance(built_operator, HarmonicTransport)  # true of every registered operator
    operator: HarmonicTransport = built_operator
    operator.to(device)
    if operator.dimension != model.projector.output_width:
        raise ValueError(
            f"transport operator dimension={operator.dimension} (k0={operator.k0}, "
            f"n_j={operator.n_j}, k_j={operator.k_j.tolist()}) != "
            f"model.projector.output_width={model.projector.output_width} -- --k0/--n-j/--k-j "
            f"must satisfy k0 + 2*sum(k_j) == projector.output_width."
        )
    print(
        f"[pretrain] transport arm: {args.transport_arm}, lambda_trans={args.lambda_trans}, "
        f"dimension={operator.dimension} (k0={operator.k0}, n_j={operator.n_j}, "
        f"k_j={operator.k_j.tolist()})",
        flush=True,
    )

    # CKPT-02: snapshotted before fit() runs, so a checkpoint's config.yaml is byte-identical to
    # the merge that actually built this run's model/optimizer.
    config_yaml_text = checkpoint.resolved_config_yaml(config, train_cfg, arm_config=arm_config)

    # THE CONFIG-DIFF DRIFT GUARD (module docstring) -- runs here, before any dataset/metadata
    # I/O, so a wiring bug or a hardcoded-default drift fails in seconds.
    with open(args.reference_config_path, encoding="utf-8") as fh:
        reference_yaml_text = fh.read()
    reference_flat = flatten_yaml(reference_yaml_text)
    expected_diff = _expected_config_diff(
        args, arm_name, n_j, k_j, predictor_overrides, reference_flat
    )
    config_diff = assert_expected_config_diff(reference_yaml_text, config_yaml_text, expected_diff)
    print(f"[pretrain] config-diff guard passed against {args.reference_config_path}", flush=True)
    print(f"[pretrain] verified diff vs. reference recipe: {config_diff}", flush=True)

    print(f"[pretrain] loading metadata from {args.data_root}", flush=True)
    metadata = load_metadata(args.data_root)
    fold_config = FoldConfig(train_folds=train_folds)
    pool_full = folds(metadata, fold_config)["train"]

    manifest_df = pd.read_parquet(
        args.manifest_path, columns=["ecg_id", "reason_code", "quality_flags"]
    )
    eligible = _restrict_to_integrity_set(pool_full, manifest_df)
    print(
        f"[pretrain] pretraining over {len(eligible)}/{len(pool_full)} pool records "
        f"({len(pool_full) - len(eligible)} dropped by the CON-04 integrity filter)",
        flush=True,
    )

    lead_stats = LeadStats.from_json(args.lead_stats_path)
    base_dataset = EcgWindowDataset(eligible, args.data_root, lead_stats=lead_stats)
    if len(base_dataset) == 0:
        raise SystemExit(
            "[pretrain] the training pool is empty after filtering -- nothing to train on"
        )

    theta_by_id, theta_meta = load_theta_tokens(args.theta_tokens_path)
    phase_dataset = PhaseTaggedDataset(
        base_dataset,
        theta_by_id,
        theta_meta,
        n_tokens=config.n_tokens,
        patch_width=int(model.encoder.config.patch_width),  # type: ignore[attr-defined]
    )

    flat_signal_count = _flat_signal_count(manifest_df, pool_full["ecg_id"])
    integrity_report = assemble_integrity_report(
        args.data_root, metadata, fold_config=fold_config, winder_repo_root=winder_root
    )

    gen_mask = generator(args.seed, "mask")
    gen_sigreg = generator(args.seed, "sigreg")
    gen_sigreg_record = generator(args.seed, "sigreg_record")
    gen_augment = generator(args.seed, "augment")
    gen_data = generator(args.seed, "data_order")

    checkpoint_meta: dict[str, Any] = {
        "winder_git_sha": git_sha(winder_root),
        "train_folds": list(fold_config.train_folds),
        "manifest_path": os.path.abspath(args.manifest_path),
        "manifest_sha256": sha256_file(args.manifest_path),
        "lead_stats_path": os.path.abspath(args.lead_stats_path),
        "lead_stats_sha256": sha256_file(args.lead_stats_path),
        "lead_stats_folds": list(lead_stats.folds),
        "flat_signal_count_pool": flat_signal_count,
        "integrity": integrity_report,
    }

    checkpoint_at_set = set(checkpoint_at_steps)
    checkpoint_at_generators: dict[str, torch.Generator] = {}
    if checkpoint_at_set:
        checkpoint_at_generators.update(
            {
                "mask": gen_mask,
                "sigreg": gen_sigreg,
                "sigreg_record": gen_sigreg_record,
                "augment": gen_augment,
                "data_order": gen_data,
            }
        )

    os.makedirs(args.artifacts_dir, exist_ok=True)
    history_path = os.path.join(args.artifacts_dir, "s2_history.jsonl")
    if not args.resume_from:
        open(history_path, "w").close()

    param_groups: list[dict[str, Any]] = [
        {
            "params": model.parameters(),
            "lr": train_cfg.lr,
            "betas": train_cfg.betas,
            "eps": train_cfg.eps,
            "weight_decay": train_cfg.weight_decay,
        }
    ]
    operator_params = list(operator.parameters())
    if operator_params:  # the cyclic arm has none; the free arm (never launched by Phase P8) would
        param_groups.append(
            {
                "params": operator_params,
                "lr": _OPERATOR_LR,
                "betas": train_cfg.betas,
                "eps": train_cfg.eps,
                "weight_decay": _OPERATOR_WEIGHT_DECAY,
            }
        )
    optimizer = torch.optim.AdamW(param_groups)

    # Loaded only after model AND optimizer both exist, so load_checkpoint's strict
    # load_state_dict calls see the exact param-group structure that saved this state -- an
    # AdamW built with fewer/more groups than the checkpoint's own optimizer_state_dict would
    # otherwise raise a confusing shape error deep inside torch rather than here.
    start_step = 0
    if args.resume_from:
        loaded = checkpoint.load_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            operator=operator,
            map_location=device,
        )
        if loaded.config_yaml != config_yaml_text:
            raise SystemExit(
                f"[pretrain] --resume-from {args.resume_from} was saved with a config.yaml "
                "that does not match this invocation's own resolved config -- resuming requires "
                "the exact same flags as the run that wrote this checkpoint, not a new recipe."
            )
        start_step = loaded.step
        gen_mask.set_state(loaded.generator_states["mask"])
        gen_sigreg.set_state(loaded.generator_states["sigreg"])
        gen_sigreg_record.set_state(loaded.generator_states["sigreg_record"])
        gen_augment.set_state(loaded.generator_states["augment"])
        gen_data.set_state(loaded.generator_states["data_order"])
        print(f"[pretrain] resumed from {args.resume_from} at step {start_step}", flush=True)

    def _on_step(m: StepMetrics) -> None:
        with open(history_path, "a") as history_file:
            history_file.write(json.dumps(asdict(m)) + "\n")
            history_file.flush()
        print(
            f"[pretrain] step={m.step} lr={m.lr:.2e} pred={m.pred_loss:.4f} "
            f"sigreg={m.sigreg_loss:.4f} total={m.total_loss:.4f} grad_norm={m.grad_norm:.4f} "
            f"trans={m.trans_loss:.4f} gain={m.trans_gain:.4f}",
            flush=True,
        )
        if (m.step + 1) in checkpoint_at_set:
            step_dir = os.path.join(args.artifacts_dir, f"checkpoint_step{m.step + 1}")
            checkpoint.save_checkpoint(
                step_dir,
                model=model,
                optimizer=optimizer,
                step=m.step + 1,
                generators=checkpoint_at_generators,
                config_yaml=config_yaml_text,
                meta=checkpoint_meta,
                operator=operator,
            )
            print(f"[pretrain] wrote mid-run checkpoint to {step_dir}", flush=True)

    # Two synchronized iterators split from one shared shuffled DataLoader (itertools.tee) --
    # mirrors the reference driver's own _real_batches_with_theta: a separately-shuffled pair of
    # loaders would desynchronize waveform and theta after the first batch. Safe only because
    # fit()'s own zip(...) consumes both in lockstep, one step of each per loop iteration.
    # `args.steps - start_step` (not args.steps): fit()'s own contract is that cfg.n_steps is the
    # SCHEDULE's total length, while how many steps THIS call runs is controlled by how many
    # batches it is handed -- start_step=0 on a fresh run makes this identical to args.steps.
    tee_a, tee_b = itertools.tee(
        _phase_tagged_batches(
            phase_dataset,
            args.batch_size,
            args.steps - start_step,
            gen_data,
            device,
            num_workers=args.num_workers,
        ),
        2,
    )
    waveform_batches = (w for w, _ in tee_a)
    theta_batches = (t for _, t in tee_b)

    metrics = fit(
        model,
        waveform_batches,
        train_cfg,
        optimizer,
        on_step=_on_step,
        start_step=start_step,
        gen_mask=gen_mask,
        gen_sigreg=gen_sigreg,
        gen_sigreg_record=gen_sigreg_record,
        gen_augment=gen_augment,
        theta_batches=theta_batches,
        operator=operator,
    )

    eval_loader = DataLoader(
        base_dataset,
        batch_size=min(args.batch_size, len(base_dataset)),
        shuffle=False,
        collate_fn=_collate_waveforms,
    )
    waveform = next(iter(eval_loader)).to(device)
    model.eval()
    with torch.no_grad():
        tokens = model.encoder.forward(waveform)
        z = model.projector.forward(tokens)
        diagnostics = spectrum_report(z.reshape(-1, z.shape[-1]))

    checkpoint_dir = os.path.join(args.artifacts_dir, "checkpoint")
    checkpoint.save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        step=metrics[-1].step + 1 if metrics else start_step,
        generators={
            "mask": gen_mask,
            "sigreg": gen_sigreg,
            "sigreg_record": gen_sigreg_record,
            "augment": gen_augment,
            "data_order": gen_data,
        },
        config_yaml=config_yaml_text,
        meta=checkpoint_meta,
        operator=operator,
    )
    print(f"[pretrain] wrote checkpoint to {checkpoint_dir}", flush=True)
    print(f"[pretrain] wrote {history_path}", flush=True)

    final_pred_loss = metrics[-1].pred_loss if metrics else None
    if final_pred_loss is not None and math.isnan(final_pred_loss):
        final_pred_loss = None

    summary: dict[str, Any] = {
        "n_steps_run": len(metrics),
        "final_pred_loss": final_pred_loss,
        "final_sigreg_loss": metrics[-1].sigreg_loss if metrics else None,
        "final_total_loss": metrics[-1].total_loss if metrics else None,
        "diagnostics": diagnostics,
        "elapsed_min": (time.time() - t0) / 60.0,
        "config_diff_vs_reference": {k: list(v) for k, v in config_diff.items()},
        "provenance": {
            "winder_git_sha": git_sha(winder_root),
            "device": str(device),
            "seed": args.seed,
            "arm_name": arm_name,
            "lambda_sig": args.lambda_sig,
            "lambda_trans": args.lambda_trans,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "checkpoint_dir": os.path.abspath(checkpoint_dir),
            "train_folds": list(fold_config.train_folds),
            "manifest_path": os.path.abspath(args.manifest_path),
            "manifest_sha256": checkpoint_meta["manifest_sha256"],
            "lead_stats_path": os.path.abspath(args.lead_stats_path),
            "lead_stats_sha256": checkpoint_meta["lead_stats_sha256"],
            "lead_stats_folds": list(lead_stats.folds),
            "flat_signal_count_pool": flat_signal_count,
            "n_records_pool_full": int(len(pool_full)),
            "n_records_integrity_eligible": int(len(eligible)),
            "integrity": integrity_report,
        },
    }
    summary_path = os.path.join(args.artifacts_dir, "s2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[pretrain] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

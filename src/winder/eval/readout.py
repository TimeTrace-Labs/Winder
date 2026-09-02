"""Frozen-checkpoint loading and encoding -- the substrate every eval numeric in this package
reads. Promoted from script-local functions into a real, importable, unit-tested library module.

**Source, and a brief-vs-reality discrepancy worth recording up front.** The design brief that
commissioned this module attributed every function here to `scripts/p1_panel_numerics.py` and
`scripts/p3_extras_numerics.py`. Verified against the actual reference-repo source
(`/home/blaised/winder-theory-exp`): `load_model_and_operator`, `read_waveforms`,
`theta_for_frame`, `encode_z`, `encode_hidden` are indeed in `p1_panel_numerics.py`, and
`operator_from_checkpoint` is indeed in `p3_extras_numerics.py` -- but `mean_features`,
`discover_seed_checkpoints`, and `final_step_from_config` actually live in
`scripts/scratch_finale_eval.py`, and `pooled_cells` actually lives in
`scripts/p6_new_coprimary_readouts.py`. Ported from their real locations; the brief's function
list itself was correct; only its script attribution was wrong.

`load_model_and_operator` and `operator_from_checkpoint` are two genuinely different loading
paths, not redundant: the former builds and loads the full `JepaModel` (needed for any encoder
forward pass); the latter constructs and loads ONLY the `HarmonicTransport` operator, skipping
the (much more expensive) encoder entirely -- used wherever only the operator's spectrum or its
`transport()` map is needed (e.g. re-deriving a gain statistic from an already-cached `z`).

Two behavioral details preserved exactly, per the design brief's own call-out:
  - `operator.requires_grad_(False)` after loading: the free arm's `omega` is an `nn.Parameter`,
    so every tensor downstream of `operator.transport(...)` would otherwise carry
    `requires_grad=True` and a later `.numpy()` call raises. Nothing in evaluation differentiates
    through the operator, so switching the whole module off is the correct fix.
  - `model.eval().to(device)` order (not `.to(device).eval()`): verified consistent across both
    `p1_panel_numerics.py::load_model_and_operator` and `p3_extras_numerics.py::build_random_init`
    -- both call `.eval()` before `.to(device)`, so this port preserves the same order rather than
    picking one arbitrarily.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from winder.config import resolve_operator_config
from winder.determinism import generator
from winder.eval.pooling import demodulated_pool, masked_mean_pool
from winder.jepa import checkpoint
from winder.jepa.dataset import EcgWindowItem
from winder.jepa.model import JepaModel, build_jepa
from winder.operators.harmonic import HarmonicTransport
from winder.operators.registry import OPERATOR_REGISTRY

__all__ = [
    "load_model_and_operator",
    "operator_from_checkpoint",
    "preflight_check_checkpoints",
    "assert_lead_stats_matches_checkpoint",
    "read_waveforms",
    "theta_for_frame",
    "encode_z",
    "encode_hidden",
    "mean_features",
    "pooled_cells",
    "discover_seed_checkpoints",
    "final_step_from_config",
]


def load_model_and_operator(
    ckpt_dir: str, *, seed: int, device: torch.device
) -> tuple[JepaModel, HarmonicTransport | None]:
    """Build+strict-load the full `JepaModel` (and, if the checkpoint declares an `arm:` section,
    the matching `HarmonicTransport` operator) from `ckpt_dir`'s own `config.yaml`/`state.pt`.

    `seed` feeds `build_jepa`'s "handshake" generator stream for the FRESH weights this
    constructs before `load_checkpoint` overwrites them in place -- only the resulting module
    structure matters, the fresh init is never read.
    """
    with open(os.path.join(ckpt_dir, checkpoint.CONFIG_FILENAME), encoding="utf-8") as fh:
        config_yaml = fh.read()
    model = build_jepa(
        checkpoint.jepa_config_from_yaml(config_yaml), generator=generator(seed, "handshake")
    )
    arm_node = checkpoint.arm_config_from_yaml(config_yaml)
    operator: HarmonicTransport | None = None
    if arm_node is not None:
        _schema_cls, operator_ctor = OPERATOR_REGISTRY[arm_node.operator_name]
        built = operator_ctor(resolve_operator_config(arm_node))
        assert isinstance(built, HarmonicTransport)
        operator = built
    checkpoint.load_checkpoint(ckpt_dir, model=model, operator=operator)
    model.eval().to(device)
    if operator is not None:
        # The FREE arm's omega is an nn.Parameter, so every tensor downstream of
        # operator.transport carries requires_grad and .numpy() on it raises. Nothing in
        # evaluation differentiates through the operator, so switching the whole module off is
        # the correct fix -- not sprinkling .detach() at each call site.
        operator.requires_grad_(False)
        operator.to(device)
    return model, operator


def operator_from_checkpoint(ckpt_dir: str) -> HarmonicTransport | None:
    """The checkpoint's declared operator, weights loaded, model side skipped entirely -- for
    analyses that need only the operator (its spectrum, or its `transport()` map applied to an
    already-cached `z`), never the encoder."""
    with open(os.path.join(ckpt_dir, checkpoint.CONFIG_FILENAME), encoding="utf-8") as fh:
        arm_node = checkpoint.arm_config_from_yaml(fh.read())
    if arm_node is None:
        return None
    _schema, ctor = OPERATOR_REGISTRY[str(arm_node.operator_name)]
    built = ctor(resolve_operator_config(arm_node))
    assert isinstance(built, HarmonicTransport)
    state = torch.load(
        os.path.join(ckpt_dir, checkpoint.STATE_FILENAME), map_location="cpu", weights_only=False
    )
    built.load_state_dict(state["operator_state_dict"])
    built.requires_grad_(False)  # free-arm omega is an nn.Parameter; see module docstring
    return built


def preflight_check_checkpoints(
    checkpoints: dict[str, str], *, seed: int, device: torch.device
) -> dict[str, str]:
    """Load-only pass over every `checkpoints` entry BEFORE an expensive eval loop -- catches a
    corrupt/incompatible checkpoint (bad state_dict, shape mismatch, missing config.yaml, ...) in
    seconds rather than partway through an hour of encoding/probing.

    Ported from `scripts/p1_panel_numerics.py::preflight_check_checkpoints` (reference repo,
    `/home/blaised/winder-theory-exp`), with the reference script's own `print(...)` progress
    lines dropped: this is now library code, so it reports failures through its return value
    only -- a caller (e.g. a script's `main`) decides whether/how to log them.

    Returns `{name: error_message}` for every entry that failed to load; an empty dict means
    every checkpoint in `checkpoints` loaded cleanly. Diagnostic only, per the reference script's
    own docstring: this is NOT a gate. A checkpoint that fails here still gets a real attempt in
    the caller's own per-checkpoint loop (and vice versa -- this is best-effort, not a guarantee
    the real loop will succeed or fail identically, e.g. a checkpoint that only OOMs once the
    full eval batch runs would pass preflight and fail later, which is expected).
    """
    failed: dict[str, str] = {}
    for name, ckpt_dir in checkpoints.items():
        try:
            model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
            del model, operator
        except Exception as e:  # noqa: BLE001 -- a bad checkpoint can raise almost anything
            failed[name] = f"{type(e).__name__}: {e}"
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return failed


def assert_lead_stats_matches_checkpoint(ckpt_dir: str, lead_stats_path: str) -> None:
    """Raise `AssertionError` unless `lead_stats_path`'s own SHA-256 matches `ckpt_dir`'s own
    `meta.json::lead_stats_sha256` -- the checkpoint's own record of which corpus-normalization
    statistics it was TRAINED with.

    Exists to make a lead-stats/checkpoint mismatch fail loudly rather than silently: evaluating
    a checkpoint's waveforms against the WRONG normalization stats produces plausible-looking but
    quietly-wrong numbers (the same failure class as a dtype mismatch), not a crash -- so this
    must be a real, executed check in an eval driver, not a comment asserting the two happen to
    agree.
    """
    meta_path = os.path.join(ckpt_dir, checkpoint.META_FILENAME)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    recorded = meta.get("lead_stats_sha256")
    if recorded is None:
        raise AssertionError(f"{meta_path} carries no 'lead_stats_sha256' field")
    with open(lead_stats_path, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    if actual != recorded:
        raise AssertionError(
            f"{ckpt_dir}: lead-stats mismatch -- this checkpoint was trained with "
            f"lead_stats_sha256={recorded!r} (per its own meta.json), but {lead_stats_path!r} "
            f"hashes to {actual!r}. Evaluating its waveforms against these stats would silently "
            f"corrupt every downstream number, not raise."
        )


def read_waveforms(dataset: Dataset[EcgWindowItem], batch_size: int = 128) -> torch.Tensor:
    """The dataset's full waveform tensor, `(N, 12, T)`, in dataset order -- one `DataLoader`
    pass with `shuffle=False`, concatenated.

    Typed against `Dataset[EcgWindowItem]` (a widened, structural type), not the reference
    script's concrete `EcgWindowDataset` -- this function only ever reads the `"waveform"` key of
    each item, so any dataset yielding that shape works, including a synthetic one in a test.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: torch.stack([it["waveform"] for it in b]),
    )
    return torch.cat(list(loader), dim=0)


def theta_for_frame(
    frame: pd.DataFrame, theta_by_id: dict[int, np.ndarray], n_tokens: int
) -> torch.Tensor:
    """`(len(frame), n_tokens)` theta lookup by `ecg_id`, NaN where a record has no phase-clock
    entry."""
    theta = np.full((len(frame), n_tokens), np.nan, dtype=np.float32)
    for i, ecg_id in enumerate(frame["ecg_id"].to_numpy()):
        row = theta_by_id.get(int(ecg_id))
        if row is not None:
            theta[i] = row
    return torch.from_numpy(theta)


@torch.no_grad()
def encode_z(
    model: JepaModel, waveforms: torch.Tensor, device: torch.device, bs: int = 128
) -> torch.Tensor:
    """`(N, T, K)` projector output -- the tensor `L_trans` and SIGReg both attach to."""
    out = []
    for start in range(0, waveforms.shape[0], bs):
        batch = waveforms[start : start + bs].to(device)
        out.append(model.projector.forward(model.encoder.forward(batch)).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_hidden(
    model: JepaModel, waveforms: torch.Tensor, device: torch.device, bs: int = 128
) -> torch.Tensor:
    """`(N, T, width)` frozen predictor hidden states -- the probe's actual readout target
    (`winder.eval.probe`'s own module docstring: pooling target is `predictor_hidden_states`,
    not the encoder's local `embed()`)."""
    out = []
    for start in range(0, waveforms.shape[0], bs):
        out.append(model.predictor_hidden_states(waveforms[start : start + bs].to(device)).cpu())
    return torch.cat(out, dim=0)


def mean_features(
    ckpt_dir: str,
    waveforms: dict[str, torch.Tensor],
    thetas: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    """`{split: (N, K)}` z/mean features via `encode_z` + `masked_mean_pool`, chunked per 512
    records so the whole-split token tensor never has to live on the heap at once."""
    model, _operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for split, wf in waveforms.items():
            rows = []
            for start in range(0, len(wf), 512):
                z = encode_z(model, wf[start : start + 512], device)
                rows.append(masked_mean_pool(z, thetas[split][start : start + 512]).numpy())
                del z
            out[split] = np.concatenate(rows, axis=0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def pooled_cells(
    ckpt_dir: str,
    waveforms: dict[str, torch.Tensor],
    thetas: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """`{split: {cell: (N, K)}}` for both readout cells (`z/mean`, `z/demodulated`).

    Encoded and pooled per 512-record chunk: pooling is per-record, so this is numerically
    identical to a whole-split `encode_z` followed by a whole-split pool, and it keeps the
    `(N, T, K)` token tensor off the heap. Raises if the checkpoint declares no transport
    operator -- `z/demodulated` is undefined without one.
    """
    model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    if operator is None:
        raise ValueError(
            f"{ckpt_dir}: no transport operator, so the z/demodulated cell is undefined"
        )
    op_cpu = operator.to("cpu")
    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for split, wf in waveforms.items():
            mean_rows, demod_rows = [], []
            for start in range(0, len(wf), 512):
                z = encode_z(model, wf[start : start + 512], device)
                th = thetas[split][start : start + 512]
                mean_rows.append(masked_mean_pool(z, th).numpy())
                demod_rows.append(demodulated_pool(z, th, op_cpu).numpy())
                del z
            out[split] = {
                "z/mean": np.concatenate(mean_rows, axis=0),
                "z/demodulated": np.concatenate(demod_rows, axis=0),
            }
    del model, operator
    return out


def final_step_from_config(ckpt_dir: str) -> int:
    """The training run's declared total step count, read from the checkpoint's own
    `config.yaml`'s `train.n_steps` field."""
    with open(os.path.join(ckpt_dir, checkpoint.CONFIG_FILENAME), encoding="utf-8") as fh:
        cfg = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.create(fh.read())))
    try:
        return int(cfg["train"]["n_steps"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"{ckpt_dir}/config.yaml has no train.n_steps ({e})") from e


def discover_seed_checkpoints(arm_dir: str) -> dict[int, str]:
    """`{step: ckpt_dir}` for every complete `checkpoint_step<N>` snapshot plus the final
    `checkpoint/` directory under `arm_dir`.

    Snapshot dirs are registered FIRST, then the final `checkpoint/` (whose step is read from
    its own `config.yaml`'s `train.n_steps` via `final_step_from_config`) -- so a step collision
    between the two is always detected, regardless of directory listing order. Raises if no
    complete checkpoint is found at all, or if the final checkpoint's step collides with a
    snapshot's.
    """
    out: dict[int, str] = {}
    final_dir: str | None = None
    for name in sorted(os.listdir(arm_dir)):
        ckpt_dir = os.path.join(arm_dir, name)
        if not os.path.isfile(os.path.join(ckpt_dir, checkpoint.STATE_FILENAME)):
            continue
        if name.startswith("checkpoint_step"):
            out[int(name.removeprefix("checkpoint_step"))] = ckpt_dir
        elif name == "checkpoint":
            final_dir = ckpt_dir
    if final_dir is not None:
        step = final_step_from_config(final_dir)
        if step in out:
            raise ValueError(f"{arm_dir}: final checkpoint step {step} collides with snapshot")
        out[step] = final_dir
    if not out:
        raise ValueError(f"{arm_dir}: no complete checkpoints found")
    return out

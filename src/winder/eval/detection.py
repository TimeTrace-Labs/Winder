"""The time-localised detection/localisation battery: inject a ground-truthed anomaly at a known
time and amplitude, encode, score with every detector x cardiac-clock combination, and aggregate
to within-record AUROC / localisation error / detection latency. Promoted from the reference
repo's `scripts/p4_localisation_numerics.py` (336 lines, `/home/blaised/winder-theory-exp`) into a
real, importable, unit-tested library module, following the same "script becomes a library
module" pattern Phase P5 used for `readout.py`/`tasks.py`/`robustness.py`/`gates.py`.

**Why this exists at all.** `winder.eval.gates.detection_gap_ci` was ported in Phase P5 but left
permanently unusable: nothing in winder-nominal produced its expected input, a `{ckpt}|{anomaly}|
{amp}|{clock}|{detector}|auroc` (+ `|record_index`) per-record dump (`gates.py`'s own module
docstring). `winder.eval.acceptance`/`scripts/eval_suite.py` both flag the gap explicitly rather
than filling it (P9's own docstring: "out of scope, flagged rather than built"). This module is
that fill: `run_checkpoint_detection_battery(..., dump_per_record=True)` emits keys in EXACTLY
that format, so `detection_gap_ci` becomes usable with zero changes to `gates.py`. `cells_for`
(below) closes the loop's other end -- the decode step `gates.py`'s own module docstring left as
"script-specific I/O glue" (`scripts/g2_detection_gap_ci.py::cells_for`, ported here near-
verbatim) that turns this module's flat dump back into `detection_gap_ci`'s nested `{(theta,
detector): {amplitude: (auroc, record_index)}}` argument shape.

**One inlined helper, following this project's own precedent rather than porting a whole module.**
The reference repo's `rpeaks_at_output_rate` lives in `winder.eval.descriptors`, a module this
port does not otherwise need (it also carries patch-window arithmetic and non-causal descriptors
this battery has no use for). `winder.eval.robustness`'s own module docstring made exactly this
call for `heart_rate_bucket` -- "a small, self-contained pure function... inlined here directly
rather than standing up all of `descriptors.py` for one classifier" -- and the same reasoning
applies here: `rpeaks_at_output_rate` is three lines, numpy-only, with no other dependency on the
rest of `descriptors.py`. Ported verbatim below, docstring included, rather than re-derived.

**Design, unchanged from the reference script.** Six perturbation families (`winder.data.perturb.
PERTURBATIONS`) x each family's own amplitude sweep, with `0.0` always prepended as a same-pipeline
sham x two cardiac-phase clocks (`offline`, needs the NEXT R-peak; `causal`, extrapolates from the
PREVIOUS beat, the online-realisable one) x the relevant detector cells (`winder.transport.
localisation`) -- skipping offline-suffixed detectors under causal theta, since "an offline
detector on causal theta" is not a real deployable configuration (reference script's own line
~243, reproduced in `run_checkpoint_detection_battery` below).

**The one deliberate deviation from the reference script, and why.** The reference script's
`sample_theta_grids` hardcodes `decimation_factor=5.0` inline. Here it is a required keyword
argument instead: the commissioning brief for this port specifically calls out confirming that
5.0 against the actual archive's own `decimation_factor` field rather than assuming it, so a
caller must pass whatever `winder.transport.dataset.load_theta_tokens`'s own metadata reports
(`build_detection_cohort` does exactly this) -- a silent decimation-factor drift between the
theta-token archive and this module can no longer be masked by a shared hardcoded constant.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from winder.data.folds import FoldConfig, folds
from winder.data.norm_stats import LeadStats
from winder.data.perturb import PERTURBATIONS
from winder.data.phase import phase_from_rpeaks
from winder.data.ptbxl import MULTIHOT_COLS, load_metadata
from winder.eval.readout import encode_z, load_model_and_operator, read_waveforms
from winder.jepa.dataset import EcgWindowDataset
from winder.jepa.model import JepaModel
from winder.operators.harmonic import HarmonicTransport
from winder.transport.dataset import load_theta_tokens
from winder.transport.localisation import (
    causal_phase_from_rpeaks,
    detection_latency,
    deviation_scores,
    identity_residual_scores,
    localisation_error,
    radial_scores,
    transport_residual_scores,
    within_record_auroc,
)

__all__ = [
    "FS",
    "DEFAULT_CAUSAL_WINDOW",
    "rpeaks_at_output_rate",
    "patch_ms_from_patch_width",
    "sample_theta_grids",
    "token_theta_from_samples",
    "score_all",
    "detection_cell_key",
    "score_one_perturbation",
    "DetectionCohort",
    "build_detection_cohort",
    "run_checkpoint_detection_battery",
    "run_detection_battery",
    "cells_for",
]

#: The decimated model-input rate this whole project tokenises at (`winder.eval.robustness`'s own
#: `theta_variants` uses the same 10 ms/sample convention). `patch_ms_from_patch_width` derives the
#: patch duration from this and a checkpoint's own `patch_width`, rather than a bare hardcoded
#: constant, so a future patch-width change cannot silently desynchronise from the reported ms.
FS = 100.0

#: Tokens of look-back for every causal detector (`winder.transport.localisation`'s `window`
#: argument) -- ~3.2 s at this project's 80 ms/token, matching the reference script's own default.
DEFAULT_CAUSAL_WINDOW = 40


def rpeaks_at_output_rate(rpeaks_native: np.ndarray, decimation_factor: float) -> np.ndarray:
    """Rescale R-peak sample positions from the native acquisition rate onto the decimated
    model-input grid (native / decimation_factor).

    Exact, not an approximation: `winder.data.decimation.decimate_to`'s own contract is "output
    sample n corresponds to input time n / fs_out with no timing shift", so a native-rate sample
    index and a decimated-rate sample index describing the same instant are related by this one
    linear rescale. `winder.data.phase.phase_from_rpeaks`'s theta is itself a ratio of two
    differences of sample indices (`(t - R_i) / (R_{i+1} - R_i)`), so rescaling every index
    (query `t` and every `R`) by the same factor leaves theta exactly unchanged -- decimation
    changes the sampling grid theta is evaluated on, never the theta value at a given instant.
    """
    return np.asarray(rpeaks_native, dtype=np.float64) / float(decimation_factor)


def patch_ms_from_patch_width(patch_width: int, fs: float = FS) -> float:
    """One token's duration in milliseconds at the decimated rate `fs` -- `patch_width=8` at
    `FS=100` gives 80 ms, matching the reference script's own hardcoded `PATCH_MS`, derived here
    instead of hardcoded so a differently-tokenised checkpoint reports the right number."""
    return patch_width * 1000.0 / fs


def sample_theta_grids(
    rpeaks_npz: str, ecg_ids: np.ndarray, n_samples: int, *, decimation_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(N, n_samples)` offline and causal theta at RAW SAMPLE resolution, for phase-restricted
    injection (an ST arc is ~80 ms wide, comparable to one token, so injecting on the token grid
    would smear it across the arc's own width).

    `decimation_factor` is a required keyword, not a hardcoded constant -- see module docstring.
    """
    d = np.load(rpeaks_npz)
    all_ids, offsets, flat = d["ecg_ids"], d["offsets"], d["rpeaks"]
    index = {int(e): i for i, e in enumerate(all_ids)}
    offline = np.full((len(ecg_ids), n_samples), np.nan)
    causal = np.full((len(ecg_ids), n_samples), np.nan)
    for row, ecg in enumerate(ecg_ids):
        i = index.get(int(ecg))
        if i is None:
            continue
        native = flat[offsets[i] : offsets[i + 1]]
        at_rate = rpeaks_at_output_rate(native, decimation_factor)
        offline[row] = phase_from_rpeaks(at_rate, n_samples)[:, 0]
        causal[row] = causal_phase_from_rpeaks(at_rate, n_samples)
    return offline, causal


def token_theta_from_samples(
    sample_theta: np.ndarray, n_tokens: int, patch_width: int
) -> torch.Tensor:
    """Theta at each token's CENTRE sample -- the same convention the transport path uses
    (`winder.eval.descriptors.theta_at_tokens(timestamp="centre")` in the reference repo),
    reproduced here from a sample grid so the offline and causal variants stay on one code path.
    """
    centres = [
        min(int((j + 0.5) * patch_width), sample_theta.shape[1] - 1) for j in range(n_tokens)
    ]
    return torch.from_numpy(sample_theta[:, centres]).float()


def score_all(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    window: int | None,
) -> dict[str, torch.Tensor]:
    """The 7 per-token scores every cell of the battery is built from -- 2 real detectors x 2
    clocks, the phase-blind deviation baseline, and the radial (norm-channel) score at both
    clocks (`winder.transport.localisation`'s own module docstring)."""
    return {
        "transport_offline": transport_residual_scores(z, theta, operator),
        "transport_causal": transport_residual_scores(
            z, theta, operator, causal=True, window=window
        ),
        "identity_offline": identity_residual_scores(z, theta),
        "identity_causal": identity_residual_scores(z, theta, causal=True, window=window),
        "deviation": deviation_scores(z, theta),
        "radial_offline": radial_scores(z, theta),
        "radial_causal": radial_scores(z, theta, causal=True, window=window),
    }


def detection_cell_key(perturbation: str, amplitude: float, theta_kind: str, detector: str) -> str:
    """`"{perturbation}|{amplitude}|{theta}|{detector}"` -- EXACTLY the cell-key format
    `winder.eval.gates.detection_gap_ci`'s own module docstring documents as the suffix after
    `{ckpt}|` in a `localisation_per_record.npz` dump. Used both for this module's own report dict
    and for the per-record dump keys, so the two can never drift apart."""
    return f"{perturbation}|{amplitude}|{theta_kind}|{detector}"


def score_one_perturbation(
    z: torch.Tensor,
    theta_tokens: Mapping[str, torch.Tensor],
    token_mask: torch.Tensor,
    operator: HarmonicTransport,
    *,
    causal_window: int | None,
    patch_ms: float,
) -> dict[str, dict[str, Any]]:
    """Score one already-encoded, already-perturbed record batch against every (theta clock,
    detector) cell, keyed `"{theta}|{detector}"`. Offline-suffixed detectors are skipped under
    causal theta -- an offline detector on causal theta is not a real online configuration
    (reference script's own line ~243).

    Each cell's dict carries `theta`, `detector`, `mean_auroc`, `median_auroc`, `n_records`,
    `localisation`, `latency`, and `_auroc` (the full `within_record_auroc` return, per-record
    AUROCs + their record indices -- consumed by `run_checkpoint_detection_battery` to build the
    per-record dump, and popped before anything is written to a report JSON).

    Factored out from `run_checkpoint_detection_battery` specifically so it is testable against
    synthetic `z`/`theta`/`token_mask`/`operator` with no model, no checkpoint, no GPU.
    """
    out: dict[str, dict[str, Any]] = {}
    for theta_kind, theta in theta_tokens.items():
        scores = score_all(z, theta, operator, causal_window)
        for det, s in scores.items():
            if theta_kind == "causal" and det.endswith("offline"):
                continue
            auroc = within_record_auroc(s, token_mask)
            out[f"{theta_kind}|{det}"] = {
                "theta": theta_kind,
                "detector": det,
                "mean_auroc": auroc["mean_auroc"],
                "median_auroc": auroc["median_auroc"],
                "n_records": auroc["n_records"],
                "localisation": localisation_error(s, token_mask, patch_ms=patch_ms),
                "latency": detection_latency(s, token_mask, patch_ms=patch_ms),
                "_auroc": auroc,
            }
    return out


@dataclass(frozen=True)
class DetectionCohort:
    """Everything one checkpoint's detection battery needs, decoded once and shared across every
    perturbation/amplitude/theta/detector cell.

    `theta_tokens` is `{"offline": ..., "causal": ...}`, each `(N, n_tokens)` -- the TOKEN-grid
    clocks the encoder's own output aligns to. `offline_samples` is the offline clock at RAW
    SAMPLE resolution, `(N, n_samples)` -- `winder.data.perturb`'s injection functions need this
    finer grid to place a lesion inside a phase arc narrower than one token (module docstring).
    """

    frame: pd.DataFrame
    ecg_ids: np.ndarray
    clean: torch.Tensor  # (N, 12, n_samples)
    offline_samples: torch.Tensor  # (N, n_samples)
    theta_tokens: dict[str, torch.Tensor]  # {"offline": (N, n_tokens), "causal": (N, n_tokens)}
    lead_std: torch.Tensor  # (12,)
    n_tokens: int
    patch_width: int
    theta_coverage_offline: float
    theta_coverage_causal: float


def build_detection_cohort(
    data_root: str,
    *,
    fold_config: FoldConfig | None = None,
    frame: pd.DataFrame | None = None,
    n_records: int,
    rpeaks_npz_path: str,
    lead_stats_path: str,
    theta_tokens_path: str,
) -> DetectionCohort:
    """Exactly one of `fold_config` or `frame` must be given.

    `fold_config`: this function derives the eval split itself, via `folds(labeled,
    fold_config)["val"]` -- mirrors the reference script's own `folds(labeled, FoldConfig())
    ["val"].head(args.n_records)`. This NEVER passes `folds()`'s own sealed-fold-release keyword
    as `True`, so `fold_config.test_fold` -- fold 10 by every `FoldConfig` this repo defines -- is
    never read regardless of which `fold_config` a caller passes. This is the path
    `run_detection_battery` and every existing validated call site use; unchanged by this
    parameter's addition.

    `frame`: a CALLER-RESOLVED frame, already the correct eval split. This is the path a
    pre-authorized fold-10 frame must take. Found this session: constructing a `FoldConfig` with
    `val_fold` pointed at the sealed fold (even with `test_fold` relabelled elsewhere) used to
    walk real sealed-fold rows out through `folds()`'s unconditionally-exposed `"val"` key with no
    warning at all -- since patched in `winder.data.folds._check_seal_invariant`, that construction
    now raises unconditionally, so this function can no longer derive a fold-10 split internally
    AT ALL, by design. A caller that has ALREADY cleared the one real gate
    (`winder.data.fold10_authorization.authorized_unseal`) must pass the resulting frame directly
    -- authorization happens exactly once, at that one call site, and the frame flows down from
    there, never re-derived.

    Raises `ValueError`, loudly, on an empty resulting split (the `fold_config` path only) --
    winder-nominal's own default `FoldConfig()` has a deliberately EMPTY `val_fold=0` sentinel
    (`winder.data.folds`'s own module docstring), so calling this with that default produces zero
    records, not a small cohort; a caller reproducing the reference repo's published numbers must
    pass `winder.data.folds.LEGACY_FOLD_CONFIG` instead (Phase P6's own acceptance gate made the
    same substitution for the same reason).
    """
    if (fold_config is None) == (frame is None):
        raise ValueError(
            "exactly one of fold_config or frame must be given -- fold_config derives the split "
            "internally (existing, validated behaviour); frame is a caller-resolved split, the "
            "only path by which an already-authorized fold-10 frame may reach this function"
        )
    frame_was_given = frame is not None
    if frame is None:
        assert fold_config is not None  # narrows for mypy; the xor check above guarantees this
        metadata = load_metadata(data_root)
        labeled = metadata.loc[metadata[list(MULTIHOT_COLS)].sum(axis=1) > 0]
        frame = folds(labeled, fold_config)["val"]
    frame = frame.head(n_records)
    if len(frame) == 0:
        raise ValueError(
            f"detection cohort's eval split is empty (fold_config={fold_config!r}, "
            f"frame_was_given={frame_was_given}) -- "
            "FoldConfig()'s own default val_fold=0 is a deliberate empty sentinel (winder.data."
            "folds module docstring). Pass fold_config=LEGACY_FOLD_CONFIG to reproduce the "
            "reference repo's published detection numbers, or a fold_config with a real "
            "val_fold once one is pre-registered, or pass frame= directly with an "
            "already-resolved, non-empty eval split."
        )

    lead_stats = LeadStats.from_json(lead_stats_path)
    _theta_by_id, theta_meta = load_theta_tokens(theta_tokens_path)
    n_tokens = cast(int, theta_meta["n_tokens"])
    patch_width = cast(int, theta_meta["patch_width"])
    decimation_factor = cast(float, theta_meta["decimation_factor"])
    lead_std = torch.tensor(lead_stats.std_mv, dtype=torch.float32)

    clean = read_waveforms(EcgWindowDataset(frame, data_root, lead_stats=lead_stats))
    ecg_ids = frame["ecg_id"].to_numpy()
    n_samples = clean.shape[2]

    offline_grid, causal_grid = sample_theta_grids(
        rpeaks_npz_path, ecg_ids, n_samples, decimation_factor=decimation_factor
    )
    theta_tokens = {
        "offline": token_theta_from_samples(offline_grid, n_tokens, patch_width),
        "causal": token_theta_from_samples(causal_grid, n_tokens, patch_width),
    }
    offline_samples = torch.from_numpy(offline_grid).float()

    return DetectionCohort(
        frame=frame,
        ecg_ids=ecg_ids,
        clean=clean,
        offline_samples=offline_samples,
        theta_tokens=theta_tokens,
        lead_std=lead_std,
        n_tokens=n_tokens,
        patch_width=patch_width,
        theta_coverage_offline=float(np.isfinite(offline_grid).mean()),
        theta_coverage_causal=float(np.isfinite(causal_grid).mean()),
    )


def run_checkpoint_detection_battery(
    model: JepaModel,
    operator: HarmonicTransport,
    cohort: DetectionCohort,
    device: torch.device,
    *,
    ckpt_name: str,
    causal_window: int = DEFAULT_CAUSAL_WINDOW,
    perturbations: Mapping[str, tuple[Any, tuple[float, ...], str]] = PERTURBATIONS,
    dump_per_record: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One checkpoint's full battery: every perturbation x its own amplitude sweep (0.0 sham
    prepended) x every admissible (theta, detector) cell, via `score_one_perturbation`.

    Returns `(per_ckpt, per_record_dump)`. `per_ckpt` is keyed by `detection_cell_key(...)`, each
    value carrying `perturbation`, `family`, `amplitude`, `theta`, `detector`, `mean_auroc`,
    `median_auroc`, `n_records`, `localisation`, `latency` -- exactly the reference script's own
    report-cell schema. `per_record_dump` is empty unless `dump_per_record=True`, in which case it
    carries `"{ckpt_name}|{cell_key}|auroc"` / `"...|record_index"` entries, ready to feed
    `winder.eval.gates.detection_gap_ci` once a caller parses them into that function's expected
    `{severity: (auroc, record_index)}` shape.
    """
    patch_ms = patch_ms_from_patch_width(cohort.patch_width)
    per_ckpt: dict[str, Any] = {}
    per_record_dump: dict[str, np.ndarray] = {}

    for pert_name, (fn, sweep, family) in perturbations.items():
        for amplitude in (0.0, *sweep):
            pert = fn(  # type: ignore[operator]
                cohort.clean,
                cohort.offline_samples,
                cohort.lead_std,
                amplitude_mv=amplitude,
                n_tokens=cohort.n_tokens,
                patch_width=cohort.patch_width,
            )
            z = encode_z(model, pert.waveform, device)
            cells = score_one_perturbation(
                z,
                cohort.theta_tokens,
                pert.token_mask,
                operator,
                causal_window=causal_window,
                patch_ms=patch_ms,
            )
            for cell in cells.values():
                auroc = cell.pop("_auroc")
                key = detection_cell_key(pert_name, amplitude, cell["theta"], cell["detector"])
                per_ckpt[key] = {
                    "perturbation": pert_name,
                    "family": family,
                    "amplitude": amplitude,
                    **cell,
                }
                if dump_per_record:
                    per_record_dump[f"{ckpt_name}|{key}|auroc"] = np.asarray(
                        auroc["per_record"], dtype=np.float32
                    )
                    per_record_dump[f"{ckpt_name}|{key}|record_index"] = np.asarray(
                        auroc["record_index"], dtype=np.int32
                    )
            del z, pert
    return per_ckpt, per_record_dump


def run_detection_battery(
    *,
    data_root: str,
    roster_dir: str,
    checkpoint_names: Sequence[str],
    rpeaks_npz_path: str,
    lead_stats_path: str,
    theta_tokens_path: str,
    fold_config: FoldConfig,
    n_records: int,
    causal_window: int = DEFAULT_CAUSAL_WINDOW,
    seed: int = 0,
    device: torch.device,
    dump_per_record: bool = False,
    perturbations: Mapping[str, tuple[Any, tuple[float, ...], str]] = PERTURBATIONS,
) -> dict[str, Any]:
    """The full multi-checkpoint battery: build one `DetectionCohort`, then run
    `run_checkpoint_detection_battery` per named roster arm, resolving `"{roster_dir}/{name}/
    checkpoint"` (bare `checkpoint/` only, matching the reference script's own roster convention --
    no `checkpoint_step<N>` snapshot support).

    A checkpoint whose directory is missing, or whose config declares no transport operator, is
    recorded under `result["skipped"][name]` (a reason string) and does NOT stop the remaining
    checkpoints -- matching the reference script's own loud-but-non-fatal `print(...)` warning for
    the same two conditions.

    `perturbations` defaults to the full `winder.data.perturb.PERTURBATIONS` registry; pass a
    subset (e.g. `{"ectopic_beat": PERTURBATIONS["ectopic_beat"]}`) to scope a run to only the
    cells a caller actually needs -- e.g. a test reproducing a handful of published numbers that
    all happen to be `ectopic_beat` cells, without paying for the other five families' sweeps.

    Returns a report dict: `{"config": {...}, "skipped": {...}, <name>: per_ckpt, ...}`, plus
    `"per_record_dump"` (a flat `{key: ndarray}` dict) iff `dump_per_record=True` and at least one
    checkpoint produced any cells.
    """
    cohort = build_detection_cohort(
        data_root,
        fold_config=fold_config,
        n_records=n_records,
        rpeaks_npz_path=rpeaks_npz_path,
        lead_stats_path=lead_stats_path,
        theta_tokens_path=theta_tokens_path,
    )
    report: dict[str, Any] = {
        "config": {
            "n_records": len(cohort.frame),
            "causal_window_tokens": causal_window,
            "patch_ms": patch_ms_from_patch_width(cohort.patch_width),
            "theta_coverage_offline": cohort.theta_coverage_offline,
            "theta_coverage_causal": cohort.theta_coverage_causal,
        },
        "skipped": {},
    }
    per_record_dump: dict[str, np.ndarray] = {}

    for name in checkpoint_names:
        ckpt_dir = os.path.join(roster_dir, name, "checkpoint")
        if not os.path.isdir(ckpt_dir):
            report["skipped"][name] = f"no checkpoint dir at {ckpt_dir} (is --roster-dir right?)"
            continue
        model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
        if operator is None:
            report["skipped"][name] = "no transport operator declared"
            del model
            continue
        operator = operator.to("cpu")
        try:
            per_ckpt, dump = run_checkpoint_detection_battery(
                model,
                operator,
                cohort,
                device,
                ckpt_name=name,
                causal_window=causal_window,
                perturbations=perturbations,
                dump_per_record=dump_per_record,
            )
            report[name] = per_ckpt
            per_record_dump.update(dump)
        finally:
            del model, operator
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if dump_per_record and per_record_dump:
        report["per_record_dump"] = per_record_dump
    return report


def cells_for(
    dump: Any, ckpt: str, anomaly: str
) -> dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Decode a flat per-record dump into `{(theta, detector): {amplitude: (auroc, record_
    index)}}` for one checkpoint and one anomaly -- EXACTLY `winder.eval.gates.detection_gap_ci`'s
    `untrained_cells` shape (its `trained_severity` argument is one specific `(theta, detector)`
    entry of this same shape, picked out by the caller).

    The missing decode step named in this module's own docstring: `run_checkpoint_detection_
    battery(..., dump_per_record=True)` / `run_detection_battery(..., dump_per_record=True)`
    already emit keys in exactly the format this function parses (`detection_cell_key`'s
    `"{ckpt}|{perturbation}|{amplitude}|{theta}|{detector}|auroc"` + `|record_index`); this
    function is the other half, turning that flat dict back into the nested shape `detection_
    gap_ci` actually takes as arguments. Ported near-verbatim from the reference repo's
    `scripts/g2_detection_gap_ci.py::cells_for` (~17 lines) -- the only change is accepting
    `dump` as EITHER a plain `{key: ndarray}` mapping (what this module's own `per_record_dump`
    already is, with no on-disk round trip required) or an npz-like object exposing `.files`
    (`numpy.lib.npyio.NpzFile`, for a caller reading a dump back off disk via `np.load(...)`).
    `dump` is typed `Any` rather than `Mapping` specifically so both duck-types resolve without a
    runtime `isinstance` branch.
    """
    keys = dump.files if hasattr(dump, "files") else list(dump.keys())
    out: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for key in keys:
        if not key.endswith("|auroc"):
            continue
        parts = key[: -len("|auroc")].split("|")
        if len(parts) != 5:
            continue
        ck, anom, amp, clk, det = parts
        if ck != ckpt or anom != anomaly:
            continue
        out[(clk, det)][amp] = (
            dump[key],
            dump[f"{key[: -len('|auroc')]}|record_index"],
        )
    return out

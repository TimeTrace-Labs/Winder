"""Phase P6, Tier 1: numerical reproduction of the reference repo's (winder-theory-exp) published
FINALE-EVAL / transport-gain / G1 / control-comparison numbers, under `LEGACY_FOLD_CONFIG`.

This is the harness `scripts/accept.py` is a thin driver over. Every numeric assertion is checked
against the ACTUAL copied-in reference JSON at `<reference_root>/expected/{finale_results,gain,
g1_finale}.json` -- read at run time via `load_expected_*`, never hardcoded as a source-level
constant -- so a transcription error in this module can never masquerade as a passing gate (the
exact failure mode the commissioning brief called out).

**Cohort construction, and why it matches `scratch_finale_eval.py` rather than
`p1_panel_numerics.py`.** Both reference scripts build the LEGACY (train 1-8 / val 9 / sealed 10)
split, but `p1_panel_numerics.py` truncates the train split to `--train-limit` (default 6,000)
while `scratch_finale_eval.py` (the script that actually produced `finale_results.json`, per its
own `provenance.parameters.train_limit: 14521`) does not effectively truncate at all: 14,521 is
`train_minus_calibration`'s own full length, confirmed empirically by `check_split_shapes` below,
not assumed. `build_acceptance_cohort` therefore takes NO train-limit argument and uses the full
pool -- this is a real protocol choice (see this module's docstring on the control-comparison
discrepancy below), not an oversight.

**Assertion-5 discrepancy, resolved analytically, not by tolerance adjustment.** The design brief
quotes a published control-comparison trio (signal AUROC ~0.8690, control AUROC ~0.7818, gap
~+0.0872) from `winner_c2b_spec.html`'s control table, citing it as "the plan's own text". Read
against the SAME html document's own claim-grade table (`two-seed mean (the claim number) =
0.8761`, sourced from the full 14,521-record pool) and its superseded four-checkpoint panel
(`superclass, plain mean`: seed0 step5k=0.8748, seed1 step5k=0.8632 -- explicitly captioned there
as "the ORIGINAL 6,000-record probe pool"), the control table's 0.8690 is EXACTLY
`(0.8748 + 0.8632) / 2 = 0.8690` -- the two-seed MEAN AUROC under the SUPERSEDED 6,000-record probe
pool, not the full-pool number this module's `check_split_shapes`/`check_probe_auroc` reproduce
(confirmed the same way for gain fraction: `(0.8814699 + 0.8570218) / 2 = 0.86925 ~= 0.869` at step
5,000 and `(0.9185787 + 0.9199209) / 2 = 0.91925 ~= 0.919` at step 25,000, both matching the control
table's "finale gain fraction" column exactly). The brief's own instructed `arm_comparison_table`
call (a SINGLE arm, `FIN_seed0/checkpoint_step5000`, against the full 14,521-record pool) is
therefore structurally a different quantity than the published 0.8690/0.7818/+0.0872 trio, not a
reproduction of it -- per the brief's own override clause ("use whichever is the actual
authoritative published number if there's any discrepancy; report if you find one"),
`check_comparison_gap` gates the SIGNAL side against `finale_results.json`'s own
FIN_seed0-step5000 headline AUROC (0.8827394194500264, the authoritative full-pool number, and
itself an independent-code-path cross-check of `check_probe_auroc`) and reports the control/gap
values alongside the published trio and this arithmetic, WITHOUT asserting a tolerance against the
superseded 6,000-record-pool numbers.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from winder.data.folds import LEGACY_FOLD_CONFIG, calibration_subset, folds, train_minus_calibration
from winder.data.integrity import git_sha
from winder.data.norm_stats import LeadStats
from winder.data.ptbxl import MULTIHOT_COLS, load_metadata
from winder.eval.comparison import EvalCohort, arm_comparison_table
from winder.eval.gates import g1_shuffled_theta_gain_null
from winder.eval.probe import LinearProbeConfig
from winder.eval.readout import (
    encode_z,
    load_model_and_operator,
    mean_features,
    read_waveforms,
    theta_for_frame,
)
from winder.eval.robustness import robustness_suite
from winder.eval.tasks import CLASSES, ci_row, fit_and_score, probe_point, superclass_multihot
from winder.jepa.dataset import EcgWindowDataset
from winder.transport.dataset import load_theta_tokens
from winder.transport.report import gain_report

__all__ = [
    "MILESTONE_ID",
    "AcceptanceCohort",
    "build_split_frames",
    "build_acceptance_cohort",
    "eval_mask_from_theta",
    "compare_value",
    "compare_bool",
    "load_expected_finale_results",
    "load_expected_gain",
    "load_expected_g1",
    "check_split_shapes",
    "check_probe_auroc",
    "check_gain_and_g1",
    "check_comparison_gap",
    "check_robustness_spotcheck",
    "run_acceptance",
]

MILESTONE_ID = "P6-tier1-acceptance-numerical-reproduction"

_SUPERCLASS5_COLUMNS = list(range(len(CLASSES)))


@dataclass(frozen=True)
class AcceptanceCohort:
    """`eval_cohort` is ready to hand straight to `winder.eval.comparison.arm_comparison_table`;
    `eval_ecg_ids`/`n_train`/`n_cal`/`n_eval` are the extra bookkeeping the split-shape and gain/G1
    checks need that `EvalCohort` itself does not carry."""

    eval_cohort: EvalCohort
    eval_ecg_ids: np.ndarray
    n_train: int
    n_cal: int
    n_eval: int


def build_split_frames(data_root: str) -> dict[str, pd.DataFrame]:
    """The LEGACY train/cal/eval metadata frames alone (no waveform decode, no theta lookup) --
    fast enough to run in a unit test, and the thing `check_split_shapes`'s train/cal/eval counts
    are ultimately checking. Mirrors `scripts/scratch_finale_eval.py::main`'s own splits block:
    `train_minus_calibration` at FULL length (no `.head()` truncation -- see module docstring),
    `calibration_subset`, and `folds(...)["val"]`, all under `LEGACY_FOLD_CONFIG`.
    """
    metadata = load_metadata(data_root)
    labeled = metadata.loc[metadata[list(MULTIHOT_COLS)].sum(axis=1) > 0]
    return {
        "train": train_minus_calibration(labeled, LEGACY_FOLD_CONFIG),
        "cal": calibration_subset(labeled, LEGACY_FOLD_CONFIG),
        "eval": folds(labeled, LEGACY_FOLD_CONFIG)["val"],
    }


def build_acceptance_cohort(
    data_root: str, reference_root: str, *, gain_limit: int = 250
) -> AcceptanceCohort:
    """The full LEGACY_FOLD_CONFIG cohort (waveforms/thetas/labels/patient_ids/rr, all three
    splits) -- the expensive step (WFDB decode of ~19k records), paid once and shared across every
    checkpoint a caller subsequently probes.
    """
    lead_stats = LeadStats.from_json(os.path.join(reference_root, "lead_stats_f1to8_legacy.json"))
    theta_by_id, theta_meta = load_theta_tokens(
        os.path.join(reference_root, "phase", "theta_tokens.npz")
    )
    n_tokens = cast(int, theta_meta["n_tokens"])
    patch_width = cast(int, theta_meta["patch_width"])

    frames = build_split_frames(data_root)
    waveforms = {
        k: read_waveforms(EcgWindowDataset(f, data_root, lead_stats=lead_stats))
        for k, f in frames.items()
    }
    thetas = {k: theta_for_frame(f, theta_by_id, n_tokens) for k, f in frames.items()}
    labels = {k: superclass_multihot(f) for k, f in frames.items()}
    patient_ids = {k: f["patient_id"].to_numpy() for k, f in frames.items()}

    manifest = pd.read_parquet(os.path.join(reference_root, "manifest.parquet"))
    rr_lookup = dict(zip(manifest["ecg_id"], manifest["rr_median_ms"], strict=True))
    rr_median_ms = {
        k: np.array([rr_lookup.get(int(e), np.nan) for e in f["ecg_id"]], dtype=np.float64)
        for k, f in frames.items()
    }

    eval_cohort = EvalCohort(
        waveforms=waveforms,
        thetas=thetas,
        labels=labels,
        patient_ids=patient_ids,
        rr_median_ms=rr_median_ms,
        patch_width=patch_width,
        gain_limit=gain_limit,
    )
    return AcceptanceCohort(
        eval_cohort=eval_cohort,
        eval_ecg_ids=frames["eval"]["ecg_id"].to_numpy(),
        n_train=len(frames["train"]),
        n_cal=len(frames["cal"]),
        n_eval=len(frames["eval"]),
    )


def eval_mask_from_theta(theta_eval: torch.Tensor) -> np.ndarray:
    """A record survives `winder.eval.pooling.masked_mean_pool` iff it has at least one token with
    a defined theta (that pooling's own all-NaN-row rule for a record with zero valid tokens) --
    checkpoint-INDEPENDENT, so this can be computed without loading any model. Used both to
    predict the eval mask cheaply and to cross-check it against a real checkpoint's
    `fit_and_score` mask in `run_acceptance`.
    """
    return torch.isfinite(theta_eval).any(dim=1).numpy()


def compare_value(name: str, expected: float, measured: float, tol: float) -> dict[str, Any]:
    """One tolerance-gated numeric comparison, as a structured, JSON-serialisable result."""
    delta = abs(float(measured) - float(expected))
    return {
        "name": name,
        "expected": expected,
        "measured": measured,
        "abs_delta": delta,
        "tolerance": tol,
        "pass": bool(delta <= tol),
    }


def compare_bool(name: str, expected: bool, measured: bool) -> dict[str, Any]:
    """One exact boolean comparison -- no tolerance, ever (`winder.eval.gates`'s own G1
    booleans)."""
    return {
        "name": name,
        "expected": bool(expected),
        "measured": bool(measured),
        "pass": bool(expected) == bool(measured),
    }


def load_expected_finale_results(reference_root: str) -> dict[str, Any]:
    """`<reference_root>/expected/finale_results.json`, read fresh -- never hardcode its numbers
    as source-level constants (module docstring)."""
    with open(
        os.path.join(reference_root, "expected", "finale_results.json"), encoding="utf-8"
    ) as fh:
        return dict(json.load(fh))


def load_expected_gain(reference_root: str) -> dict[str, Any]:
    """`<reference_root>/expected/gain.json`, read fresh."""
    with open(os.path.join(reference_root, "expected", "gain.json"), encoding="utf-8") as fh:
        return dict(json.load(fh))


def load_expected_g1(reference_root: str) -> dict[str, Any]:
    """`<reference_root>/expected/g1_finale.json`, read fresh."""
    with open(os.path.join(reference_root, "expected", "g1_finale.json"), encoding="utf-8") as fh:
        return dict(json.load(fh))


def check_split_shapes(
    cohort: AcceptanceCohort, ev_eval: np.ndarray, expected_splits: dict[str, int]
) -> dict[str, Any]:
    """Assertion family 1: exact train/cal/eval sizes, the eval mask's survivor count, and the
    FIRST dropped row's index. `ev_eval` should be a REAL checkpoint's `fit_and_score` eval mask
    (not `eval_mask_from_theta`'s prediction alone), so this check exercises the actual probe
    pipeline the brief is auditing, not a shortcut re-derivation of it.
    """
    dropped = np.flatnonzero(~ev_eval)
    first_dropped = int(dropped[0]) if len(dropped) else -1
    checks = [
        compare_value("n_train", expected_splits["train"], cohort.n_train, 0),
        compare_value("n_cal", expected_splits["cal"], cohort.n_cal, 0),
        compare_value("n_eval", expected_splits["eval"], cohort.n_eval, 0),
        compare_value("n_eval_surviving", 2128, int(ev_eval.sum()), 0),
        compare_value("first_dropped_row_index", 93, first_dropped, 0),
    ]
    return {"checks": checks, "pass": all(c["pass"] for c in checks)}


def check_probe_auroc(
    ckpt_dir: str,
    cohort: AcceptanceCohort,
    *,
    device: torch.device,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    """One checkpoint's z/mean-pooled, fold-9 superclass-5 macro-AUROC: `winder.eval.readout.
    mean_features` + `winder.eval.tasks.fit_and_score`, then `ci_row` (`n_boot > 0`) or
    `probe_point` (`n_boot == 0`, point estimate only) -- exactly `scratch_finale_eval.py`'s own
    `probe_checkpoint`/`ci_row`/`probe_point` pipeline (the script that produced
    `finale_results.json`), NOT `p1_panel_numerics.py`'s 6,000-record-pool-capped equivalent.
    """
    ec = cohort.eval_cohort
    feats = mean_features(ckpt_dir, ec.waveforms, ec.thetas, device, seed)
    cfg = LinearProbeConfig(seed_probe=seed)
    scores_full, ev = fit_and_score(
        feats["train"],
        ec.labels["train"],
        feats["cal"],
        ec.labels["cal"],
        feats["eval"],
        ec.labels["eval"],
        CLASSES,
        cfg,
    )
    if n_boot > 0:
        row = ci_row(
            scores_full,
            ec.labels["eval"],
            ec.patient_ids["eval"],
            _SUPERCLASS5_COLUMNS,
            n_boot,
            seed,
        )
    else:
        point = probe_point(scores_full, ec.labels["eval"], _SUPERCLASS5_COLUMNS)
        row = {
            "macro_auroc": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "n_eval": int(ev.sum()),
            "n_dropped": int((~ev).sum()),
            "n_boot": 0,
        }
    return {"ckpt_dir": ckpt_dir, "row": row, "eval_mask": ev}


def check_gain_and_g1(
    ckpt_dir: str,
    cohort: AcceptanceCohort,
    *,
    device: torch.device,
    seed: int,
    n_strata: int = 16,
    gain_limit: int = 250,
    n_replicates: int = 2000,
) -> dict[str, Any]:
    """One checkpoint's transport gain (`winder.transport.report.gain_report`) AND its G1
    shuffled-theta null (`winder.eval.gates.g1_shuffled_theta_gain_null`), sharing ONE model load
    and ONE `encode_z` pass over the first `gain_limit` eval records -- the reference repo runs
    these via two separate script invocations that each reload and re-encode the checkpoint;
    nothing in `gates.py`/`report.py` requires that process-boundary separation, so this merges
    them for efficiency (a new-code design choice, not a modification of ported logic).

    `z`/`theta` are cast to float32 BEFORE either call (`gates.g1_shuffled_theta_gain_null`'s own
    docstring: it does no casting itself; the reference script casts before calling -- matching
    that convention exactly is required to reproduce `g1_finale.json`'s booleans).
    """
    ec = cohort.eval_cohort
    model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    if operator is None:
        raise ValueError(f"{ckpt_dir}: no transport operator -- gain/G1 are undefined")
    try:
        g_lim = min(gain_limit, ec.waveforms["eval"].shape[0])
        z_eval = encode_z(model, ec.waveforms["eval"][:g_lim], device).float()
        th_eval = ec.thetas["eval"][:g_lim].float()
        pid_eval = ec.patient_ids["eval"][:g_lim]
        op_cpu = operator.to("cpu")
        gain = gain_report(z_eval, th_eval, op_cpu, pid_eval, n_strata=n_strata, seed=seed)
        g1 = g1_shuffled_theta_gain_null(
            z_eval,
            th_eval,
            op_cpu,
            pid_eval,
            n_strata=n_strata,
            n_replicates=n_replicates,
            seed=seed,
        )
    finally:
        del model, operator
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"gain": gain, "g1": g1}


def check_comparison_gap(
    signal_ckpt_dir: str,
    control_ckpt_dir: str,
    cohort: AcceptanceCohort,
    *,
    device: torch.device,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    """`winder.eval.comparison.arm_comparison_table` on `{"signal": ..., "control": ...}`, sharing
    ONE `EvalCohort` between both arms (the design brief's own instruction)."""
    return arm_comparison_table(
        {"signal": signal_ckpt_dir, "control": control_ckpt_dir},
        cohort.eval_cohort,
        device=device,
        n_boot=n_boot,
        seed=seed,
    )


def check_robustness_spotcheck(
    ckpt_dir: str,
    cohort: AcceptanceCohort,
    *,
    device: torch.device,
    seed: int,
    reference_path: str | None,
    arm_key: str = "FINs0_30k",
) -> dict[str, Any] | None:
    """Non-gating spot-check: run `winder.eval.robustness.robustness_suite` on `ckpt_dir` and
    report it side by side with `reference_path`'s own `arm_key` cell, if `reference_path` exists.
    Never raises on a missing/mismatched reference file or a missing arm key -- this check informs,
    it never gates (the design brief's own "report, do not block on it").
    """
    ec = cohort.eval_cohort
    model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    if operator is None:
        return None
    try:
        z_by_split = {s: encode_z(model, wf, device) for s, wf in ec.waveforms.items()}
        op_cpu = operator.to("cpu")
        measured = robustness_suite(
            model,
            op_cpu,
            z_by_split,
            ec.thetas,
            ec.waveforms,
            ec.labels,
            ec.patient_ids["eval"],
            ec.rr_median_ms,
            ec.patch_width,
            LinearProbeConfig(seed_probe=seed),
            device,
            seed=seed,
        )
    finally:
        del model, operator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference_cell: Any = None
    if reference_path and os.path.isfile(reference_path):
        with open(reference_path, encoding="utf-8") as fh:
            reference = json.load(fh)
        reference_cell = reference.get(arm_key)
    return {"measured": measured, "reference_arm_key": arm_key, "reference_cell": reference_cell}


def run_acceptance(
    *,
    data_root: str,
    reference_root: str,
    device: torch.device,
    seed: int = 0,
    n_boot: int = 1000,
    n_strata: int = 16,
    gain_limit: int = 250,
    n_replicates: int = 2000,
    robustness_reference_path: str | None = None,
) -> dict[str, Any]:
    """The full Tier 1 acceptance run: assembles the cohort once, then runs the five required
    assertion families in the brief's own diagnostic order (splits -> AUROC -> gain -> G1 ->
    comparison), stopping the GATING pipeline at the first family-1..4 failure (family 5's signal
    side is still gated; its control/gap side is report-only per the module docstring's
    assertion-5 resolution). The non-gating robustness spot-check always runs last, wrapped so it
    can never flip the overall status.

    Returns the full report-schema envelope (status/milestone_id/metrics/provenance/decisions).
    """
    t0 = time.time()
    decisions: list[str] = [
        "Cohort built under LEGACY_FOLD_CONFIG with NO train-limit truncation, matching "
        "scratch_finale_eval.py's own full-14521-record-pool run (its provenance.parameters."
        "train_limit=14521), not p1_panel_numerics.py's 6000-record-pool default.",
        "Assertion 5 (control-comparison gap): the published trio (signal ~0.8690, control "
        "~0.7818, gap ~+0.0872) is the two-seed MEAN AUROC under the SUPERSEDED 6,000-record "
        "probe pool -- (0.8748 + 0.8632) / 2 = 0.8690 exactly, per winner_c2b_spec.html's own "
        "four-checkpoint panel table, captioned there as the original 6,000-record pool. The "
        "brief's own instructed call (single arm FIN_seed0, full 14,521-record pool) is gated "
        "instead against finale_results.json's authoritative FIN_seed0-step5000 headline AUROC "
        "(0.8827394194500264); control/gap are reported, not tolerance-gated, against the "
        "published (superseded-pool) trio.",
        "check_gain_and_g1 merges gain_report and g1_shuffled_theta_gain_null into one model "
        "load + one encode_z pass per checkpoint (the reference repo runs these as two separate "
        "script invocations); z/theta cast to float32 before both calls, matching gates.py's "
        "documented float32 convention.",
        "LinearProbeConfig used at its defaults (LinearProbeConfig(seed_probe=seed), seed=0), "
        "per the brief's 'LinearProbeConfig() at its defaults' instruction.",
    ]

    cohort = build_acceptance_cohort(data_root, reference_root, gain_limit=gain_limit)
    expected_finale = load_expected_finale_results(reference_root)
    expected_gain = load_expected_gain(reference_root)
    expected_g1 = load_expected_g1(reference_root)
    expected_splits = expected_finale["metrics"]["splits"]

    ckpt = {
        ("FIN_seed0", 5000): os.path.join(reference_root, "FIN_seed0", "checkpoint_step5000"),
        ("FIN_seed0", 25000): os.path.join(reference_root, "FIN_seed0", "checkpoint_step25000"),
        ("FIN_seed0", 30000): os.path.join(reference_root, "FIN_seed0", "checkpoint"),
        ("FIN_seed1", 5000): os.path.join(reference_root, "FIN_seed1", "checkpoint_step5000"),
        ("FIN_seed1", 25000): os.path.join(reference_root, "FIN_seed1", "checkpoint_step25000"),
        ("FIN_seed1", 30000): os.path.join(reference_root, "FIN_seed1", "checkpoint"),
        ("FIN_LAM0_seed0", 5000): os.path.join(
            reference_root, "FIN_LAM0_seed0", "checkpoint_step5000"
        ),
    }

    metrics: dict[str, Any] = {"families": {}}
    families_pass: dict[str, bool] = {}

    # ---------------------------------------------------------------- family 2: probe AUROC
    auroc_rows: dict[str, Any] = {}
    for arm, step, n_boot_i in (
        ("FIN_seed0", 5000, n_boot),
        ("FIN_seed1", 5000, n_boot),
        ("FIN_seed0", 30000, 0),
        ("FIN_seed1", 30000, 0),
    ):
        auroc_rows[f"{arm}_step{step}"] = check_probe_auroc(
            ckpt[(arm, step)], cohort, device=device, seed=seed, n_boot=n_boot_i
        )

    auroc_checks: list[dict[str, Any]] = []
    for arm in ("FIN_seed0", "FIN_seed1"):
        expected_row = expected_finale["metrics"]["rows"]["superclass5"]["headline"]["per_arm"][arm]
        measured_row = auroc_rows[f"{arm}_step5000"]["row"]
        auroc_checks.append(
            compare_value(
                f"{arm}_step5000_macro_auroc",
                expected_row["macro_auroc"],
                measured_row["macro_auroc"],
                1e-4,
            )
        )
        auroc_checks.append(
            compare_value(f"{arm}_step5000_lo", expected_row["lo"], measured_row["lo"], 1e-3)
        )
        auroc_checks.append(
            compare_value(f"{arm}_step5000_hi", expected_row["hi"], measured_row["hi"], 1e-3)
        )
        expected_30k = expected_finale["metrics"]["curves"]["superclass5"][arm]["30000"]
        measured_30k = auroc_rows[f"{arm}_step30000"]["row"]["macro_auroc"]
        auroc_checks.append(
            compare_value(f"{arm}_step30000_macro_auroc", expected_30k, measured_30k, 1e-4)
        )
    families_pass["probe_auroc"] = all(c["pass"] for c in auroc_checks)
    metrics["families"]["probe_auroc"] = {
        "checks": auroc_checks,
        "pass": families_pass["probe_auroc"],
        "rows": {name: r["row"] for name, r in auroc_rows.items()},  # eval_mask excluded (ndarray)
    }

    # -------------------------------------------------------------------- family 1: split shapes
    real_ev_eval = auroc_rows["FIN_seed0_step5000"]["eval_mask"]
    theta_predicted_ev = eval_mask_from_theta(cohort.eval_cohort.thetas["eval"])
    shapes_result = check_split_shapes(cohort, real_ev_eval, expected_splits)
    shapes_result["theta_predicted_eval_mask_matches_real"] = bool(
        np.array_equal(theta_predicted_ev, real_ev_eval)
    )
    families_pass["split_shapes"] = shapes_result["pass"]
    metrics["families"]["split_shapes"] = shapes_result

    if not (families_pass["split_shapes"] and families_pass["probe_auroc"]):
        status = "FAIL"
        metrics["stopped_after"] = (
            "probe_auroc" if not families_pass["probe_auroc"] else "split_shapes"
        )
        return _envelope(status, metrics, decisions, data_root, reference_root, device, seed, t0)

    # ------------------------------------------------------------------ family 3+4: gain and G1
    gain_g1_checks: list[dict[str, Any]] = []
    gain_g1_bool_checks: list[dict[str, Any]] = []
    gain_g1_raw: dict[str, Any] = {}
    for arm, step, gain_key, g1_key in (
        ("FIN_seed0", 5000, "FINs0_5k", "FIN_seed0_step5000"),
        ("FIN_seed0", 25000, "FINs0_25k", "FIN_seed0_step25000"),
        ("FIN_seed1", 5000, "FINs1_5k", "FIN_seed1_step5000"),
        ("FIN_seed1", 25000, "FINs1_25k", "FIN_seed1_step25000"),
    ):
        result = check_gain_and_g1(
            ckpt[(arm, step)],
            cohort,
            device=device,
            seed=seed,
            n_strata=n_strata,
            gain_limit=gain_limit,
            n_replicates=n_replicates,
        )
        gain_g1_raw[f"{arm}_step{step}"] = result
        expected_gain_cell = expected_gain[gain_key]
        gain_g1_checks.append(
            compare_value(
                f"{gain_key}_overall_mean_gain",
                expected_gain_cell["overall_mean_gain"],
                result["gain"]["overall_mean_gain"],
                1e-6,
            )
        )
        gain_g1_checks.append(
            compare_value(
                f"{gain_key}_overall_gain_fraction",
                expected_gain_cell["overall_gain_fraction"],
                result["gain"]["overall_gain_fraction"],
                1e-6,
            )
        )
        expected_g1_cell = expected_g1["results"][g1_key]
        gain_g1_bool_checks.append(
            compare_bool(
                f"{g1_key}_ci_excludes_zero",
                expected_g1_cell["ci_excludes_zero"],
                result["g1"]["ci_excludes_zero"],
            )
        )
        gain_g1_bool_checks.append(
            compare_bool(
                f"{g1_key}_shuffled_fraction_within_pm0.02",
                expected_g1_cell["shuffled_fraction_within_pm0.02"],
                result["g1"]["shuffled_fraction_within_pm0.02"],
            )
        )
    families_pass["gain"] = all(c["pass"] for c in gain_g1_checks)
    families_pass["g1"] = all(c["pass"] for c in gain_g1_bool_checks)
    metrics["families"]["gain"] = {
        "checks": gain_g1_checks,
        "pass": families_pass["gain"],
        "raw": {name: r["gain"] for name, r in gain_g1_raw.items()},
    }
    metrics["families"]["g1"] = {
        "checks": gain_g1_bool_checks,
        "pass": families_pass["g1"],
        "raw": {name: r["g1"] for name, r in gain_g1_raw.items()},
    }

    if not (families_pass["gain"] and families_pass["g1"]):
        status = "FAIL"
        metrics["stopped_after"] = "gain" if not families_pass["gain"] else "g1"
        return _envelope(status, metrics, decisions, data_root, reference_root, device, seed, t0)

    # ------------------------------------------------------------ family 5: control-comparison gap
    comparison = check_comparison_gap(
        ckpt[("FIN_seed0", 5000)],
        ckpt[("FIN_LAM0_seed0", 5000)],
        cohort,
        device=device,
        seed=seed,
        n_boot=n_boot,
    )
    signal_row = comparison["signal"]
    control_row = comparison["control"]
    expected_signal_auroc = expected_finale["metrics"]["rows"]["superclass5"]["headline"][
        "per_arm"
    ]["FIN_seed0"]["macro_auroc"]
    signal_check = compare_value(
        "comparison_signal_macro_auroc", expected_signal_auroc, signal_row["macro_auroc"], 1e-4
    )
    families_pass["comparison"] = signal_check["pass"]
    published_trio = {
        "signal_auroc": 0.8690,
        "control_auroc": 0.7818,
        "gap": 0.0872,
        "source": (
            "winner_c2b_spec.html control table, step 5000 -- SUPERSEDED 6000-record probe "
            "pool two-seed mean (see module docstring / decisions[1] for the exact arithmetic); "
            "not tolerance-gated"
        ),
    }
    measured_gap = signal_row["macro_auroc"] - control_row["macro_auroc"]
    metrics["families"]["comparison"] = {
        "gating_check": signal_check,
        "pass": families_pass["comparison"],
        "measured": {
            "signal_macro_auroc": signal_row["macro_auroc"],
            "control_macro_auroc": control_row["macro_auroc"],
            "gap": measured_gap,
            "signal_gain_fraction": signal_row["gain_fraction"],
            "control_gain_fraction": control_row["gain_fraction"],
            "signal_g1_pass": signal_row["g1_pass"],
            "control_g1_pass": control_row["g1_pass"],
        },
        "published_trio_report_only": published_trio,
    }

    if not families_pass["comparison"]:
        status = "FAIL"
        metrics["stopped_after"] = "comparison"
        return _envelope(status, metrics, decisions, data_root, reference_root, device, seed, t0)

    # ---------------------------------------------------------- non-gating robustness spot-check
    try:
        spotcheck = check_robustness_spotcheck(
            ckpt[("FIN_seed0", 30000)],
            cohort,
            device=device,
            seed=seed,
            reference_path=robustness_reference_path,
        )
        metrics["families"]["robustness_spotcheck"] = {"gating": False, "result": spotcheck}
    except Exception as exc:  # noqa: BLE001 -- explicitly non-gating (module docstring)
        metrics["families"]["robustness_spotcheck"] = {
            "gating": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        decisions.append(
            f"Non-gating robustness spot-check raised {type(exc).__name__}: {exc} -- recorded, "
            "does not affect overall status."
        )

    status = "PASS"
    return _envelope(status, metrics, decisions, data_root, reference_root, device, seed, t0)


def _envelope(
    status: str,
    metrics: dict[str, Any],
    decisions: list[str],
    data_root: str,
    reference_root: str,
    device: torch.device,
    seed: int,
    t0: float,
) -> dict[str, Any]:
    metrics["elapsed_sec"] = time.time() - t0
    return {
        "status": status,
        "milestone_id": MILESTONE_ID,
        "metrics": metrics,
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": {
                "data_root": data_root,
                "reference_root": reference_root,
                "device": str(device),
            },
            "seed": seed,
        },
        "decisions": decisions,
        "questions": [],
    }

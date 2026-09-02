"""Phase P9: post-training eval suite over the 4 real checkpoints written by Phase P8
(`signal_seed0`, `signal_seed1`, `control_seed0`, `control_seed1`, each 30,000 steps, 12
checkpoints).

**Every number this script produces is fold-9-contaminated and diagnostic only.** Fold 9 is BOTH
the training data for these 4 checkpoints AND the only fold `LEGACY_FOLD_CONFIG` can construct
without opening the sealed fold 10 (`winder.data.folds`'s own module docstring). This is the
plan's own accepted design, not a bug: every JSON this script writes carries `"split_status":
"train_contaminated"` and `"headline": false` at its top level, unconditionally. Fold 10 is never
touched here (`tests/test_folds.py::test_no_call_site_unseals` enforces that `folds()`'s own
sealed-fold-release keyword, set to `True`, appears as a call site nowhere in `src/`/`scripts/`;
nothing in this module calls it).

**Task 1 -- preflight.** `preflight_check_checkpoints` (`winder.eval.readout`) is run over all 48
checkpoint dirs (4 arms x 12 steps) before any expensive encoding, so a corrupt/incompatible
checkpoint is caught in seconds, not partway through an hour of probing.

**The lead-stats trap.** These checkpoints were TRAINED with folds-1-9 normalization stats
(`artifacts/lead_stats_f1to9.json`), not Phase P6's LEGACY folds-1-8 stats
(`artifacts/reference/lead_stats_f1to8_legacy.json`, deliberately reproducing the OLD protocol
for the acceptance gate). `winder.eval.readout.assert_lead_stats_matches_checkpoint` is run
against every one of the 48 checkpoint dirs, comparing the checkpoint's own recorded
`meta.json::lead_stats_sha256` against the ACTUAL sha256 of the lead-stats file this script is
about to load waveforms with -- a real, executed assertion, not a comment. A mismatch on any
checkpoint STOPS the run (`status: "FAIL"`) before any encoding happens, rather than silently
producing plausible-looking numbers computed against the wrong corpus statistics.

**Task 2 -- the eval battery, "focus"-scoped like `p1_panel_numerics.py`'s own convention.** The
FULL battery (probe AUROC with a patient-clustered CI + robustness suite + G1 gate + transport
gain/geometry report) runs only at `checkpoint_step5000` and the final checkpoint, for all 4 arms
(8 full-battery runs). A cheap AUROC-ONLY point-estimate curve (the `z/mean` cell, no bootstrap,
no robustness) runs across all 12 steps for all 4 arms (48 points) -- mirroring
`finale_results.json`'s own "curves" convention (point macro-AUROC per step, no CI). This keeps
wall-clock sane while still answering "how does it track over training" cheaply and "what's the
full picture" at the two steps that matter most.

Out of scope, flagged rather than built: a detection/localisation battery. `winder.eval.gates.
detection_gap_ci` needs a severity-swept anomaly-injection dump per (clock, detector) cell that
was never ported into winder-nominal (`gates.py`'s own module docstring: that dump format
"does not exist anywhere in winder-nominal"). Per this brief's own instruction, this gap is
reported, not filled with new plumbing.

**Task 3 -- the 4-arm comparison table.** `winder.eval.comparison.arm_comparison_table` (the
function Phase P6 validated against the published ~0.087 control-gap numbers BEFORE it was ever
pointed at new arms) is called on `{signal_seed0, signal_seed1, control_seed0, control_seed1}` at
both focus steps. This is the first signal-vs-control comparison run under one consistent
protocol in this whole project -- still fold-9-contaminated, same caveat as everything else here.
Reported plainly: no framing as resolving Phase P6's own control-gap protocol-mismatch finding,
no crowned headline step (`winder.eval.tasks.select_step` is deliberately NOT called here -- that
question is parked for the CTO, not decided by this script). `pairwise_deltas` is likewise not
called: a signal-vs-control delta CI is exactly the parked framing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from winder.data.integrity import git_sha
from winder.data.norm_stats import LeadStats
from winder.eval.acceptance import build_split_frames
from winder.eval.comparison import EvalCohort, arm_comparison_table
from winder.eval.gates import g1_shuffled_theta_gain_null
from winder.eval.pooling import masked_mean_pool
from winder.eval.probe import LinearProbeConfig
from winder.eval.readout import (
    assert_lead_stats_matches_checkpoint,
    discover_seed_checkpoints,
    encode_z,
    load_model_and_operator,
    mean_features,
    preflight_check_checkpoints,
    read_waveforms,
    theta_for_frame,
)
from winder.eval.robustness import robustness_suite
from winder.eval.tasks import CLASSES, ci_row, fit_and_score, probe_point, superclass_multihot
from winder.jepa.dataset import EcgWindowDataset
from winder.paths import default_data_root
from winder.transport.dataset import load_theta_tokens
from winder.transport.report import gain_report, geometry_report, operator_report

MILESTONE_ID = "P9-post-training-eval-suite"

#: Literal constants, per the design brief: every JSON this script emits carries these two keys
#: at its TOP level, unconditionally -- not just documented in a docstring.
SPLIT_STATUS = "train_contaminated"
HEADLINE = False

#: The 4 real arms Phase P8 trained -- never hardcoded elsewhere in this module (this is the one
#: place the roster naming convention lives, matching `winder.ablations`'s own convention).
DEFAULT_ARMS: tuple[str, ...] = ("signal_seed0", "signal_seed1", "control_seed0", "control_seed1")

_CLASS_COLUMNS = list(range(len(CLASSES)))


# ============================================================================ cohort construction


def build_p9_cohort(
    data_root: str, artifacts_dir: str, lead_stats_path: str, *, train_limit: int = 0
) -> tuple[EvalCohort, dict[str, Any]]:
    """The `LEGACY_FOLD_CONFIG` eval cohort (train/cal/eval metadata splits), waveforms decoded
    against `lead_stats_path` -- winder-nominal's OWN `lead_stats_f1to9.json`, matching what the
    4 real checkpoints were actually trained with (the lead-stats trap this module's docstring
    describes), NOT `winder.eval.acceptance.build_acceptance_cohort`'s hardcoded legacy
    folds-1-8 stats. `winder.eval.acceptance.build_split_frames` is reused for the metadata split
    itself (stats-free, so no trap there); everything downstream of it (waveform decode, theta
    lookup, manifest RR lookup) is assembled fresh here against winder-nominal's own top-level
    `manifest.parquet`/`phase/theta_tokens.npz`, not `artifacts/reference/`'s copied-in old-repo
    artifacts (those exist only to reproduce the OLD protocol's published numbers in Phase P6).

    `train_limit` (0 = no truncation) exists for fast tests only; the real P9 run uses the full
    pool, matching `build_acceptance_cohort`'s own no-truncation precedent (not
    `p1_panel_numerics.py`'s 6,000-record-pool default).
    """
    frames = build_split_frames(data_root)
    if train_limit:
        frames = {**frames, "train": frames["train"].head(train_limit)}

    lead_stats = LeadStats.from_json(lead_stats_path)
    theta_by_id, theta_meta = load_theta_tokens(
        os.path.join(artifacts_dir, "phase", "theta_tokens.npz")
    )
    n_tokens = cast(int, theta_meta["n_tokens"])
    patch_width = cast(int, theta_meta["patch_width"])

    waveforms = {
        k: read_waveforms(EcgWindowDataset(f, data_root, lead_stats=lead_stats))
        for k, f in frames.items()
    }
    thetas = {k: theta_for_frame(f, theta_by_id, n_tokens) for k, f in frames.items()}
    labels = {k: superclass_multihot(f) for k, f in frames.items()}
    patient_ids = {k: f["patient_id"].to_numpy() for k, f in frames.items()}

    manifest = pd.read_parquet(os.path.join(artifacts_dir, "manifest.parquet"))
    rr_lookup = dict(zip(manifest["ecg_id"], manifest["rr_median_ms"], strict=True))
    rr_median_ms = {
        k: np.array([rr_lookup.get(int(e), np.nan) for e in f["ecg_id"]], dtype=np.float64)
        for k, f in frames.items()
    }

    cohort = EvalCohort(
        waveforms=waveforms,
        thetas=thetas,
        labels=labels,
        patient_ids=patient_ids,
        rr_median_ms=rr_median_ms,
        patch_width=patch_width,
        gain_limit=250,
    )
    bookkeeping = {
        "n_train": len(frames["train"]),
        "n_cal": len(frames["cal"]),
        "n_eval": len(frames["eval"]),
        "lead_stats_path": lead_stats_path,
    }
    return cohort, bookkeeping


# =================================================================================== focus steps


def resolve_focus_steps(focus_labels: list[str], steps: dict[int, str]) -> dict[str, int]:
    """Map each requested focus label (`"5000"`, `"final"`, ...) to an actual step key present in
    `steps`. `"final"` resolves to `max(steps)` -- `discover_seed_checkpoints`'s own final-vs-
    snapshot collision check already guarantees there is exactly one candidate for the true final
    step. Raises `ValueError` (named, actionable) if a literal step label is not present."""
    resolved: dict[str, int] = {}
    for label in focus_labels:
        if label == "final":
            resolved[label] = max(steps)
        else:
            step = int(label)
            if step not in steps:
                raise ValueError(f"requested focus step {step} not found among {sorted(steps)}")
            resolved[label] = step
    return resolved


# ==================================================================== Task 2a: AUROC-only curves


def auroc_curve_for_arm(
    steps: dict[int, str], cohort: EvalCohort, *, device: torch.device, seed: int
) -> dict[str, Any]:
    """One `z/mean`-cell, point-estimate (no bootstrap) macro-AUROC per step -- the cheap "how
    does it track over training" curve, matching `finale_results.json`'s own curves convention.
    A per-step failure is recorded as `{"error": ...}` and does NOT stop the remaining steps."""
    cfg = LinearProbeConfig(seed_probe=seed)
    curve: dict[str, Any] = {}
    for step in sorted(steps):
        try:
            feats = mean_features(steps[step], cohort.waveforms, cohort.thetas, device, seed)
            scores_full, ev = fit_and_score(
                feats["train"],
                cohort.labels["train"],
                feats["cal"],
                cohort.labels["cal"],
                feats["eval"],
                cohort.labels["eval"],
                CLASSES,
                cfg,
            )
            point = probe_point(scores_full, cohort.labels["eval"], _CLASS_COLUMNS)
            curve[str(step)] = {
                "cell": "z/mean",
                "macro_auroc": point,
                "n_eval": int(ev.sum()),
                "n_dropped": int((~ev).sum()),
            }
        except Exception as e:  # noqa: BLE001 -- one bad step must not sink the whole curve
            curve[str(step)] = {"error": f"{type(e).__name__}: {e}"}
    return curve


# ======================================================================= Task 2b: full battery


def full_battery_for_checkpoint(
    name: str,
    ckpt_dir: str,
    cohort: EvalCohort,
    *,
    device: torch.device,
    seed: int,
    n_boot: int,
    n_strata: int,
    gain_limit: int,
    n_replicates: int,
    geometry_limit: int,
) -> dict[str, Any]:
    """Operator spectrum + geometry report + transport gain + G1 shuffled-theta null + `z/mean`
    probe AUROC (with a patient-clustered CI) + the full robustness suite -- one checkpoint, one
    model load, one `encode_z` pass per split shared across every piece below.

    `z`/`theta` are cast to float32 before `gain_report`/`g1_shuffled_theta_gain_null`
    (`winder.eval.gates`'s own documented convention -- it does no casting itself); the operator
    is moved to CPU before every call below, matching every tensor it is applied against (every
    `encode_z` output is already `.cpu()`-resident by construction).
    """
    model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    if operator is None:
        raise ValueError(f"{name} ({ckpt_dir}): no transport operator -- full battery undefined")
    try:
        z_by_split = {s: encode_z(model, wf, device) for s, wf in cohort.waveforms.items()}
        op_report = operator_report(name, operator)
        op_cpu = operator.to("cpu")

        g_lim = min(geometry_limit, z_by_split["eval"].shape[0])
        geometry = geometry_report(
            z_by_split["eval"][:g_lim].double(), cohort.thetas["eval"][:g_lim].double(), op_cpu
        )

        gl = min(gain_limit, z_by_split["eval"].shape[0])
        z_gain = z_by_split["eval"][:gl].float()
        th_gain = cohort.thetas["eval"][:gl].float()
        pid_gain = cohort.patient_ids["eval"][:gl]
        gain = gain_report(z_gain, th_gain, op_cpu, pid_gain, n_strata=n_strata, seed=seed)
        g1 = g1_shuffled_theta_gain_null(
            z_gain,
            th_gain,
            op_cpu,
            pid_gain,
            n_strata=n_strata,
            n_replicates=n_replicates,
            seed=seed,
        )

        feats = {
            s: masked_mean_pool(z_by_split[s], cohort.thetas[s]).numpy()
            for s in ("train", "cal", "eval")
        }
        scores_full, _ev = fit_and_score(
            feats["train"],
            cohort.labels["train"],
            feats["cal"],
            cohort.labels["cal"],
            feats["eval"],
            cohort.labels["eval"],
            CLASSES,
            LinearProbeConfig(seed_probe=seed),
        )
        probe_z_mean = ci_row(
            scores_full,
            cohort.labels["eval"],
            cohort.patient_ids["eval"],
            _CLASS_COLUMNS,
            n_boot,
            seed,
        )

        robustness = robustness_suite(
            model,
            op_cpu,
            z_by_split,
            cohort.thetas,
            cohort.waveforms,
            cohort.labels,
            cohort.patient_ids["eval"],
            cohort.rr_median_ms,
            cohort.patch_width,
            LinearProbeConfig(seed_probe=seed),
            device,
            seed=seed,
        )
    finally:
        del model, operator
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "operator": op_report,
        "geometry": geometry,
        "gain": gain,
        "g1": g1,
        "probe_z_mean": probe_z_mean,
        "robustness": robustness,
    }


# ================================================================================== atomic write


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write `payload` to `path` via a sibling `.tmp` + `os.replace` -- atomic on the same
    filesystem, so a crash mid-write (or mid-run, since this is called after every stage) can
    never leave a truncated/corrupted report on disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    os.replace(tmp_path, path)


def _envelope(
    status: str, metrics: dict[str, Any], decisions: list[str], params: dict[str, Any], seed: int
) -> dict[str, Any]:
    return {
        "status": status,
        "milestone_id": MILESTONE_ID,
        "split_status": SPLIT_STATUS,
        "headline": HEADLINE,
        "metrics": metrics,
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": params,
            "seed": seed,
        },
        "decisions": [
            "Every number in this report is fold-9-contaminated: fold 9 is both training data "
            "for these 4 checkpoints and the only fold LEGACY_FOLD_CONFIG can construct without "
            "opening the sealed fold 10 -- the plan's own accepted design for this phase, not a "
            "bug. Nothing here is a headline number.",
            "Cohort built via winder.eval.acceptance.build_split_frames (LEGACY_FOLD_CONFIG "
            "metadata split) + winder-nominal's OWN top-level artifacts/lead_stats_f1to9.json, "
            "artifacts/manifest.parquet, artifacts/phase/theta_tokens.npz -- deliberately NOT "
            "winder.eval.acceptance.build_acceptance_cohort, which hardcodes the LEGACY "
            "folds-1-8 lead stats for Phase P6's own old-protocol reproduction.",
            "Detection/localisation battery (winder.eval.gates.detection_gap_ci) is out of "
            "scope: it needs a severity-swept anomaly-injection per-record dump that was never "
            "ported into winder-nominal. Flagged, not built -- see this module's own docstring.",
            "winder.eval.tasks.select_step and winder.eval.comparison.pairwise_deltas are "
            "deliberately NOT called: crowning a headline step or a signal-vs-control delta CI "
            "is exactly the framing this phase's brief parks for the CTO.",
            *decisions,
        ],
        "questions": [],
    }


# ======================================================================================== main


def main(argv: list[str] | None = None) -> int:
    """Parse args, run preflight -> lead-stats assertion -> AUROC curves -> full battery ->
    4-arm comparison table, write the report JSON after every stage (atomically), return 0 iff
    the run reached completion without a lead-stats mismatch or a fatal top-level error (per-arm/
    per-checkpoint failures inside a stage are recorded, not fatal -- see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument(
        "--lead-stats-path", default=None, help="default <artifacts-dir>/lead_stats_f1to9.json"
    )
    ap.add_argument("--out", default="artifacts/reports/p9_eval_suite.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-strata", type=int, default=16)
    ap.add_argument("--gain-limit", type=int, default=250)
    ap.add_argument("--n-replicates", type=int, default=2000)
    ap.add_argument("--geometry-limit", type=int, default=1200)
    ap.add_argument("--focus-steps", default="5000,final")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--train-limit", type=int, default=0, help="0 = full pool (the real run)")
    args = ap.parse_args(argv)

    t0 = time.time()
    device = torch.device(args.device)
    roster_dir = args.roster_dir or os.path.join(args.artifacts_dir, "roster")
    lead_stats_path = args.lead_stats_path or os.path.join(
        args.artifacts_dir, "lead_stats_f1to9.json"
    )
    arm_names = [a for a in args.arms.split(",") if a]
    focus_labels = [s for s in args.focus_steps.split(",") if s]
    params = {
        "data_root": args.data_root,
        "artifacts_dir": args.artifacts_dir,
        "roster_dir": roster_dir,
        "lead_stats_path": lead_stats_path,
        "device": str(device),
        "n_boot": args.n_boot,
        "n_strata": args.n_strata,
        "gain_limit": args.gain_limit,
        "n_replicates": args.n_replicates,
        "geometry_limit": args.geometry_limit,
        "focus_steps": focus_labels,
        "arms": arm_names,
        "train_limit": args.train_limit,
    }

    arm_dirs = {name: os.path.join(roster_dir, name) for name in arm_names}
    for name, d in arm_dirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"roster arm dir not found: {d} (arm={name!r})")

    # -------------------------------------------------- Task 1: discovery + preflight (all 48)
    per_arm_steps: dict[str, dict[int, str]] = {
        name: discover_seed_checkpoints(d) for name, d in arm_dirs.items()
    }
    all_checkpoints: dict[str, str] = {
        f"{name}/step{step}": ckpt_dir
        for name, steps in per_arm_steps.items()
        for step, ckpt_dir in steps.items()
    }
    preflight_failed = preflight_check_checkpoints(all_checkpoints, seed=args.seed, device=device)
    preflight_report = {
        "n_checkpoints": len(all_checkpoints),
        "n_ok": len(all_checkpoints) - len(preflight_failed),
        "failed": preflight_failed,
    }

    metrics: dict[str, Any] = {"preflight": preflight_report}
    _write_json_atomic(args.out, _envelope("RUNNING", metrics, [], params, args.seed))

    # ------------------------------------------ the lead-stats trap: a real, executed assertion
    hash_failures: dict[str, str] = {}
    for key, ckpt_dir in all_checkpoints.items():
        try:
            assert_lead_stats_matches_checkpoint(ckpt_dir, lead_stats_path)
        except AssertionError as e:
            hash_failures[key] = str(e)
    metrics["lead_stats_hash_check"] = {
        "lead_stats_path": lead_stats_path,
        "n_checked": len(all_checkpoints),
        "n_mismatched": len(hash_failures),
        "mismatched": hash_failures,
    }
    if hash_failures:
        report = _envelope(
            "FAIL",
            metrics,
            [
                f"STOPPED before any encoding: {len(hash_failures)} checkpoint(s) declare a "
                "lead_stats_sha256 that does not match the actual sha256 of the lead-stats file "
                "this run was about to use -- evaluating against it would silently corrupt every "
                "downstream number."
            ],
            params,
            args.seed,
        )
        _write_json_atomic(args.out, report)
        print(f"[eval_suite] status=FAIL (lead-stats mismatch) wrote {args.out}", flush=True)
        return 1

    # ---------------------------------------------------------------------- build cohort once
    cohort, bookkeeping = build_p9_cohort(
        args.data_root, args.artifacts_dir, lead_stats_path, train_limit=args.train_limit
    )
    metrics["cohort"] = bookkeeping
    _write_json_atomic(args.out, _envelope("RUNNING", metrics, [], params, args.seed))

    # --------------------------------------------------- Task 2a: AUROC-only curves, all 4 arms
    curves: dict[str, Any] = {}
    for name in arm_names:
        curves[name] = auroc_curve_for_arm(
            per_arm_steps[name], cohort, device=device, seed=args.seed
        )
    metrics["auroc_curves"] = curves
    _write_json_atomic(args.out, _envelope("RUNNING", metrics, [], params, args.seed))

    # -------------------------------------------------- Task 2b: full battery at focus steps
    # Resolved once per arm and reused by Task 3 below, so "final" (max(steps)) cannot resolve
    # to two different literal step numbers across the two tasks.
    focus_by_arm = {
        name: resolve_focus_steps(focus_labels, per_arm_steps[name]) for name in arm_names
    }
    battery: dict[str, Any] = {}
    for name in arm_names:
        steps = per_arm_steps[name]
        battery[name] = {}
        for label, step in focus_by_arm[name].items():
            key = f"{name}_step{step}"
            try:
                battery[name][label] = {
                    "step": step,
                    **full_battery_for_checkpoint(
                        key,
                        steps[step],
                        cohort,
                        device=device,
                        seed=args.seed,
                        n_boot=args.n_boot,
                        n_strata=args.n_strata,
                        gain_limit=args.gain_limit,
                        n_replicates=args.n_replicates,
                        geometry_limit=args.geometry_limit,
                    ),
                }
            except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole battery
                battery[name][label] = {"step": step, "error": f"{type(e).__name__}: {e}"}
    metrics["full_battery"] = battery
    _write_json_atomic(args.out, _envelope("RUNNING", metrics, [], params, args.seed))

    # --------------------------------------------- Task 3: 4-arm comparison table, both steps
    comparison: dict[str, Any] = {}
    for label in focus_labels:
        try:
            arms_at_label = {
                name: per_arm_steps[name][focus_by_arm[name][label]] for name in arm_names
            }
            table = arm_comparison_table(
                arms_at_label,
                cohort,
                device=device,
                n_boot=args.n_boot,
                n_strata=args.n_strata,
                n_replicates=args.n_replicates,
                seed=args.seed,
            )
            # "_scores" carries a full ndarray per arm (winder.eval.comparison's own docstring) --
            # not JSON-serialisable and not a headline number; dropped before writing to disk.
            comparison[label] = {
                name: {k: v for k, v in row.items() if k != "_scores"}
                for name, row in table.items()
            }
        except Exception as e:  # noqa: BLE001 -- one bad step must not lose the other's table
            comparison[label] = {"error": f"{type(e).__name__}: {e}"}
    metrics["comparison_table"] = comparison

    status = "PASS"
    report = _envelope(status, metrics, [], params, args.seed)
    report["metrics"]["elapsed_sec"] = time.time() - t0
    _write_json_atomic(args.out, report)
    print(f"[eval_suite] status={status} wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

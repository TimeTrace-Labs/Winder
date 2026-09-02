"""Phase P7's own smoke: 200 CPU steps, all 4 planned (name, lambda_trans, seed) combos, run as
FOUR SEPARATE `uv run python scripts/pretrain.py` subprocesses (matching how Phase P8's real GPU
launches will actually be invoked, unlike `scripts/run_ablation.py`'s in-process call -- this
smoke's whole point is proving the ACTUAL CLI entrypoint works end-to-end, subprocess boundary
included), then verifies every claim this phase's design brief asks for against the four runs'
own on-disk outputs.

Uses the LEGACY (folds 1-8) lead-stats/manifest/theta artifacts under `artifacts/reference/` --
refitting `lead_stats` on folds 1-9 is Phase P8's own job, not this smoke's; this is a deliberate,
harmless simplification for proving the pipeline works mechanically, not a statistics-correctness
claim (module docstring mirrors `scripts/pretrain.py`'s own such note).

Mirrors `scripts/accept.py`'s own pattern: a thin driver writing a report JSON in this project's
standard schema, invoked manually (not part of `uv run pytest`) -- exactly like `accept.py`'s own
Tier 1 numerical reproduction isn't bundled into the regular test gate either.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Any

import torch

from winder.paths import default_data_root

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_ROOT = default_data_root()
_REFERENCE_ROOT = os.path.join(_REPO_ROOT, "artifacts", "reference")
_LEGACY_LEAD_STATS = os.path.join(_REFERENCE_ROOT, "lead_stats_f1to8_legacy.json")
_MANIFEST_PATH = os.path.join(_REFERENCE_ROOT, "manifest.parquet")
_THETA_TOKENS_PATH = os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz")

#: (name, lambda_trans, seed) -- Phase P8's own four planned combos (build plan's launch table).
COMBOS: tuple[tuple[str, float, int], ...] = (
    ("signal_seed0", 1.0, 0),
    ("signal_seed1", 1.0, 1),
    ("control_seed0", 0.0, 0),
    ("control_seed1", 0.0, 1),
)

#: Per-combo expected config-diff key set against artifacts/reference/FIN_seed0/.../config.yaml,
#: at this smoke's own reduced --steps -- the four-field allowed-diff set this phase's own
#: investigation found (train.lambda_trans, train.seed_pretrain, arm.seed, arm.name), plus
#: train.n_steps (present for every combo here, since none runs the reference's own 30000).
_EXPECTED_DIFF_KEYS: dict[str, set[str]] = {
    "signal_seed0": {"train.n_steps"},
    "signal_seed1": {"train.n_steps", "train.seed_pretrain", "arm.seed", "arm.name"},
    "control_seed0": {"train.n_steps", "train.lambda_trans"},
    "control_seed1": {
        "train.n_steps",
        "train.lambda_trans",
        "train.seed_pretrain",
        "arm.seed",
        "arm.name",
    },
}

#: StepMetrics fields that must be finite in every arm, regardless of lambda_trans.
_ALWAYS_FINITE = (
    "lr",
    "pred_loss",
    "persistence_loss",
    "sigreg_loss",
    "total_loss",
    "grad_norm",
    "cutoff_mean",
)
#: Additionally finite only when lambda_trans != 0.0 (the transport block actually runs).
_SIGNAL_ONLY_FINITE = (
    "trans_loss",
    "trans_floor",
    "trans_gain",
    "trans_directional",
    "closure_residual",
)
#: NaN "not applicable" everywhere in this recipe (sigreg_frame="raw", lambda_sig_record=0.0,
#: transport_radial_weight=0.0 -- none exposed as CLI flags here).
_ALWAYS_NAN = ("trans_radial", "theta_valid_frac", "sigreg_n_records", "sigreg_record_loss")

#: StepMetrics fields expected BITWISE IDENTICAL between the signal and control arm at the same
#: seed, step 0: neither the transport block (RNG-free) nor lambda_trans's own weighting touches
#: mask/sigreg/augment/data-order draws or the predictor/regularizer forward values themselves --
#: only total_loss/grad_norm/trans_* legitimately differ.
_DETERMINISM_INVARIANT_FIELDS = (
    "pred_loss",
    "persistence_loss",
    "sigreg_loss",
    "n_context",
    "n_target",
    "cutoff_mean",
)


def _combo_argv(
    *,
    name: str,
    lambda_trans: float,
    seed: int,
    steps: int,
    batch_size: int,
    checkpoint_at: str,
    data_root: str,
    artifacts_dir: str,
) -> list[str]:
    """The crowned recipe's fixed flags at this smoke's reduced step count and legacy artifacts
    -- built directly (not via `winder.ablations.resolve_arm`, whose common flags bake in the
    REAL launch's `--steps 30000`/`--device cuda`/folds-1-9 lead-stats path, none of which this
    smoke uses)."""
    return [
        "--batch-size",
        str(batch_size),
        "--steps",
        str(steps),
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--artifacts-dir",
        artifacts_dir,
        "--data-root",
        data_root,
        "--train-folds",
        "1,2,3,4,5,6,7,8,9",
        "--lead-stats-path",
        _LEGACY_LEAD_STATS,
        "--manifest-path",
        _MANIFEST_PATH,
        "--theta-tokens-path",
        _THETA_TOKENS_PATH,
        "--lambda-sig",
        "0.15",
        "--checkpoint-at",
        checkpoint_at,
        "--transport-arm",
        "cyclic",
        "--lambda-trans",
        str(lambda_trans),
        "--k0",
        "4",
        "--n-j",
        "1,2,3,4,5,6,7,8,9,10",
        "--k-j",
        "24,24,20,16,12,10,8,6,4,2",
        "--encoder-name",
        "conv_trunk",
        "--predictor-json",
        '{"n_layers":4}',
        "--augment",
        "gauss,powerline,wander,ampmod,leaddrop,leadgain",
        "--augment-prob",
        "0.5",
    ]


def _run_one_arm(
    *, name: str, lambda_trans: float, seed: int, args: argparse.Namespace, artifacts_root: str
) -> dict[str, Any]:
    """Launches `scripts/pretrain.py` as a real subprocess (module docstring), tees its output to
    `<artifacts_dir>/run.log`, and writes `<artifacts_dir>/EXIT_CODE` -- an explicit, on-disk exit
    status a caller reads back rather than inferring success from a log tail or a piped tee's own
    (unreliable) exit code."""
    artifacts_dir = os.path.join(artifacts_root, name)
    os.makedirs(artifacts_dir, exist_ok=True)
    argv = _combo_argv(
        name=name,
        lambda_trans=lambda_trans,
        seed=seed,
        steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_at=args.checkpoint_at,
        data_root=args.data_root,
        artifacts_dir=artifacts_dir,
    )
    cmd = ["uv", "run", "python", "scripts/pretrain.py", *argv]
    log_path = os.path.join(artifacts_dir, "run.log")
    print(f"[p7_smoke] launching {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    exit_code_path = os.path.join(artifacts_dir, "EXIT_CODE")
    with open(exit_code_path, "w", encoding="utf-8") as fh:
        fh.write(str(proc.returncode))
    print(f"[p7_smoke] {name} exited {proc.returncode} in {elapsed:.1f}s", flush=True)
    return {
        "name": name,
        "artifacts_dir": artifacts_dir,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "log_path": log_path,
    }


def _verify_arm(name: str, artifacts_dir: str, steps: int, checkpoint_at: str) -> dict[str, Any]:
    """Every claim this phase's brief makes about a single arm's own outputs: config-diff guard
    key set, history completeness, checkpoint strict-load structure, and the per-arm NaN pattern.
    Returns a dict of named boolean checks plus enough raw data for the cross-arm checks below."""
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    summary_path = os.path.join(artifacts_dir, "s2_summary.json")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    actual_diff_keys = set(summary["config_diff_vs_reference"])
    checks["config_diff_matches_expected_key_set"] = actual_diff_keys == _EXPECTED_DIFF_KEYS[name]
    detail["config_diff_keys"] = sorted(actual_diff_keys)

    history_path = os.path.join(artifacts_dir, "s2_history.jsonl")
    with open(history_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    checks["history_has_exactly_n_steps_rows"] = len(rows) == steps
    checks["history_steps_are_contiguous_from_zero"] = [r["step"] for r in rows] == list(
        range(steps)
    )
    lambda_trans = summary["provenance"]["lambda_trans"]
    signal_arm = lambda_trans != 0.0
    finite_fields = _ALWAYS_FINITE + (_SIGNAL_ONLY_FINITE if signal_arm else ())
    nan_fields = _ALWAYS_NAN + (() if signal_arm else _SIGNAL_ONLY_FINITE)
    no_bad_nan = all(
        (row[f] is not None and not math.isnan(row[f])) for row in rows for f in finite_fields
    )
    nan_where_expected = all(
        (row[f] is None or math.isnan(row[f])) for row in rows for f in nan_fields
    )
    checks["no_nan_in_required_finite_fields"] = no_bad_nan
    checks["nan_sentinel_fields_are_nan_as_expected"] = nan_where_expected
    detail["step0_row"] = rows[0]

    mid_run_steps = sorted({int(tok) for tok in checkpoint_at.split(",") if tok})
    mid_run_dirs = [f"checkpoint_step{n}" for n in mid_run_steps]
    for step_dir in (*mid_run_dirs, "checkpoint"):
        path = os.path.join(artifacts_dir, step_dir, "state.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        checks[f"{step_dir}_has_80_model_keys"] = len(state["model_state_dict"]) == 80
        checks[f"{step_dir}_has_operator_state_dict"] = "operator_state_dict" in state
        if step_dir == "checkpoint":
            detail["final_model_state_shapes"] = {
                k: list(v.shape) for k, v in state["model_state_dict"].items()
            }

    return {"checks": checks, "detail": detail}


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Runs all 4 combos as real subprocesses, verifies every per-arm claim, then the two
    cross-arm claims (structural parity, step-0 determinism) between the seed-0 signal/control
    pair. Returns the full report dict; never raises on a failed arm (recorded in the report
    instead), so every arm's diagnostics are always captured."""
    os.makedirs(args.artifacts_root, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {}
    for name, lambda_trans, seed in COMBOS:
        runs[name] = _run_one_arm(
            name=name,
            lambda_trans=lambda_trans,
            seed=seed,
            args=args,
            artifacts_root=args.artifacts_root,
        )

    per_arm_verification: dict[str, Any] = {}
    for name, _lambda_trans, _seed in COMBOS:
        if runs[name]["exit_code"] != 0:
            per_arm_verification[name] = {"skipped": "non-zero exit code, see run.log"}
            continue
        per_arm_verification[name] = _verify_arm(
            name, runs[name]["artifacts_dir"], args.steps, args.checkpoint_at
        )

    cross_arm: dict[str, Any] = {}
    if all(runs[c[0]]["exit_code"] == 0 for c in COMBOS):
        signal_shapes = per_arm_verification["signal_seed0"]["detail"]["final_model_state_shapes"]
        control_shapes = per_arm_verification["control_seed0"]["detail"]["final_model_state_shapes"]
        cross_arm["all_else_equal_same_keys"] = set(signal_shapes) == set(control_shapes)
        cross_arm["all_else_equal_same_shapes"] = all(
            signal_shapes[k] == control_shapes[k] for k in signal_shapes
        )

        signal_row0 = per_arm_verification["signal_seed0"]["detail"]["step0_row"]
        control_row0 = per_arm_verification["control_seed0"]["detail"]["step0_row"]
        cross_arm["step0_determinism"] = {
            f: (signal_row0[f], control_row0[f]) for f in _DETERMINISM_INVARIANT_FIELDS
        }
        cross_arm["step0_determinism_matches_bitwise"] = all(
            signal_row0[f] == control_row0[f] for f in _DETERMINISM_INVARIANT_FIELDS
        )

    all_checks_passed = all(
        all(v["checks"].values()) for v in per_arm_verification.values() if "checks" in v
    ) and all(v for v in cross_arm.values() if isinstance(v, bool))
    all_ran = all(runs[c[0]]["exit_code"] == 0 for c in COMBOS)
    status = "PASS" if (all_ran and all_checks_passed) else "FAIL"

    return {
        "status": status,
        "milestone_id": "P7_smoke",
        "metrics": {
            "runs": runs,
            "per_arm_verification": per_arm_verification,
            "cross_arm": cross_arm,
        },
        "provenance": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "checkpoint_at": args.checkpoint_at,
            "data_root": args.data_root,
            "python": sys.version,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--checkpoint-at", default="100")
    ap.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    ap.add_argument("--artifacts-root", default=os.path.join("artifacts", "smoke_p7"))
    ap.add_argument("--out", default=os.path.join("artifacts", "reports", "p7_smoke.json"))
    args = ap.parse_args(argv)

    report = run_smoke(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"[p7_smoke] status={report['status']} wrote {args.out}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

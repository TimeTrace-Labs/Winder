"""Thin argparse driver over `winder.eval.detection.run_detection_battery` -- the
detection/localisation battery `scripts/p9_eval_suite.py` flagged as out of scope (its own module
docstring: "a severity-swept anomaly-injection dump... was never ported into winder-nominal").

All the actual injection/scoring/aggregation logic lives in `src/winder/eval/detection.py`, which
is independently unit-tested (`tests/test_eval_detection.py`) -- this script only parses
arguments, resolves default paths, calls `run_detection_battery`, and writes the report JSON
atomically.

**Fold 10 is never touched.** `--fold-config nominal` (the default) uses `winder.data.folds.
FoldConfig()`'s own deliberately EMPTY `val_fold=0` sentinel; `--fold-config legacy` uses
`LEGACY_FOLD_CONFIG` (train 1-8 / val 9 / sealed 10) -- the substitution Phase P6's own
acceptance gate already had to make to reproduce the reference repo's original protocol.
`folds()`'s own sealed-fold-release keyword is passed as `True` NOWHERE in this script or in
`winder.eval.detection` (`tests/test_folds.py::test_no_call_site_unseals` enforces this as a
standing repo invariant).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

from winder.data.folds import LEGACY_FOLD_CONFIG, FoldConfig
from winder.data.integrity import git_sha
from winder.eval.detection import run_detection_battery
from winder.paths import default_data_root

MILESTONE_ID = "detection-localisation-battery-port"

#: Never a headline number: this driver's own job (per its commissioning brief) is porting and
#: validating against OLD, already-published numbers -- not producing a new claim.
HEADLINE = False


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write `payload` to `path` via a sibling `.tmp` + `os.replace` -- atomic on the same
    filesystem, matching `scripts/eval_suite.py`'s own convention."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    os.replace(tmp_path, path)


def _envelope(
    status: str,
    split_status: str,
    metrics: dict[str, Any],
    decisions: list[str],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "milestone_id": MILESTONE_ID,
        "split_status": split_status,
        "headline": HEADLINE,
        "metrics": metrics,
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": params,
            "seed": seed,
        },
        "decisions": decisions,
        "questions": [],
    }


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the battery, write the report JSON (+ a sibling per-record `.npz` when
    `--dump-per-record` is set), return 0 iff at least one requested checkpoint produced results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument(
        "--fold-config",
        choices=("nominal", "legacy"),
        default="nominal",
        help="'nominal' (default): winder.data.folds.FoldConfig()'s own empty val_fold=0 "
        "sentinel -- safe, never touches fold 10, but the val split is empty until a real "
        "protocol is pre-registered. 'legacy': LEGACY_FOLD_CONFIG (train 1-8/val 9/sealed 10), "
        "for reproducing the reference repo's own published detection numbers. Neither ever "
        "passes folds()'s own sealed-fold-release keyword as True; fold 10 stays sealed either "
        "way.",
    )
    ap.add_argument(
        "--rpeaks-npz", default=None, help="default <root>/phase/rpeaks.npz (root: see below)"
    )
    ap.add_argument(
        "--theta-tokens-path",
        default=None,
        help="default <root>/phase/theta_tokens.npz",
    )
    ap.add_argument(
        "--lead-stats-path",
        default=None,
        help="default <root>/lead_stats_f1to8_legacy.json (legacy) or "
        "<artifacts-dir>/lead_stats_f1to9.json (nominal)",
    )
    ap.add_argument(
        "--checkpoints",
        default="",
        help="comma-separated roster arm names, each resolved to <roster-dir>/<name>/checkpoint "
        "(required -- an empty value fails fast with status=FAIL, see below)",
    )
    ap.add_argument("--n-records", type=int, default=400)
    ap.add_argument("--causal-window", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dump-per-record",
        action="store_true",
        help="also write a sibling <out>_per_record.npz: per-record AUROCs and their record "
        "indices per cell, keyed for winder.eval.gates.detection_gap_ci",
    )
    ap.add_argument("--out", default="artifacts/reports/detection_battery.json")
    args = ap.parse_args(argv)

    fold_config = LEGACY_FOLD_CONFIG if args.fold_config == "legacy" else FoldConfig()
    # "root" is where the phase-clock/lead-stats artifacts this run's own protocol expects live:
    # the reference-reproduction artifacts under <artifacts-dir>/reference for "legacy", or this
    # repo's own top-level <artifacts-dir> for "nominal".
    root = (
        os.path.join(args.artifacts_dir, "reference")
        if args.fold_config == "legacy"
        else args.artifacts_dir
    )
    roster_dir = args.roster_dir or os.path.join(args.artifacts_dir, "roster")
    rpeaks_npz_path = args.rpeaks_npz or os.path.join(root, "phase", "rpeaks.npz")
    theta_tokens_path = args.theta_tokens_path or os.path.join(root, "phase", "theta_tokens.npz")
    lead_stats_path = args.lead_stats_path or (
        os.path.join(root, "lead_stats_f1to8_legacy.json")
        if args.fold_config == "legacy"
        else os.path.join(args.artifacts_dir, "lead_stats_f1to9.json")
    )
    checkpoint_names = [c for c in args.checkpoints.split(",") if c]
    split_status = "legacy_reference_reproduction" if args.fold_config == "legacy" else "nominal"

    params = {
        "data_root": args.data_root,
        "artifacts_dir": args.artifacts_dir,
        "roster_dir": roster_dir,
        "fold_config": args.fold_config,
        "rpeaks_npz_path": rpeaks_npz_path,
        "theta_tokens_path": theta_tokens_path,
        "lead_stats_path": lead_stats_path,
        "checkpoints": checkpoint_names,
        "n_records": args.n_records,
        "causal_window": args.causal_window,
        "device": args.device,
        "dump_per_record": args.dump_per_record,
    }

    if not checkpoint_names:
        report = _envelope(
            "FAIL",
            split_status,
            {},
            ["--checkpoints must name at least one roster arm; none was given."],
            params,
            args.seed,
        )
        _write_json_atomic(args.out, report)
        print(f"[detection_battery] status=FAIL (no --checkpoints) wrote {args.out}", flush=True)
        return 1

    metrics = run_detection_battery(
        data_root=args.data_root,
        roster_dir=roster_dir,
        checkpoint_names=checkpoint_names,
        rpeaks_npz_path=rpeaks_npz_path,
        lead_stats_path=lead_stats_path,
        theta_tokens_path=theta_tokens_path,
        fold_config=fold_config,
        n_records=args.n_records,
        causal_window=args.causal_window,
        seed=args.seed,
        device=torch.device(args.device),
        dump_per_record=args.dump_per_record,
    )

    per_record_dump = metrics.pop("per_record_dump", None)
    if per_record_dump:
        npz_path = os.path.splitext(args.out)[0] + "_per_record.npz"
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez_compressed(npz_path, **per_record_dump)
        metrics["per_record_dump_path"] = npz_path

    ran = [k for k in metrics if k not in ("config", "skipped")]
    status = "PASS" if ran else "FAIL"
    decisions = [
        "Fold 10 is never touched: 'nominal' (default) uses winder.data.folds.FoldConfig()'s "
        "own empty val_fold=0 sentinel; 'legacy' uses LEGACY_FOLD_CONFIG (train 1-8/val 9/sealed "
        "10). folds()'s own sealed-fold-release keyword is passed as True nowhere in this "
        "script or in winder.eval.detection.",
        f"skipped checkpoints: {metrics['skipped']}"
        if metrics.get("skipped")
        else "no checkpoints skipped.",
    ]
    report = _envelope(status, split_status, metrics, decisions, params, args.seed)
    _write_json_atomic(args.out, report)
    print(f"[detection_battery] status={status} wrote {args.out}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

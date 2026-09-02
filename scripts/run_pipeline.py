#!/usr/bin/env python3
"""One entry point chaining fetch -> manifest -> phase tokens -> lead stats -> train -> eval, so
a bare clone has a single documented command rather than six scripts a reader has to sequence by
reading each one's own docstring.

    uv run python scripts/run_pipeline.py --arms signal,control --seeds 0

Idempotent by construction, at STAGE granularity: before running a stage, checks whether its own
output already exists and skips it if so (printing why) -- rerunning after a requeue or a partial
failure only does the work still missing, it does not silently re-fetch 21,799 records or re-run
a 4-hour training arm that already completed. `--force` disables every skip check for one run.

Every stage is invoked via its own script's `main(argv)`, in-process (matching
`scripts/run_ablation.py`'s own precedent: a subprocess would re-pay every import for no benefit,
since this driver and every stage script already share one process's worth of already-imported
modules). Each stage's own exit code is checked and this driver stops at the first nonzero one --
`build_phase_tokens.py` returning 1 means its own halt_recommended calibration check fired, a
real finding to surface, not a plumbing failure to retry past.

Training and eval are the two stages this driver does NOT skip on output presence: a roster
arm's own `checkpoint_step*` completeness is exactly `--resume-from`'s job (`pretrain.py`), not
this driver's, and eval always re-runs because rerunning it is normally the point.
"""

from __future__ import annotations

import argparse
import os
import time

import build_manifest
import build_phase_tokens
import eval_suite
import fetch_ptbxl
import fit_lead_stats
import run_ablation

from winder.ablations import ABLATION_ARMS
from winder.paths import default_artifacts_dir, default_data_root


def _stage(name: str) -> None:
    print(f"\n[run_pipeline] === {name} ===", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--artifacts-dir", default=default_artifacts_dir())
    ap.add_argument(
        "--arms",
        default="signal,control",
        help=f"comma-separated arms from {sorted(ABLATION_ARMS)}",
    )
    ap.add_argument("--seeds", default="0,1", help="comma-separated seeds, e.g. '0,1'")
    ap.add_argument("--train-folds", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--force", action="store_true", help="ignore existing outputs, rerun every stage"
    )
    ap.add_argument(
        "--skip-fetch",
        action="store_true",
        help="assume --data-root is already populated; do not attempt to fetch it",
    )
    ap.add_argument(
        "--skip-eval", action="store_true", help="stop after training, do not run eval_suite"
    )
    args = ap.parse_args(argv)

    arm_names = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arm_names if a not in ABLATION_ARMS]
    if unknown:
        ap.error(f"unknown arm(s) {unknown} -- registered arms are {sorted(ABLATION_ARMS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s]

    manifest_path = os.path.join(args.artifacts_dir, "manifest.parquet")
    theta_tokens_path = os.path.join(args.artifacts_dir, "phase", "theta_tokens.npz")
    lead_stats_path = os.path.join(args.artifacts_dir, "lead_stats_f1to9.json")

    t0 = time.time()

    # ------------------------------------------------------------------------------------- fetch
    if args.skip_fetch:
        _stage("fetch (skipped: --skip-fetch)")
    elif not args.force and os.path.isfile(os.path.join(args.data_root, "ptbxl_database.csv")):
        _stage(f"fetch (skipped: {args.data_root}/ptbxl_database.csv already present)")
    else:
        _stage("fetch")
        rc = fetch_ptbxl.main(["--data-root", args.data_root])
        if rc != 0:
            print(f"[run_pipeline] fetch_ptbxl failed (exit {rc})", flush=True)
            return rc

    # ---------------------------------------------------------------------------------- manifest
    if not args.force and os.path.isfile(manifest_path):
        _stage(f"manifest (skipped: {manifest_path} already present)")
    else:
        _stage("manifest")
        rc = build_manifest.main(
            ["--data-root", args.data_root, "--artifacts-dir", args.artifacts_dir]
        )
        if rc != 0:
            print(f"[run_pipeline] build_manifest failed (exit {rc})", flush=True)
            return rc

    # ----------------------------------------------------------------------------- phase tokens
    if not args.force and os.path.isfile(theta_tokens_path):
        _stage(f"phase tokens (skipped: {theta_tokens_path} already present)")
    else:
        _stage("phase tokens")
        rc = build_phase_tokens.main(["--artifacts-dir", args.artifacts_dir])
        if rc != 0:
            print(
                f"[run_pipeline] build_phase_tokens exited {rc} -- halt_recommended fired on "
                "this cohort's own calibration check; see artifacts/phase/m0_calibration.json "
                "for the uniformity/independence numbers that triggered it. Not a plumbing bug.",
                flush=True,
            )
            return rc

    # --------------------------------------------------------------------------------- lead stats
    if not args.force and os.path.isfile(lead_stats_path):
        _stage(f"lead stats (skipped: {lead_stats_path} already present)")
    else:
        _stage("lead stats")
        rc = fit_lead_stats.main(
            [
                "--data-root",
                args.data_root,
                "--train-folds",
                args.train_folds,
                "--out-path",
                lead_stats_path,
            ]
        )
        if rc != 0:
            print(f"[run_pipeline] fit_lead_stats failed (exit {rc})", flush=True)
            return rc

    # -------------------------------------------------------------------------------------- train
    for arm in arm_names:
        for seed in seeds:
            arm_dir = os.path.join(args.artifacts_dir, "roster", f"{arm}_seed{seed}")
            _stage(f"train {arm}_seed{seed}")
            rc = run_ablation.main(
                [
                    arm,
                    "--seed",
                    str(seed),
                    "--artifacts-dir",
                    arm_dir,
                    "--artifacts-base",
                    args.artifacts_dir,
                    "--device",
                    args.device,
                ]
            )
            if rc != 0:
                print(
                    f"[run_pipeline] run_ablation {arm}_seed{seed} failed (exit {rc})", flush=True
                )
                return rc

    # --------------------------------------------------------------------------------------- eval
    if args.skip_eval:
        _stage("eval (skipped: --skip-eval)")
    else:
        _stage("eval")
        arm_seed_names = [f"{a}_seed{s}" for a in arm_names for s in seeds]
        rc = eval_suite.main(
            [
                "--data-root",
                args.data_root,
                "--artifacts-dir",
                args.artifacts_dir,
                "--device",
                args.device,
                "--arms",
                ",".join(arm_seed_names),
            ]
        )
        if rc != 0:
            print(f"[run_pipeline] eval_suite failed (exit {rc})", flush=True)
            return rc

    print(f"\n[run_pipeline] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

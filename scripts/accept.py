"""Phase P6, Tier 1: thin argparse driver over `winder.eval.acceptance.run_acceptance`.

All the actual reproduction logic (cohort construction, the five assertion families, the
non-gating robustness spot-check) lives in `src/winder/eval/acceptance.py`, which is independently
unit-tested (`tests/test_eval_acceptance.py`) -- this script only parses arguments, calls it, and
writes the report JSON to `--out`.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from winder.eval.acceptance import run_acceptance
from winder.paths import default_data_root


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the Tier 1 acceptance harness, write the report JSON, return 0 iff PASS."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--reference-root", default="artifacts/reference")
    ap.add_argument("--out", default="artifacts/reports/p6_tier1_acceptance.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-strata", type=int, default=16)
    ap.add_argument("--gain-limit", type=int, default=250)
    ap.add_argument("--n-replicates", type=int, default=2000)
    ap.add_argument(
        "--robustness-reference",
        default=None,
        help="optional robustness.json to spot-check FIN_seed0/checkpoint (30k) against",
    )
    args = ap.parse_args(argv)

    report = run_acceptance(
        data_root=args.data_root,
        reference_root=args.reference_root,
        device=torch.device(args.device),
        seed=args.seed,
        n_boot=args.n_boot,
        n_strata=args.n_strata,
        gain_limit=args.gain_limit,
        n_replicates=args.n_replicates,
        robustness_reference_path=args.robustness_reference,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"[accept] status={report['status']} wrote {args.out}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

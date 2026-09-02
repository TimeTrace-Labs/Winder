"""Thin driver over `winder.ablations.resolve_arm` + `scripts/pretrain.py`'s own `main()`.

`uv run python scripts/run_ablation.py <name> --seed N [--artifacts-dir DIR]` resolves `<name>`
against `winder.ablations.ABLATION_ARMS` (defaulting `--artifacts-dir` to
`artifacts/roster/<name>_seed<N>`, matching Phase P8's own naming convention) and calls
`pretrain.main` with the resolved argv IN-PROCESS -- a `subprocess` call would fork a second
Python interpreter and re-pay every import (torch, the encoder registry, ...) for no benefit here:
this driver and `pretrain.py` already share one process's worth of already-imported modules, and
`pretrain.main`'s own `SystemExit`-on-error convention (from `argparse.ArgumentParser.error`)
propagates through an in-process call exactly as it would through a subprocess's exit code, so
there is nothing a subprocess boundary would add for this use.
"""

from __future__ import annotations

import argparse
import os

import pretrain

from winder.ablations import ABLATION_ARMS, resolve_arm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", choices=sorted(ABLATION_ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--artifacts-dir",
        default=None,
        help="default artifacts/roster/<name>_seed<N>, matching Phase P8's own roster layout",
    )
    ap.add_argument(
        "--artifacts-base",
        default=None,
        help="where the crowned recipe's fixed inputs live (default: winder.paths."
        "default_artifacts_dir(), i.e. $WINDER_ARTIFACTS_DIR or 'artifacts') -- unrelated to "
        "--artifacts-dir, this run's own output location",
    )
    ap.add_argument(
        "--device", default="cuda", help="passed through to pretrain.py; 'cuda' matches Phase P8"
    )
    args = ap.parse_args(argv)

    artifacts_dir = args.artifacts_dir or os.path.join(
        "artifacts", "roster", f"{args.name}_seed{args.seed}"
    )
    resolved_argv = resolve_arm(
        args.name,
        seed=args.seed,
        artifacts_dir=artifacts_dir,
        artifacts_base=args.artifacts_base,
        device=args.device,
    )
    print(f"[run_ablation] {args.name} seed={args.seed} -> pretrain.py {resolved_argv}", flush=True)
    return pretrain.main(resolved_argv)


if __name__ == "__main__":
    raise SystemExit(main())

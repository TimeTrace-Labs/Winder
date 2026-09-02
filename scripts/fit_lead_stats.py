"""Fit corpus-level per-lead normalization statistics from PTB-XL's records500/ release -- a lean
driver over already-ported `winder.data.norm_stats.fit_lead_stats` (Phase P2), parameterised over
`--train-folds` so ONE script produces both the legacy folds-1-8 refit and Phase P8's own
folds-1-9 refit, rather than two near-duplicate scripts (see the reference repo's
`scripts/s1_lead_stats.py`, whose fold set was a hardcoded `FoldConfig()` default -- that
shortcut is exactly what this driver's `--train-folds` flag removes).

Reads `winder.data.ptbxl.load_metadata`'s join, restricts to `--train-folds` directly (not via
`winder.data.folds.folds()`, which also computes val/test splits this driver has no use for),
reads each record from `records500/` and decimates it to 100 Hz via
`winder.data.ptbxl.read_and_decimate_500hz` (DATA-04 -- see that module's docstring), and fits
`fit_lead_stats` over them. No integrity-manifest filtering happens here (mirrors the reference
script exactly): a record whose waveform is actually unreadable/non-finite fails loudly inside
`read_and_decimate_500hz`/`fit_lead_stats` rather than being silently skipped.

NOT stochastic, so no `--seed`: `fit_lead_stats` accumulates plain running sums over `records` in
the fixed order `load_metadata` returns (ascending `ecg_id`) -- no RNG is constructed or consumed
anywhere in this path, unlike e.g. `scripts/pretrain.py`'s data-order shuffling. Two runs over the
same `--data-root`/`--train-folds` are therefore bitwise identical by construction, with no seed
to pin.

`--train-folds` is validated BEFORE any signal is read: a value outside
`winder.data.folds.FoldConfig().train_folds` (the sealed-fold/leak boundary
`winder.data.norm_stats.LeadStats.folds`'s own field validator also enforces) fails in
milliseconds naming the bad folds, rather than after a ~20k-record, multi-minute fit ends in a
pydantic `ValidationError` on construction.

Outputs:
  <--out-path>                          the fitted `LeadStats`, JSON (LeadStats.to_json)
  <--out-path stem>_summary.json        counts, timing, provenance (git SHA, data root, integrity
                                         report) -- sibling to --out-path, mirroring
                                         `scripts/s1_lead_stats.py`'s own `s1_summary.json`
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from winder.data.folds import FoldConfig
from winder.data.integrity import assemble_integrity_report, git_sha
from winder.data.norm_stats import fit_lead_stats
from winder.data.ptbxl import LEAD_ORDER, load_metadata, read_and_decimate_500hz
from winder.paths import default_data_root

_DEFAULT_DATA_ROOT = default_data_root()


def _parse_int_csv(raw: str, flag_name: str) -> list[int]:
    """`"1,2,3"` -> `[1, 2, 3]`, `ValueError` naming `flag_name` on a malformed token."""
    try:
        return [int(tok) for tok in raw.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{flag_name} must be a comma-separated list of ints, got {raw!r}"
        ) from exc


def _validate_train_folds(train_folds: tuple[int, ...]) -> None:
    """Fails fast, before any I/O, if `train_folds` reaches outside
    `FoldConfig().train_folds` -- the same boundary `LeadStats.folds`'s own field validator
    enforces (this project's non-leaking training-fold set: excludes the sealed test fold 10 and
    any future validation fold). Catching this here, rather than letting the eventual
    `LeadStats(...)` construction raise, turns a typo into a millisecond failure instead of one
    that surfaces only after the full multi-minute fit completes."""
    allowed = set(FoldConfig().train_folds)
    leaked = set(train_folds) - allowed
    if leaked:
        raise ValueError(
            f"--train-folds {sorted(train_folds)} includes fold(s) {sorted(leaked)} outside "
            f"the non-leaking training-fold set {sorted(allowed)} (FoldConfig().train_folds) -- "
            f"this would leak the sealed test fold or a future validation fold into the fitted "
            f"normalization statistics."
        )


def _iter_signals(metadata: pd.DataFrame, data_root: str) -> Iterator[np.ndarray]:
    """Yields one `(1000, 12)` decimated-to-100Hz signal per `metadata` row, in row order."""
    for filename_hr in metadata["filename_hr"]:
        hea_path = os.path.join(data_root, str(filename_hr) + ".hea")
        yield read_and_decimate_500hz(hea_path, expected_sig_name=LEAD_ORDER)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=_DEFAULT_DATA_ROOT,
        help="PTB-XL root (records500/, ptbxl_database.csv)",
    )
    ap.add_argument(
        "--train-folds",
        required=True,
        help="comma-separated PTB-XL strat_fold ints to fit over, e.g. '1,2,3,4,5,6,7,8,9'",
    )
    ap.add_argument("--out-path", required=True, help="output path for the fitted LeadStats JSON")
    args = ap.parse_args(argv)

    try:
        train_folds = tuple(sorted(set(_parse_int_csv(args.train_folds, "--train-folds"))))
    except ValueError as exc:
        ap.error(str(exc))
    if not train_folds:
        ap.error("--train-folds must name at least one fold")
    try:
        _validate_train_folds(train_folds)
    except ValueError as exc:
        ap.error(str(exc))

    t0 = time.time()
    winder_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"[fit_lead_stats] loading metadata from {args.data_root}", flush=True)
    metadata = load_metadata(args.data_root)
    train = metadata.loc[metadata["strat_fold"].isin(train_folds)]
    print(
        f"[fit_lead_stats] fitting lead stats over {len(train)} records in folds {train_folds}",
        flush=True,
    )

    stats = fit_lead_stats(
        _iter_signals(train, args.data_root),
        leads=LEAD_ORDER,
        fs=100,
        folds=train_folds,
        winder_git_sha=git_sha(winder_root),
    )

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(out_dir, exist_ok=True)
    stats.to_json(args.out_path)
    print(f"[fit_lead_stats] wrote {args.out_path}", flush=True)

    fold_config = FoldConfig(train_folds=train_folds)
    report = assemble_integrity_report(
        args.data_root, metadata, fold_config=fold_config, winder_repo_root=winder_root
    )
    summary: dict[str, Any] = {
        "n_records_fit": stats.n_records,
        "n_samples_fit": stats.n_samples,
        "train_folds": list(train_folds),
        "mean_mv": list(stats.mean_mv),
        "std_mv": list(stats.std_mv),
        "elapsed_min": (time.time() - t0) / 60.0,
        "provenance": {
            "winder_git_sha": stats.winder_git_sha,
            "data_root": os.path.abspath(args.data_root),
            "out_path": os.path.abspath(args.out_path),
            "integrity": report,
        },
    }
    summary_path = os.path.splitext(args.out_path)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[fit_lead_stats] wrote {summary_path}", flush=True)

    print("\n===== SUMMARY =====", flush=True)
    print(f"records fit: {stats.n_records}  samples: {stats.n_samples}", flush=True)
    print(f"mean_mv: {[round(m, 4) for m in stats.mean_mv]}", flush=True)
    print(f"std_mv:  {[round(s, 4) for s in stats.std_mv]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

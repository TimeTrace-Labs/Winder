#!/usr/bin/env python3
"""Fetch PTB-XL 1.0.3 from PhysioNet into a `--data-root` directory `build_manifest.py`/
`fit_lead_stats.py`/`pretrain.py` can read directly -- the one pipeline stage with no producing
script in this repo before this file existed (a fresh clone had no way to obtain the raw corpus
it needs).

Fetches only what this codebase reads: `ptbxl_database.csv`, `scp_statements.csv`, and every
`records500/<subdir>/<ecg_id>_hr.{hea,dat}` pair named by the metadata's own `filename_hr` column.
`records100/` is never fetched -- nothing in this repo reads it; the 100 Hz grid is produced by
decimating 500 Hz (`winder.data.decimation`), not downloaded pre-decimated.

The two metadata CSVs are verified against the sha256 pair already recorded in this repo's own
provenance (`artifacts/lead_stats_f1to9_summary.json`'s `integrity.sha256`, and the consumed
fold-10 authorization record) -- a mismatch here means the fetched file is not the same PTB-XL
release the published numbers were computed on, and this script refuses to proceed rather than
silently building a manifest against different data. Per-record .hea/.dat pairs are NOT
individually checksummed (PhysioNet publishes no per-file hash for these): `build_manifest.py`'s
own downstream QC (WRONG_SHAPE/NAN exclusion, `wfdb_io.read_record`'s header/checksum validation
where present) is this pipeline's actual defence against a corrupted waveform file, exercised on
every record it touches regardless of how that file arrived on disk.

Idempotent and resumable: a file already present at its expected final path and non-empty is
skipped, not re-fetched -- so a killed or interrupted run can be re-launched and only fetches what
is still missing. `--workers` threads (I/O-bound HTTP, not CPU-bound like build_manifest.py's
ProcessPoolExecutor) fetch records concurrently; the two metadata CSVs are always fetched serially,
first, since every record's URL is derived from `ptbxl_database.csv`'s own `filename_hr` column.

    uv run python scripts/fetch_ptbxl.py --data-root ~/data/ptbxl
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"

#: Recorded once, from this repo's own `artifacts/lead_stats_f1to9_summary.json`
#: (`integrity.sha256`) and cross-checked against the consumed fold-10 authorization record's
#: `frozen_inputs.metadata_sha256` -- both agree. A fetched CSV that does not hash to this is not
#: the PTB-XL 1.0.3 release the published numbers were computed on.
_EXPECTED_CSV_SHA256 = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}

_EXPECTED_N_RECORDS = 21799


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str, *, retries: int = 3) -> None:
    """`urllib`, not `requests` -- this repo has no HTTP client dependency and one file's worth of
    stdlib code is not worth adding one. Retries on any exception (PhysioNet occasionally resets
    a connection mid-transfer under concurrent load); the last attempt's exception propagates."""
    tmp = dest + ".part"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
                out.write(resp.read())
            os.replace(tmp, dest)
            return
        except Exception as exc:  # noqa: BLE001 -- retried below; only the last one propagates
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    if os.path.exists(tmp):
        os.remove(tmp)
    assert last_exc is not None
    raise last_exc


def _fetch_metadata_csv(name: str, data_root: str) -> None:
    dest = os.path.join(data_root, name)
    if os.path.isfile(dest) and _sha256_file(dest) == _EXPECTED_CSV_SHA256[name]:
        print(f"[fetch_ptbxl] {name}: already present, sha256 verified", flush=True)
        return
    print(f"[fetch_ptbxl] fetching {name}...", flush=True)
    _download(f"{_BASE_URL}/{name}", dest)
    actual = _sha256_file(dest)
    expected = _EXPECTED_CSV_SHA256[name]
    if actual != expected:
        raise ValueError(
            f"{name}: sha256 mismatch after download -- got {actual}, expected {expected}. "
            "This is not the PTB-XL 1.0.3 release this repo's published numbers were computed "
            "on; refusing to proceed with a manifest built against different data."
        )
    print(f"[fetch_ptbxl] {name}: sha256 verified", flush=True)


def _fetch_one_record(stem: str, data_root: str) -> str | None:
    """`stem` is a `filename_hr` value, e.g. `records500/00000/00001_hr` -- fetches the `.hea` and
    `.dat` pair. Returns an error string on failure (for the caller to collect), None on success
    or if both files were already present."""
    hea_dest = os.path.join(data_root, stem + ".hea")
    dat_dest = os.path.join(data_root, stem + ".dat")
    if os.path.isfile(hea_dest) and os.path.getsize(hea_dest) > 0:
        if os.path.isfile(dat_dest) and os.path.getsize(dat_dest) > 0:
            return None
    os.makedirs(os.path.dirname(hea_dest), exist_ok=True)
    try:
        _download(f"{_BASE_URL}/{stem}.hea", hea_dest)
        _download(f"{_BASE_URL}/{stem}.dat", dat_dest)
    except Exception as exc:  # noqa: BLE001 -- collected and reported by the caller
        return f"{stem}: {type(exc).__name__}: {exc}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="destination directory (created if absent)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent record downloads")
    ap.add_argument(
        "--limit", type=int, default=0, help="fetch only the first N records (smoke, 0 = all)"
    )
    args = ap.parse_args(argv)

    os.makedirs(args.data_root, exist_ok=True)

    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        _fetch_metadata_csv(name, args.data_root)

    meta = pd.read_csv(
        os.path.join(args.data_root, "ptbxl_database.csv"), usecols=["ecg_id", "filename_hr"]
    )
    if len(meta) != _EXPECTED_N_RECORDS:
        raise ValueError(
            f"ptbxl_database.csv has {len(meta)} rows, expected {_EXPECTED_N_RECORDS} -- "
            "the sha256 check above already confirmed this is the right file, so this would "
            "indicate a bug in this script's row count, not a data problem."
        )
    stems = meta["filename_hr"].tolist()
    if args.limit:
        stems = stems[: args.limit]

    print(f"[fetch_ptbxl] fetching {len(stems)} records500/ .hea+.dat pairs...", flush=True)
    t0 = time.time()
    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch_one_record, s, args.data_root): s for s in stems}
        for future in as_completed(futures):
            done += 1
            err = future.result()
            if err is not None:
                failures.append(err)
            if done % 1000 == 0 or done == len(stems):
                print(f"[fetch_ptbxl] {done}/{len(stems)} records", flush=True)

    elapsed = time.time() - t0
    print(f"[fetch_ptbxl] done in {elapsed:.0f}s, {len(failures)} failure(s)", flush=True)
    if failures:
        for f in failures[:20]:
            print(f"[fetch_ptbxl] FAILED: {f}", flush=True)
        if len(failures) > 20:
            print(f"[fetch_ptbxl] ... and {len(failures) - 20} more", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

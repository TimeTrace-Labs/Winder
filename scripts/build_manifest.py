#!/usr/bin/env python3
"""Phase extraction, QC, and the per-record manifest, for the real PTB-XL corpus.

Ported near-verbatim from ttl-phase's `scripts/s0_phase_manifest.py` (pinned commit, see
notes/build.md). This is where "data access" (regenerate, not import ttl-phase's artifacts)
actually happens: R-peak detection runs directly on the raw 500 Hz signal, never on a decimated
one, so this script depends only on `winder.data.wfdb_io`, `winder.data.phase`,
`winder.data.ptbxl`, `winder.data.manifest` -- not `winder.data.decimation`.

Outputs (under --artifacts-dir, default ./artifacts, gitignored):
  manifest.parquet         per-record ledger, every record accounted for
  phase/rpeaks.npz         ragged refined R-peaks + offsets + ecg_ids (INCLUDED only) --
                           the single source of truth for the phase clock; theta/bin_id
                           for any B are derived on demand via phase.phase_from_rpeaks /
                           phase.bin_phase, never cached at full resolution here.
  phase/s0_summary.json    yields, reason table, superclass yield, provenance

Bugs the reference repo fixed here, carried forward by this port (see winder.data.manifest's own
module docstring for the fuller account):
  sex is written as float('nan') for a missing value, matching RecordRow.sex's documented type --
      not -1.
  every flag phase.py can emit is mapped to a manifest reason code, in an explicit precedence
      order, and _assert_flag_coverage() checks this is total against phase.ALL_FLAGS *before*
      processing a single record -- not discovered after the fact by a record silently carrying
      an unmapped flag through to status="included".

Parallelism: the worker function is pure numpy (wfdb_io + phase, no pandas/pyarrow import),
and every task carries its own absolute .hea path rather than resolving one from a module
global -- so a spawned worker process needs nothing from this module's import-time state.
Metadata assembly (pandas) happens only in the parent, after collecting worker results.
"""

import argparse
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from winder.data.manifest import NO_REASON, Manifest
from winder.data.phase import ALL_FLAGS as PHASE_ALL_FLAGS
from winder.data.phase import (
    FLAG_FLAT_SIGNAL,
    FLAG_HIGH_RR_CV,
    FLAG_IMPLAUSIBLE_RR,
    FLAG_LOW_CONFIDENCE,
    FLAG_LOW_YIELD,
    FLAG_NO_BEATS,
    FLAG_RR_OUTLIERS,
    FLAG_TOO_FEW_BEATS,
    DetectorParams,
    PhaseQCConfig,
    extract_phase,
)
from winder.data.ptbxl import MULTIHOT_COLS, load_metadata
from winder.data.wfdb_io import read_record
from winder.paths import default_data_root

# ------------------------------------------------ flag -> manifest reason code
# Precedence matters: report the most basic failure. A flat signal explains why beat
# detection failed, so it's checked first; NO_BEATS is folded into TOO_FEW_BEATS (0 < any
# positive minimum). LOW_CONFIDENCE sits last because it's opt-in (inactive by default,
# see PhaseQCConfig.min_detector_confidence) and least specific about the cause.
_FLAG_PRECEDENCE: tuple[tuple[str, str], ...] = (
    (FLAG_FLAT_SIGNAL, "FLAT_SIGNAL"),
    (FLAG_NO_BEATS, "TOO_FEW_BEATS"),
    (FLAG_TOO_FEW_BEATS, "TOO_FEW_BEATS"),
    (FLAG_IMPLAUSIBLE_RR, "IMPLAUSIBLE_RR"),
    (FLAG_RR_OUTLIERS, "RR_OUTLIERS"),
    (FLAG_HIGH_RR_CV, "HIGH_RR_CV"),
    (FLAG_LOW_YIELD, "LOW_PHASE_YIELD"),
    (FLAG_LOW_CONFIDENCE, "LOW_CONFIDENCE"),
)


def _assert_flag_coverage() -> None:
    """Every flag phase.py can emit must map to a reason code. Run once, before touching
    any record -- a silent gap here is exactly how a flagged record used to slip through
    to status="included" unnoticed."""
    mapped = {flag for flag, _ in _FLAG_PRECEDENCE}
    unmapped = set(PHASE_ALL_FLAGS) - mapped
    if unmapped:
        raise AssertionError(
            f"phase.py can emit flags with no manifest reason-code mapping: "
            f"{sorted(unmapped)}. A record carrying one of these would be silently "
            f"included. Add a mapping entry in _FLAG_PRECEDENCE (and, if needed, a new "
            f"code to manifest.REASON_CODES)."
        )


def _reason_for(flags: list[str]) -> str | None:
    """First matching reason code in precedence order, or None if nothing mapped fired."""
    for flag, reason in _FLAG_PRECEDENCE:
        if flag in flags:
            return reason
    return None


def _process_one(
    task: tuple[int, str], qc: PhaseQCConfig, detector: DetectorParams, jitter: bool
) -> dict[str, Any]:
    """One record. Returns a plain, picklable dict. Pure numpy: no pandas/pyarrow import,
    no module-global path -- everything the worker needs is in `task`."""
    ecg_id, hea_path = task
    out: dict[str, Any] = {"ecg_id": ecg_id}
    try:
        sig, _hdr = read_record(hea_path)
    except Exception as exc:
        out.update(
            status="excluded",
            reason_code="READ_ERROR",
            reason_detail=f"{type(exc).__name__}: {exc}"[:200],
        )
        return out

    if sig.shape != (5000, 12):
        out.update(status="excluded", reason_code="WRONG_SHAPE", reason_detail=str(sig.shape))
        return out
    if not np.isfinite(sig).all():
        out.update(status="excluded", reason_code="NAN", reason_detail="non-finite sample")
        return out

    try:
        pr = extract_phase(
            sig, fs=500, qc=qc, params=detector, estimate_jitter=jitter, jitter_seed=ecg_id
        )
    except Exception:
        # Matches ttl-phase's own convention of filing a phase-extraction crash under
        # READ_ERROR (a mild misnomer, inherited rather than fixed here) -- the traceback in
        # reason_detail is what actually makes it diagnosable.
        out.update(
            status="excluded",
            reason_code="READ_ERROR",
            reason_detail=("phase: " + traceback.format_exc())[:200],
        )
        return out

    q = pr.quality
    flags = list(q["flags"])
    reason = _reason_for(flags)
    out.update(
        status=("excluded" if reason else "included"),
        reason_code=(reason or NO_REASON),
        reason_detail=";".join(flags),
        n_beats=int(pr.n_beats),
        phase_yield=float(q["phase_yield"]),
        rr_mean_ms=float(q["rr_mean_ms"]),
        rr_median_ms=float(q["rr_median_ms"]),
        rr_sd_ms=float(q["rr_sd_ms"]),
        rr_cv=float(q["rr_cv"]),
        jitter_ms=float(q["jitter_ms"]),
        quality_flags=flags,
        rpeaks=np.asarray(pr.rpeaks, dtype=np.float64),
    )
    return out


def _str_or_empty(v: Any) -> str:
    return "" if pd.isna(v) else str(v)


def _float_or_nan(v: Any) -> float:
    return float("nan") if pd.isna(v) else float(v)


def _git_sha(root: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=default_data_root(),
        help="PTB-XL root (ptbxl_database.csv, scp_statements.csv, records500/)",
    )
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N records (smoke)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--no-jitter", action="store_true", help="skip jitter_estimate (faster smoke runs)"
    )
    args = ap.parse_args(argv)

    _assert_flag_coverage()

    art = args.artifacts_dir
    os.makedirs(os.path.join(art, "phase"), exist_ok=True)

    t0 = time.time()
    meta = load_metadata(args.data_root)
    print(f"[build_manifest] metadata rows: {len(meta)}", flush=True)
    print(
        f"[build_manifest] superclass histogram:\n{meta['superclass'].value_counts()}", flush=True
    )

    if args.limit:
        meta = meta.iloc[: args.limit].copy()
    # column-vectorized rather than itertuples(): itertuples() gives each attribute a huge
    # per-row union type across pandas-stubs' dtype inference, which is both slow to
    # type-check and imprecise.
    ecg_ids = meta["ecg_id"].to_numpy()
    filenames = meta["filename_hr"].to_numpy()
    tasks = [
        (int(eid), os.path.join(args.data_root, str(fname) + ".hea"))
        for eid, fname in zip(ecg_ids, filenames, strict=True)
    ]

    qc = PhaseQCConfig()
    detector = DetectorParams()
    jitter = not args.no_jitter

    results: list[dict[str, Any]] = []
    done = 0
    n = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        mapped = ex.map(_process_one, tasks, [qc] * n, [detector] * n, [jitter] * n, chunksize=32)
        for res in mapped:
            results.append(res)
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(
                    f"[build_manifest] {done}/{n}  {el:.0f}s  {done / el:.1f} rec/s "
                    f"eta {(n - done) / (done / el) / 60:.1f} min",
                    flush=True,
                )
    print(f"[build_manifest] extraction done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- manifest + rpeaks archive ----
    man = Manifest()
    meta_idx = meta.set_index("ecg_id")
    rp_all: list[np.ndarray] = []
    rp_off = [0]
    rp_ids: list[int] = []
    for r in sorted(results, key=lambda z: z["ecg_id"]):
        eid = r["ecg_id"]
        m = meta_idx.loc[eid]
        fields: dict[str, Any] = dict(
            patient_id=int(m["patient_id"]),
            strat_fold=int(m["strat_fold"]),
            superclass=str(m["superclass"]),
            superclasses=tuple(int(m[c]) for c in MULTIHOT_COLS),
            # age=300 (293 records) is PTB-XL's own HIPAA de-identification top-code for
            # patients aged >=90, not a missing-value sentinel: the raw value carries real
            # information ("90 or older") that a NaN would destroy, and there is no
            # analogous schema contradiction to fix -- passed through unmodified.
            age=_float_or_nan(m["age"]),
            sex=_float_or_nan(m["sex"]),  # NaN for missing, never -1
            device=_str_or_empty(m.get("device")),
            site=_str_or_empty(m.get("site")),
        )
        for k in (
            "n_beats",
            "phase_yield",
            "rr_mean_ms",
            "rr_median_ms",
            "rr_sd_ms",
            "rr_cv",
            "jitter_ms",
            "quality_flags",
        ):
            if k in r:
                fields[k] = r[k]
        if r["status"] == "included":
            man.add_included(eid, **fields)
            rp = r["rpeaks"]
            rp_all.append(rp)
            rp_off.append(rp_off[-1] + len(rp))
            rp_ids.append(eid)
        else:
            man.add_excluded(
                eid,
                reason_code=r["reason_code"],
                reason_detail=r.get("reason_detail", ""),
                **fields,
            )

    man.assert_accounts_for(len(tasks))
    mp = man.to_parquet(os.path.join(art, "manifest.parquet"))
    print(
        f"[build_manifest] manifest -> {mp}  included={man.n_included} excluded={man.n_excluded}",
        flush=True,
    )

    np.savez_compressed(
        os.path.join(art, "phase", "rpeaks.npz"),
        rpeaks=np.concatenate(rp_all) if rp_all else np.zeros(0),
        offsets=np.array(rp_off, dtype=np.int64),
        ecg_ids=np.array(rp_ids, dtype=np.int64),
        fs=np.array(500),
    )
    print(f"[build_manifest] rpeaks -> {art}/phase/rpeaks.npz  ({len(rp_ids)} records)", flush=True)

    # ---- summary + provenance ----
    d = man.to_dataframe()
    inc = d[d.status == "included"]
    summary: dict[str, Any] = {
        "n_total": int(len(tasks)),
        "n_included": int(man.n_included),
        "n_excluded": int(man.n_excluded),
        "inclusion_rate": float(man.n_included / len(tasks)),
        "reason_table": man.reason_table().reset_index().to_dict(orient="records"),
        "yield_by_superclass": man.summary(by="superclass").reset_index().to_dict(orient="records"),
        "n_beats": {k: float(v) for k, v in inc.n_beats.describe().items()},
        "rr_median_ms": {k: float(v) for k, v in inc.rr_median_ms.describe().items()},
        "phase_yield": {k: float(v) for k, v in inc.phase_yield.describe().items()},
        "jitter_ms": {k: float(v) for k, v in inc.jitter_ms.describe().items()},
        "total_beats": int(inc.n_beats.sum()),
        "elapsed_min": (time.time() - t0) / 60.0,
        "provenance": {
            "winder_git_sha": _git_sha(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data_root": os.path.abspath(args.data_root),
            "detector_params": detector.__dict__,
            "qc_config": {k: v for k, v in vars(qc).items()},
            "jitter_estimated": jitter,
            "limit": args.limit or None,
        },
    }
    with open(os.path.join(art, "phase", "s0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n[build_manifest] ===== SUMMARY =====")
    print(
        f"included {summary['n_included']}/{summary['n_total']} "
        f"({summary['inclusion_rate']:.3%}), total beats {summary['total_beats']:,}"
    )
    print(f"beats/record  median {summary['n_beats']['50%']:.1f}")
    print(f"RR median     {summary['rr_median_ms']['50%']:.1f} ms")
    print(f"phase yield   median {summary['phase_yield']['50%']:.4f}")
    print(f"JITTER (ms)   median {summary['jitter_ms']['50%']:.3f}")
    print("\nexclusion reasons:")
    print(man.reason_table().to_string())
    print("\nyield by superclass:")
    print(man.summary(by="superclass").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

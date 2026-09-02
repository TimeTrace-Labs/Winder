"""Tests for scripts/fit_lead_stats.py: the parameterised `--train-folds` lead-stats fitting
driver (Phase P8 prep, Task 1). Mirrors tests/test_s1_lead_stats.py's idiom (a synthetic PTB-XL
root, records500-only, DATA-04's decimate-to-100Hz bridge), extended to cover the fold-set CLI
flag and the fast fail-before-any-I/O validation this driver adds on top of the reference script.

`tests/conftest.py` puts `scripts/` on `sys.path`, so `import fit_lead_stats` here means
`scripts/fit_lead_stats.py`, not the `winder` package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fit_lead_stats
import numpy as np
import pandas as pd
import pytest

from winder.data.decimation import decimate_to
from winder.data.norm_stats import LeadStats
from winder.data.ptbxl import LEAD_ORDER
from winder.data.wfdb_io import read_record, write_format16


def _real_500hz_signal() -> np.ndarray:
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "wfdb")
    hea_path = sorted(
        os.path.join(fixtures_dir, f) for f in os.listdir(fixtures_dir) if f.endswith(".hea")
    )[0]
    sig, header = read_record(hea_path)
    assert header.fs == 500 and sig.shape == (5000, 12)
    return sig


def _write_records500_record(tmp_path: Path, stem_rel: str, sig: np.ndarray) -> None:
    stem = tmp_path / stem_rel
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_format16(str(stem), sig.astype(np.float64), fs=500, sig_name=list(LEAD_ORDER))


# --------------------------------------------------------------------------- _iter_signals


def test_iter_signals_reads_records500_and_decimates_to_100hz(tmp_path: Path) -> None:
    sig500 = _real_500hz_signal()
    _write_records500_record(tmp_path, "records500/00000/00001_hr", sig500)
    metadata = pd.DataFrame({"filename_hr": ["records500/00000/00001_hr"]})

    signals = list(fit_lead_stats._iter_signals(metadata, str(tmp_path)))
    assert len(signals) == 1
    assert signals[0].shape == (1000, 12)
    assert np.allclose(signals[0], decimate_to(sig500, 500, 100))


# --------------------------------------------------------------------------- _validate_train_folds


def test_validate_train_folds_accepts_the_default_nine_fold_pool() -> None:
    fit_lead_stats._validate_train_folds((1, 2, 3, 4, 5, 6, 7, 8, 9))  # must not raise


def test_validate_train_folds_accepts_the_legacy_eight_fold_subset() -> None:
    fit_lead_stats._validate_train_folds((1, 2, 3, 4, 5, 6, 7, 8))  # must not raise


def test_validate_train_folds_rejects_the_sealed_test_fold() -> None:
    with pytest.raises(ValueError, match=r"outside the non-leaking training-fold set"):
        fit_lead_stats._validate_train_folds((1, 2, 10))


def test_validate_train_folds_rejects_fold_zero() -> None:
    with pytest.raises(ValueError, match=r"outside the non-leaking training-fold set"):
        fit_lead_stats._validate_train_folds((0, 1, 2))


# --------------------------------------------------------------------------- main, end to end


def _write_synthetic_ptbxl_root(tmp_path: Path, n_records: int = 4) -> str:
    """`n_records` records spread over folds 1..9 in round-robin (record `i` -> fold
    `((i - 1) % 9) + 1`), so a `--train-folds` subset can be exercised against a mix of
    in-pool and out-of-pool records."""
    root = tmp_path / "ptbxl"
    sig500 = _real_500hz_signal()
    rows = []
    for i in range(1, n_records + 1):
        stem_rel = f"records500/00000/{i:05d}_hr"
        _write_records500_record(root, stem_rel, sig500 + float(i))  # distinct per record
        fold = ((i - 1) % 9) + 1
        rows.append(
            {
                "ecg_id": i,
                "patient_id": 100 + i,
                "age": 50,
                "sex": 0,
                "height": np.nan,
                "weight": 70.0,
                "device": "CS-12   E",
                "site": 0.0,
                "recording_date": "1984-11-09 09:17:34",
                "scp_codes": "{'NORM': 100.0, 'SR': 0.0}",
                "strat_fold": fold,
                "filename_lr": f"records100/00000/{i:05d}_lr",
                "filename_hr": stem_rel,
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        [
            {
                "": "NORM",
                "description": "normal ecg",
                "diagnostic": 1.0,
                "form": np.nan,
                "rhythm": np.nan,
                "diagnostic_class": "NORM",
                "diagnostic_subclass": "NORM",
            }
        ]
    ).to_csv(root / "scp_statements.csv", index=False)
    return str(root)


def test_main_end_to_end_restricts_to_the_requested_train_folds(tmp_path: Path) -> None:
    root = _write_synthetic_ptbxl_root(tmp_path, n_records=9)  # one record per fold 1..9
    out_path = tmp_path / "artifacts" / "lead_stats_subset.json"

    exit_code = fit_lead_stats.main(
        ["--data-root", root, "--train-folds", "1,2,3", "--out-path", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.is_file()

    stats = LeadStats.from_json(str(out_path))
    assert stats.folds == (1, 2, 3)
    assert stats.n_records == 3  # only the 3 records in folds 1-3
    assert stats.n_samples == 3 * 1000

    summary_path = tmp_path / "artifacts" / "lead_stats_subset_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["n_records_fit"] == 3
    assert summary["train_folds"] == [1, 2, 3]


def test_main_fitting_folds_1_to_9_covers_every_record_in_a_nine_fold_synthetic_root(
    tmp_path: Path,
) -> None:
    root = _write_synthetic_ptbxl_root(tmp_path, n_records=9)
    out_path = tmp_path / "lead_stats_f1to9.json"

    exit_code = fit_lead_stats.main(
        ["--data-root", root, "--train-folds", "1,2,3,4,5,6,7,8,9", "--out-path", str(out_path)]
    )
    assert exit_code == 0
    stats = LeadStats.from_json(str(out_path))
    assert stats.folds == (1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert stats.n_records == 9


def test_main_rejects_a_leaked_sealed_fold_before_touching_any_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--data-root` points nowhere real -- if the fold-leak check were ever skipped, this would
    instead fail on a `FileNotFoundError` from `load_metadata` (a different, distinguishing
    failure), not silently pass."""
    with pytest.raises(SystemExit):
        fit_lead_stats.main(
            [
                "--data-root",
                "/definitely/not/a/real/path",
                "--train-folds",
                "1,2,10",
                "--out-path",
                str(tmp_path / "unused.json"),
            ]
        )


def test_main_rejects_malformed_train_folds() -> None:
    with pytest.raises(SystemExit):
        fit_lead_stats.main(
            [
                "--data-root",
                "/definitely/not/a/real/path",
                "--train-folds",
                "1,two,3",
                "--out-path",
                "/tmp/unused.json",
            ]
        )


def test_main_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    """`fit_lead_stats` has no RNG (module docstring) -- two runs over the same inputs must
    produce byte-identical mean_mv/std_mv, never merely close."""
    root = _write_synthetic_ptbxl_root(tmp_path, n_records=5)
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    argv_a = ["--data-root", root, "--train-folds", "1,2,3,4,5,6,7,8,9", "--out-path", str(out_a)]
    argv_b = ["--data-root", root, "--train-folds", "1,2,3,4,5,6,7,8,9", "--out-path", str(out_b)]
    assert fit_lead_stats.main(argv_a) == 0
    assert fit_lead_stats.main(argv_b) == 0
    stats_a = LeadStats.from_json(str(out_a))
    stats_b = LeadStats.from_json(str(out_b))
    assert stats_a.mean_mv == stats_b.mean_mv
    assert stats_a.std_mv == stats_b.std_mv

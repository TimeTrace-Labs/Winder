import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from winder.data.decimation import decimate_to
from winder.data.norm_stats import LeadStats, fit_lead_stats
from winder.data.normalization import CorpusStatsNormConfig
from winder.data.ptbxl import LEAD_ORDER
from winder.data.wfdb_io import read_record


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "leads": LEAD_ORDER,
        "mean_mv": tuple(0.0 for _ in LEAD_ORDER),
        "std_mv": tuple(1.0 for _ in LEAD_ORDER),
        "fs": 100,
        "folds": (1, 2, 3),
        "n_records": 10,
        "n_samples": 10_000,
        "winder_git_sha": "deadbeef",
        "created_utc": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_valid_lead_stats_constructs() -> None:
    stats = LeadStats(**_valid_kwargs())
    assert stats.leads == LEAD_ORDER


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="matching lengths"):
        LeadStats(**_valid_kwargs(mean_mv=(0.0,) * 5))


def test_rejects_wrong_lead_order() -> None:
    with pytest.raises(ValidationError, match="LEAD_ORDER"):
        LeadStats(**_valid_kwargs(leads=tuple(reversed(LEAD_ORDER))))


def test_rejects_non_finite_mean() -> None:
    bad = (float("nan"), *[0.0] * (len(LEAD_ORDER) - 1))
    with pytest.raises(ValidationError, match="finite"):
        LeadStats(**_valid_kwargs(mean_mv=bad))


def test_rejects_nonpositive_std() -> None:
    bad = (0.0, *[1.0] * (len(LEAD_ORDER) - 1))
    with pytest.raises(ValidationError, match="std_mv"):
        LeadStats(**_valid_kwargs(std_mv=bad))


def test_rejects_validation_or_sealed_fold_leakage() -> None:
    # winder-nominal deviation: FoldConfig()'s default train_folds is (1..9) here (val_fold
    # moved to the sentinel 0 -- see winder.data.folds), so fold 9 is now a legitimate
    # training fold and no longer exercises this check. Fold 0 (the new val_fold) plays the
    # role fold 9 played against the reference repo's original default.
    with pytest.raises(ValidationError, match="training folds"):
        LeadStats(**_valid_kwargs(folds=(1, 2, 10)))
    with pytest.raises(ValidationError, match="training folds"):
        LeadStats(**_valid_kwargs(folds=(1, 0)))


def test_json_round_trip(tmp_path: Path) -> None:
    stats = LeadStats(**_valid_kwargs())
    path = stats.to_json(str(tmp_path / "lead_stats.json"))
    loaded = LeadStats.from_json(path)
    assert loaded == stats


def test_to_norm_config_bridge() -> None:
    stats = LeadStats(
        **_valid_kwargs(
            mean_mv=tuple(float(i) for i in range(len(LEAD_ORDER))),
            std_mv=tuple(float(i + 1) for i in range(len(LEAD_ORDER))),
        )
    )
    config = stats.to_norm_config()
    assert isinstance(config, CorpusStatsNormConfig)
    assert list(config.mean_mv) == list(stats.mean_mv)
    assert list(config.std_mv) == list(stats.std_mv)


def test_fit_lead_stats_matches_direct_numpy_computation() -> None:
    """Real signal statistics on the 10 committed fixture records, decimated to 100 Hz -- the
    idiom from tests/test_decimation.py:72-86."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "wfdb")
    hea_paths = sorted(glob.glob(os.path.join(fixtures_dir, "*.hea")))
    assert len(hea_paths) == 10
    signals = []
    for p in hea_paths:
        sig, header = read_record(p)
        signals.append(decimate_to(sig, header.fs, 100))

    stats = fit_lead_stats(iter(signals), leads=LEAD_ORDER, fs=100, folds=(1, 2, 3))
    assert stats.n_records == 10

    all_samples = np.concatenate(signals, axis=0).astype(np.float64)
    expected_mean = all_samples.mean(axis=0)
    expected_std = all_samples.std(axis=0)
    np.testing.assert_allclose(stats.mean_mv, expected_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.std_mv, expected_std, rtol=1e-7, atol=1e-7)


def test_fit_lead_stats_empty_raises() -> None:
    with pytest.raises(ValueError, match="no records"):
        fit_lead_stats(iter([]), leads=LEAD_ORDER, fs=100, folds=(1,))


def test_fit_lead_stats_wrong_lead_count_raises() -> None:
    bad = np.ones((100, 5))
    with pytest.raises(ValueError, match="expected each record"):
        fit_lead_stats(iter([bad]), leads=LEAD_ORDER, fs=100, folds=(1,))


def test_fit_lead_stats_dedupes_and_sorts_folds() -> None:
    sig = np.random.default_rng(0).normal(size=(100, len(LEAD_ORDER)))
    stats = fit_lead_stats(iter([sig]), leads=LEAD_ORDER, fs=100, folds=(3, 1, 3, 2))
    assert stats.folds == (1, 2, 3)

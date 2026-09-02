import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from winder.data.decimation import decimate_to
from winder.data.norm_stats import LeadStats
from winder.data.normalization import CorpusStatsNormConfig
from winder.data.ptbxl import LEAD_ORDER, MULTIHOT_COLS
from winder.data.wfdb_io import read_record, write_format16
from winder.jepa.dataset import EcgWindowDataset


def _raw_lead_stats() -> LeadStats:
    """mean=0/std=1 per lead -- a no-op affine transform, used by tests whose own concern is
    decimation/shape/error-propagation, not normalization, so they can keep asserting against
    the un-normalized `decimate_to` output unchanged."""
    return LeadStats(
        leads=LEAD_ORDER,
        mean_mv=tuple(0.0 for _ in LEAD_ORDER),
        std_mv=tuple(1.0 for _ in LEAD_ORDER),
        fs=100,
        folds=(1, 2, 3),
        n_records=10,
        n_samples=10_000,
        winder_git_sha="deadbeef",
        created_utc="2026-01-01T00:00:00+00:00",
    )


def _lead_stats(**overrides: Any) -> LeadStats:
    """A fitted-artifact fixture with a non-trivial (non-0/1) mean/std per lead, so a test
    comparing normalized output against raw input actually exercises the affine transform
    rather than coincidentally matching a no-op mean=0/std=1 config."""
    base: dict[str, Any] = {
        "leads": LEAD_ORDER,
        "mean_mv": tuple(0.1 * i for i in range(len(LEAD_ORDER))),
        "std_mv": tuple(2.0 + 0.5 * i for i in range(len(LEAD_ORDER))),
        "fs": 100,
        "folds": (1, 2, 3),
        "n_records": 10,
        "n_samples": 10_000,
        "winder_git_sha": "deadbeef",
        "created_utc": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return LeadStats(**base)


def test_missing_required_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        EcgWindowDataset(pd.DataFrame({"ecg_id": [1]}), data_root="/tmp", lead_stats=_lead_stats())


def test_missing_lead_stats_argument_raises() -> None:
    """DATA-02: lead_stats has no default -- omitting it is a TypeError from Python's own
    signature enforcement, not a dataset that silently returns an unnormalized waveform."""
    with pytest.raises(TypeError):
        EcgWindowDataset(pd.DataFrame({"ecg_id": [1]}), data_root="/tmp")  # type: ignore[call-arg]


def test_rejects_a_lead_stats_of_the_wrong_type() -> None:
    """Passing the resolved config (or anything else) instead of the fitted LeadStats artifact
    must raise loudly, not silently skip normalization or crash inside __getitem__."""
    bad = CorpusStatsNormConfig(mean_mv=[0.0] * len(LEAD_ORDER), std_mv=[1.0] * len(LEAD_ORDER))
    metadata = pd.DataFrame([_metadata_row("records500/00000/00099_hr")])
    with pytest.raises(TypeError, match="LeadStats"):
        EcgWindowDataset(metadata, data_root="/tmp", lead_stats=bad)  # type: ignore[arg-type]


def test_rejects_lead_stats_fitted_at_the_wrong_rate() -> None:
    metadata = pd.DataFrame([_metadata_row("records500/00000/00098_hr")])
    with pytest.raises(ValueError, match="fs=500"):
        EcgWindowDataset(metadata, data_root="/tmp", lead_stats=_lead_stats(fs=500))


def test_no_alternate_normalization_parameter_exists() -> None:
    """Statically documents CM-08's guarantee: the only normalization-shaped constructor
    parameter is `lead_stats: LeadStats` -- there is no `norm`/`mode`/`config` parameter a
    caller could point at PerBeatNormConfig or the raw/"perbeat" string tags."""
    import inspect

    params = inspect.signature(EcgWindowDataset.__init__).parameters
    assert set(params) == {"self", "metadata", "data_root", "lead_stats"}
    assert params["lead_stats"].annotation is LeadStats


def _metadata_row(
    filename_hr: str, *, filename_lr: str = "records100/00000/absent_lr", **sc_overrides: int
) -> dict[str, object]:
    row: dict[str, object] = {
        "ecg_id": 1,
        "patient_id": 7,
        "strat_fold": 3,
        "filename_lr": filename_lr,
        "filename_hr": filename_hr,
    }
    for c in MULTIHOT_COLS:
        row[c] = sc_overrides.get(c, 0)
    return row


def _write_fake_records500_record(tmp_path: Path, stem_rel: str, sig: np.ndarray) -> None:
    stem = tmp_path / stem_rel
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_format16(str(stem), sig.astype(np.float64), fs=500, sig_name=list(LEAD_ORDER))


def _real_500hz_signal() -> np.ndarray:
    """Real signal statistics, not synthetic noise -- one of the 10 committed fixture
    records, already at 500 Hz natively (no decimation needed to produce it)."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "wfdb")
    hea_path = sorted(
        os.path.join(fixtures_dir, f) for f in os.listdir(fixtures_dir) if f.endswith(".hea")
    )[0]
    sig, header = read_record(hea_path)
    assert header.fs == 500 and sig.shape == (5000, 12)
    return sig


def test_reads_a_records500_record_and_decimates_to_100hz(tmp_path: Path) -> None:
    sig500 = _real_500hz_signal()
    _write_fake_records500_record(tmp_path, "records500/00000/00001_hr", sig500)
    metadata = pd.DataFrame([_metadata_row("records500/00000/00001_hr")])

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())
    assert len(dataset) == 1
    item = dataset[0]

    assert item["waveform"].shape == (12, 1000)
    assert item["waveform"].dtype == torch.float32
    assert item["ecg_id"] == 1
    assert item["patient_id"] == 7
    assert item["strat_fold"] == 3
    assert item["labels"].shape == (5,)
    assert item["has_label"] is False  # all sc_* columns are 0

    expected = decimate_to(sig500, 500, 100).T
    assert np.allclose(item["waveform"].numpy(), expected)


def test_loads_correctly_when_no_native_records100_file_exists_at_all(tmp_path: Path) -> None:
    """DATA-04's bridge: `EcgWindowDataset` never reads `records100/` at all, so a record
    whose `records100/` file is absent -- true for 98.6% of the corpus locally -- still
    loads correctly via `records500/` + `decimate_to`. `data_root` here has no `records100/`
    directory whatsoever, not just a missing file for this one record."""
    sig500 = _real_500hz_signal()
    _write_fake_records500_record(tmp_path, "records500/00000/00042_hr", sig500)
    assert not (tmp_path / "records100").exists()
    metadata = pd.DataFrame(
        [_metadata_row("records500/00000/00042_hr", filename_lr="records100/00000/00042_lr")]
    )

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())
    item = dataset[0]
    assert item["waveform"].shape == (12, 1000)


def test_has_label_true_when_any_superclass_is_set(tmp_path: Path) -> None:
    _write_fake_records500_record(tmp_path, "records500/00000/00002_hr", _real_500hz_signal())
    metadata = pd.DataFrame([_metadata_row("records500/00000/00002_hr", sc_NORM=1)])

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())
    item = dataset[0]
    assert item["has_label"] is True
    assert item["labels"][0] == 1.0  # sc_NORM is the first of MULTIHOT_COLS


def test_wrong_shape_raises(tmp_path: Path) -> None:
    bad_sig = np.zeros((500, 12))  # wrong sample count -- records500/ should be exactly 5000
    _write_fake_records500_record(tmp_path, "records500/00000/00003_hr", bad_sig)
    metadata = pd.DataFrame([_metadata_row("records500/00000/00003_hr")])

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())
    with pytest.raises(ValueError, match=r"expected \(5000, 12\)"):
        dataset[0]


def test_wrong_source_fs_raises(tmp_path: Path) -> None:
    """A records500/ header that doesn't actually declare 500 Hz must raise rather than be
    silently decimated at the wrong ratio -- mirrors ttl-phase's own `iter_signals` fs
    assertion (src/data/ptbxl.py:518)."""
    stem = tmp_path / "records500" / "00000" / "00004_hr"
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_format16(str(stem), np.zeros((5000, 12)), fs=250, sig_name=list(LEAD_ORDER))
    metadata = pd.DataFrame([_metadata_row("records500/00000/00004_hr")])

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())
    with pytest.raises(ValueError, match="expected 500"):
        dataset[0]


def test_non_finite_waveform_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # read_and_decimate_500hz (winder.data.ptbxl) is where read_record is actually called now
    # -- EcgWindowDataset delegates to it rather than calling read_record itself.
    import winder.data.ptbxl as ptbxl_module

    def _fake_read_record(
        hea_path: str, expected_sig_name: object = None
    ) -> tuple[np.ndarray, object]:
        class _Header:
            fs = 500

        return np.full((5000, 12), np.nan, dtype=np.float32), _Header()

    monkeypatch.setattr(ptbxl_module, "read_record", _fake_read_record)
    metadata = pd.DataFrame([_metadata_row("does/not/matter")])
    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())
    with pytest.raises(ValueError, match="non-finite"):
        dataset[0]


# --------------------------------------------------------------------- DATA-02: normalization
def test_normalization_is_applied_and_matches_a_manual_zscore(tmp_path: Path) -> None:
    """Values must actually change from the raw decimated signal, in exactly the direction a
    per-lead z-score predicts -- not merely "some transform happened"."""
    sig500 = _real_500hz_signal()
    _write_fake_records500_record(tmp_path, "records500/00000/00005_hr", sig500)
    metadata = pd.DataFrame([_metadata_row("records500/00000/00005_hr")])
    stats = _lead_stats()

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=stats)
    item = dataset[0]

    decimated = decimate_to(sig500, 500, 100)  # (1000, 12), raw (un-normalized)
    mean = np.asarray(stats.mean_mv)
    std = np.asarray(stats.std_mv)
    expected_normalized = ((decimated - mean) / std).T  # (12, 1000)

    assert np.allclose(item["waveform"].numpy(), expected_normalized, atol=1e-5)
    # Sanity check the transform is non-trivial: normalized output must actually differ from
    # the raw decimated signal (guards against a no-op mean=0/std=1 slipping into the fixture).
    assert not np.allclose(item["waveform"].numpy(), decimated.T, atol=1e-3)


def test_normalization_uses_the_exact_stats_passed_in_not_a_refit(tmp_path: Path) -> None:
    """Two datasets over the identical record, differing only in which LeadStats they were
    constructed with, must produce different normalized output -- proof the class applies the
    caller's own stats rather than deriving/caching its own from the data it reads."""
    sig500 = _real_500hz_signal()
    _write_fake_records500_record(tmp_path, "records500/00000/00006_hr", sig500)
    metadata = pd.DataFrame([_metadata_row("records500/00000/00006_hr")])

    stats_a = _lead_stats(
        mean_mv=tuple(0.0 for _ in LEAD_ORDER), std_mv=tuple(1.0 for _ in LEAD_ORDER)
    )
    stats_b = _lead_stats(
        mean_mv=tuple(5.0 for _ in LEAD_ORDER), std_mv=tuple(3.0 for _ in LEAD_ORDER)
    )

    item_a = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=stats_a)[0]
    item_b = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=stats_b)[0]

    assert not np.allclose(item_a["waveform"].numpy(), item_b["waveform"].numpy())
    decimated = decimate_to(sig500, 500, 100)
    expected_b = ((decimated - 5.0) / 3.0).T
    assert np.allclose(item_b["waveform"].numpy(), expected_b, atol=1e-5)


def test_normalization_survives_a_json_round_trip(tmp_path: Path) -> None:
    """DATA-02's acceptance is against a *loaded* lead_stats.json, not an in-memory LeadStats --
    exercise the actual to_json -> from_json bridge s2_pretrain_jepa.py will use."""
    sig500 = _real_500hz_signal()
    _write_fake_records500_record(tmp_path, "records500/00000/00007_hr", sig500)
    metadata = pd.DataFrame([_metadata_row("records500/00000/00007_hr")])

    stats = _lead_stats()
    stats_path = stats.to_json(str(tmp_path / "lead_stats.json"))
    loaded = LeadStats.from_json(stats_path)

    dataset = EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=loaded)
    item = dataset[0]

    decimated = decimate_to(sig500, 500, 100)
    mean = np.asarray(stats.mean_mv)
    std = np.asarray(stats.std_mv)
    expected_normalized = ((decimated - mean) / std).T
    assert np.allclose(item["waveform"].numpy(), expected_normalized, atol=1e-5)

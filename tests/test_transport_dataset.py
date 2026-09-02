import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from winder.data.norm_stats import LeadStats
from winder.data.ptbxl import LEAD_ORDER, MULTIHOT_COLS
from winder.data.wfdb_io import read_record, write_format16
from winder.jepa.dataset import EcgWindowDataset
from winder.transport.dataset import PhaseTaggedDataset, load_theta_tokens

N_TOKENS = 125
PATCH_WIDTH = 8


def _raw_lead_stats() -> LeadStats:
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


def _metadata_row(ecg_id: int, filename_hr: str) -> dict[str, object]:
    row: dict[str, object] = {
        "ecg_id": ecg_id,
        "patient_id": 7,
        "strat_fold": 3,
        "filename_lr": "records100/00000/absent_lr",
        "filename_hr": filename_hr,
    }
    for c in MULTIHOT_COLS:
        row[c] = 0
    return row


def _write_fake_records500_record(tmp_path: Path, stem_rel: str, sig: np.ndarray) -> None:
    stem = tmp_path / stem_rel
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_format16(str(stem), sig.astype(np.float64), fs=500, sig_name=list(LEAD_ORDER))


def _real_500hz_signal() -> np.ndarray:
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "wfdb")
    hea_path = sorted(
        os.path.join(fixtures_dir, f) for f in os.listdir(fixtures_dir) if f.endswith(".hea")
    )[0]
    sig, header = read_record(hea_path)
    assert header.fs == 500 and sig.shape == (5000, 12)
    return sig


def _build_base(tmp_path: Path, ecg_ids: list[int]) -> EcgWindowDataset:
    sig500 = _real_500hz_signal()
    rows = []
    for ecg_id in ecg_ids:
        stem = f"records500/00000/{ecg_id:05d}_hr"
        _write_fake_records500_record(tmp_path, stem, sig500)
        rows.append(_metadata_row(ecg_id, stem))
    metadata = pd.DataFrame(rows)
    return EcgWindowDataset(metadata, data_root=str(tmp_path), lead_stats=_raw_lead_stats())


def _write_theta_tokens(tmp_path: Path, ecg_id_to_theta: dict[int, np.ndarray]) -> str:
    npz_path = str(tmp_path / "theta_tokens.npz")
    ecg_ids = np.array(sorted(ecg_id_to_theta))
    theta = np.stack([ecg_id_to_theta[i] for i in ecg_ids])
    np.savez(
        npz_path,
        ecg_ids=ecg_ids,
        theta=theta,
        patch_width=PATCH_WIDTH,
        n_tokens=N_TOKENS,
        decimation_factor=5.0,
        timestamp="centre",
    )
    return npz_path


# ==================================================================== M2-C1: waveform identity


def test_wrapped_item_matches_base_dataset_on_every_shared_field(tmp_path: Path) -> None:
    base = _build_base(tmp_path, [1, 2])
    theta_npz = _write_theta_tokens(tmp_path, {1: np.linspace(0, 6.0, N_TOKENS, dtype=np.float32)})
    by_id, meta = load_theta_tokens(theta_npz)
    wrapped = PhaseTaggedDataset(base, by_id, meta, n_tokens=N_TOKENS, patch_width=PATCH_WIDTH)

    for i in range(len(base)):
        base_item = base[i]
        wrapped_item = wrapped[i]
        assert torch.equal(wrapped_item["waveform"], base_item["waveform"])
        assert torch.equal(wrapped_item["labels"], base_item["labels"])
        assert wrapped_item["ecg_id"] == base_item["ecg_id"]
        assert wrapped_item["patient_id"] == base_item["patient_id"]
        assert wrapped_item["strat_fold"] == base_item["strat_fold"]
        assert wrapped_item["has_label"] == base_item["has_label"]


def test_len_delegates_to_base(tmp_path: Path) -> None:
    base = _build_base(tmp_path, [1, 2, 3])
    theta_npz = _write_theta_tokens(tmp_path, {1: np.zeros(N_TOKENS, dtype=np.float32)})
    by_id, meta = load_theta_tokens(theta_npz)
    wrapped = PhaseTaggedDataset(base, by_id, meta, n_tokens=N_TOKENS, patch_width=PATCH_WIDTH)
    assert len(wrapped) == len(base) == 3


# ============================================================= theta attachment, present/absent


def test_theta_attached_for_a_record_present_in_the_lookup(tmp_path: Path) -> None:
    base = _build_base(tmp_path, [1])
    expected_theta = np.linspace(0, 6.0, N_TOKENS, dtype=np.float32)
    theta_npz = _write_theta_tokens(tmp_path, {1: expected_theta})
    by_id, meta = load_theta_tokens(theta_npz)
    wrapped = PhaseTaggedDataset(base, by_id, meta, n_tokens=N_TOKENS, patch_width=PATCH_WIDTH)

    item = wrapped[0]
    assert item["theta"].shape == (N_TOKENS,)
    assert item["theta"].dtype == torch.float32
    np.testing.assert_allclose(item["theta"].numpy(), expected_theta)


def test_theta_is_all_nan_for_a_record_absent_from_the_lookup(tmp_path: Path) -> None:
    """A record excluded from M0's phase-QC pool (e.g. HIGH_RR_CV) still gets an item -- the
    transport loss excludes NaN-theta tokens on its own, so this must not raise."""
    base = _build_base(tmp_path, [1, 99])  # ecg_id=99 has no entry below
    theta_npz = _write_theta_tokens(tmp_path, {1: np.zeros(N_TOKENS, dtype=np.float32)})
    by_id, meta = load_theta_tokens(theta_npz)
    wrapped = PhaseTaggedDataset(base, by_id, meta, n_tokens=N_TOKENS, patch_width=PATCH_WIDTH)

    ids = [wrapped[i]["ecg_id"] for i in range(len(wrapped))]
    absent_index = ids.index(99)
    theta = wrapped[absent_index]["theta"]
    assert theta.shape == (N_TOKENS,)
    assert torch.all(torch.isnan(theta))


# ==================================================================== M2-A1: grid agreement


def test_n_tokens_mismatch_raises(tmp_path: Path) -> None:
    base = _build_base(tmp_path, [1])
    theta_npz = _write_theta_tokens(tmp_path, {1: np.zeros(N_TOKENS, dtype=np.float32)})
    by_id, meta = load_theta_tokens(theta_npz)
    with pytest.raises(ValueError, match="n_tokens"):
        PhaseTaggedDataset(base, by_id, meta, n_tokens=250, patch_width=PATCH_WIDTH)


def test_patch_width_mismatch_raises(tmp_path: Path) -> None:
    base = _build_base(tmp_path, [1])
    theta_npz = _write_theta_tokens(tmp_path, {1: np.zeros(N_TOKENS, dtype=np.float32)})
    by_id, meta = load_theta_tokens(theta_npz)
    with pytest.raises(ValueError, match="patch_width"):
        PhaseTaggedDataset(base, by_id, meta, n_tokens=N_TOKENS, patch_width=16)


# ================================================================== M2-C2: shuffle-order identity


def test_dataloader_shuffle_order_is_unaffected_by_wrapping(tmp_path: Path) -> None:
    """DataLoader's shuffle permutation is generated from len(dataset) and the generator alone,
    independent of what __getitem__ returns -- wrapping cannot alter shuffle order by
    construction, but this locks the claim in as an executable check rather than an assumption.
    num_workers>0 is not exercised here: worker-process spawning is orthogonal to this class's
    own logic and would only add flakiness/cost to a fast unit test."""
    ecg_ids = [1, 2, 3, 4, 5]
    base = _build_base(tmp_path, ecg_ids)
    theta_npz = _write_theta_tokens(
        tmp_path, {i: np.zeros(N_TOKENS, dtype=np.float32) for i in ecg_ids}
    )
    by_id, meta = load_theta_tokens(theta_npz)
    wrapped = PhaseTaggedDataset(base, by_id, meta, n_tokens=N_TOKENS, patch_width=PATCH_WIDTH)

    gen_a = torch.Generator().manual_seed(0)
    gen_b = torch.Generator().manual_seed(0)
    order_base = [
        item["ecg_id"] for item in DataLoader(base, batch_size=1, shuffle=True, generator=gen_a)
    ]
    order_wrapped = [
        item["ecg_id"] for item in DataLoader(wrapped, batch_size=1, shuffle=True, generator=gen_b)
    ]
    assert order_base == order_wrapped

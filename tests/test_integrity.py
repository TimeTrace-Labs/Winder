import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from winder.data.folds import FoldConfig
from winder.data.integrity import assemble_integrity_report, config_hash, git_sha, sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_file(str(path)) == expected


def test_config_hash_is_deterministic_and_content_sensitive() -> None:
    a = config_hash("seed: 0\n")
    b = config_hash("seed: 0\n")
    c = config_hash("seed: 1\n")
    assert a == b
    assert a != c


def test_git_sha_returns_a_full_sha_for_this_repo() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = git_sha(repo_root)
    assert sha is not None
    assert len(sha) == 40


def test_git_sha_returns_none_for_a_non_git_directory(tmp_path: Path) -> None:
    assert git_sha(str(tmp_path)) is None


def _toy_metadata(n_patients: int = 20) -> pd.DataFrame:
    rows = [
        {"ecg_id": pid, "patient_id": pid, "strat_fold": (pid % 10) + 1}
        for pid in range(n_patients)
    ]
    return pd.DataFrame(rows)


def test_assemble_integrity_report_structure(tmp_path: Path) -> None:
    metadata = _toy_metadata()
    report = assemble_integrity_report(str(tmp_path), metadata, fold_config=FoldConfig())
    assert report["dataset_version"] == "1.0.3"
    assert report["n_records_total"] == 20
    assert "train" in report["splits"]
    assert "val" in report["splits"]
    assert "test" not in report["splits"]  # sealed by default
    assert report["sha256"] == {}  # no CSVs present at tmp_path
    assert report["winder_git_sha"] is None  # not requested
    assert report["config_hash"] is None  # not requested


def test_assemble_integrity_report_includes_csv_hashes(tmp_path: Path) -> None:
    (tmp_path / "ptbxl_database.csv").write_text("a,b\n1,2\n")
    metadata = _toy_metadata()
    report = assemble_integrity_report(str(tmp_path), metadata)
    expected = sha256_file(str(tmp_path / "ptbxl_database.csv"))
    assert report["sha256"]["ptbxl_database.csv"] == expected


def test_assemble_integrity_report_includes_git_sha_and_config_hash(tmp_path: Path) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata = _toy_metadata()
    report = assemble_integrity_report(
        str(tmp_path), metadata, winder_repo_root=repo_root, config_yaml="seed: 0\n"
    )
    assert report["winder_git_sha"] is not None
    assert report["config_hash"] == config_hash("seed: 0\n")


def test_assemble_integrity_report_splits_are_patient_disjoint(tmp_path: Path) -> None:
    """assemble_integrity_report calls folds(), which asserts patient-disjointness
    unconditionally -- a patient split across train and val must raise, not silently pass.

    winder-nominal deviation: strat_fold=0 is used here instead of the reference repo's
    original fold-9 literal, because FoldConfig()'s default train_folds is (1..9) here
    (val_fold moved to the sentinel 0 -- see winder.data.folds) -- fold 9 is now a train
    fold, not val, so it no longer collides across two different splits. Fold 0 (the new
    val_fold) reproduces the same cross-split collision this test is checking for.
    """
    metadata = pd.DataFrame(
        [
            {"ecg_id": 1, "patient_id": 1, "strat_fold": 1},  # train
            {"ecg_id": 2, "patient_id": 1, "strat_fold": 0},  # val -- same patient!
        ]
    )
    with pytest.raises(ValueError, match="not patient-disjoint"):
        assemble_integrity_report(str(tmp_path), metadata)

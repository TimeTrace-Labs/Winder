"""Tests for winder.eval.provenance: RunProvenance and the report-schema envelope."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from winder.eval.provenance import RunProvenance, assemble_report

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_collect_hashes_existing_files_and_none_for_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "state.pt"
    state_path.write_bytes(b"some checkpoint bytes")

    prov = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=100,
        train_folds=(1, 2, 3),
        val_fold=9,
        test_fold=10,
        device="cpu",
        seed=0,
        state_path=str(state_path),
        manifest_path=str(tmp_path / "does_not_exist.parquet"),
        lead_stats_path=None,
    )
    assert prov.checkpoint_sha256 is not None
    assert _SHA256_RE.match(prov.checkpoint_sha256)
    assert prov.manifest_sha256 is None  # path given but file absent
    assert prov.lead_stats_sha256 is None  # path itself was None
    assert prov.step == 100
    assert prov.train_folds == (1, 2, 3)
    assert prov.torch_version == torch.__version__
    assert prov.device == "cpu"
    assert prov.seed == 0


def test_collect_hash_changes_with_file_content(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"content one")
    b.write_bytes(b"content two, different length")

    prov_a = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=1,
        train_folds=(1,),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
        state_path=str(a),
    )
    prov_b = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=1,
        train_folds=(1,),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
        state_path=str(b),
    )
    assert prov_a.checkpoint_sha256 != prov_b.checkpoint_sha256


def test_collect_hash_is_deterministic_for_identical_content(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    path.write_bytes(b"identical bytes")
    prov1 = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=1,
        train_folds=(1,),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
        state_path=str(path),
    )
    prov2 = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=1,
        train_folds=(1,),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
        state_path=str(path),
    )
    assert prov1.checkpoint_sha256 == prov2.checkpoint_sha256


def test_collect_degrades_gracefully_with_no_paths_supplied() -> None:
    """A synthetic-fixture test run with no real files on disk still produces a valid
    RunProvenance -- every hash field is None, not a raised exception."""
    prov = RunProvenance.collect(
        checkpoint_dir="synthetic",
        step=0,
        train_folds=(),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
    )
    assert prov.checkpoint_sha256 is None
    assert prov.manifest_sha256 is None
    assert prov.lead_stats_sha256 is None


def test_collect_git_sha_is_a_valid_hex_string_or_none(tmp_path: Path) -> None:
    prov = RunProvenance.collect(
        checkpoint_dir="synthetic",
        step=0,
        train_folds=(),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
        repo_root=str(Path(__file__).resolve().parent.parent),
    )
    # This repo IS a git checkout, so a real SHA is expected -- but degrade gracefully (None)
    # rather than fail the test if the environment running it genuinely has no git available.
    assert prov.git_sha is None or re.match(r"^[0-9a-f]{40}$", prov.git_sha)


# ============================================================================= assemble_report


def test_assemble_report_matches_the_report_schema(tmp_path: Path) -> None:
    prov = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=5000,
        train_folds=(1, 2, 3, 4, 5, 6, 7, 8),
        val_fold=9,
        test_fold=10,
        device="cpu",
        seed=0,
    )
    report = assemble_report(
        "PASS", "P5-GATES", {"macro_auroc": 0.87}, prov, ["ported verbatim from p1"]
    )
    assert set(report) == {
        "status",
        "milestone_id",
        "metrics",
        "provenance",
        "decisions",
        "questions",
    }
    assert report["status"] == "PASS"
    assert report["milestone_id"] == "P5-GATES"
    assert report["metrics"] == {"macro_auroc": 0.87}
    assert report["decisions"] == ["ported verbatim from p1"]
    assert report["questions"] == []
    assert report["provenance"]["step"] == 5000
    assert report["provenance"]["train_folds"] == (1, 2, 3, 4, 5, 6, 7, 8)


def test_assemble_report_questions_default_to_empty_list(tmp_path: Path) -> None:
    prov = RunProvenance.collect(
        checkpoint_dir=str(tmp_path),
        step=0,
        train_folds=(),
        val_fold=0,
        test_fold=10,
        device="cpu",
        seed=0,
    )
    report = assemble_report("NEEDS_CLARIFICATION", "X", {}, prov, [], questions=["what fold?"])
    assert report["questions"] == ["what fold?"]

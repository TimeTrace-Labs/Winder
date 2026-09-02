"""Tests for the gate-4 prep helper, `scripts/print_fold10_authorization_template.py`.

`build_template` is exercised against a synthetic, tmp_path-based artifacts layout -- fast,
no real PTB-XL/checkpoint data needed, matching this repo's own convention elsewhere. The one
real-file dependency (`authorized_sha256`, computed from the actual `scripts/
fold10_nominal_eval.py` on disk) is asserted against a fresh, independent hash computation, not
hardcoded, so it stays correct even as that file changes.
"""

import hashlib
import re
from pathlib import Path

import pytest
from fold10_nominal_eval import DEFAULT_ARMS, DEFAULT_STEPS, frozen_parameters_dict
from print_fold10_authorization_template import (
    ACCEPTANCE_STATEMENT,
    AUTHORIZED_SCRIPT,
    build_template,
)

from winder.jepa import checkpoint as jepa_checkpoint


def _build_synthetic_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """A `data_root` and `artifacts_dir` with every file `build_template` reads, populated with
    arbitrary (but non-empty) bytes -- content correctness of each individual hash is already
    covered by `test_resolve_frozen_inputs_hashes_every_file_and_all_checkpoints`; this fixture's
    job is just to exercise `build_template`'s own orchestration end to end."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "ptbxl_database.csv").write_bytes(b"ptbxl")
    (data_root / "scp_statements.csv").write_bytes(b"scp")

    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "phase").mkdir(parents=True)
    (artifacts_dir / "reference" / "phase").mkdir(parents=True)
    (artifacts_dir / "lead_stats_f1to9.json").write_bytes(b"lead_stats")
    (artifacts_dir / "reference" / "phase" / "rpeaks.npz").write_bytes(b"rpeaks")
    (artifacts_dir / "phase" / "theta_tokens.npz").write_bytes(b"theta_tokens")
    (artifacts_dir / "manifest.parquet").write_bytes(b"manifest")

    roster_dir = artifacts_dir / "roster"
    for arm in DEFAULT_ARMS:
        for step in DEFAULT_STEPS:
            d = roster_dir / arm / f"checkpoint_step{step}"
            d.mkdir(parents=True)
            (d / jepa_checkpoint.STATE_FILENAME).write_bytes(f"state-{arm}-{step}".encode())
            (d / jepa_checkpoint.CONFIG_FILENAME).write_bytes(f"config-{arm}-{step}".encode())
            (d / jepa_checkpoint.META_FILENAME).write_bytes(f"meta-{arm}-{step}".encode())

    return data_root, artifacts_dir


def test_build_template_resolves_all_twelve_real_event_checkpoints(tmp_path: Path) -> None:
    data_root, artifacts_dir = _build_synthetic_artifacts(tmp_path)
    template, checkpoints = build_template(
        data_root=str(data_root),
        artifacts_dir=str(artifacts_dir),
        roster_dir=None,
        lead_stats_path=None,
        rpeaks_npz_path=None,
        theta_tokens_path=None,
        manifest_path=None,
    )
    assert len(checkpoints) == len(DEFAULT_ARMS) * len(DEFAULT_STEPS) == 12
    assert set(template["frozen_inputs"]["checkpoint_sha256"].keys()) == set(checkpoints.keys())


def test_build_template_authorized_sha256_matches_a_fresh_independent_hash(
    tmp_path: Path,
) -> None:
    data_root, artifacts_dir = _build_synthetic_artifacts(tmp_path)
    template, _ = build_template(
        data_root=str(data_root),
        artifacts_dir=str(artifacts_dir),
        roster_dir=None,
        lead_stats_path=None,
        rpeaks_npz_path=None,
        theta_tokens_path=None,
        manifest_path=None,
    )
    with open(AUTHORIZED_SCRIPT, "rb") as fh:
        expected = hashlib.sha256(fh.read()).hexdigest()
    assert template["authorized_script"] == AUTHORIZED_SCRIPT
    assert template["authorized_sha256"] == expected


def test_build_template_frozen_parameters_matches_the_shared_source_of_truth(
    tmp_path: Path,
) -> None:
    data_root, artifacts_dir = _build_synthetic_artifacts(tmp_path)
    template, _ = build_template(
        data_root=str(data_root),
        artifacts_dir=str(artifacts_dir),
        roster_dir=None,
        lead_stats_path=None,
        rpeaks_npz_path=None,
        theta_tokens_path=None,
        manifest_path=None,
    )
    assert template["frozen_parameters"] == frozen_parameters_dict()


def test_build_template_leaves_the_four_human_decision_fields_as_explicit_placeholders(
    tmp_path: Path,
) -> None:
    """These four fields are not derivable from any file on disk -- must never be silently
    guessed or defaulted to something that looks plausible."""
    data_root, artifacts_dir = _build_synthetic_artifacts(tmp_path)
    template, _ = build_template(
        data_root=str(data_root),
        artifacts_dir=str(artifacts_dir),
        roster_dir=None,
        lead_stats_path=None,
        rpeaks_npz_path=None,
        theta_tokens_path=None,
        manifest_path=None,
    )
    for field in (
        "preregistration_doc_commit",
        "second_reviewer_correspondence_commit",
        "authorized_by",
        "authorized_at",
    ):
        assert str(template[field]).startswith("<FILL IN")
    assert template["acceptance_statement"] == ACCEPTANCE_STATEMENT


@pytest.mark.parametrize(
    "pattern",
    [
        r'open\([^)]*["\']w',
        r"\.write_text\(",
        r"\.write_bytes\(",
        r"json\.dump\(",
    ],
)
def test_script_source_contains_no_write_mode_file_operation(pattern: str) -> None:
    """Static invariant, gate-4's own non-negotiable: this script may only ever print a
    candidate record, never write `artifacts/fold10_authorization.json` (or anything else) to
    disk. A future edit that adds a write call anywhere in this file must fail this test."""
    source = Path("scripts/print_fold10_authorization_template.py").read_text(encoding="utf-8")
    assert not re.search(pattern, source), f"found a write-mode file operation matching {pattern}"

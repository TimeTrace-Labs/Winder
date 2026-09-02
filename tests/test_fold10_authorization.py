"""Tests for the fold-10 unseal authorization gate.

`_verify_authorization` is pure and fully unit-testable with synthetic files under `tmp_path` --
none of these tests unseal real fold 10 data. `authorized_unseal` itself is exercised only via
the "no authorization record exists" path (today's real, expected state); actually authorizing a
successful unseal in a test would require a real sign-off record, which must never exist outside
the one real event.
"""

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from winder.data.fold10_authorization import (
    _REPO_ROOT,
    AuthorizationError,
    _consume_authorization_record,
    _verify_authorization,
    authorized_unseal,
    load_frozen_inputs,
    load_frozen_parameters,
)


def _write_caller_script(tmp_path: Path, content: str = "# a caller script\n") -> Path:
    p = tmp_path / "caller.py"
    p.write_text(content, encoding="utf-8")
    return p


def _write_record(
    path: Path,
    *,
    authorized_script: str,
    authorized_sha256: str,
    preregistration_doc_commit: str = "abc123",
    second_reviewer_correspondence_commit: str = "def456",
    authorized_by: str = "test",
    authorized_at: str = "2026-01-01T00:00:00Z",
    extra: dict | None = None,
) -> None:
    record = {
        "authorized_script": authorized_script,
        "authorized_sha256": authorized_sha256,
        "preregistration_doc_commit": preregistration_doc_commit,
        "second_reviewer_correspondence_commit": second_reviewer_correspondence_commit,
        "authorized_by": authorized_by,
        "authorized_at": authorized_at,
    }
    if extra:
        record.update(extra)
    path.write_text(json.dumps(record), encoding="utf-8")


# --------------------------------------------------------------- today's real, expected state
def test_authorized_unseal_raises_when_no_authorization_record_exists() -> None:
    """Pins today's real repo state: `artifacts/fold10_authorization.json` does not exist, so
    `authorized_unseal` must refuse. This is the test that proves adding this module did not
    weaken anything -- fold 10 is exactly as sealed after this file exists as before."""
    real_path = _REPO_ROOT / "artifacts" / "fold10_authorization.json"
    assert not real_path.is_file(), (
        "artifacts/fold10_authorization.json exists -- fold 10 may have been unsealed. "
        "This test intentionally fails loudly rather than silently skip."
    )
    import pandas as pd

    df = pd.DataFrame({"ecg_id": [1], "patient_id": [1], "strat_fold": [10]})
    with pytest.raises(AuthorizationError, match="no authorization record"):
        authorized_unseal(df)


# --------------------------------------------------------- _verify_authorization: negative paths
def test_verify_authorization_raises_when_record_file_absent(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    with pytest.raises(AuthorizationError, match="no authorization record"):
        _verify_authorization(caller, caller, tmp_path / "does_not_exist.json")


def test_verify_authorization_raises_on_malformed_json(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    record_path = tmp_path / "auth.json"
    record_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="not valid JSON"):
        _verify_authorization(caller, caller, record_path)


def test_verify_authorization_raises_on_missing_required_field(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    record_path = tmp_path / "auth.json"
    record_path.write_text(json.dumps({"authorized_script": "x.py"}), encoding="utf-8")
    with pytest.raises(AuthorizationError, match="missing required field"):
        _verify_authorization(caller, caller, record_path)


def test_verify_authorization_raises_when_authorized_script_is_a_different_file(
    tmp_path: Path,
) -> None:
    """The record names a script other than the one actually calling -- must never authorize."""
    caller = _write_caller_script(tmp_path)
    other = tmp_path / "some_other_script.py"
    other.write_text(caller.read_text(), encoding="utf-8")  # even with IDENTICAL content
    record_path = tmp_path / "auth.json"
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    _write_record(
        record_path,
        authorized_script=str(other.relative_to(_REPO_ROOT))
        if other.is_relative_to(_REPO_ROOT)
        else str(other),
        authorized_sha256=sha,
    )
    # authorized_script is resolved relative to the repo root inside _verify_authorization;
    # using an absolute tmp_path-based string here still exercises the "different file" branch
    # because tmp_path is never under _REPO_ROOT.
    with pytest.raises(AuthorizationError, match="covers .* but the actual calling script is"):
        _verify_authorization(caller, caller, record_path)


def test_verify_authorization_raises_when_content_hash_does_not_match(tmp_path: Path) -> None:
    """The load-bearing case: script was reviewed, then edited. Must refuse, not re-approve."""
    caller = _write_caller_script(tmp_path, content="# original, reviewed content\n")
    record_path = tmp_path / "auth.json"
    wrong_sha = hashlib.sha256(b"this is not the caller's content").hexdigest()
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=wrong_sha)
    with pytest.raises(AuthorizationError, match="has changed since authorization"):
        _verify_authorization(caller, caller, record_path)


def test_verify_authorization_detects_a_post_signoff_edit(tmp_path: Path) -> None:
    """Same idea as above, phrased as the actual scenario: sign off on version A, then the file
    on disk becomes version B before it runs."""
    caller = _write_caller_script(tmp_path, content="version A\n")
    original_sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=original_sha)

    # Sanity: at this point, verification succeeds (proves the positive path is reachable).
    result = _verify_authorization(caller, caller, record_path)
    assert result.authorized_by == "test"

    # Now edit the file post-signoff.
    caller.write_text("version B -- edited after review\n", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="has changed since authorization"):
        _verify_authorization(caller, caller, record_path)


# --------------------------------------------------------- _verify_authorization: positive path
def test_verify_authorization_succeeds_when_everything_matches(tmp_path: Path) -> None:
    caller = _write_caller_script(
        tmp_path,
        content=textwrap.dedent("""\
        # a fully reviewed, hash-pinned caller script
        print("hello")
    """),
    )
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(
        record_path,
        authorized_script=str(rel),
        authorized_sha256=sha,
        authorized_by="the project lead",
        preregistration_doc_commit="deadbeef",
        second_reviewer_correspondence_commit="cafef00d",
    )
    result = _verify_authorization(caller, caller, record_path)
    assert result.authorized_by == "the project lead"
    assert result.preregistration_doc_commit == "deadbeef"
    assert result.second_reviewer_correspondence_commit == "cafef00d"


# ------------------------------------------------------- _verify_authorization: argv0 binding
def test_verify_authorization_raises_when_argv0_is_not_the_authorized_script(
    tmp_path: Path,
) -> None:
    """gate-3's B3 finding: the calling FILE matching the record is necessary but not sufficient
    -- the process's own entry point (sys.argv[0]) must ALSO be that file, or this proves nothing
    about whether main()'s own CLI-validated, frozen-parameter path was actually taken. Simulates
    a test file/notebook/REPL importing and calling a function that lives in the authorized
    script: caller_path matches the record, but argv0_path (some other process) does not."""
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=sha)

    not_the_script = tmp_path / "pytest_or_a_notebook_kernel"
    not_the_script.write_text("# not the authorized script\n", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="running process's own entry point"):
        _verify_authorization(caller, not_the_script, record_path)


def test_verify_authorization_succeeds_when_argv0_also_matches(tmp_path: Path) -> None:
    """Companion positive case: when argv0_path equals the authorized script too (the real
    `python scripts/fold10_nominal_eval.py ...` invocation shape), verification still succeeds --
    the new check does not break the legitimate path."""
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=sha)
    result = _verify_authorization(caller, caller, record_path)
    assert result.authorized_by == "test"


# ---------------------------------------------------------------------- load_frozen_parameters
def test_load_frozen_parameters_raises_when_record_absent(tmp_path: Path) -> None:
    with pytest.raises(AuthorizationError, match="no authorization record"):
        load_frozen_parameters(tmp_path / "does_not_exist.json")


def test_load_frozen_parameters_raises_when_field_missing(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=sha)
    with pytest.raises(AuthorizationError, match="frozen_parameters"):
        load_frozen_parameters(record_path)


def test_load_frozen_parameters_returns_the_pinned_block(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    frozen = {
        "n_strata": 16,
        "gain_limit": 250,
        "n_replicates": 2000,
        "geometry_limit": 1200,
        "causal_window": 40,
        "detection_n_records": 400,
    }
    _write_record(
        record_path,
        authorized_script=str(rel),
        authorized_sha256=sha,
        extra={"frozen_parameters": frozen},
    )
    assert load_frozen_parameters(record_path) == frozen


# -------------------------------------------------------------------------- load_frozen_inputs
def test_load_frozen_inputs_raises_when_record_absent(tmp_path: Path) -> None:
    with pytest.raises(AuthorizationError, match="no authorization record"):
        load_frozen_inputs(tmp_path / "does_not_exist.json")


def test_load_frozen_inputs_raises_when_field_missing(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=sha)
    with pytest.raises(AuthorizationError, match="frozen_inputs"):
        load_frozen_inputs(record_path)


def test_load_frozen_inputs_returns_the_pinned_block(tmp_path: Path) -> None:
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    frozen = {
        "data_root": "/some/data/root",
        "roster_dir": "/some/roster/dir",
        "metadata_sha256": {"ptbxl_database.csv": "aaa", "scp_statements.csv": "bbb"},
        "lead_stats_sha256": "ccc",
        "rpeaks_npz_sha256": "ddd",
        "theta_tokens_npz_sha256": "eee",
        "checkpoint_sha256": {"signal_seed0/step5000": "fff"},
    }
    _write_record(
        record_path,
        authorized_script=str(rel),
        authorized_sha256=sha,
        extra={"frozen_inputs": frozen},
    )
    assert load_frozen_inputs(record_path) == frozen


# ------------------------------------------------------------------ _consume_authorization_record
def test_consume_authorization_record_renames_to_a_consumed_sibling(tmp_path: Path) -> None:
    record_path = tmp_path / "fold10_authorization.json"
    record_path.write_text('{"hello": "world"}', encoding="utf-8")
    consumed_path = _consume_authorization_record(record_path, now="20260819T000000Z")
    assert consumed_path == tmp_path / "fold10_authorization.consumed.20260819T000000Z.json"
    assert not record_path.exists()
    assert consumed_path.is_file()
    assert json.loads(consumed_path.read_text(encoding="utf-8")) == {"hello": "world"}


def test_consumed_record_no_longer_authorizes_a_second_call(tmp_path: Path) -> None:
    """The actual point of consumption: after it fires, `_verify_authorization` against the
    original path must fail exactly like "no authorization record exists" -- proving a second
    invocation cannot succeed against the same record."""
    caller = _write_caller_script(tmp_path)
    sha = hashlib.sha256(caller.read_bytes()).hexdigest()
    record_path = tmp_path / "auth.json"
    rel = caller.relative_to(_REPO_ROOT) if caller.is_relative_to(_REPO_ROOT) else caller
    _write_record(record_path, authorized_script=str(rel), authorized_sha256=sha)

    # Sanity: valid before consumption.
    result = _verify_authorization(caller, caller, record_path)
    assert result.authorized_by == "test"

    _consume_authorization_record(record_path, now="20260819T000000Z")
    with pytest.raises(AuthorizationError, match="no authorization record"):
        _verify_authorization(caller, caller, record_path)


def test_authorized_unseal_real_repo_state_still_refuses_with_no_record(tmp_path: Path) -> None:
    """Companion to the existing no-record test: consumption logic must not accidentally create
    or touch any file when there was never a record to consume."""
    real_path = _REPO_ROOT / "artifacts" / "fold10_authorization.json"
    assert not real_path.is_file()
    consumed_glob = list((_REPO_ROOT / "artifacts").glob("fold10_authorization.consumed.*.json"))
    assert consumed_glob == [], (
        f"unexpected consumed authorization record(s) on disk: {consumed_glob} -- fold 10 may "
        "have been unsealed for real. This test intentionally fails loudly rather than silently "
        "skip."
    )

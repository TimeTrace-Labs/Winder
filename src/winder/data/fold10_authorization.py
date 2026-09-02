"""The one, single, hash-gated path by which fold 10 may ever be unsealed for real.

`winder.data.folds.folds(..., unseal=True)` has no opinion about WHO is calling it or WHY --
that is deliberate, `folds()` is a general-purpose split function and should stay one. This
module is the campaign-specific gate in front of it: `authorized_unseal(df, cfg)` is a drop-in
replacement for `folds(df, cfg, unseal=True)` that additionally requires a pre-recorded,
content-hash-pinned sign-off before it will do anything.

**Why a hash pin, not a filename check.** A filename alone ("only scripts/fold10_nominal_eval.py
may unseal") proves nothing about what that file currently CONTAINS -- it could be reviewed once,
approved, and silently edited afterward. Pinning the SHA-256 of the file's content at sign-off
time closes that gap: if the file changes even one byte after authorization, this raises rather
than unseal.

**Why call-stack introspection, not a caller-supplied path string.** A caller could pass any
string it likes as "this is who I am" -- that proves nothing. `inspect.stack()` reports the REAL
file that contains the currently-executing calling code, which cannot be spoofed by an argument.

**Why the process entry point (`sys.argv[0]`) is checked too, not just the calling file.**
A gate-3 blind review (2026-08-18) found a real gap: `inspect.stack()` only proves WHICH FILE's
code is calling -- it says nothing about HOW that code was reached. `resolve_target_fold_frames`
lives in `scripts/fold10_nominal_eval.py`, so `inspect.stack()[1]` always names that file,
regardless of whether the actual trigger was `python scripts/fold10_nominal_eval.py
--target-fold 10 ...` (gate 1's own CLI, with `main()`'s frozen-parameter checks already run) or
some other entry point importing and calling `resolve_target_fold_frames` directly -- a test file,
a notebook, a REPL -- bypassing every one of `main()`'s own guards. `tests/
test_fold10_nominal_eval.py` already does exactly this today (harmlessly, only because no
authorization record exists yet). Requiring `Path(sys.argv[0]).resolve()` to equal the authorized
script's own path closes this: it binds the authorization to "this file's code, running AS the
main program," not merely to "this file's code, called from wherever."

**What this does NOT defend against.** This is not a security boundary against a determined
adversary with arbitrary code execution (monkeypatching `inspect`/`sys.argv`, etc.) -- that is not
the threat model. The threat model is the one this whole project's seal exists for: an accidental
or self-motivated shortcut taken under time pressure by someone who has no wish to actually defeat
the check, just to move faster. Against that threat model, "the file that ran, as the process's
own entry point, is provably the file that was reviewed, byte for byte" is exactly the right bar.

**The authorization record.** A JSON file at `_AUTHORIZATION_PATH` (repo-root-relative,
`artifacts/fold10_authorization.json`), which does NOT exist today and must not be created before
a pre-registered protocol and its review ceremony are both complete:

    {
      "authorized_script": "scripts/fold10_nominal_eval.py",
      "authorized_sha256": "<sha256 of that script's exact content at sign-off time>",
      "preregistration_doc_commit": "<git commit hash of the frozen pre-registration document>",
      "second_reviewer_correspondence_commit": "<git commit hash of the blind second-agent review>",
      "authorized_by": "<name>",
      "authorized_at": "<ISO-8601 timestamp>",
      "acceptance_statement": "I accept this number whatever it is.",
      "frozen_parameters": {
        "n_strata": 16, "gain_limit": 250, "n_replicates": 2000, "geometry_limit": 1200,
        "causal_window": 40, "detection_n_records": 400
      },
      "frozen_inputs": {
        "data_root": "<path, path-trusted, not hashed -- see fold10_nominal_eval.py>",
        "roster_dir": "<path>",
        "metadata_sha256": {"ptbxl_database.csv": "<sha256>", "scp_statements.csv": "<sha256>"},
        "lead_stats_sha256": "<sha256>",
        "rpeaks_npz_sha256": "<sha256>",
        "theta_tokens_npz_sha256": "<sha256>",
        "manifest_sha256": "<sha256>",
        "checkpoint_sha256": {
          "signal_seed0/step5000": {
            "state.pt": "<sha256>", "config.yaml": "<sha256>", "meta.json": "<sha256>"
          },
          "...": "..."
        }
      }
    }

`frozen_parameters` closes gate-3's B4 finding: these six statistical/scale parameters were
previously CLI-configurable with no record of which values backed "the" event.
`load_frozen_parameters` (below) reads this block; `scripts/fold10_nominal_eval.py`'s own
`main()` refuses to run at `--target-fold 10` unless every resolved CLI value matches it exactly.

`frozen_inputs` closes gate-3's A-2 finding (a second, independent blind review, 2026-08-18):
`frozen_parameters` alone left `--roster-dir`/`--data-root`/`--lead-stats-path`/etc. unpinned, so
a caller could repoint them at different underlying checkpoints or data while passing every other
check. `load_frozen_inputs` (below) reads this block; `main()` refuses to run at
`--target-fold 10` unless every resolved content hash (and the two path strings) matches exactly.
`data_root`'s own waveform corpus is deliberately path-trusted, not content-hashed (hashing every
WFDB record is impractical) -- its content-level coverage is the metadata hash, the pre-existing
lead-stats trap, and the acceptance gate's own split-shape checks.

**Extended after a third, independent blind review (2026-08-19, gate-3 round 3) found the first
version of this fix was itself incomplete.** `scripts/fold10_nominal_eval.py::build_fold10_style_
cohort` took `artifacts_dir` and re-derived `theta_tokens.npz`/`manifest.parquet`'s paths
internally, rather than using the already-resolved, already-hash-checked path variables `main()`
had computed -- a live-constructed `--artifacts-dir` repoint bypassed the theta-tokens check
entirely (decorative, not enforced) and `manifest.parquet` had no entry in this schema at all,
feeding an unreviewed file into every scored quantity that reads `rr_median_ms`. Fixed at the
source: that function now takes `theta_tokens_path`/`manifest_path` as explicit parameters, never
a directory (see its own docstring for the invariant this establishes); `manifest_sha256` and
per-checkpoint `config.yaml`/`meta.json` hashes (also read on this path, also previously
unhashed) were added to this schema, with the complete input surface re-enumerated by tracing a
real run (`strace -f -e trace=openat`) rather than reasoned about from memory.

Its absence is the seal. `tests/test_fold10_authorization.py`'s
`test_authorized_unseal_raises_when_no_authorization_record_exists` pins that today's repo, with
no such file present, still raises -- proving this module does not weaken anything that exists
today.

**The record is single-use, consumed at the moment of a successful unseal (gate-3 B.5 finding,
closed 2026-08-18).** A prior draft of this module relied on a documentation-only rule ("delete
the record after the event") that nothing enforced -- an independent review correctly noted that
nothing stopped a second `--target-fold 10` invocation from succeeding against a still-valid,
undeleted record, which compounds with the `frozen_inputs` gap above into exactly the "rerun
smuggles in a protocol change" failure mode `notes/fold10_preregistration.md`'s rerun rule names
as most dangerous. `authorized_unseal` now atomically renames the record to a
`fold10_authorization.consumed.<UTC-stamp>.json` sibling (`_consume_authorization_record`)
immediately after `_verify_authorization` succeeds -- renamed, not deleted, so the audit trail
survives and is meant to be committed alongside the event's outputs. A second invocation then
fails on the ordinary "no authorization record" path, exactly as if none had ever existed. A
rerun (bug-fix path only, never a scope change) requires a fresh record from a fresh ceremony,
never resurrecting a consumed one.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from winder.data.folds import FoldConfig, folds

__all__ = [
    "AuthorizationError",
    "authorized_unseal",
    "load_frozen_inputs",
    "load_frozen_parameters",
]

#: Repo-root-relative path to the sign-off record. Hardcoded, not caller-suppliable -- a caller
#: must not be able to point this at an alternate, unreviewed record.
_AUTHORIZATION_PATH = "artifacts/fold10_authorization.json"

_REPO_ROOT = Path(__file__).resolve().parents[3]


class AuthorizationError(RuntimeError):
    """Fold 10 stays sealed: no valid, current sign-off record covers the calling script."""


@dataclass(frozen=True)
class _VerifiedAuthorization:
    authorized_script: str
    preregistration_doc_commit: str
    second_reviewer_correspondence_commit: str
    authorized_by: str
    authorized_at: str


def _verify_authorization(
    caller_path: Path, argv0_path: Path, authorization_json_path: Path
) -> _VerifiedAuthorization:
    """Pure, fully unit-testable core check. Raises `AuthorizationError` with a specific,
    diagnosable reason on every failure path; never silently returns a partial result.

    `caller_path` and the record's own `authorized_script` field are compared repo-root-relative,
    both resolved through `Path.resolve()` first, so a `./`-prefixed or symlinked path cannot
    slip past the comparison by string-formatting alone. `argv0_path` (the resolved process entry
    point, `sys.argv[0]`) must ALSO equal the authorized script -- this is what binds the
    authorization to "running this file as the main program," not merely to "this file's code
    happened to be on the call stack" (see module docstring: this is what forecloses a test file,
    notebook, or REPL importing and calling a function that lives in the authorized script).
    """
    if not authorization_json_path.is_file():
        raise AuthorizationError(
            f"no authorization record at {authorization_json_path} -- fold 10 stays sealed. "
            f"This is the expected, correct state until a pre-registered protocol and its "
            f"review ceremony are both complete."
        )
    try:
        record: dict[str, Any] = json.loads(authorization_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AuthorizationError(
            f"authorization record at {authorization_json_path} is not valid JSON: {e}"
        ) from e

    required = {
        "authorized_script",
        "authorized_sha256",
        "preregistration_doc_commit",
        "second_reviewer_correspondence_commit",
        "authorized_by",
        "authorized_at",
    }
    missing = required - record.keys()
    if missing:
        raise AuthorizationError(
            f"authorization record at {authorization_json_path} is missing required field(s): "
            f"{sorted(missing)}"
        )

    authorized_script_path = (_REPO_ROOT / str(record["authorized_script"])).resolve()
    if authorized_script_path != caller_path.resolve():
        raise AuthorizationError(
            f"authorization record covers {authorized_script_path}, but the actual calling "
            f"script is {caller_path.resolve()} -- these must be the exact same file. "
            f"A sign-off for one script never authorizes another."
        )

    if authorized_script_path != argv0_path.resolve():
        raise AuthorizationError(
            f"authorization record covers {authorized_script_path}, and the calling code does "
            f"live there, but the running process's own entry point (sys.argv[0]) is "
            f"{argv0_path.resolve()} -- fold 10 may only be unsealed by running "
            f"{record['authorized_script']} directly as the main program "
            f"(e.g. `python {record['authorized_script']} --target-fold 10 ...`), never by "
            f"importing and calling its functions from another entry point (a test file, a "
            f"notebook, a REPL). This is what binds the authorization to gate 1's own "
            f"CLI-validated, frozen-parameter invocation."
        )

    if not caller_path.is_file():
        raise AuthorizationError(f"calling script {caller_path} does not exist on disk")
    actual_sha256 = hashlib.sha256(caller_path.read_bytes()).hexdigest()
    if actual_sha256 != record["authorized_sha256"]:
        raise AuthorizationError(
            f"{caller_path} has changed since authorization: recorded sha256="
            f"{record['authorized_sha256']}, actual sha256={actual_sha256}. Re-run the full "
            f"review ceremony and re-authorize before running -- do not re-sign the old hash."
        )

    return _VerifiedAuthorization(
        authorized_script=str(record["authorized_script"]),
        preregistration_doc_commit=str(record["preregistration_doc_commit"]),
        second_reviewer_correspondence_commit=str(record["second_reviewer_correspondence_commit"]),
        authorized_by=str(record["authorized_by"]),
        authorized_at=str(record["authorized_at"]),
    )


def _consume_authorization_record(authorization_json_path: Path, *, now: str) -> Path:
    """Pure, unit-testable core of gate-3's B.5 fix: atomically renames the record to a
    `.consumed.<now>.json` sibling, so a second `--target-fold 10` invocation can never succeed
    against the same record. Renaming (not deleting) preserves the audit trail -- the consumed
    file is meant to be committed alongside the event's outputs. `now` is caller-supplied purely
    for testability; `authorized_unseal` passes the real current UTC stamp.
    """
    consumed_path = authorization_json_path.with_name(
        f"{authorization_json_path.stem}.consumed.{now}{authorization_json_path.suffix}"
    )
    authorization_json_path.replace(consumed_path)
    return consumed_path


def authorized_unseal(df: pd.DataFrame, cfg: FoldConfig | None = None) -> dict[str, pd.DataFrame]:
    """Drop-in replacement for `folds(df, cfg, unseal=True)`, gated on a hash-pinned sign-off
    record that must name, byte-for-byte, the actual file calling this function.

    Raises `AuthorizationError` (never silently falls back to the sealed default) if no such
    record exists, or if it exists but does not match the calling script exactly. On success, the
    record is immediately consumed (renamed) -- see `_consume_authorization_record` -- so it can
    authorize exactly one call, ever.
    """
    caller_frame = inspect.stack()[1]
    caller_path = Path(caller_frame.filename).resolve()
    argv0_path = Path(sys.argv[0]).resolve()
    record_path = _REPO_ROOT / _AUTHORIZATION_PATH
    authorization = _verify_authorization(caller_path, argv0_path, record_path)
    consumed_path = _consume_authorization_record(
        record_path, now=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    # `folds()` itself still prints the stderr banner and raises the UserWarning below --
    # this function ADDS a gate in front of it, it does not replace the existing one.
    result = folds(df, cfg, unseal=True)
    print(
        f"\n[fold10_authorization] unseal authorized by {authorization.authorized_by} at "
        f"{authorization.authorized_at}, pre-registration doc commit "
        f"{authorization.preregistration_doc_commit}, second-reviewer correspondence commit "
        f"{authorization.second_reviewer_correspondence_commit}. Record consumed -> "
        f"{consumed_path}.\n"
    )
    return result


def load_frozen_parameters(authorization_json_path: Path | None = None) -> dict[str, int]:
    """Reads the authorization record's `frozen_parameters` block -- the six statistical/scale
    parameters (`n_strata`, `gain_limit`, `n_replicates`, `geometry_limit`, `causal_window`,
    `detection_n_records`) gate-3's B4 finding flagged as unpinned. Raises `AuthorizationError`
    (never returns a default) if the record is absent or lacks the field -- a `target_fold == 10`
    run must never silently proceed with unpinned parameters.
    """
    path = authorization_json_path or (_REPO_ROOT / _AUTHORIZATION_PATH)
    if not path.is_file():
        raise AuthorizationError(
            f"no authorization record at {path} -- cannot read frozen_parameters. Fold 10 stays "
            f"sealed until a pre-registered protocol and its review ceremony are both complete."
        )
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "frozen_parameters" not in record:
        raise AuthorizationError(
            f"authorization record at {path} has no 'frozen_parameters' field -- it must name "
            f"the exact n_strata/gain_limit/n_replicates/geometry_limit/causal_window/"
            f"detection_n_records values this event is pinned to."
        )
    return dict(record["frozen_parameters"])


def load_frozen_inputs(authorization_json_path: Path | None = None) -> dict[str, Any]:
    """Reads the authorization record's `frozen_inputs` block -- the discrete file-content hashes
    (checkpoints, lead-stats, rpeaks, theta-tokens, metadata) and the `data_root`/`roster_dir`
    path strings gate-3's A-2 finding flagged as unpinned. Raises `AuthorizationError` (never
    returns a default) if the record is absent or lacks the field -- a `target_fold == 10` run
    must never silently proceed with unpinned data/checkpoint provenance.
    """
    path = authorization_json_path or (_REPO_ROOT / _AUTHORIZATION_PATH)
    if not path.is_file():
        raise AuthorizationError(
            f"no authorization record at {path} -- cannot read frozen_inputs. Fold 10 stays "
            f"sealed until a pre-registered protocol and its review ceremony are both complete."
        )
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "frozen_inputs" not in record:
        raise AuthorizationError(
            f"authorization record at {path} has no 'frozen_inputs' field -- it must name the "
            f"exact data_root/roster_dir/checkpoint/lead_stats/rpeaks/theta_tokens values this "
            f"event is pinned to."
        )
    return dict(record["frozen_inputs"])

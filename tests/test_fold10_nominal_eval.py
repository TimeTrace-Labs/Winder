"""Fast, no-real-data tests for `scripts/fold10_nominal_eval.py`'s one shared branch point,
`resolve_target_fold_frames`.

Everything here runs against a tiny synthetic in-memory frame (`_toy_metadata`, matching
`tests/test_folds.py`'s own fixture shape) -- no checkpoint, no waveform decode, no GPU, and
(load-bearing) no real fold-10 data is ever touched: the `target_fold == 10` case is exercised
only via its own real, current refusal (no `artifacts/fold10_authorization.json` exists on this
repo today), never via a successful unseal.
"""

import hashlib
import inspect
import warnings
from pathlib import Path

import pandas as pd
import pytest
from fold10_nominal_eval import (
    DEFAULT_CAUSAL_WINDOW,
    _check_target_fold_10_frozen_inputs,
    _check_target_fold_10_frozen_parameters,
    _count_battery_errors,
    _resolve_frozen_inputs,
    build_fold10_style_cohort,
    frozen_parameters_dict,
    resolve_checkpoints,
    resolve_default_paths,
    resolve_target_fold_frames,
)

from winder.data.fold10_authorization import _REPO_ROOT, AuthorizationError
from winder.data.folds import LEGACY_FOLD_CONFIG
from winder.jepa import checkpoint as jepa_checkpoint


def _toy_metadata(n_patients: int = 40, records_per_patient: int = 1) -> pd.DataFrame:
    """10 folds, patients assigned round-robin -- same shape as `tests/test_folds.py`'s own
    fixture, so this file's assumptions about `folds()`'s behaviour stay aligned with the tests
    that pin it directly."""
    rows = []
    ecg_id = 1
    for pid in range(n_patients):
        fold = (pid % 10) + 1
        for _ in range(records_per_patient):
            rows.append({"ecg_id": ecg_id, "patient_id": pid, "strat_fold": fold})
            ecg_id += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------- target_fold != 10: the "val" branch


def test_target_fold_9_reads_eval_frame_via_val_key() -> None:
    """`target_fold=9` must resolve `eval_frame` to exactly fold 9's rows, via the same `"val"`
    key `winder.eval.acceptance.build_split_frames` already uses for the identical purpose --
    never through the sealed-fold-release path."""
    df = _toy_metadata(n_patients=100, records_per_patient=2)
    result = resolve_target_fold_frames(df, target_fold=9)
    assert set(result.eval_frame["strat_fold"].unique()) == {9}
    assert set(result.train_frame["strat_fold"].unique()) <= {1, 2, 3, 4, 5, 6, 7, 8}
    assert set(result.cal_frame["strat_fold"].unique()) <= {1, 2, 3, 4, 5, 6, 7, 8}
    # train/cal are patient-disjoint from eval by construction (folds() checks this itself).
    assert set(result.train_frame["patient_id"]).isdisjoint(set(result.eval_frame["patient_id"]))
    assert set(result.cal_frame["patient_id"]).isdisjoint(set(result.eval_frame["patient_id"]))


def test_target_fold_9_never_warns_sealed_fold() -> None:
    """The load-bearing safety property of the "val"-branch design: unlike a sealed-release call,
    this path never prints or warns "SEALED FOLD" for any target_fold != 10 -- proving it truly
    never exercises `folds()`'s own sealed-release code path."""
    df = _toy_metadata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_target_fold_frames(df, target_fold=9)  # must not raise/warn


def test_target_fold_9_frames_match_legacy_fold_config_exactly() -> None:
    """`target_fold=9`'s train_folds exclusion (`tuple(f for f in range(1, 10) if f != 9)`)
    reduces to exactly `LEGACY_FOLD_CONFIG`'s own train set -- this is not a coincidence to pin
    loosely: `target_fold=9`'s resolved frames should be row-for-row identical to what
    `LEGACY_FOLD_CONFIG` itself produces, since `FoldConfig`'s own calibration_frac/
    calibration_seed defaults already match `LEGACY_FOLD_CONFIG`'s explicit values. (There is no
    longer a `detection_fold_config` field to compare directly -- `resolve_target_fold_frames`
    now returns only frames, reused as-is for both the probe and detection cohorts; see its own
    docstring for why.)"""
    from winder.data.folds import calibration_subset, folds, train_minus_calibration

    df = _toy_metadata(n_patients=100, records_per_patient=2)
    result = resolve_target_fold_frames(df, target_fold=9)
    expected_eval = folds(df, LEGACY_FOLD_CONFIG)["val"]
    expected_train = train_minus_calibration(df, LEGACY_FOLD_CONFIG)
    expected_cal = calibration_subset(df, LEGACY_FOLD_CONFIG)
    assert list(result.eval_frame["ecg_id"]) == list(expected_eval["ecg_id"])
    assert list(result.train_frame["ecg_id"]) == list(expected_train["ecg_id"])
    assert list(result.cal_frame["ecg_id"]) == list(expected_cal["ecg_id"])


def test_target_fold_5_also_excludes_itself_from_train_folds() -> None:
    """Any non-10 target_fold in 1..9 must exclude itself from train_folds -- required by
    folds()'s own seal invariant and patient-disjointness check, not merely stylistic (module
    docstring). Checked via the observable frames, since `train_folds`/`val_fold` are no longer
    exposed as a returned field."""
    df = _toy_metadata()
    result = resolve_target_fold_frames(df, target_fold=5)
    assert set(result.eval_frame["strat_fold"].unique()) == {5}
    assert 5 not in set(result.train_frame["strat_fold"].unique())
    assert 5 not in set(result.cal_frame["strat_fold"].unique())


# --------------------------------------------------------- target_fold == 10: the sealed branch


def test_target_fold_10_requires_authorization_that_does_not_exist_today() -> None:
    """The single most important test in this file: today's real repo state (no
    `artifacts/fold10_authorization.json`) must make `target_fold=10` refuse, even against a
    synthetic frame that would otherwise happily produce a "fold 10" split. This is the routing
    proof that `target_fold=10` really does take the `authorized_unseal` branch, not a mock of
    it -- `resolve_target_fold_frames` is exercised for real, only its (absent) authorization
    record is real too.
    """
    real_record_path = _REPO_ROOT / "artifacts" / "fold10_authorization.json"
    assert not real_record_path.is_file(), (
        "artifacts/fold10_authorization.json exists -- fold 10 may have been unsealed. "
        "This test intentionally fails loudly rather than silently skip."
    )
    df = _toy_metadata()
    with pytest.raises(AuthorizationError, match="no authorization record"):
        resolve_target_fold_frames(df, target_fold=10)


def test_target_fold_10_never_reaches_val_fold_ten_config_without_authorization() -> None:
    """Companion to the above: `authorized_unseal` raises before `resolve_target_fold_frames`
    ever constructs or returns anything, so calling with target_fold=10 today must raise BEFORE
    returning any `TargetFoldFrames` at all -- there is no partial-success path that hands back
    frames derived from fold 10."""
    df = _toy_metadata()
    with pytest.raises(AuthorizationError):
        result = resolve_target_fold_frames(df, target_fold=10)
        raise AssertionError(f"should never reach here: got {result!r}")


# ---------------------------------------------- gate-3 B4 fix: frozen-parameter enforcement


def test_frozen_parameters_check_passes_when_resolved_matches_exactly() -> None:
    frozen = {
        "n_strata": 16,
        "gain_limit": 250,
        "n_replicates": 2000,
        "geometry_limit": 1200,
        "causal_window": 40,
        "detection_n_records": 400,
    }
    _check_target_fold_10_frozen_parameters(dict(frozen), dict(frozen))  # must not raise


def test_frozen_parameters_check_raises_on_any_single_mismatch() -> None:
    frozen = {
        "n_strata": 16,
        "gain_limit": 250,
        "n_replicates": 2000,
        "geometry_limit": 1200,
        "causal_window": 40,
        "detection_n_records": 400,
    }
    resolved = dict(frozen)
    resolved["detection_n_records"] = 40  # a caller quietly using a smaller cohort
    with pytest.raises(SystemExit, match="frozen_parameters"):
        _check_target_fold_10_frozen_parameters(resolved, frozen)


# --------------------------------------------------- gate-3 status-field fix: error propagation


def test_count_battery_errors_is_zero_on_a_clean_run() -> None:
    metrics = {
        "preflight": {"failed": {}},
        "detection_cohort": {"n_records": 32},
        "full_battery": {"signal_seed0": {"5000": {"step": 5000, "auroc": 0.9}}},
    }
    assert _count_battery_errors(metrics) == 0


def test_count_battery_errors_counts_a_failed_battery_cell() -> None:
    """The regression this test guards: a per-cell exception used to be swallowed into the cell's
    own JSON without ever flipping the top-level status away from PASS."""
    metrics = {
        "preflight": {"failed": {}},
        "detection_cohort": {"n_records": 32},
        "full_battery": {
            "signal_seed0": {
                "5000": {"step": 5000, "auroc": 0.9},
                "20000": {"step": 20000, "error": "RuntimeError: boom"},
            }
        },
    }
    assert _count_battery_errors(metrics) == 1


def test_count_battery_errors_counts_preflight_and_detection_cohort_failures_too() -> None:
    metrics = {
        "preflight": {"failed": {"signal_seed0/step5000": "checkpoint missing"}},
        "detection_cohort": {"error": "ValueError: empty frame"},
        "full_battery": {},
    }
    assert _count_battery_errors(metrics) == 2


# --------------------------------------------------- gate-3 A-2 fix: frozen-input enforcement


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_resolve_frozen_inputs_hashes_every_file_and_all_checkpoints(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "ptbxl_database.csv").write_bytes(b"ptbxl content")
    (data_root / "scp_statements.csv").write_bytes(b"scp content")

    lead_stats_path = tmp_path / "lead_stats.json"
    lead_stats_path.write_bytes(b"lead stats content")
    rpeaks_npz_path = tmp_path / "rpeaks.npz"
    rpeaks_npz_path.write_bytes(b"rpeaks content")
    theta_tokens_path = tmp_path / "theta_tokens.npz"
    theta_tokens_path.write_bytes(b"theta tokens content")
    manifest_path = tmp_path / "manifest.parquet"
    manifest_path.write_bytes(b"manifest content")

    roster_dir = tmp_path / "roster"
    ckpt_a = roster_dir / "signal_seed0" / "step5000"
    ckpt_b = roster_dir / "control_seed0" / "step5000"
    ckpt_a.mkdir(parents=True)
    ckpt_b.mkdir(parents=True)
    (ckpt_a / jepa_checkpoint.STATE_FILENAME).write_bytes(b"weights a")
    (ckpt_a / jepa_checkpoint.CONFIG_FILENAME).write_bytes(b"config a")
    (ckpt_a / jepa_checkpoint.META_FILENAME).write_bytes(b"meta a")
    (ckpt_b / jepa_checkpoint.STATE_FILENAME).write_bytes(b"weights b")
    (ckpt_b / jepa_checkpoint.CONFIG_FILENAME).write_bytes(b"config b")
    (ckpt_b / jepa_checkpoint.META_FILENAME).write_bytes(b"meta b")

    result = _resolve_frozen_inputs(
        data_root=str(data_root),
        roster_dir=str(roster_dir),
        lead_stats_path=str(lead_stats_path),
        rpeaks_npz_path=str(rpeaks_npz_path),
        theta_tokens_path=str(theta_tokens_path),
        manifest_path=str(manifest_path),
        checkpoints={"signal_seed0/step5000": str(ckpt_a), "control_seed0/step5000": str(ckpt_b)},
    )

    assert result["data_root"] == str(data_root)
    assert result["roster_dir"] == str(roster_dir)
    assert result["metadata_sha256"] == {
        "ptbxl_database.csv": _sha256_bytes(b"ptbxl content"),
        "scp_statements.csv": _sha256_bytes(b"scp content"),
    }
    assert result["lead_stats_sha256"] == _sha256_bytes(b"lead stats content")
    assert result["rpeaks_npz_sha256"] == _sha256_bytes(b"rpeaks content")
    assert result["theta_tokens_npz_sha256"] == _sha256_bytes(b"theta tokens content")
    assert result["manifest_sha256"] == _sha256_bytes(b"manifest content")
    assert result["checkpoint_sha256"] == {
        "signal_seed0/step5000": {
            "state.pt": _sha256_bytes(b"weights a"),
            "config.yaml": _sha256_bytes(b"config a"),
            "meta.json": _sha256_bytes(b"meta a"),
        },
        "control_seed0/step5000": {
            "state.pt": _sha256_bytes(b"weights b"),
            "config.yaml": _sha256_bytes(b"config b"),
            "meta.json": _sha256_bytes(b"meta b"),
        },
    }


def test_check_frozen_inputs_passes_when_resolved_matches_exactly() -> None:
    frozen = {
        "data_root": "/d",
        "roster_dir": "/r",
        "metadata_sha256": {"ptbxl_database.csv": "a", "scp_statements.csv": "b"},
        "lead_stats_sha256": "c",
        "rpeaks_npz_sha256": "d",
        "theta_tokens_npz_sha256": "e",
        "manifest_sha256": "m",
        "checkpoint_sha256": {
            "signal_seed0/step5000": {"state.pt": "f", "config.yaml": "g", "meta.json": "h"}
        },
    }
    _check_target_fold_10_frozen_inputs(dict(frozen), dict(frozen))  # must not raise


def test_check_frozen_inputs_raises_when_a_checkpoint_hash_differs() -> None:
    """The exact scenario gate-3's A-2 finding named: --roster-dir repointed at a different
    checkpoint while every other check (script hash, arms/steps, frozen_parameters) still
    passes."""
    frozen = {
        "data_root": "/d",
        "roster_dir": "/r",
        "metadata_sha256": {"ptbxl_database.csv": "a", "scp_statements.csv": "b"},
        "lead_stats_sha256": "c",
        "rpeaks_npz_sha256": "d",
        "theta_tokens_npz_sha256": "e",
        "manifest_sha256": "m",
        "checkpoint_sha256": {
            "signal_seed0/step5000": {"state.pt": "f", "config.yaml": "g", "meta.json": "h"}
        },
    }
    resolved = dict(frozen)
    resolved["checkpoint_sha256"] = {
        "signal_seed0/step5000": {"state.pt": "DIFFERENT", "config.yaml": "g", "meta.json": "h"}
    }
    with pytest.raises(SystemExit, match="frozen_inputs"):
        _check_target_fold_10_frozen_inputs(resolved, frozen)


def test_check_frozen_inputs_raises_when_a_checkpoint_config_differs() -> None:
    """`config.yaml`/`meta.json` are hashed alongside `state.pt` now (gate-3 round 3) -- a
    shape-compatible config/meta drift that a strict state-dict load would not itself catch must
    still be caught here."""
    frozen = {
        "data_root": "/d",
        "roster_dir": "/r",
        "metadata_sha256": {"ptbxl_database.csv": "a", "scp_statements.csv": "b"},
        "lead_stats_sha256": "c",
        "rpeaks_npz_sha256": "d",
        "theta_tokens_npz_sha256": "e",
        "manifest_sha256": "m",
        "checkpoint_sha256": {
            "signal_seed0/step5000": {"state.pt": "f", "config.yaml": "g", "meta.json": "h"}
        },
    }
    resolved = dict(frozen)
    resolved["checkpoint_sha256"] = {
        "signal_seed0/step5000": {"state.pt": "f", "config.yaml": "DIFFERENT", "meta.json": "h"}
    }
    with pytest.raises(SystemExit, match="frozen_inputs"):
        _check_target_fold_10_frozen_inputs(resolved, frozen)


def test_check_frozen_inputs_raises_when_manifest_hash_differs() -> None:
    """The exact scenario gate-3's round-3 finding named: manifest.parquet was entirely absent
    from the schema before this fix."""
    frozen = {
        "data_root": "/d",
        "roster_dir": "/r",
        "metadata_sha256": {"ptbxl_database.csv": "a", "scp_statements.csv": "b"},
        "lead_stats_sha256": "c",
        "rpeaks_npz_sha256": "d",
        "theta_tokens_npz_sha256": "e",
        "manifest_sha256": "m",
        "checkpoint_sha256": {
            "signal_seed0/step5000": {"state.pt": "f", "config.yaml": "g", "meta.json": "h"}
        },
    }
    resolved = dict(frozen)
    resolved["manifest_sha256"] = "DIFFERENT"
    with pytest.raises(SystemExit, match="frozen_inputs"):
        _check_target_fold_10_frozen_inputs(resolved, frozen)


def test_check_frozen_inputs_raises_when_roster_dir_path_differs() -> None:
    frozen = {
        "data_root": "/d",
        "roster_dir": "/r",
        "metadata_sha256": {"ptbxl_database.csv": "a", "scp_statements.csv": "b"},
        "lead_stats_sha256": "c",
        "rpeaks_npz_sha256": "d",
        "theta_tokens_npz_sha256": "e",
        "manifest_sha256": "m",
        "checkpoint_sha256": {
            "signal_seed0/step5000": {"state.pt": "f", "config.yaml": "g", "meta.json": "h"}
        },
    }
    resolved = dict(frozen)
    resolved["roster_dir"] = "/some/other/roster"
    with pytest.raises(SystemExit, match="frozen_inputs"):
        _check_target_fold_10_frozen_inputs(resolved, frozen)


# ------------------------------ gate-3 round-3 invariant: no raw directory on the fold-10 path


def test_build_fold10_style_cohort_takes_no_artifacts_dir_parameter() -> None:
    """The exact bug round 3 found and constructed a live bypass against: `build_fold10_style_
    cohort` used to take `artifacts_dir` and re-derive `theta_tokens.npz`/`manifest.parquet`'s
    paths internally, decorating around the already-hash-checked `theta_tokens_path` variable
    `main()` had computed. Encodes the fix's own invariant as a signature check, cheap enough to
    run on every commit, so a regression can't creep back in via a plausible-looking convenience
    parameter."""
    params = set(inspect.signature(build_fold10_style_cohort).parameters)
    assert "artifacts_dir" not in params
    assert {"theta_tokens_path", "manifest_path"} <= params


# ------------------------------------------------------- gate-4 prep: shared resolution helpers


def test_frozen_parameters_dict_matches_the_argparse_defaults() -> None:
    """The single source of truth both this file's own CLI defaults and
    scripts/print_fold10_authorization_template.py's frozen_parameters block draw from."""
    assert frozen_parameters_dict() == {
        "n_strata": 16,
        "gain_limit": 250,
        "n_replicates": 2000,
        "geometry_limit": 1200,
        "causal_window": DEFAULT_CAUSAL_WINDOW,
        "detection_n_records": 400,
    }


def test_resolve_default_paths_derives_from_artifacts_dir_when_unset() -> None:
    result = resolve_default_paths("some/artifacts")
    assert result == {
        "roster_dir": "some/artifacts/roster",
        "lead_stats_path": "some/artifacts/lead_stats_f1to9.json",
        "rpeaks_npz_path": "some/artifacts/reference/phase/rpeaks.npz",
        "theta_tokens_path": "some/artifacts/phase/theta_tokens.npz",
        "manifest_path": "some/artifacts/manifest.parquet",
    }


def test_resolve_default_paths_honours_explicit_overrides() -> None:
    result = resolve_default_paths(
        "some/artifacts",
        roster_dir="/explicit/roster",
        lead_stats_path="/explicit/lead_stats.json",
        rpeaks_npz_path="/explicit/rpeaks.npz",
        theta_tokens_path="/explicit/theta_tokens.npz",
        manifest_path="/explicit/manifest.parquet",
    )
    assert result == {
        "roster_dir": "/explicit/roster",
        "lead_stats_path": "/explicit/lead_stats.json",
        "rpeaks_npz_path": "/explicit/rpeaks.npz",
        "theta_tokens_path": "/explicit/theta_tokens.npz",
        "manifest_path": "/explicit/manifest.parquet",
    }


def test_resolve_checkpoints_maps_every_requested_arm_and_step(tmp_path: Path) -> None:
    roster_dir = tmp_path / "roster"
    for arm in ("signal_seed0", "control_seed0"):
        for step_dir in ("checkpoint_step5000", "checkpoint_step20000"):
            d = roster_dir / arm / step_dir
            d.mkdir(parents=True)
            (d / jepa_checkpoint.STATE_FILENAME).write_bytes(b"x")

    checkpoints = resolve_checkpoints(str(roster_dir), ["signal_seed0", "control_seed0"], [5000])
    assert checkpoints == {
        "signal_seed0/step5000": str(roster_dir / "signal_seed0" / "checkpoint_step5000"),
        "control_seed0/step5000": str(roster_dir / "control_seed0" / "checkpoint_step5000"),
    }


def test_resolve_checkpoints_raises_when_arm_dir_missing(tmp_path: Path) -> None:
    roster_dir = tmp_path / "roster"
    roster_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="roster arm dir not found"):
        resolve_checkpoints(str(roster_dir), ["signal_seed0"], [5000])


def test_resolve_checkpoints_raises_when_requested_step_missing(tmp_path: Path) -> None:
    roster_dir = tmp_path / "roster"
    d = roster_dir / "signal_seed0" / "checkpoint_step5000"
    d.mkdir(parents=True)
    (d / jepa_checkpoint.STATE_FILENAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="requested step 20000 not found"):
        resolve_checkpoints(str(roster_dir), ["signal_seed0"], [20000])

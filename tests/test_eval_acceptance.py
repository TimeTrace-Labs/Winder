"""Tests for winder.eval.acceptance: the Phase P6 Tier 1 numerical-reproduction harness.

Split into two tiers, matching this project's own precedent
(`tests/test_acceptance_structure.py`): fast, always-run unit tests for the pure comparison/
bookkeeping logic (no I/O, no model, no GPU), and skip-gated integration tests that touch real
PTB-XL metadata / the copied-in reference artifacts when present -- never the full numeric
reproduction itself (that lives in `scripts/accept.py`'s own report, not in pytest, per the design
brief's own "the gate requires... pytest... green" alongside a separately-run acceptance script).
"""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import pytest
import torch

from winder.data.folds import LEGACY_FOLD_CONFIG
from winder.eval.acceptance import (
    AcceptanceCohort,
    build_split_frames,
    check_split_shapes,
    compare_bool,
    compare_value,
    eval_mask_from_theta,
    load_expected_finale_results,
    load_expected_g1,
    load_expected_gain,
)
from winder.eval.comparison import EvalCohort
from winder.paths import default_data_root

# ============================================================================ compare_value/bool


def test_compare_value_pass_at_and_inside_tolerance() -> None:
    assert compare_value("x", 1.0, 1.0, 0.0)["pass"] is True
    assert compare_value("x", 1.0, 1.00005, 1e-4)["pass"] is True  # exactly at the boundary


def test_compare_value_fails_just_outside_tolerance() -> None:
    result = compare_value("x", 1.0, 1.00011, 1e-4)
    assert result["pass"] is False
    assert result["abs_delta"] == pytest.approx(0.00011)
    assert result["name"] == "x"


def test_compare_value_zero_tolerance_requires_exact_match() -> None:
    assert compare_value("n", 14521, 14521, 0)["pass"] is True
    assert compare_value("n", 14521, 14520, 0)["pass"] is False


def test_compare_bool_exact_match_only() -> None:
    assert compare_bool("g1", True, True)["pass"] is True
    assert compare_bool("g1", False, False)["pass"] is True
    assert compare_bool("g1", True, False)["pass"] is False


# ==================================================================== eval_mask_from_theta


def test_eval_mask_from_theta_flags_all_nan_rows_only() -> None:
    theta = torch.tensor(
        [
            [0.1, 0.2, float("nan")],  # at least one finite -> survives
            [float("nan"), float("nan"), float("nan")],  # all NaN -> dropped
            [float("nan"), 1.5, float("nan")],  # one finite -> survives
        ]
    )
    mask = eval_mask_from_theta(theta)
    np.testing.assert_array_equal(mask, np.array([True, False, True]))


def test_eval_mask_from_theta_all_finite_survives_entirely() -> None:
    theta = torch.zeros(5, 4)
    mask = eval_mask_from_theta(theta)
    assert mask.all()


# =========================================================================== check_split_shapes


def _dummy_cohort(n_train: int, n_cal: int, n_eval: int) -> AcceptanceCohort:
    empty_eval_cohort = EvalCohort(
        waveforms={}, thetas={}, labels={}, patient_ids={}, rr_median_ms={}, patch_width=80
    )
    return AcceptanceCohort(
        eval_cohort=empty_eval_cohort,
        eval_ecg_ids=np.array([]),
        n_train=n_train,
        n_cal=n_cal,
        n_eval=n_eval,
    )


def test_check_split_shapes_passes_on_matching_counts_and_first_dropped_index() -> None:
    cohort = _dummy_cohort(14521, 2563, 2146)
    ev = np.ones(2146, dtype=bool)
    ev[93:111] = False  # 18 consecutive drops starting at 93 -> matches n_eval_surviving=2128
    result = check_split_shapes(cohort, ev, {"train": 14521, "cal": 2563, "eval": 2146})
    assert result["pass"] is True
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["first_dropped_row_index"]["measured"] == 93
    assert by_name["n_eval_surviving"]["measured"] == 2128


def test_check_split_shapes_fails_loudly_on_wrong_train_count() -> None:
    """Edge/failure mode: a train-pool-size mismatch (e.g. a stray train-limit truncation) must
    fail this check, not pass silently -- the exact bug class this gate exists to catch."""
    cohort = _dummy_cohort(6000, 2563, 2146)  # p1_panel_numerics.py's 6000-record-pool default
    ev = np.ones(2146, dtype=bool)
    ev[93:111] = False
    result = check_split_shapes(cohort, ev, {"train": 14521, "cal": 2563, "eval": 2146})
    assert result["pass"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["n_train"]["pass"] is False
    assert by_name["n_train"]["abs_delta"] == 8521


def test_check_split_shapes_no_dropped_rows_reports_sentinel_index() -> None:
    """Edge case: every eval row survives -- `first_dropped_row_index` must not crash on an empty
    `dropped` array, and should report the -1 sentinel rather than a spurious index."""
    cohort = _dummy_cohort(14521, 2563, 2146)
    ev = np.ones(2146, dtype=bool)
    result = check_split_shapes(cohort, ev, {"train": 14521, "cal": 2563, "eval": 2146})
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["first_dropped_row_index"]["measured"] == -1
    assert by_name["first_dropped_row_index"]["pass"] is False  # expected 93, sentinel != 93


# ========================================================== skip-gated: real metadata / artifacts

_PTBXL_ROOT = default_data_root()
_HAS_PTBXL_ROOT = os.path.isfile(os.path.join(_PTBXL_ROOT, "ptbxl_database.csv"))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFERENCE_ROOT = os.path.join(_REPO_ROOT, "artifacts", "reference")
_HAS_EXPECTED = os.path.isdir(os.path.join(_REFERENCE_ROOT, "expected"))
_HAS_THETA_TOKENS = os.path.isfile(os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz"))


@pytest.mark.skipif(not _HAS_PTBXL_ROOT, reason=f"PTB-XL data root not found at {_PTBXL_ROOT}")
def test_build_split_frames_matches_published_split_shapes() -> None:
    """Fast (metadata-only, no waveform decode) reproduction of assertion family 1's train/cal/
    eval counts -- the LEGACY_FOLD_CONFIG protocol itself, independent of any checkpoint."""
    frames = build_split_frames(_PTBXL_ROOT)
    assert len(frames["train"]) == 14521
    assert len(frames["cal"]) == 2563
    assert len(frames["eval"]) == 2146
    assert LEGACY_FOLD_CONFIG.train_folds == (1, 2, 3, 4, 5, 6, 7, 8)
    assert LEGACY_FOLD_CONFIG.val_fold == 9


@pytest.mark.skipif(
    not (_HAS_PTBXL_ROOT and _HAS_THETA_TOKENS),
    reason="PTB-XL data root or artifacts/reference/phase/theta_tokens.npz not found",
)
def test_eval_mask_from_theta_matches_published_dropped_row_count() -> None:
    """The checkpoint-independent theta-derived eval mask alone reproduces n_eval_surviving=2128
    and first_dropped_row_index=93 -- verified directly against real theta data, not asserted."""
    from winder.eval.readout import theta_for_frame
    from winder.transport.dataset import load_theta_tokens

    frames = build_split_frames(_PTBXL_ROOT)
    theta_by_id, theta_meta = load_theta_tokens(
        os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz")
    )
    theta_eval = theta_for_frame(frames["eval"], theta_by_id, cast(int, theta_meta["n_tokens"]))
    mask = eval_mask_from_theta(theta_eval)
    assert int(mask.sum()) == 2128
    dropped = np.flatnonzero(~mask)
    assert int(dropped[0]) == 93


@pytest.mark.skipif(not _HAS_EXPECTED, reason=f"{_REFERENCE_ROOT}/expected not found")
def test_load_expected_json_files_parse_and_carry_the_expected_top_level_shape() -> None:
    """Guards against schema drift in the copied-in reference JSONs: if their shape changes, this
    fails here (fast, no GPU) instead of deep inside `run_acceptance`'s KeyError."""
    finale = load_expected_finale_results(_REFERENCE_ROOT)
    assert finale["metrics"]["splits"] == {"train": 14521, "cal": 2563, "eval": 2146}
    assert "FIN_seed0" in finale["metrics"]["rows"]["superclass5"]["headline"]["per_arm"]

    gain = load_expected_gain(_REFERENCE_ROOT)
    assert set(gain) == {"FINs0_5k", "FINs0_25k", "FINs1_5k", "FINs1_25k"}

    g1 = load_expected_g1(_REFERENCE_ROOT)
    assert set(g1["results"]) == {
        "FIN_seed0_step5000",
        "FIN_seed0_step25000",
        "FIN_seed1_step5000",
        "FIN_seed1_step25000",
    }

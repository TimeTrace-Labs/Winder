"""Tests for winder.eval.gates: the G1 shuffled-theta transport-gain null and the patient-
clustered detection-gap CI.

`test_g1_gate_fails_on_a_phase_blind_model` is the exact test the design brief names: on
synthetic latents with NO phase structure at all, `g1_shuffled_theta_gain_null` must report
`ci_excludes_zero is False` -- proof the gate has teeth, not just that it returns some boolean.
`test_g1_shuffled_theta_gain_null_accepts_an_exactly_equivariant_model` is the positive control
this project's own precedent (`tests/test_transport_delta_gain.py`) always pairs with a
phase-blind negative control, so a reader can see the gate correctly ACCEPT a genuinely
phase-structured model, not merely reject a null one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from winder.eval.gates import (
    G1_SHUFFLED_FRACTION_BAND,
    corrected_by_record,
    detection_gap_ci,
    g1_accept,
    g1_shuffled_theta_gain_null,
    permute_theta_within_record,
)
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig

_K0, _N_J, _K_J = 2, [1, 2, 3], [1, 2, 1]  # K = 2 + 2*4 = 10


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


# ======================================================================== g1_accept, the gate


def test_g1_accept_requires_both_conditions() -> None:
    assert g1_accept(True, 0.0) is True
    assert g1_accept(False, 0.0) is False  # CI does not exclude zero
    assert g1_accept(True, 0.5) is False  # shuffled fraction way outside the band


def test_g1_accept_band_boundary_is_inclusive() -> None:
    assert g1_accept(True, G1_SHUFFLED_FRACTION_BAND) is True
    assert g1_accept(True, G1_SHUFFLED_FRACTION_BAND + 1e-9) is False


# ============================================================= permute_theta_within_record


def test_permute_theta_within_record_preserves_mask_and_marginal() -> None:
    theta = np.array(
        [
            [0.1, 0.2, np.nan, 0.4, 0.5],
            [np.nan, np.nan, 1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    out = permute_theta_within_record(theta, seed=0)
    assert np.array_equal(
        np.isnan(out), np.isnan(theta)
    )  # NaN mask preserved position-for-position
    for i in range(theta.shape[0]):
        finite_in = np.sort(theta[i][np.isfinite(theta[i])])
        finite_out = np.sort(out[i][np.isfinite(out[i])])
        np.testing.assert_array_equal(finite_in, finite_out)  # same multiset per record


def test_permute_theta_within_record_actually_reorders_when_possible() -> None:
    """Not vacuous: with many records of >1 finite value, at least one record's order changes."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * math.pi, size=(20, 8))
    out = permute_theta_within_record(theta, seed=1)
    assert not np.array_equal(theta, out)
    # every row is still a permutation of itself
    for i in range(theta.shape[0]):
        np.testing.assert_array_equal(np.sort(theta[i]), np.sort(out[i]))


def test_permute_theta_within_record_is_deterministic_under_a_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * math.pi, size=(10, 6))
    a = permute_theta_within_record(theta, seed=7)
    b = permute_theta_within_record(theta, seed=7)
    np.testing.assert_array_equal(a, b)


# ==================================================================== g1_shuffled_theta_gain_null


def test_g1_shuffled_theta_gain_null_accepts_an_exactly_equivariant_model() -> None:
    """Positive control: z_t = R_{theta_t} z_0 exactly, theta drawn independently PER RECORD
    (not a shared grid -- a shared identical grid across every record leaves enough residual
    cross-record structure after an in-record shuffle that the shuffled fraction lands just
    outside the +/-0.02 band, measured empirically). The true gain is at its ceiling and
    shuffling theta destroys the correspondence between rotation and data, so the gate must
    ACCEPT -- both booleans True."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    n, t = 24, 32
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)
    patient_ids = np.arange(n)

    result = g1_shuffled_theta_gain_null(
        z, theta, operator, patient_ids, n_strata=8, n_replicates=500, seed=0
    )
    assert result["true_overall_gain_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert result["ci_excludes_zero"] is True
    assert result["shuffled_fraction_within_pm0.02"] is True
    assert result["g1_pass"] is True


def test_g1_gate_fails_on_a_phase_blind_model() -> None:
    """The required test: theta plays NO role in generating z at all (drawn independently), so
    there is no cardiac-clock structure for a shuffle to destroy. A phase-blind model must NOT
    pass this gate -- `ci_excludes_zero` must be False, proving the gate has teeth rather than
    just returning some boolean."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(1)
    n, t = 16, 24
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z = torch.randn(
        n, t, operator.dimension, dtype=torch.float64, generator=gen
    )  # independent of theta
    patient_ids = np.arange(n)

    result = g1_shuffled_theta_gain_null(
        z, theta, operator, patient_ids, n_strata=8, n_replicates=500, seed=0
    )
    assert result["ci_excludes_zero"] is False
    assert result["g1_pass"] is False


def test_g1_shuffled_theta_gain_null_same_seed_reproduces_exactly() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    n, t = 10, 16
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    patient_ids = np.arange(n)

    a = g1_shuffled_theta_gain_null(z, theta, operator, patient_ids, n_strata=8, seed=3)
    b = g1_shuffled_theta_gain_null(z, theta, operator, patient_ids, n_strata=8, seed=3)
    assert a == b


# =========================================================================== corrected_by_record


def test_corrected_by_record_subtracts_severity_zero_per_record() -> None:
    sev = {
        "0.0": (np.array([0.5, 0.5, 0.5]), np.array([0, 1, 2])),
        "1.0": (np.array([0.9, 0.7, 0.6]), np.array([0, 1, 2])),
    }
    out = corrected_by_record(sev)
    assert out is not None
    per_record, peak = out
    assert peak == "1.0"
    assert per_record == pytest.approx({0: 0.4, 1: 0.2, 2: 0.1})


def test_corrected_by_record_peak_chosen_by_raw_finite_mean_not_corrected_mean() -> None:
    """Two non-zero severities: '0.5' has the larger RAW mean AUROC but a SMALLER corrected mean
    once severity-0 is subtracted (a high but flat null on those records); '1.0' has a lower raw
    mean but is chosen only if the selection rule uses corrected means -- it must not be."""
    sev = {
        "0.0": (np.array([0.6, 0.6]), np.array([0, 1])),
        "0.5": (np.array([0.95, 0.95]), np.array([0, 1])),  # raw mean 0.95, corrected mean 0.35
        "1.0": (np.array([0.7, 0.7]), np.array([0, 1])),  # raw mean 0.70, corrected mean 0.10
    }
    out = corrected_by_record(sev)
    assert out is not None
    _per_record, peak = out
    assert peak == "0.5"  # the larger RAW mean wins, not the larger corrected one


def test_corrected_by_record_returns_none_without_severity_zero() -> None:
    sev = {"1.0": (np.array([0.9]), np.array([0]))}
    assert corrected_by_record(sev) is None


def test_corrected_by_record_returns_none_with_only_severity_zero() -> None:
    sev = {"0.0": (np.array([0.5]), np.array([0]))}
    assert corrected_by_record(sev) is None


def test_corrected_by_record_drops_non_overlapping_records() -> None:
    sev = {
        "0.0": (np.array([0.5, 0.5]), np.array([0, 1])),
        "1.0": (np.array([0.9, 0.9, 0.9]), np.array([0, 1, 2])),  # record 2 has no severity-0
    }
    out = corrected_by_record(sev)
    assert out is not None
    per_record, _peak = out
    assert set(per_record) == {0, 1}  # record 2 excluded -- no baseline to correct against


# ============================================================================= detection_gap_ci


def test_detection_gap_ci_computes_the_baseline_corrected_gap() -> None:
    trained = {
        "0.0": (np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 2, 3])),
        "1.0": (np.array([0.9, 0.8, 0.9, 0.8]), np.array([0, 1, 2, 3])),  # +0.4/+0.3/+0.4/+0.3
    }
    untrained = {
        ("offline", "matched_filter"): {
            "0.0": (np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 2, 3])),
            "1.0": (np.array([0.6, 0.6, 0.6, 0.6]), np.array([0, 1, 2, 3])),  # +0.1 flat
        }
    }
    patient_ids = np.array([10, 10, 20, 20])
    out = detection_gap_ci(trained, untrained, patient_ids, n_replicates=200, seed=0)
    assert out is not None
    assert out["untrained_best_detector"] == "matched_filter/offline"
    assert out["gap"]["mean"] == pytest.approx(
        0.35 - 0.1, abs=1e-9
    )  # mean(trained) - mean(untrained)
    assert out["n_records_paired"] == 4
    assert out["ci_excludes_zero"] is True  # a uniform +0.25 gap on every record


def test_detection_gap_ci_prefers_the_untrained_cell_with_the_larger_corrected_mean() -> None:
    trained = {
        "0.0": (np.array([0.5, 0.5]), np.array([0, 1])),
        "1.0": (np.array([0.9, 0.9]), np.array([0, 1])),
    }
    weak_cell = {
        "0.0": (np.array([0.5, 0.5]), np.array([0, 1])),
        "1.0": (np.array([0.55, 0.55]), np.array([0, 1])),  # corrected +0.05
    }
    strong_cell = {
        "0.0": (np.array([0.5, 0.5]), np.array([0, 1])),
        "1.0": (np.array([0.8, 0.8]), np.array([0, 1])),  # corrected +0.30, the adversarial pick
    }
    untrained = {("a", "weak"): weak_cell, ("b", "strong"): strong_cell}
    patient_ids = np.array([1, 2])
    out = detection_gap_ci(trained, untrained, patient_ids, n_replicates=100, seed=0)
    assert out is not None
    assert out["untrained_best_detector"] == "strong/b"


def test_detection_gap_ci_returns_none_when_trained_cell_unresolvable() -> None:
    trained = {"1.0": (np.array([0.9]), np.array([0]))}  # no severity-0 at all
    untrained = {
        ("a", "b"): {
            "0.0": (np.array([0.5]), np.array([0])),
            "1.0": (np.array([0.9]), np.array([0])),
        }
    }
    assert detection_gap_ci(trained, untrained, np.array([1]), n_replicates=10, seed=0) is None


def test_detection_gap_ci_returns_none_when_no_untrained_cell_resolves() -> None:
    trained = {"0.0": (np.array([0.5]), np.array([0])), "1.0": (np.array([0.9]), np.array([0]))}
    untrained = {("a", "b"): {"1.0": (np.array([0.9]), np.array([0]))}}  # no severity-0
    assert detection_gap_ci(trained, untrained, np.array([1]), n_replicates=10, seed=0) is None

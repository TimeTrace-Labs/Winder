"""Tests for winder.transport.report: the per-checkpoint operator/geometry/gain reports."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.transport.delta_gain import delta_stratified_gain
from winder.transport.report import gain_report, geometry_report, operator_report

_K0, _N_J, _K_J = 2, [1, 2, 3], [1, 2, 1]  # K = 2 + 2*4 = 10


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


# ============================================================================= operator_report


def test_operator_report_none_operator_reports_has_operator_false() -> None:
    out = operator_report("control_arm", None)
    assert out == {"checkpoint": "control_arm", "has_operator": False}


def test_operator_report_matches_the_operators_own_spectrum() -> None:
    operator = _toy_operator()
    out = operator_report("cyclic_arm", operator)
    assert out["has_operator"] is True
    assert out["k0"] == _K0
    assert out["n_j"] == _N_J
    assert out["k_j"] == _K_J
    assert out["dimension"] == operator.dimension
    assert out["omega"] == pytest.approx([float(v) for v in _N_J])  # cyclic: omega frozen at n_j
    assert out["omega_minus_n"] == pytest.approx([0.0, 0.0, 0.0])
    assert out["closure_residual"] == pytest.approx(0.0, abs=1e-6)  # cyclic closes exactly
    assert out["learnable_omega"] is False


def test_operator_report_reflects_a_drifted_free_operator() -> None:
    operator = FreeOperator(FreeOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))
    with torch.no_grad():
        operator.omega.mul_(0.9)
    out = operator_report("free_arm", operator)
    assert out["learnable_omega"] is True
    assert out["distance_to_nearest_integer"][0] == pytest.approx(0.1, abs=1e-6)
    assert out["closure_residual"] > 0.5  # a drifted omega does not close


# ============================================================================= geometry_report


def test_geometry_report_has_the_expected_top_level_shape() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    n, t = 6, 16
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    out = geometry_report(z, theta, operator)
    assert set(out) == {"pooled", "trajectory", "binned_means_invariant_block", "loops"}
    assert set(out["pooled"]) == {"token_level_spectrum", "mean_pooled", "demodulated"}
    assert set(out["loops"]) == {str(n) for n in _N_J}
    for j_label, loop in out["loops"].items():
        assert "error" in loop or set(loop) == {
            "harmonic_index",
            "n_j",
            "k_j",
            "reference_bin",
            "real",
            "imag",
            "block_norm",
            "residual_norm",
            "coherence",
        }, j_label


def test_geometry_report_loop_projection_recovers_a_clean_circle_under_exact_equivariance() -> None:
    """An end-to-end sanity check: exactly-equivariant data should trace a near-perfect circle
    (coherence close to 1) in every harmonic's own loop projection."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(1)
    n, t = 8, 32
    theta = (torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)).expand(n, t).contiguous()
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    out = geometry_report(z, theta, operator)
    for j_label, loop in out["loops"].items():
        assert "error" not in loop, j_label
        coherence = np.asarray(loop["coherence"])
        finite = coherence[np.isfinite(coherence)]
        assert finite.min() > 0.99, (j_label, finite.min())


# ================================================================================== gain_report


def test_gain_report_matches_delta_stratified_gain_and_adds_a_bootstrap() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    n, t = 10, 12
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    patient_ids = np.repeat(np.arange(n // 2), 2)  # two records per patient

    direct = delta_stratified_gain(z, theta, operator, n_strata=8)
    report = gain_report(z, theta, operator, patient_ids, n_strata=8, seed=0)

    assert report["overall_mean_gain"] == pytest.approx(direct.overall_mean_gain)
    assert report["overall_gain_fraction"] == pytest.approx(direct.overall_gain_fraction)
    assert report["mean_gain"] == pytest.approx(direct.mean_gain, nan_ok=True)
    assert set(report["bootstrap"]) == {"mean", "lo", "hi", "n_clusters", "n_values"}
    assert report["bootstrap"]["n_clusters"] == n // 2
    assert report["bootstrap"]["lo"] <= report["bootstrap"]["mean"] <= report["bootstrap"]["hi"]


def test_gain_report_is_deterministic_under_a_fixed_seed() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(3)
    n, t = 8, 10
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    patient_ids = np.arange(n)

    a = gain_report(z, theta, operator, patient_ids, n_strata=6, seed=7)
    b = gain_report(z, theta, operator, patient_ids, n_strata=6, seed=7)
    assert a == b

import math

import numpy as np
import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.transport.geometry import (
    fisher_separation,
    harmonic_energy_spectrum,
    phase_resolved_trajectory,
    pooled_geometry_report,
)

_K0, _N_J, _K_J = 2, [1, 2, 3], [1, 2, 1]  # K = 2 + 2*4 = 10


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


# ======================================================================= harmonic_energy_spectrum


def test_harmonic_energy_spectrum_sums_over_a_harmonics_own_planes() -> None:
    """The gauge-invariant contraction: harmonic n=2 has k_j=2 planes (4 dims) and its reported
    energy must be their SUM, not any single plane's."""
    operator = _toy_operator()
    z = torch.zeros(1, 10)
    z[0, :2] = 1.0  # invariant block: energy 2
    z[0, 2:4] = 2.0  # harmonic n=1 (1 plane): energy 8
    z[0, 4:8] = 1.0  # harmonic n=2 (2 planes): energy 4
    z[0, 8:10] = 0.0  # harmonic n=3: energy 0

    spec = harmonic_energy_spectrum(z, operator)
    assert spec["invariant_energy"] == pytest.approx(2.0 / 10 * 10)  # per-dim mean * dims
    # per_dim_energy is a MEAN over rows, then summed over that block's dims:
    assert spec["invariant_energy"] == pytest.approx(2.0)
    assert spec["harmonic_energy"][0] == pytest.approx(8.0)
    assert spec["harmonic_energy"][1] == pytest.approx(4.0)
    assert spec["harmonic_energy"][2] == pytest.approx(0.0)


def test_harmonic_energy_spectrum_shares_sum_to_one() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(50, 10, generator=gen)
    spec = harmonic_energy_spectrum(z, operator)
    total = spec["invariant_share"] + sum(spec["harmonic_share"])
    assert total == pytest.approx(1.0)


def test_harmonic_energy_spectrum_isotropic_reference_matches_dimension_shares() -> None:
    operator = _toy_operator()
    z = torch.randn(10, 10)
    spec = harmonic_energy_spectrum(z, operator)
    assert spec["invariant_share_isotropic_reference"] == pytest.approx(2 / 10)
    assert spec["harmonic_share_isotropic_reference"] == pytest.approx([2 / 10, 4 / 10, 2 / 10])


def test_isotropic_latent_gives_shares_matching_the_isotropic_reference() -> None:
    """Sanity: for genuinely isotropic z, each block's energy share is its dimension share."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(200_000, 10, generator=gen)
    spec = harmonic_energy_spectrum(z, operator)
    assert spec["invariant_share"] == pytest.approx(
        spec["invariant_share_isotropic_reference"], abs=0.01
    )
    for share, ref in zip(
        spec["harmonic_share"], spec["harmonic_share_isotropic_reference"], strict=True
    ):
        assert share == pytest.approx(ref, abs=0.01)


# ========================================================== pooled_geometry_report: Prop 4.1 / 4.2


def test_mean_pooling_annihilates_harmonics_while_demodulation_preserves_them() -> None:
    """The central before/after claim, on an exactly-equivariant synthetic trajectory: mean
    pooling's harmonic shares collapse to ~0 (Prop 4.1) while demodulation's match the token-
    level spectrum (Prop 4.2)."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    n, t = 30, 60
    theta = (torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)).expand(n, t).contiguous()
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    report = pooled_geometry_report(z, theta, operator)
    mean_shares = report["mean_pooled"]["spectrum"]["harmonic_share"]
    demod_shares = report["demodulated"]["spectrum"]["harmonic_share"]

    assert sum(mean_shares) < 1e-12  # every harmonic annihilated
    assert report["mean_pooled"]["spectrum"]["invariant_share"] == pytest.approx(1.0)
    assert sum(demod_shares) > 0.3  # harmonics survive demodulation


def test_demodulation_raises_effective_rank_over_mean_pooling() -> None:
    """Prop 4.1 leaves a K0-dimensional readout; Prop 4.2 retains all K. On an exactly-
    equivariant trajectory the mean-pooled covariance is rank <= K0 while the demodulated one
    is full-rank -- the quantitative form of 'mean pooling erases structure'."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(3)
    n, t = 200, 60
    theta = (torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)).expand(n, t).contiguous()
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    report = pooled_geometry_report(z, theta, operator)
    mean_er = report["mean_pooled"]["geometry"]["effective_rank"]
    demod_er = report["demodulated"]["geometry"]["effective_rank"]
    assert mean_er < _K0 + 0.5  # collapsed to the invariant block's own dimension
    assert demod_er > 8.0  # close to the full K=10
    assert demod_er > mean_er


def test_pooled_geometry_report_handles_nan_theta_rows() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(4)
    n, t = 20, 30
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    theta[0] = float("nan")  # one record with no valid tokens at all
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)

    report = pooled_geometry_report(z, theta, operator)
    # the all-NaN record is dropped from both poolings' geometry, not propagated
    assert report["mean_pooled"]["geometry"]["n_records"] == n - 1
    assert report["demodulated"]["geometry"]["n_records"] == n - 1
    assert math.isfinite(report["demodulated"]["geometry"]["effective_rank"])


# ================================================================================ fisher_separation


def test_fisher_separation_is_large_for_well_separated_classes() -> None:
    gen = np.random.default_rng(0)
    n, k = 200, 8
    features = np.concatenate([gen.normal(0, 1, (n, k)), gen.normal(0, 1, (n, k)) + 10.0], axis=0)
    labels = np.concatenate([np.zeros((n, 1)), np.ones((n, 1))], axis=0)
    result = fisher_separation(features, labels)
    assert result["mean"] > 10.0


def test_fisher_separation_is_near_zero_for_identical_classes() -> None:
    gen = np.random.default_rng(1)
    n, k = 400, 8
    features = gen.normal(0, 1, (2 * n, k))
    labels = np.concatenate([np.zeros((n, 1)), np.ones((n, 1))], axis=0)  # labels carry no signal
    result = fisher_separation(features, labels)
    assert result["mean"] < 0.05


def test_fisher_separation_nan_for_a_degenerate_column() -> None:
    features = np.random.default_rng(2).normal(0, 1, (50, 4))
    labels = np.zeros((50, 2))
    labels[:, 0] = 1.0  # column 0 all-positive, column 1 all-negative -- both degenerate
    result = fisher_separation(features, labels)
    assert all(math.isnan(r) for r in result["per_class"])
    assert math.isnan(result["mean"])


def test_fisher_separation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="rows"):
        fisher_separation(np.zeros((10, 4)), np.zeros((9, 2)))


# ===================================================================== phase_resolved_trajectory


def test_phase_resolved_trajectory_invariant_block_is_constant_under_exact_equivariance() -> None:
    """By construction the invariant block does not move with phase -- so its std across phase
    bins is ~0 on an exactly-equivariant trajectory. On a trained checkpoint this same number
    measures how far the encoder is from that idealisation."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(5)
    n, t = 40, 120
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    traj = phase_resolved_trajectory(z, theta, operator, n_bins=12)
    # z0 varies across records, so the binned MEAN invariant block is the same average vector in
    # every bin -- its across-bin std reflects only sampling noise, not phase dependence.
    assert traj["invariant_block_std_across_bins"] < 0.5
    assert traj["n_bins"] == 12
    assert sum(traj["bin_counts"]) == n * t


def test_phase_resolved_trajectory_bins_cover_the_full_cycle() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(6)
    n, t = 10, 240
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)

    traj = phase_resolved_trajectory(z, theta, operator, n_bins=8)
    assert all(c > 0 for c in traj["bin_counts"])  # uniform theta populates every bin
    assert len(traj["binned_means"]) == 8
    assert len(traj["spectrum_per_bin"]) == 8


def test_phase_resolved_trajectory_marks_empty_bins() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(7)
    n, t = 5, 20
    theta = torch.full((n, t), 0.1, dtype=torch.float64)  # every token in bin 0
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)

    traj = phase_resolved_trajectory(z, theta, operator, n_bins=8)
    assert traj["bin_counts"][0] == n * t
    assert all(c == 0 for c in traj["bin_counts"][1:])
    assert traj["spectrum_per_bin"][0] is not None
    assert all(s is None for s in traj["spectrum_per_bin"][1:])

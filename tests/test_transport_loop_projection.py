import math

import numpy as np
import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.transport.geometry import (
    _harmonic_block_as_complex,
    harmonic_loop_projection,
    phase_resolved_trajectory,
)

_K0, _N_J, _K_J = 2, [1, 2, 3], [3, 2, 1]  # K = 2 + 2*6 = 14


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


def _random_unitary(k: int, seed: int) -> torch.Tensor:
    """A Haar-ish `U(k)` element: QR of a complex Gaussian, phase-fixed so R has positive
    diagonal (otherwise Q is unitary but not uniformly distributed -- irrelevant here, but it
    keeps the draw reproducible and well conditioned)."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.complex(
        torch.randn(k, k, dtype=torch.float64, generator=gen),
        torch.randn(k, k, dtype=torch.float64, generator=gen),
    )
    q, r = torch.linalg.qr(a)
    scaled: torch.Tensor = q * (r.diagonal().sgn()).unsqueeze(0)
    return scaled


def _apply_gauge(
    vectors: torch.Tensor, operator: CyclicOperator, j: int, g: torch.Tensor
) -> torch.Tensor:
    """Rewrite harmonic block `j` of every row of `(n, K)` as `g @ v`, leaving all other blocks
    byte-identical -- the explicit realisation of the `U(k_j)` gauge freedom the projection is
    supposed to be blind to."""
    out = vectors.clone().to(torch.float64)
    offset = operator.k0 + 2 * int(operator.k_j[:j].sum())
    width = 2 * int(operator.k_j[j])
    v = _harmonic_block_as_complex(vectors, operator, j)  # (n, k_j)
    rotated = v @ g.transpose(0, 1)  # (g v)_a = sum_b g_ab v_b
    planes = torch.stack([rotated.real, rotated.imag], dim=-1)  # (n, k_j, 2)
    out[..., offset : offset + width] = planes.reshape(*rotated.shape[:-1], width)
    return out


def _bin_thetas(n_bins: int) -> torch.Tensor:
    """Bin CENTRES, matching phase_resolved_trajectory's own left-edge binning of [0, 2*pi)."""
    return (torch.arange(n_bins, dtype=torch.float64) + 0.5) * (2 * math.pi / n_bins)


# =========================================================== the property the plot rests on


def test_projection_is_invariant_under_the_blocks_own_unitary_gauge() -> None:
    """The whole reason this function exists. `U(k_j)` commutes with `R_Delta` on block `j`, so
    it is unobservable -- any quantity claimed to be interpretable must not move under it."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    binned = torch.randn(8, operator.dimension, dtype=torch.float64, generator=gen)

    j = 0  # k_j = 3, so the gauge group here is a genuinely non-trivial U(3)
    base = harmonic_loop_projection(binned, operator, harmonic_index=j)
    g = _random_unitary(int(operator.k_j[j]), seed=1)
    gauged = harmonic_loop_projection(
        _apply_gauge(binned, operator, j, g), operator, harmonic_index=j
    )

    np.testing.assert_allclose(base["real"], gauged["real"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(base["imag"], gauged["imag"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(base["block_norm"], gauged["block_norm"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(base["coherence"], gauged["coherence"], rtol=0, atol=1e-10)


def test_a_naive_single_plane_readout_is_NOT_gauge_invariant() -> None:
    """Negative control proving the previous test has teeth: plotting plane 0's raw (x, y) --
    the obvious thing to do, and what the projection deliberately avoids -- moves substantially
    under the same unobservable gauge change."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    binned = torch.randn(8, operator.dimension, dtype=torch.float64, generator=gen)

    j, offset = 0, _K0
    g = _random_unitary(int(operator.k_j[j]), seed=3)
    gauged = _apply_gauge(binned, operator, j, g)
    before = binned[:, offset : offset + 2]
    after = gauged[:, offset : offset + 2]
    assert float((after - before).norm()) > 0.1 * float(before.norm())


# ============================================================ what the loop actually measures


def test_exact_equivariance_traces_a_circle_wound_n_j_times() -> None:
    """Under `v_b = exp(i n_j theta_b) w`, the projection is `exp(i n_j (theta_b - theta_ref))`
    times a constant radius: constant modulus, and a total unwrapped winding of exactly
    `2*pi*n_j` around the cycle."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(4)
    n_bins = 16
    theta = _bin_thetas(n_bins)
    z0 = torch.randn(1, operator.dimension, dtype=torch.float64, generator=gen)
    binned = operator.transport(z0.expand(n_bins, -1), theta)

    for j, n_j in enumerate(_N_J):
        proj = harmonic_loop_projection(binned, operator, harmonic_index=j)
        radius = np.hypot(proj["real"], proj["imag"])
        np.testing.assert_allclose(radius, radius[0], rtol=1e-10)

        angles = np.unwrap(np.arctan2(proj["imag"], proj["real"]))
        step = 2 * math.pi * n_j / n_bins
        assert angles[-1] - angles[0] == pytest.approx(step * (n_bins - 1), abs=1e-8)
        # and the SPACING is uniform, which is the part a real checkpoint has to earn
        np.testing.assert_allclose(np.diff(angles), step, atol=1e-8)

        np.testing.assert_allclose(proj["residual_norm"], 0.0, atol=1e-10)
        np.testing.assert_allclose(proj["coherence"], 1.0, atol=1e-10)


def test_a_phase_blind_latent_collapses_to_one_point_and_draws_no_loop() -> None:
    """The null the figure is read against: identical bin means give `p_b = ||v||` real and
    positive in every bin -- zero imaginary part, zero enclosed area, no winding."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(5)
    n_bins = 8
    one = torch.randn(1, operator.dimension, dtype=torch.float64, generator=gen)
    binned = one.expand(n_bins, -1).contiguous()

    proj = harmonic_loop_projection(binned, operator, harmonic_index=0)
    np.testing.assert_allclose(proj["imag"], 0.0, atol=1e-12)
    np.testing.assert_allclose(proj["real"], proj["real"][0], rtol=1e-12)
    np.testing.assert_allclose(proj["coherence"], 1.0, atol=1e-12)


def test_the_reference_bin_lands_on_the_positive_real_axis_at_its_own_norm() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(6)
    binned = torch.randn(8, operator.dimension, dtype=torch.float64, generator=gen)

    for ref in (0, 3, 7):
        proj = harmonic_loop_projection(binned, operator, harmonic_index=1, reference_bin=ref)
        assert proj["real"][ref] == pytest.approx(proj["block_norm"][ref], rel=1e-12)
        assert proj["imag"][ref] == pytest.approx(0.0, abs=1e-12)
        assert proj["coherence"][ref] == pytest.approx(1.0, rel=1e-12)


def test_independent_bin_content_leaves_a_positive_out_of_line_residual() -> None:
    """`residual_norm` is what distinguishes "the bins move around one circle" from "the bins
    point in unrelated directions that happen to project onto something". Under exact
    equivariance it is 0 (test above); for unrelated bins it must not be."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(7)
    binned = torch.randn(8, operator.dimension, dtype=torch.float64, generator=gen)

    proj = harmonic_loop_projection(binned, operator, harmonic_index=0)
    residual = np.asarray(proj["residual_norm"])
    assert residual[0] == pytest.approx(0.0, abs=1e-12)  # the reference explains itself exactly
    assert (residual[1:] > 1e-6).all()
    # Pythagoras must close: |p|^2 + residual^2 == ||v||^2
    modulus_sq = np.asarray(proj["real"]) ** 2 + np.asarray(proj["imag"]) ** 2
    np.testing.assert_allclose(
        modulus_sq + residual**2, np.asarray(proj["block_norm"]) ** 2, rtol=1e-10
    )


def test_coherence_never_exceeds_one() -> None:
    """Cauchy-Schwarz, as an executable check on the normalisation."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(8)
    for seed_offset in range(5):
        binned = torch.randn(10, operator.dimension, dtype=torch.float64, generator=gen) * float(
            seed_offset + 1
        )
        proj = harmonic_loop_projection(binned, operator, harmonic_index=2)
        assert max(proj["coherence"]) <= 1.0 + 1e-12


# ================================================================ composition and error cases


def test_consumes_phase_resolved_trajectory_output_directly() -> None:
    """The two functions are meant to chain: trajectory produces `(n_bins, K)` binned means,
    projection turns one block of them into a plottable loop."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(9)
    n, t, n_bins = 30, 60, 8
    theta = (torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)).expand(n, t).contiguous()
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    traj = phase_resolved_trajectory(z, theta, operator, n_bins=n_bins)
    binned = torch.tensor(traj["binned_means"], dtype=torch.float64)
    proj = harmonic_loop_projection(binned, operator, harmonic_index=0)
    assert len(proj["real"]) == n_bins
    angles = np.unwrap(np.arctan2(proj["imag"], proj["real"]))
    assert (np.diff(angles) > 0).all()  # monotone bin ordering around the loop


def test_zero_energy_reference_bin_raises_rather_than_dividing_by_zero() -> None:
    operator = _toy_operator()
    binned = torch.randn(8, operator.dimension, dtype=torch.float64)
    offset = operator.k0
    binned[2, offset : offset + 2 * int(operator.k_j[0])] = 0.0
    with pytest.raises(ValueError, match="zero or non-finite"):
        harmonic_loop_projection(binned, operator, harmonic_index=0, reference_bin=2)


def test_shape_and_index_errors_raise() -> None:
    operator = _toy_operator()
    good = torch.randn(8, operator.dimension, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\(n_bins, K\)"):
        harmonic_loop_projection(torch.randn(2, 8, operator.dimension), operator)
    with pytest.raises(ValueError, match="operator dimension"):
        harmonic_loop_projection(torch.randn(8, 3), operator)
    with pytest.raises(ValueError, match="harmonic_index"):
        harmonic_loop_projection(good, operator, harmonic_index=len(_N_J))
    with pytest.raises(ValueError, match="reference_bin"):
        harmonic_loop_projection(good, operator, reference_bin=8)

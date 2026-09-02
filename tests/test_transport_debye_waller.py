import math

import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.transport.debye_waller import (
    DebyeWallerCurve,
    debye_waller_curve,
    fit_debye_waller_slope,
    harmonic_block_amplitudes,
)

_K0, _N_J, _K_J = 2, [1, 2, 3, 4], [1, 1, 1, 1]  # K = 2 + 2*4 = 10


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


# ==================================================================== harmonic_block_amplitudes


def test_harmonic_block_amplitudes_reads_each_planes_own_norm() -> None:
    operator = _toy_operator()
    pooled = torch.zeros(1, 10)
    pooled[0, 2:4] = torch.tensor([3.0, 4.0])  # harmonic n=1: norm 5
    pooled[0, 4:6] = torch.tensor([0.0, 2.0])  # harmonic n=2: norm 2
    pooled[0, 6:8] = torch.tensor([1.0, 0.0])  # harmonic n=3: norm 1
    pooled[0, 8:10] = torch.tensor([0.0, 0.0])  # harmonic n=4: norm 0

    amps = harmonic_block_amplitudes(pooled, operator)
    torch.testing.assert_close(amps, torch.tensor([5.0, 2.0, 1.0, 0.0]), atol=1e-6, rtol=0)


def test_harmonic_block_amplitudes_averages_over_records() -> None:
    operator = _toy_operator()
    pooled = torch.zeros(2, 10)
    pooled[0, 2:4] = torch.tensor([3.0, 4.0])  # norm 5
    pooled[1, 2:4] = torch.tensor([0.0, 1.0])  # norm 1
    amps = harmonic_block_amplitudes(pooled, operator)
    assert float(amps[0]) == pytest.approx(3.0)  # mean(5, 1) = 3


# ================================================= debye_waller_curve: the exact-prediction check


def test_debye_waller_curve_at_sigma_zero_recovers_the_clean_amplitude() -> None:
    """Prop 4.2 exactness (sigma=0, no injected jitter): the pooled amplitude at every harmonic
    equals z0's own block norm exactly, on an exactly-equivariant trajectory."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    t = 50
    theta = torch.rand(t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z0 = torch.randn(1, operator.dimension, dtype=torch.float64, generator=gen)
    tokens = operator.transport(z0.expand(t, -1), theta).unsqueeze(0)  # (1, T, K)

    curve = debye_waller_curve(tokens, theta.unsqueeze(0), operator, sigmas=[0.0], generator=gen)
    expected = harmonic_block_amplitudes(z0, operator)
    torch.testing.assert_close(torch.tensor(curve.amplitudes[0]), expected, atol=1e-9, rtol=1e-6)


def test_debye_waller_slope_is_minus_one_half_on_synthetic_data() -> None:
    """The derivation, checked directly: for an exactly-equivariant trajectory, demodulating
    with theta + N(0, sigma^2) gives, in expectation, R_{-noise} @ z0 per token; averaging over
    many independent noise draws converges (law of large numbers) to
    E[R(-n*noise)] @ z0_block_n = exp(-n^2*sigma^2/2) * z0_block_n (E[cos(n*eps)] is exactly the
    Gaussian characteristic function at frequency n; E[sin(n*eps)] = 0 by symmetry) -- so the
    fitted slope of log(amplitude_ratio) against n^2*sigma^2 must be -1/2, a large-T Monte Carlo
    convergence check on the theory's own closed-form prediction, not a fit to arbitrary data."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(1)
    t = 20_000  # large T for tight law-of-large-numbers convergence
    theta = torch.rand(t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z0 = torch.randn(1, operator.dimension, dtype=torch.float64, generator=gen)
    tokens = operator.transport(z0.expand(t, -1), theta).unsqueeze(0)

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4]
    curve = debye_waller_curve(tokens, theta.unsqueeze(0), operator, sigmas=sigmas, generator=gen)
    fit = fit_debye_waller_slope(curve)

    assert fit.slope == pytest.approx(-0.5, abs=0.03)
    assert fit.intercept == pytest.approx(0.0, abs=0.05)
    assert fit.r_squared > 0.95
    assert fit.n_points == len(_N_J) * (len(sigmas) - 1)


def test_debye_waller_slope_scales_with_n_squared_not_n() -> None:
    """A weaker, structurally distinct sanity check: at a FIXED sigma, higher harmonics decay
    MORE than lower ones (a monotonicity any n^2-scaling law implies), so a bug that used |n|
    instead of n^2 (or ignored n entirely) would still be caught even if the exact slope fit
    were loose."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    t = 20_000
    theta = torch.rand(t, dtype=torch.float64, generator=gen) * 2 * math.pi
    z0 = torch.randn(1, operator.dimension, dtype=torch.float64, generator=gen)
    tokens = operator.transport(z0.expand(t, -1), theta).unsqueeze(0)

    curve = debye_waller_curve(
        tokens, theta.unsqueeze(0), operator, sigmas=[0.0, 0.3], generator=gen
    )
    clean, jittered = curve.amplitudes[0], curve.amplitudes[1]
    ratios = [j / c for j, c in zip(jittered, clean, strict=True)]
    assert ratios == sorted(ratios, reverse=True)  # n=1 retains most, n=4 retains least


def test_debye_waller_curve_excludes_nan_theta_tokens() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(3)
    t = 200
    theta = torch.rand(t, dtype=torch.float64, generator=gen) * 2 * math.pi
    theta[::4] = float("nan")  # 25% missing, like real data
    z0 = torch.randn(1, operator.dimension, dtype=torch.float64, generator=gen)
    tokens = operator.transport(z0.expand(t, -1), theta).unsqueeze(0)

    # Must not raise, and sigma=0 must still recover the clean amplitude (proving NaN tokens
    # were excluded from the pool rather than propagating NaN into the whole record).
    curve = debye_waller_curve(tokens, theta.unsqueeze(0), operator, sigmas=[0.0], generator=gen)
    assert all(math.isfinite(a) for a in curve.amplitudes[0])


# ========================================================================= fit_debye_waller_slope


def test_fit_handles_fewer_than_two_points() -> None:
    curve = DebyeWallerCurve(sigmas=[0.0], n_j=[1, 2], amplitudes=[[1.0, 1.0]])
    fit = fit_debye_waller_slope(curve)
    assert math.isnan(fit.slope)
    assert fit.n_points == 0


def test_fit_skips_nonpositive_reference_or_ratio() -> None:
    # harmonic 0 has zero clean amplitude (nothing to take a ratio against); harmonic 1 is normal.
    curve = DebyeWallerCurve(
        sigmas=[0.0, 0.1],
        n_j=[1, 2],
        amplitudes=[[0.0, 1.0], [0.0, math.exp(-0.5 * 4 * 0.01)]],
    )
    fit = fit_debye_waller_slope(curve)
    assert fit.n_points == 1  # only harmonic 2's point survives

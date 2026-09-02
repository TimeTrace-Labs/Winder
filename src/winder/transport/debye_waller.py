"""The Debye-Waller check (notes/internal/phase_equivariance_notes_v13.pdf A.6): harmonic `n` is
attenuated by `exp(-n^2 * sigma_theta^2 / 2)` when the phase clock carries Gaussian jitter of
std `sigma_theta`. This gives a falsifiable, SLOPE-FIXED prediction rather than a fitted one:
injecting a further KNOWN jitter `sigma` on top of a trained checkpoint's own (already-jittery)
phase clock and regressing `log(amplitude_ratio)` against `n^2 * sigma^2` must return a slope of
`-1/2` -- if the recovered demodulated amplitude is genuine phase-locked harmonic content, not
some other artifact that merely correlates with theta.

Method: demodulate-and-pool (`winder.eval.pooling.demodulated_pool`) using `theta + N(0,
sigma^2)` in place of the checkpoint's own theta (`sigma=0` recovers the unperturbed estimate),
then take the RMS norm of each harmonic block's own 2-D planes as that harmonic's amplitude at
that `sigma`. Ratio against the `sigma=0` amplitude removes the harmonic's own (unknown, real)
clean magnitude from the regression -- only the DECAY, whose functional form the theory actually
predicts, is being tested.
"""

import math
from dataclasses import dataclass

import torch

from winder.eval.pooling import demodulated_pool
from winder.operators.harmonic import HarmonicTransport

__all__ = [
    "DebyeWallerCurve",
    "DebyeWallerFit",
    "harmonic_block_amplitudes",
    "debye_waller_curve",
    "fit_debye_waller_slope",
]


def harmonic_block_amplitudes(pooled: torch.Tensor, operator: HarmonicTransport) -> torch.Tensor:
    """`(N, K)` demodulated-and-pooled vectors -> `(n_harmonics,)`: mean, over records and over
    a harmonic's own `k_j` planes, of each plane's 2-D norm. The invariant block (`k0` dims) has
    no harmonic index and is not included."""
    amplitudes = []
    offset = operator.k0
    for k_j in operator.k_j.tolist():
        width = 2 * int(k_j)
        block = pooled[:, offset : offset + width]
        planes = block.reshape(block.shape[0], int(k_j), 2)
        plane_amplitude = planes.norm(dim=-1)  # (N, k_j)
        amplitudes.append(float(plane_amplitude.mean()))
        offset += width
    return torch.tensor(amplitudes)


@dataclass(frozen=True)
class DebyeWallerCurve:
    sigmas: list[float]
    n_j: list[int]
    amplitudes: list[list[float]]  # amplitudes[i][j] = amplitude of harmonic n_j[j] at sigmas[i]


def debye_waller_curve(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    sigmas: list[float],
    *,
    generator: torch.Generator,
) -> DebyeWallerCurve:
    """`z`: `(N, T, K)`, `theta`: `(N, T)` (NaN where undefined) -- a real checkpoint's own
    tokens. `sigmas` should include 0.0 (the unperturbed reference every ratio is taken
    against). Only VALID (finite-theta) entries are perturbed; NaN stays NaN, so
    `demodulated_pool`'s own exclusion is unaffected by this injection.
    """
    valid = torch.isfinite(theta)
    amplitudes_per_sigma = []
    for sigma in sigmas:
        noise = torch.randn(theta.shape, generator=generator) * sigma
        noisy_theta = torch.where(valid, theta + noise, theta)
        pooled = demodulated_pool(z, noisy_theta, operator)
        finite_rows = pooled[torch.isfinite(pooled).all(dim=-1)]
        amplitudes_per_sigma.append(harmonic_block_amplitudes(finite_rows, operator).tolist())
    return DebyeWallerCurve(
        sigmas=list(sigmas), n_j=list(operator.n_j), amplitudes=amplitudes_per_sigma
    )


@dataclass(frozen=True)
class DebyeWallerFit:
    slope: float  # predicted exactly -0.5
    intercept: float  # predicted exactly 0.0 (ratio is 1 at sigma=0 by construction)
    r_squared: float
    n_points: int


def fit_debye_waller_slope(curve: DebyeWallerCurve) -> DebyeWallerFit:
    """Ordinary least squares of `log(amplitude_ratio)` against `n^2 * sigma^2`, pooled over
    every `(harmonic, sigma>0)` pair -- `sigma=0` points are excluded (ratio is trivially 1,
    `x=0`, and would only add zero-information leverage at the origin)."""
    zero_idx = curve.sigmas.index(0.0)
    reference = curve.amplitudes[zero_idx]

    xs: list[float] = []
    ys: list[float] = []
    for i, sigma in enumerate(curve.sigmas):
        if sigma == 0.0:
            continue
        for j, n_j in enumerate(curve.n_j):
            ref = reference[j]
            if ref <= 0:
                continue
            ratio = curve.amplitudes[i][j] / ref
            if ratio <= 0:
                continue
            xs.append((n_j**2) * (sigma**2))
            ys.append(math.log(ratio))

    n = len(xs)
    if n < 2:
        return DebyeWallerFit(
            slope=float("nan"), intercept=float("nan"), r_squared=float("nan"), n_points=n
        )

    x_arr = torch.tensor(xs, dtype=torch.float64)
    y_arr = torch.tensor(ys, dtype=torch.float64)
    x_mean, y_mean = x_arr.mean(), y_arr.mean()
    x_centered, y_centered = x_arr - x_mean, y_arr - y_mean
    denom = float((x_centered * x_centered).sum())
    if denom == 0.0:
        return DebyeWallerFit(
            slope=float("nan"), intercept=float("nan"), r_squared=float("nan"), n_points=n
        )

    slope = float((x_centered * y_centered).sum()) / denom
    intercept = float(y_mean - slope * x_mean)
    predicted = intercept + slope * x_arr
    ss_res = float(((y_arr - predicted) ** 2).sum())
    ss_tot = float((y_centered * y_centered).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return DebyeWallerFit(slope=slope, intercept=intercept, r_squared=r_squared, n_points=n)

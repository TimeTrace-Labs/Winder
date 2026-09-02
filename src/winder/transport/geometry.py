"""Latent-geometry structure, before and after demodulation.

Every quantity here is chosen to be GAUGE-INVARIANT under the stabiliser the operator's own
normal form leaves free (notes/internal/phase_equivariance_notes_v13.pdf §2.4.1, A.0.5): with
`Q = I` and multiplicity `k_j > 1` per harmonic, the basis is unidentifiable up to
`O(K0) x U(k_1) x ... x U(k_m)` -- rotating within the invariant block, or within a single
harmonic's own `k_j` planes, changes `R_Delta` not at all. So:

- **Safe to interpret**: energy SUMMED over a harmonic's own planes (`harmonic_energy_spectrum`),
  the invariant-vs-harmonic split, any spectrum of the whole covariance (eigenvalues are
  basis-free), and effective/stable rank. These are functions of gauge-invariant contractions.
- **NOT safe to interpret**: any individual coordinate's value, any single plane's amplitude in
  isolation, the "direction" of a harmonic block. A plot of coordinate 37 means nothing.

The before/after comparison this module exists for is `masked_mean_pool` vs `demodulated_pool`
(`winder.eval.pooling`), on the SAME token set (valid theta only), from the SAME checkpoint --
so the only thing that differs is the combination rule, not the data. Prop 4.1 says mean pooling
annihilates every harmonic and leaves a `K0`-dimensional readout; Prop 4.2 says demodulation
recovers all `K`. `pooled_geometry_report` measures whether that actually happens on a trained
checkpoint, rather than assuming the propositions' idealised conditions hold.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from winder.eval.pooling import demodulated_pool, masked_mean_pool
from winder.jepa.diagnostics import covariance, effective_rank, stable_rank
from winder.operators.harmonic import HarmonicTransport

__all__ = [
    "harmonic_energy_spectrum",
    "PooledGeometry",
    "pooled_geometry_report",
    "fisher_separation",
    "phase_resolved_trajectory",
    "harmonic_loop_projection",
]


def harmonic_energy_spectrum(z: torch.Tensor, operator: HarmonicTransport) -> dict[str, Any]:
    """Energy carried by the invariant block and by each harmonic index, summed over that
    harmonic's own `k_j` planes (the gauge-invariant contraction -- module docstring).

    `z` is `(..., K)`; every leading dim is averaged over. Returns absolute mean energies plus
    their shares of the total, and an isotropic reference share for each block (its own
    dimension count over `K`) -- the number a perfectly isotropic latent would give, so a share
    can be read as "n times isotropic" rather than as a bare fraction.
    """
    flat = z.reshape(-1, z.shape[-1]).to(torch.float64)
    k = flat.shape[-1]
    per_dim_energy = flat.square().mean(dim=0)  # (K,)

    k0_energy = float(per_dim_energy[: operator.k0].sum())
    harmonic_energies = []
    offset = operator.k0
    for k_j in operator.k_j.tolist():
        width = 2 * int(k_j)
        harmonic_energies.append(float(per_dim_energy[offset : offset + width].sum()))
        offset += width

    total = k0_energy + sum(harmonic_energies)
    total = total if total > 0 else 1.0
    return {
        "n_j": list(operator.n_j),
        "k0": operator.k0,
        "k_j": [int(v) for v in operator.k_j.tolist()],
        "invariant_energy": k0_energy,
        "harmonic_energy": harmonic_energies,
        "invariant_share": k0_energy / total,
        "harmonic_share": [e / total for e in harmonic_energies],
        "invariant_share_isotropic_reference": operator.k0 / k,
        "harmonic_share_isotropic_reference": [2 * int(v) / k for v in operator.k_j.tolist()],
    }


@dataclass(frozen=True)
class PooledGeometry:
    """Rank/spectrum of one pooled record-embedding matrix `(N, K)`."""

    n_records: int
    effective_rank: float
    stable_rank: float
    eigenvalues: list[float]  # descending, of the centred covariance
    mean_norm: float


def _pooled_geometry(pooled: torch.Tensor) -> PooledGeometry:
    finite = pooled[torch.isfinite(pooled).all(dim=-1)]
    cov = covariance(finite)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0).flip(0)  # descending
    return PooledGeometry(
        n_records=int(finite.shape[0]),
        effective_rank=effective_rank(cov),
        stable_rank=stable_rank(cov),
        eigenvalues=[float(v) for v in eigvals],
        mean_norm=float(finite.to(torch.float64).mean(dim=0).norm()),
    )


def pooled_geometry_report(
    z: torch.Tensor, theta: torch.Tensor, operator: HarmonicTransport
) -> dict[str, Any]:
    """The before/after-demodulation comparison, on one `(N, T, K)` token tensor and its
    `(N, T)` theta. Both poolings see exactly the same valid-theta token set, so the only
    difference is the combination rule (module docstring).

    Also reports each pooling's own harmonic energy spectrum: Prop 4.1 predicts the mean-pooled
    embedding's harmonic shares collapse toward zero (leaving the invariant block dominant),
    while Prop 4.2 predicts demodulation preserves them.
    """
    mean_pooled = masked_mean_pool(z, theta)
    demodulated = demodulated_pool(z, theta, operator)
    return {
        "token_level_spectrum": harmonic_energy_spectrum(z, operator),
        "mean_pooled": {
            "geometry": _pooled_geometry(mean_pooled).__dict__,
            "spectrum": harmonic_energy_spectrum(
                mean_pooled[torch.isfinite(mean_pooled).all(dim=-1)], operator
            ),
        },
        "demodulated": {
            "geometry": _pooled_geometry(demodulated).__dict__,
            "spectrum": harmonic_energy_spectrum(
                demodulated[torch.isfinite(demodulated).all(dim=-1)], operator
            ),
        },
    }


def fisher_separation(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Multi-label between-class vs within-class scatter, per label column, as a class-separation
    measure that does NOT involve fitting a probe -- so it reports the geometry's own structure
    rather than a particular optimiser's ability to exploit it.

    For each binary label column: `trace(between) / trace(within)`, where `between` is the
    squared distance between the positive and negative class means and `within` is the pooled
    within-class variance. Higher = the two classes' point clouds are further apart relative to
    their own spread. Reported per column plus the mean over columns; a column with no positives
    or no negatives yields NaN and is excluded from the mean.
    """
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"features and labels must both be 2-D, got {features.shape} and {labels.shape}"
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"features has {features.shape[0]} rows but labels has {labels.shape[0]}")
    ratios = []
    for col in range(labels.shape[1]):
        positive = features[labels[:, col] > 0.5]
        negative = features[labels[:, col] <= 0.5]
        if len(positive) < 2 or len(negative) < 2:
            ratios.append(float("nan"))
            continue
        between = float(np.sum((positive.mean(axis=0) - negative.mean(axis=0)) ** 2))
        within = float(positive.var(axis=0).sum() + negative.var(axis=0).sum())
        ratios.append(between / within if within > 0 else float("nan"))
    finite = [r for r in ratios if np.isfinite(r)]
    return {
        "per_class": ratios,
        "mean": float(np.mean(finite)) if finite else float("nan"),
    }


def phase_resolved_trajectory(
    z: torch.Tensor, theta: torch.Tensor, operator: HarmonicTransport, *, n_bins: int = 24
) -> dict[str, Any]:
    """The latent's own average path around the cardiac cycle: bin every valid token by its
    theta, average `z` within each bin, and report both the raw binned means and their
    per-harmonic energies.

    This is the "does it actually go round?" plot's data. Under exact equivariance a harmonic-n
    block's binned mean traces a circle completed `n` times over the cycle; a phase-invariant
    (collapsed) latent traces a single stationary point. The invariant block should be
    approximately constant across bins BY CONSTRUCTION (that is what makes it invariant) -- its
    variation across bins is therefore a direct, interpretable measure of how far the trained
    encoder is from the equivariant idealisation.
    """
    valid = torch.isfinite(theta)
    z_flat = z[valid].to(torch.float64)  # (M, K)
    theta_flat = theta[valid].to(torch.float64)
    bin_idx = torch.clamp((theta_flat / (2 * np.pi / n_bins)).long(), 0, n_bins - 1)

    binned_means = []
    counts = []
    for b in range(n_bins):
        rows = z_flat[bin_idx == b]
        counts.append(int(rows.shape[0]))
        binned_means.append(
            rows.mean(dim=0) if rows.shape[0] > 0 else torch.full((z.shape[-1],), float("nan"))
        )
    binned = torch.stack(binned_means)  # (n_bins, K)

    # Needs >= 2 populated bins to have an across-bin spread at all -- one bin (or none) is not
    # a degenerate answer to report as 0.0, it is an undefined one.
    populated = binned[torch.isfinite(binned).all(dim=-1)]
    invariant_across_bins = (
        float(populated[:, : operator.k0].std(dim=0).mean())
        if len(populated) >= 2
        else float("nan")
    )
    return {
        "n_bins": n_bins,
        "bin_counts": counts,
        "binned_means": binned.tolist(),
        "spectrum_per_bin": [
            harmonic_energy_spectrum(binned[b : b + 1], operator)
            if torch.isfinite(binned[b]).all()
            else None
            for b in range(n_bins)
        ],
        "invariant_block_std_across_bins": invariant_across_bins,
    }


def _harmonic_block_as_complex(
    vectors: torch.Tensor, operator: HarmonicTransport, j: int
) -> torch.Tensor:
    """`(n, K) -> (n, k_j)` complex: harmonic index `j`'s own block, with each of its `k_j`
    2-D planes read as one complex number `x + iy`.

    This identification is what makes the block's transport a SCALAR: `R(n_j * Delta)` acting on
    plane `(x, y)` is exactly multiplication by `exp(i * n_j * Delta)` on `x + iy`, the same
    scalar on every one of the `k_j` planes (`HarmonicTransport`'s docstring: multiplicity ties
    planes to a common rate). The block's gauge group is therefore the full `U(k_j)` of
    complex-linear isometries -- everything commuting with scalar multiplication.
    """
    offset = operator.k0 + 2 * int(operator.k_j[:j].sum())
    width = 2 * int(operator.k_j[j])
    block = vectors[..., offset : offset + width].to(torch.float64)
    planes = block.reshape(*block.shape[:-1], width // 2, 2)
    return torch.complex(planes[..., 0], planes[..., 1])


def harmonic_loop_projection(
    binned_means: torch.Tensor,
    operator: HarmonicTransport,
    *,
    harmonic_index: int = 0,
    reference_bin: int = 0,
) -> dict[str, Any]:
    """A GAUGE-INVARIANT 2-D picture of how one harmonic block's binned mean moves around the
    cardiac cycle -- the honest version of a "latent trajectory loop" plot.

    `binned_means` is `(n_bins, K)`, e.g. `phase_resolved_trajectory`'s own output. Writing
    `v_b` for bin `b`'s harmonic-`j` block as a complex vector (`_harmonic_block_as_complex`),
    with `u = v_ref / ||v_ref||`:

        p_b = <u, v_b>  = sum_i conj(u_i) * v_{b,i}    (Hermitian, so complex-valued)

    and the plot is `(Re p_b, Im p_b)`.

    **Why not just plot two coordinates.** With multiplicity `k_j = 21` the basis inside a
    harmonic block is unidentifiable up to `U(21)` (module docstring), so plane 0's own `(x, y)`
    is meaningless -- a different-but-equivalent checkpoint would draw a different picture.
    `p_b` is invariant: replacing every `v_b` by `g v_b` for `g` in `U(k_j)` also replaces `u` by
    `g u`, and `<g u, g v_b> = <u, v_b>` exactly. What is plotted is thus a property of the
    representation, not of the arbitrary basis it was stored in.

    **What is discovered versus what is forced.** Nothing here is true by construction. `v_b` is
    an average of RAW encoder outputs in phase bin `b` -- no demodulation, no operator applied.
    A phase-blind encoder gives the same `v_b` in every bin, hence `p_b = ||v||` real and
    positive for all `b`: a single point on the positive real axis, no loop at all. A loop
    appears only if the encoder's own harmonic-`j` block genuinely rotates with theta. Under
    exact equivariance, `v_b = exp(i * n_j * theta_b) w` for a common `w`, giving
    `p_b = exp(i * n_j * (theta_b - theta_ref)) * ||w||` -- a circle of radius `||w||` traversed
    `n_j` times over the cycle, in bin order and with uniform spacing. Both the winding and the
    spacing are therefore measurements.

    Also returned per bin:
      `residual_norm` -- `sqrt(||v_b||^2 - |p_b|^2)`, the part of bin `b`'s mean lying OUTSIDE
        the complex line through the reference direction (zero under exact equivariance).
      `coherence` -- `|p_b| / ||v_b||` in `[0, 1]`, the fraction of bin `b`'s block-mean norm the
        single reference direction explains. Both are gauge-invariant for the same reason `p_b`
        is.
    """
    if binned_means.ndim != 2:
        raise ValueError(f"binned_means must be (n_bins, K), got {tuple(binned_means.shape)}")
    if binned_means.shape[-1] != operator.dimension:
        raise ValueError(
            f"binned_means' last dim {binned_means.shape[-1]} != operator dimension "
            f"{operator.dimension}"
        )
    n_bins = binned_means.shape[0]
    if not 0 <= harmonic_index < len(operator.n_j):
        raise ValueError(
            f"harmonic_index {harmonic_index} out of range for {len(operator.n_j)} harmonics"
        )
    if not 0 <= reference_bin < n_bins:
        raise ValueError(f"reference_bin {reference_bin} out of range for {n_bins} bins")

    v = _harmonic_block_as_complex(binned_means, operator, harmonic_index)  # (n_bins, k_j)
    norms = v.abs().square().sum(dim=-1).sqrt()  # (n_bins,), real
    ref_norm = norms[reference_bin]
    if not torch.isfinite(ref_norm) or float(ref_norm) == 0.0:
        raise ValueError(
            f"reference_bin {reference_bin} has zero or non-finite harmonic-{harmonic_index} "
            f"energy -- it cannot define a direction; pick a populated bin"
        )
    u = v[reference_bin] / ref_norm.to(v.dtype)
    p = (u.conj().unsqueeze(0) * v).sum(dim=-1)  # (n_bins,) complex

    # Residual as the norm of the ORTHOGONAL COMPONENT, not as sqrt(||v||^2 - |p|^2). The two are
    # equal in exact arithmetic, but the latter subtracts two nearly-equal positives whenever the
    # residual is small -- catastrophic cancellation that floors the answer at ~sqrt(eps)*||v||
    # (~6e-8 in float64), which is exactly the regime the equivariant case lives in and exactly
    # where a "is this zero?" reading matters most. Subtracting the projection first is
    # cancellation-free and reaches ~eps*||v||.
    residual = (v - p.unsqueeze(-1) * u.unsqueeze(0)).abs().square().sum(dim=-1).sqrt()
    return {
        "harmonic_index": harmonic_index,
        "n_j": int(operator.n_j[harmonic_index]),
        "k_j": int(operator.k_j[harmonic_index]),
        "reference_bin": reference_bin,
        "real": [float(x) for x in p.real],
        "imag": [float(x) for x in p.imag],
        "block_norm": [float(x) for x in norms],
        "residual_norm": [float(x) for x in residual],
        "coherence": [
            float(a / b) if float(b) > 0 else float("nan")
            for a, b in zip(p.abs(), norms, strict=True)
        ],
    }

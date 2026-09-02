"""Collapse-guard diagnostics for the transport arm -- cheap, pure functions computed from a
`z` batch, the operator, or the predictor, meant to be read every few hundred steps of a real
run rather than wired into `winder.jepa.train.train_step`'s own hot path.

Two failure modes this module exists to catch, neither of which `StepMetrics.trans_gain`/
`closure_residual` (`winder.jepa.train`) catches on its own:

- **Invariant collapse**: the encoder satisfies the transport loss perfectly by putting all its
  energy in the phase-invariant block, where nothing has to move (`winder.operators.harmonic`'s
  own module docstring). `k0_energy_fraction`/`block_energy_participation_ratio` measure this
  directly; `winder.jepa.regularizers.SigReg` is the structural guard against it (a rank-k0
  latent is maximally anisotropic, so SIGReg penalises it heavily), verified as a math gate in
  `tests/test_transport_diagnostics.py`, not merely assumed.
- **omega -> 0 collapse** (the free arm only): `R_delta -> I` for every delta, so
  `closure_residual == 0` -- the SAME value it takes when omega closes correctly at the declared
  integers. `closure_residual` alone cannot distinguish "closed at the right frequencies" from
  "collapsed to no rotation at all" -- `omega_summary`'s `min_abs_omega` is what catches this
  (the predecessor prototype's own free arm collapsed to `omega ~ 0.086`, see
  `winder.operators.harmonic`'s module docstring).
"""

from dataclasses import dataclass

import torch
from torch import nn

from winder.operators.harmonic import HarmonicTransport

__all__ = [
    "k0_energy_fraction",
    "block_energy_participation_ratio",
    "OmegaSummary",
    "omega_summary",
    "ln_gamma_cv",
]


def k0_energy_fraction(z: torch.Tensor, k0: int) -> float:
    """Fraction of total squared l2 norm carried by the first `k0` (invariant-block)
    coordinates of `z`'s last dimension, averaged over every leading-dim slice. Isotropic
    reference: `k0 / K` (uniform energy spread over all K dims). `k0 == 0` returns 0.0 exactly
    (no invariant block exists to collapse into)."""
    if k0 == 0:
        return 0.0
    total = z.square().sum(dim=-1)
    k0_energy = z[..., :k0].square().sum(dim=-1)
    frac = k0_energy / total.clamp_min(1e-12)
    return float(frac.mean())


def block_energy_participation_ratio(z: torch.Tensor, operator: HarmonicTransport) -> float:
    """Participation ratio (`1 / sum(p_i^2)`, `p_i` = block `i`'s own share of total energy)
    over the operator's blocks: one invariant block (`k0` dims) plus one block PER HARMONIC
    INDEX, pooling that harmonic's own `k_j`-fold multiplicity together (those `k_j` planes
    share one physical rotation rate -- `HarmonicTransport`'s own module docstring on what
    multiplicity means). Ranges from 1 (all energy concentrated in a single block) to
    `n_max + 1` (perfectly even split across every block) -- a low value alongside a high
    `k0_energy_fraction` is the invariant-collapse signature specifically; a low value with
    energy elsewhere indicates a *different* harmonic dominating instead.
    """
    energies = [z[..., : operator.k0].square().sum(dim=-1)]
    offset = operator.k0
    for k_j in operator.k_j.tolist():
        width = 2 * int(k_j)
        energies.append(z[..., offset : offset + width].square().sum(dim=-1))
        offset += width
    stacked = torch.stack(energies, dim=-1)  # (..., n_blocks)
    total = stacked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    shares = stacked / total
    participation_ratio = 1.0 / shares.square().sum(dim=-1).clamp_min(1e-12)
    return float(participation_ratio.mean())


@dataclass(frozen=True)
class OmegaSummary:
    min_abs_omega: float  # the omega -> 0 collapse canary (see module docstring)
    max_int_dist: float  # max_j |omega_j - round(omega_j)| -- general non-closure, not omega -> 0
    omega: list[float]  # raw per-harmonic-index values, for reading which harmonic moved


def omega_summary(operator: HarmonicTransport) -> OmegaSummary:
    """Trivial (`max_int_dist` ~ 0, `min_abs_omega` == the smallest declared integer) for the
    cyclic arm, since its omega never moves -- meaningful for the free arm, whose omega is a
    learned parameter."""
    omega = operator.omega.detach()
    return OmegaSummary(
        min_abs_omega=float(omega.abs().min()),
        max_int_dist=float((omega - omega.round()).abs().max()),
        omega=[float(v) for v in omega.tolist()],
    )


def ln_gamma_cv(module: nn.Module) -> float:
    """Coefficient of variation (`std / mean`) of every `nn.LayerNorm`'s learned affine scale
    (`.weight`, "gamma") found anywhere inside `module`, pooled across every LayerNorm --
    bounds how much of `O(K)` the `Q = I` restriction (`winder.transport.loss`'s module
    docstring) would actually cost if this module's own output were ever demodulated. At a
    fresh init, gamma == 1 everywhere and LayerNorm's affine step is exactly
    conjugation-invariant under ANY orthogonal Q; growing CV means the invariant subgroup
    shrinks toward signed permutations only. `NaN` if `module` contains no LayerNorm at all.
    """
    gammas = [m.weight.detach().flatten() for m in module.modules() if isinstance(m, nn.LayerNorm)]
    if not gammas:
        return float("nan")
    all_gamma = torch.cat(gammas)
    mean = float(all_gamma.mean())
    if mean == 0.0:
        return float("nan")
    return abs(float(all_gamma.std()) / mean)

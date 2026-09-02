"""`HarmonicTransport`: the transport operator's normal form (notes/internal/
phase_equivariance_notes_v13.pdf Eq. 6), with Q = I.

R_Delta = I_{k0} (+) direct-sum over harmonic index j of k_j copies of R(n_j * Delta), where
R(phi) is the ordinary 2x2 rotation matrix. `CyclicOperator`/`FreeOperator` (this package's two
registered arms) are both thin subclasses of this one class, differing in exactly one
constructor argument (`learnable_omega`) -- everything else, including the applied maths, is
shared.

**Q = I is not a simplification, it is exact.** The note's Eq. 6 is `R_Delta = Q B_Delta Q^T`
for an orthogonal change of basis Q; the transport loss (Eq. 13) is `1 - cos<R_Delta z_src,
z_tgt>` on l2-normalised arguments. Both `cos<.,.>` and the l2 normaliser (including its epsilon
clamp, which is a function of the norm alone) are O(K)-invariant, so `Q^T` commutes through the
whole loss: `L_trans(Q B Q^T; z) == L_trans(B; Q^T z)` for any Q. Since this MVP's projector
(`winder.jepa.projector.MlpProjectionHead`) ends in an unconstrained `nn.Linear` with no
normalisation anywhere downstream of it, `Q^T` is absorbable into that layer's own weight/bias
with no loss of generality for the transport loss alone -- see
`winder/transport/loss.py`'s module docstring for the one place this does NOT extend (the
predictor's LayerNorms), which is a bounded, measured restriction on the *joint* objective, not
on the operator.

Applied as an O(K) elementwise block rotation, never a dense K x K matmul: each of the
`sum(k_j)` harmonic planes is rotated by its own 2x2 formula directly on the (x, y) pair those
two coordinates carry. This is O(N*K) per call, with no `(N, K, K)` intermediate.

Multiplicity ties planes together, not apart: `k_j` copies of harmonic `n_j` are `k_j`
independent 2-D planes rotating at the exact SAME rate (the representation-theoretic meaning of
"multiplicity of the same irreducible representation" -- notes A.0.4, "assigning frequency n_j a
multiplicity k_j gives it a 2k_j-dimensional subspace"). The free arm therefore learns one omega
PER HARMONIC INDEX (`len(n_j)` learnable scalars), broadcast across that harmonic's k_j planes --
not one independent omega per plane. Untying them would let multiplicity-mates drift apart,
which is a different (and much less constrained) hypothesis than the note's own construction.
"""

import math

import torch
from torch import nn

from winder.operators.base import TransportOperator

__all__ = ["HarmonicTransport"]


def _validate_spectrum(k0: int, n_j: list[int], k_j: list[int]) -> None:
    if k0 < 0:
        raise ValueError(f"k0 must be >= 0, got {k0}")
    if len(n_j) == 0:
        raise ValueError(
            "n_j must be non-empty: a phase-invariant-only operator (m=0) transports nothing "
            "and is not what this class is for"
        )
    if len(n_j) != len(k_j):
        raise ValueError(f"n_j and k_j must have equal length, got {len(n_j)} and {len(k_j)}")
    if n_j != list(range(1, len(n_j) + 1)):
        raise ValueError(
            f"n_j must be the contiguous set {{1, ..., {len(n_j)}}} in ascending order (Eq. 23's "
            f"contiguity constraint -- no gaps, no repeats: a gapped or repeated spectrum buys "
            f"fine periodicity without localisation, notes §7.1), got {n_j}"
        )
    if any(k <= 0 for k in k_j):
        raise ValueError(f"every multiplicity k_j must be positive, got {k_j}")


class HarmonicTransport(nn.Module, TransportOperator):
    # Explicit annotations, not left to nn.Module's generic __getattr__ fallback (typed
    # `Tensor | Module`): `omega` is assigned via one of two branches below (`nn.Parameter` or
    # `register_buffer`, neither of which gives mypy a static attribute type on its own), and
    # `k_j` is always a buffer. Declaring both here is the standard way to make an
    # `nn.Module`'s dynamically-registered tensor attributes typecheck as plain `Tensor`.
    omega: torch.Tensor
    k_j: torch.Tensor

    def __init__(self, k0: int, n_j: list[int], k_j: list[int], *, learnable_omega: bool) -> None:
        super().__init__()
        _validate_spectrum(k0, n_j, k_j)
        self.k0 = k0
        self.n_j = list(n_j)
        self.learnable_omega = learnable_omega
        omega0 = torch.tensor(n_j, dtype=torch.float32)
        self.register_buffer("k_j", torch.tensor(k_j, dtype=torch.int64))
        if learnable_omega:
            self.omega = nn.Parameter(omega0.clone())
        else:
            self.register_buffer("omega", omega0.clone())

    def reset_parameters_deterministic(self, gen: torch.Generator) -> None:
        """No-op: `omega` is initialised at construction to the declared integer frequencies,
        deterministically and independent of `gen` -- mirrors
        `winder.jepa.predictor.RelativePositionBias`'s own zero-init hook, satisfying
        `winder.determinism.init_parameters`'s closed-vocabulary check for a raw `nn.Parameter`
        no standard layer type covers. There is nothing here for a generator draw to do."""
        return

    @property
    def dimension(self) -> int:
        """K = k0 + 2*sum(k_j): the width of latent this operator acts on."""
        return self.k0 + 2 * int(self.k_j.sum())

    def _omega_per_plane(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.repeat_interleave(self.omega, self.k_j).to(dtype=dtype, device=device)

    def transport(self, z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """z: (..., K), delta: (...,) matching z's leading dims -> R_delta @ z, (..., K).

        `K` must equal `self.dimension`. `delta` broadcasts elementwise over every plane (not
        matrix-multiplied): plane j's angle is `delta * omega_j`, applied via the closed-form 2x2
        rotation directly on that plane's (x, y) pair.
        """
        if z.shape[-1] != self.dimension:
            raise ValueError(f"z's last dim {z.shape[-1]} != operator dimension {self.dimension}")
        if tuple(delta.shape) != tuple(z.shape[:-1]):
            raise ValueError(
                f"delta shape {tuple(delta.shape)} must equal z's leading dims "
                f"{tuple(z.shape[:-1])}"
            )
        z0 = z[..., : self.k0]
        zh = z[..., self.k0 :]
        m = zh.shape[-1] // 2
        planes = zh.reshape(*zh.shape[:-1], m, 2)
        omega_plane = self._omega_per_plane(dtype=z.dtype, device=z.device)
        angle = delta.unsqueeze(-1) * omega_plane  # (..., m)
        cos, sin = angle.cos(), angle.sin()
        x, y = planes[..., 0], planes[..., 1]
        rotated = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1).reshape(*zh.shape)
        return torch.cat([z0, rotated], dim=-1)

    def closure_residual(self) -> torch.Tensor:
        """||R_{2*pi} - I||_F, from the closed form `sqrt(8 * sum_j k_j * sin^2(pi * omega_j))`
        -- derived directly from the parameters at Delta = 2*pi exactly, never from an argmin
        over a sampled Delta grid (that statistic measures a different, wrong quantity -- see
        `winder/transport/loss.py`'s module docstring for the failure this avoids). Each plane's
        own `||R(2*pi*omega_j) - I||_F^2 == 8*sin^2(pi*omega_j)` by direct expansion of the 2x2
        rotation-minus-identity Frobenius norm; blocks are orthogonal, so the total is the sum
        over planes (k_j copies each), and the invariant block contributes exactly 0."""
        k_j = self.k_j.to(self.omega.dtype)
        return torch.sqrt(8.0 * (k_j * torch.sin(math.pi * self.omega).square()).sum())

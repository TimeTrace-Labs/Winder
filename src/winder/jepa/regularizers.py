"""SIGReg: `Regularizer`'s MVP implementation, and `NoRegularizer` for the code-path-free
ablation arm.

Numerics follow the design spec's own reference pseudocode (Sec 11.4) verbatim: `M` random unit
directions project the flattened embeddings, an empirical characteristic function is compared
against the standard-Gaussian target `exp(-t**2/2)` at `J` trapezoid-quadrature knots on `[0,
t_max]` (not LeJEPA's own published `[-5, 5]`), weighted by the target CF itself, summed and
scaled by `N`, averaged over directions. Real (cos/sin) arithmetic, not complex `exp`, matching
the reference pseudocode exactly and avoiding complex-autograd/mypy friction.

**Not comparable across repos.** The sibling `ttl-phase`'s own SIGReg implementation uses yet a
*third* quadrature grid (`linspace(0.2, 4.0, 17)`). SIGReg magnitudes here, in `ttl-phase`, and in
LeJEPA's own paper are not on the same scale -- do not compare raw values across any of them.

Measured on this grid (`N = 64*250 = 16000`, `K = 256`, `M = 256`): isotropic `randn` (the
target) lands near 1.0-1.1; a fully collapsed (constant) input lands near 6000+; a rank-2
dimensional collapse lands near 2000+. The floor is **not zero** -- Theorem 6's `O(1/N)` bias is
multiplied by the statistic's own leading `N` factor, leaving an `N`-independent residual. A
canary asserting "near zero" would fail against a correct implementation; assert
order-of-magnitude separation instead.

The `[0, 3]`-vs-LeJEPA's-published-`[-5, 5]` grid deviation above was re-checked directly against
the paper (arXiv:2511.08544, Algorithm 1) via a 3-reviewer panel (biostats / research-engineering /
adversarial-critic lenses, majority vote) rather than left as an unexamined difference: 2-of-3
upheld the current grid, since the paper's own weighting `exp(-t**2/2)` puts negligible mass past
`t=3` (~5e-3 of the dropped tail against O(1)-O(1e4) statistic values), holding `n_knots=17` fixed
while widening `t_max` only coarsens quadrature spacing where the statistic's curvature is
actually concentrated (near the origin), and absolute SIGReg scale is -- per the disclaimer above
-- already non-comparable across implementations, so paper-exact truncation range buys no
downstream benefit that empirical `lambda_sig` calibration doesn't already handle.

Stronger evidence for the same verdict surfaced afterward, directly from the paper's own ablation
(Table 1a, ImageNet-1K/ViT-Large linear-probe accuracy): integration domain `[-3, 3]` scores
73.71-75.02% across quadrature-point counts, statistically on par with their own recommended
`[-5, 5]` (73.95-74.16%) -- both clearly ahead of `[-1, 1]` (71.82-72.88%). This grid's `t_max=3.0`
sits inside the paper's own validated-fine range, not outside it; the panel's verdict wasn't just
argued from first principles, it matches what LeJEPA's own authors measured.

Directions are drawn fresh from the passed `generator` on every call and immediately discarded
(no gradient, per `Regularizer`'s contract) -- resampling every call is load-bearing, not
incidental: LeJEPA Sec 4.3 notes that resampling even `M=16` directions per step outperforms a
fixed set of thousands, because the cumulative number of directions seen grows across training
steps even though any single call only tests a few.

`__call__` accepts `(N, K)` (as above) or `(T, N, K)`. A 3-D call is the per-timestep reduction
architecture-primer.html §"The reduction is being corrected" fixes on: the identical statistic is
computed independently for each of the `T` leading slices (`N` = that slice's own row count, e.g.
batch size, not `N*T` pooled), then averaged over `T`. Directions are drawn ONCE per call and
shared across every slice -- the per-*call* resampling above stays load-bearing; only the
within-call sharing across `T` is new (the shared-directions default (architecture-primer.html §9)).
`(N, K)` is exactly the `T=1` special case of this and returns bit-identical results to the
pre-widening implementation. `train.py`'s call site passes `(T, B, K)` (transposed from the natural
`(B, T, K)` token layout) rather than flattening `B` and `T` together, so `N` = batch size, matching
LeJEPA/LeWorldModel rather than pooling every token of every record into one statistic.

Runs at fp32 *or better*, never worse: this MVP's training protocol runs the rest of the network
under bf16 autocast, and SIGReg promotes anything below fp32 up to it for numerical stability --
but a caller already passing fp64 (e.g. `torch.autograd.gradcheck`) is left at fp64, not silently
downcast.
"""

from dataclasses import dataclass

import torch

from winder.jepa.base import Regularizer

__all__ = ["SigRegConfig", "SigReg", "NoRegularizerConfig", "NoRegularizer"]


@dataclass
class SigRegConfig:
    n_directions: int = 256
    n_knots: int = 17
    t_max: float = 3.0
    chunk: int = 32


class SigReg(Regularizer):
    def __init__(self, config: SigRegConfig) -> None:
        if config.n_directions <= 0:
            raise ValueError(f"n_directions must be positive, got {config.n_directions}")
        if config.n_knots < 2:
            raise ValueError(f"n_knots must be >= 2, got {config.n_knots}")
        if config.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {config.t_max}")
        self.config = config

    def __call__(self, z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        if z.ndim == 2:
            z3 = z.unsqueeze(0)  # (1, N, K): the pooled/single-statistic case, T=1
        elif z.ndim == 3:
            z3 = z
        else:
            raise ValueError(f"z must be 2-D (N, K) or 3-D (T, N, K), got shape {tuple(z.shape)}")
        # A floor, not a forced downcast: promote anything below fp32 (e.g. bf16 from an
        # autocast context) up to fp32, but leave fp32/fp64 input as-is -- float64 callers
        # (e.g. torch.autograd.gradcheck) must not be silently downcast to fp32.
        compute_dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
        Z = z3.to(compute_dtype)
        t_steps, n, d = Z.shape
        knots = self.config.n_knots
        t_max = self.config.t_max
        n_directions = self.config.n_directions
        chunk = self.config.chunk if self.config.chunk > 0 else n_directions

        t = torch.linspace(0.0, t_max, knots, dtype=compute_dtype, device=Z.device)
        dt = t_max / (knots - 1)
        trap = torch.full_like(t, 2.0 * dt)
        trap[0] = dt
        trap[-1] = dt
        phi0 = torch.exp(-0.5 * t.square())
        quad_weight = trap * phi0

        # Drawn on CPU (matching `generator`'s device) then moved -- a CPU-seeded generator
        # cannot drive a CUDA tensor's RNG op directly. Same pattern as SeededDropout. One
        # shared draw for the whole call, including every T-slice of a 3-D input (module
        # docstring's "directions shared across timesteps" default).
        directions = torch.randn(d, n_directions, generator=generator, dtype=compute_dtype)
        directions = directions.to(Z.device)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

        per_timestep_total = Z.new_zeros(t_steps)
        for start in range(0, n_directions, chunk):
            slice_dirs = directions[:, start : start + chunk]
            projected = Z @ slice_dirs  # (T, N, chunk) -- batched matmul, one per T-slice
            phase = projected.unsqueeze(-1) * t  # (T, N, chunk, J)
            real = phase.cos().mean(dim=1)  # (T, chunk, J) -- mean over N, per T-slice
            imag = phase.sin().mean(dim=1)
            err = (real - phi0).square() + imag.square()
            statistic = n * (err @ quad_weight)  # (T, chunk)
            per_timestep_total = per_timestep_total + statistic.sum(dim=1)  # (T,)
        per_timestep = per_timestep_total / n_directions  # (T,): one statistic per T-slice
        return per_timestep.mean()  # average over T -- identity when T=1


@dataclass
class NoRegularizerConfig:
    """No-op: the ablation arm with SIGReg structurally absent from the computation graph.

    Distinct from `lambda_sig=0` (the spec's collapse-control ablation, which keeps `sigreg`
    registered and its raw value logged, just weighted out of the total loss) -- `NoRegularizer`
    is for a caller that wants no regularizer code path at all, not merely a zero-weighted one.
    """


class NoRegularizer(Regularizer):
    def __init__(self, config: NoRegularizerConfig) -> None:
        self.config = config

    def __call__(self, z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        if z.ndim not in (2, 3):
            raise ValueError(f"z must be 2-D (N, K) or 3-D (T, N, K), got shape {tuple(z.shape)}")
        return z.new_zeros(())

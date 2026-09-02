"""Embedding-distribution diagnostics: stable rank, effective rank, and a batch-accumulating
running moment estimator.

Loss curves alone cannot reveal dimensional collapse: with no target encoder or stop-gradient
(see `winder.jepa.model`'s module docstring), a shrinking prediction loss and a flat SIGReg value
are consistent with a slow rank slide. These diagnostics are the canary for that -- logged
alongside the two losses every training step, per the design spec (Sec 14), never used to gate
anything.

Report on *two* different tensors, for the same reason SIGReg's attach site and the eval surface
differ (see `winder.jepa.model`): the projected tokens (what SIGReg actually constrains) and the
mean-pooled encoder output (what the probe actually reads) can diverge, and inspecting only one
would miss that. The pooled-representation diagnostic is batch-size-limited if computed per
batch -- `B` samples in a `K`-dimensional space cap its rank at `B-1` regardless of collapse
(centering removes one degree of freedom); `RunningMoments` exists so that diagnostic can be
accumulated across many batches instead of read off a single one.
"""

import torch

__all__ = ["covariance", "stable_rank", "effective_rank", "spectrum_report", "RunningMoments"]


def covariance(z: torch.Tensor, *, center: bool = True) -> torch.Tensor:
    """`(N, K) -> (K, K)` population covariance (divides by `N`, not `N-1`, matching the design
    spec's own `Sigma = (1/N)(Z-Zbar)^T(Z-Zbar)`), computed in float64 regardless of `z`'s dtype
    -- `effective_rank`'s eigendecomposition is numerically sensitive to the surrounding
    compute's precision (e.g. bf16 autocast) in a way the raw embeddings are not."""
    if z.ndim != 2:
        raise ValueError(f"z must be 2-D (N, K), got shape {tuple(z.shape)}")
    x = z.detach().to(torch.float64)
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    return (x.T @ x) / n


def stable_rank(cov: torch.Tensor) -> float:
    """`tr(Sigma)^2 / (tr(Sigma^2) + eps)` -- no eigendecomposition needed."""
    trace = torch.trace(cov)
    trace_sq = torch.trace(cov @ cov)
    return float((trace * trace) / (trace_sq + 1e-12))


def effective_rank(cov: torch.Tensor) -> float:
    """`exp(-sum p_i log(p_i + eps))`, `p_i = lambda_i / sum(lambda)`, `lambda` the eigenvalues
    of a symmetric PSD covariance matrix."""
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum()
    if float(total) <= 0.0:
        return 0.0
    p = eigvals / total
    entropy = -(p * (p + 1e-12).log()).sum()
    return float(entropy.exp())


def spectrum_report(z: torch.Tensor) -> dict[str, float]:
    """Both diagnostics plus supporting numbers, over a single `(N, K)` batch. SIGReg targets
    `N(0, I)`, so `mean_norm` is reported separately from the *centred* covariance spectrum --
    they answer different questions (is it centred? is it isotropic?)."""
    cov = covariance(z)
    n, k = z.shape
    return {
        "n": float(n),
        "k": float(k),
        "mean_norm": float(z.detach().to(torch.float64).mean(dim=0).norm()),
        "stable_rank": stable_rank(cov),
        "effective_rank": effective_rank(cov),
    }


class RunningMoments:
    """Accumulates mean and (population) covariance across many batches via Chan et al.'s
    parallel/batched generalization of Welford's algorithm, so a pooled-representation rank
    diagnostic is not capped by a single batch's size."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self._m2 = torch.zeros(dim, dim, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be (batch, {self.dim}), got {tuple(x.shape)}")
        x64 = x.detach().to(torch.float64)
        n_b = x64.shape[0]
        if n_b == 0:
            return
        mean_b = x64.mean(dim=0)
        centered = x64 - mean_b
        m2_b = centered.T @ centered

        if self.n == 0:
            self.mean, self._m2, self.n = mean_b, m2_b, n_b
            return

        n_a = self.n
        delta = mean_b - self.mean
        n = n_a + n_b
        self.mean = self.mean + delta * (n_b / n)
        self._m2 = self._m2 + m2_b + torch.outer(delta, delta) * (n_a * n_b / n)
        self.n = n

    @property
    def covariance(self) -> torch.Tensor:
        if self.n < 2:
            raise ValueError(f"need at least 2 samples to estimate a covariance, got n={self.n}")
        return self._m2 / self.n

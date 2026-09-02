"""E2-06: three ways to collapse a record's RF-valid token sequence `(T, K)` into one embedding
`(K,)`, compared on the SAME frozen checkpoint with the probe otherwise held fixed
(`winder.eval.probe.fit_linear_probe`, unchanged).

`mean_pool` is `embed_records`'s own existing reduction (`tokens.mean(dim=1)`), reproduced here
rather than imported so this module has no dependency on `probe.py`'s batching/device-handling --
`tests/test_eval_pooling.py::test_mean_pool_matches_embed_records_reduction` is what pins the two
to the same number, not a shared code path.

`last_token_pool` is a LOCAL summary, not a whole-record one: under a causal encoder, token
`T-1`'s own receptive field is exactly the record's final raw-sample window and it never reads
anything earlier. Under the retired `ResidualCnnEncoder` that window was 113 samples (CM-01,
=~1.13s at 100Hz); under `PatchEncoder` (architecture-primer.html §5-6) it is exactly one
`patch_width`-sample patch (80ms), narrower still. Any AUROC delta against `mean_pool` on a
superclass whose evidence lives outside that window (e.g. a diffuse repolarization pattern) partly
reflects that horizon difference, not a "pooling mechanism" difference in isolation -- report both
together, not the delta alone.

`causal_attention_pool` is a FIXED (no learned parameters, no fitting) dot-product attention pool:
the record's own last valid token stands in as the query (under a causal encoder it is the
position with the largest receptive field, i.e. the most temporally-informed single token, the
natural query for "what does the rest of the causal record path point back to"), attending over
every valid token as keys/values, `softmax(q . k_i / sqrt(K))`. Because the query is drawn from
the same set it attends over, this pool can numerically degenerate toward `last_token_pool` if the
softmax saturates onto the query token itself (self-similarity is always the largest available
dot product before any competing signal) -- `attention_saturation` exists to measure exactly that
risk on real data, rather than assume the three pooling variants are actually distinct.

`masked_mean_pool`/`demodulated_pool` (M6, the phase-equivariance MVP's readout matrix) are a
second pair, restricted to tokens with a defined cardiac phase (`theta` finite) rather than every
token unconditionally: `masked_mean_pool` is the fair comparison point for `demodulated_pool` --
same token SET, different combination rule -- which is why it is not simply `mean_pool` with a
mask multiplied in (that would still divide by the WRONG count). `demodulated_pool` is
Proposition 4.2 of notes/internal/phase_equivariance_notes_v13.pdf: undo each valid token's own
known rotation (`operator.transport(tokens, -theta)`) before averaging, so harmonics the operator
was trained to move add IN PHASE instead of cancelling -- exactly what `mean_pool`/
`masked_mean_pool` erase by Proposition 4.1. Both return an all-NaN row for a record with zero
valid tokens; callers must drop such rows before fitting/scoring a probe, the same as any other
missing-feature row.
"""

import math
from typing import Literal

import torch

from winder.operators.base import TransportOperator

__all__ = [
    "PoolMethod",
    "mean_pool",
    "last_token_pool",
    "causal_attention_pool",
    "attention_saturation",
    "masked_mean_pool",
    "demodulated_pool",
]

PoolMethod = Literal["mean", "last_token", "causal_attention"]


def mean_pool(tokens: torch.Tensor) -> torch.Tensor:
    """`(N, T, K) -> (N, K)`: unweighted mean over the token axis."""
    return tokens.mean(dim=1)


def last_token_pool(tokens: torch.Tensor) -> torch.Tensor:
    """`(N, T, K) -> (N, K)`: the final token only (see module docstring for why this is a
    LOCAL, not whole-record, summary under a causal encoder)."""
    return tokens[:, -1, :]


def causal_attention_pool(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(N, T, K) -> (pooled (N, K), weights (N, T))`, `weights` summing to 1 along `T`.

    Query is `tokens[:, -1, :]` (see module docstring); no learned parameters, so this pool
    requires no fitting and is exactly as "frozen" as `mean_pool`/`last_token_pool`.
    """
    query = tokens[:, -1, :]
    k = tokens.shape[-1]
    logits = torch.einsum("ntk,nk->nt", tokens, query) / math.sqrt(k)
    weights = torch.softmax(logits, dim=-1)
    pooled = torch.einsum("nt,ntk->nk", weights, tokens)
    return pooled, weights


def attention_saturation(weights: torch.Tensor) -> dict[str, float]:
    """Diagnostics deciding whether `causal_attention_pool` is a genuinely distinct third
    variant or has numerically collapsed onto `last_token_pool` (module docstring's degeneracy
    risk): `mean_max_weight` (mean, over records, of each record's own largest attention weight
    -- near 1.0 means every record's pool is dominated by a single token) and
    `mean_effective_attended_tokens` (mean, over records, of `exp(entropy(weights))` -- the
    number of tokens the softmax spreads mass over in an "effective count" sense; near 1.0 is the
    same collapse, near `T` is close to uniform/mean-pool-like).

    `weights` is `(N, T)`, one softmax distribution per record (as returned by
    `causal_attention_pool`).
    """
    max_weight = weights.max(dim=-1).values
    entropy = -(weights * (weights + 1e-12).log()).sum(dim=-1)
    effective = entropy.exp()
    return {
        "mean_max_weight": float(max_weight.mean()),
        "mean_effective_attended_tokens": float(effective.mean()),
    }


def _valid_masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(N, T, K), (N, T) bool -> (N, K): mean over valid positions only; an all-NaN row where a
    record has none. Shared arithmetic core of masked_mean_pool/demodulated_pool below."""
    masked = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = masked.sum(dim=1) / counts
    has_any = valid.any(dim=1)
    return torch.where(has_any.unsqueeze(-1), pooled, torch.full_like(pooled, float("nan")))


def masked_mean_pool(tokens: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """`(N, T, K)`, `(N, T)` -> `(N, K)`: `mean_pool` restricted to tokens with a defined theta
    (finite, not NaN) -- the fair comparison point for `demodulated_pool` (module docstring)."""
    if theta.shape != tokens.shape[:2]:
        raise ValueError(
            f"theta shape {tuple(theta.shape)} must equal tokens' leading dims "
            f"{tuple(tokens.shape[:2])}"
        )
    return _valid_masked_mean(tokens, torch.isfinite(theta))


def demodulated_pool(
    tokens: torch.Tensor, theta: torch.Tensor, operator: TransportOperator
) -> torch.Tensor:
    """`(N, T, K)`, `(N, T)` -> `(N, K)`: Proposition 4.2's coherent estimator (module
    docstring) -- `R_{-theta_t}` applied to every valid token, then averaged."""
    if theta.shape != tokens.shape[:2]:
        raise ValueError(
            f"theta shape {tuple(theta.shape)} must equal tokens' leading dims "
            f"{tuple(tokens.shape[:2])}"
        )
    valid = torch.isfinite(theta)
    theta_filled = torch.where(valid, theta, torch.zeros_like(theta))
    demodulated = operator.transport(tokens, -theta_filled)
    return _valid_masked_mean(demodulated, valid)

"""The MVP `Predictor`: a causal pre-LayerNorm Transformer over the natural token order.

No context/target concatenation and no separate position-index bookkeeping (an earlier design
pass built exactly that; the design spec's masking scheme makes it unnecessary): masked positions
are replaced *in place*, at their own natural sequence position, by a single learned
`mask_token`. Every token -- masked or not -- keeps the position it already had, so relative bias
is simply `clip(i - j, -max_distance, max_distance)` over the fixed `0..n_tokens-1` order. There
is no absolute positional embedding anywhere in this module; position enters only through that
relative bias table.

Causal (CM-02): a lower-triangular mask is added to the relative bias before every call to
`F.scaled_dot_product_attention`, so position `i`'s output depends only on positions `<= i` --
perturbing a later token's input never changes an earlier token's prediction, the exact inverse of
`tests/test_jepa_predictor.py`'s old `test_bidirectional_not_causal`. `rel_pos_max_distance`
defaults to `n_tokens - 1` (249), not 64: `winder.jepa.masking.CausalBlockMaskSampler` samples a
gap `g` up to 74 tokens, so a target token can sit up to 75 tokens after the cutoff it is
predicted from -- a table clipped at 64 would collapse every context token beyond that distance
onto the same bias entry, making the horizon itself unrepresentable (CM-05).

Dropout uses `winder.jepa.seeded_dropout.SeededDropout`, not `nn.Dropout` -- see that module's
docstring for why. `F.scaled_dot_product_attention`'s own internal attention-weight dropout
(`dropout_p`) is left at its default (0): the design spec's single `dropout` field applies at the
two residual connections only (`u' = u + Dropout[MHA(LN(u))]`, `u'' = u' + Dropout[MLP(LN(u'))]`),
not to attention weights themselves.
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from winder.jepa.base import Predictor
from winder.jepa.seeded_dropout import SeededDropout

__all__ = ["RelativePositionBias", "TransformerPredictorConfig", "TransformerPredictor"]


class RelativePositionBias(nn.Module):
    """Learned per-head bias indexed by `clip(i - j, -max_distance, max_distance)`:
    `2 * max_distance + 1` entries per head, added directly to attention logits as an additive
    mask. This is the *only* way position enters this predictor."""

    def __init__(self, n_heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.table = nn.Parameter(torch.zeros(n_heads, 2 * max_distance + 1))

    def reset_parameters_deterministic(self, gen: torch.Generator) -> None:
        """Zero-init, matching construction -- see `winder.determinism.init_parameters`, whose
        closed-vocabulary check this hook satisfies for a raw `nn.Parameter` no standard layer
        type covers."""
        nn.init.zeros_(self.table)

    def forward(self, n_tokens: int, *, device: torch.device) -> torch.Tensor:
        positions = torch.arange(n_tokens, device=device)
        rel = positions[:, None] - positions[None, :]  # (S, S), i - j
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.table[:, rel]  # (n_heads, S, S)
        return bias.unsqueeze(0)  # (1, n_heads, S, S) -- broadcasts over the batch dim in SDPA


class TransformerBlock(nn.Module):
    """One pre-LayerNorm block: `u + Dropout[Attn(LN(u))]`, then `u + Dropout[MLP(LN(u))]`."""

    def __init__(self, config: "TransformerPredictorConfig", *, dropout_seed: int) -> None:
        super().__init__()
        width = config.width
        if width % config.n_heads != 0:
            raise ValueError(f"width={width} not divisible by n_heads={config.n_heads}")
        self.n_heads = config.n_heads
        self.head_width = width // config.n_heads

        self.norm1 = nn.LayerNorm(width, eps=config.layernorm_eps)
        self.qkv = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)
        self.attn_dropout = SeededDropout(config.dropout, seed=dropout_seed)

        self.norm2 = nn.LayerNorm(width, eps=config.layernorm_eps)
        self.mlp_fc1 = nn.Linear(width, config.feedforward_width)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(config.feedforward_width, width)
        self.mlp_dropout = SeededDropout(config.dropout, seed=dropout_seed + 1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, s, w = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, s, 3, self.n_heads, self.head_width).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, n_heads, s, head_width)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(b, s, w)
        x = x + self.attn_dropout(self.out_proj(attn))
        h = self.norm2(x)
        mlp = self.mlp_fc2(self.mlp_act(self.mlp_fc1(h)))
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, x + self.mlp_dropout(mlp))


@dataclass
class TransformerPredictorConfig:
    width: int = 256
    n_layers: int = 2
    n_heads: int = 4
    feedforward_width: int = 1024
    dropout: float = 0.1
    rel_pos_max_distance: int = 249
    layernorm_eps: float = 1e-5


class TransformerPredictor(nn.Module, Predictor):
    def __init__(self, config: TransformerPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.rel_bias = RelativePositionBias(config.n_heads, config.rel_pos_max_distance)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, dropout_seed=1000 * i) for i in range(config.n_layers)]
        )
        self.out_proj = nn.Linear(config.width, config.width)

    def reset_parameters_deterministic(self, gen: torch.Generator) -> None:
        """Resets only `mask_token` (this module's own direct parameter) -- see
        `winder.determinism.init_parameters`. Child modules (`rel_bias`, `blocks`, `out_proj`)
        are visited separately by that function's own recursive walk and are not affected by
        this hook."""
        nn.init.zeros_(self.mask_token)

    @property
    def width(self) -> int:
        return self.config.width

    def forward(self, z_ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if z_ctx.ndim != 3 or z_ctx.shape[-1] != self.config.width:
            raise ValueError(f"z_ctx must be (B, S, {self.config.width}), got {tuple(z_ctx.shape)}")
        if tuple(mask.shape) != tuple(z_ctx.shape[:2]):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != z_ctx's leading dims {tuple(z_ctx.shape[:2])}"
            )
        b, s, w = z_ctx.shape
        mask_token = self.mask_token.expand(b, s, w)
        # `mask` comes from the mask sampler, which has no device concept of its own (CPU
        # tensors throughout) -- move it onto z_ctx's device before it meets mask_token/z_ctx in
        # torch.where, matching this function's own device=q.device convention below.
        q = torch.where(mask.unsqueeze(-1).to(z_ctx.device), mask_token, z_ctx)
        rel_bias = self.rel_bias.forward(s, device=q.device)
        # Causal (CM-02): built once per forward, not per block. `causal[i, j]` is True iff
        # `j <= i`; masked-out (future) positions get an additive -inf so softmax zeroes them.
        causal = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
        attn_mask = rel_bias.masked_fill(~causal, float("-inf"))
        for block in self.blocks:
            q = block(q, attn_mask)
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.out_proj(q))

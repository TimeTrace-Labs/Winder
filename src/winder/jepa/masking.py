"""One-step-ahead token masking: `MaskSampler`'s only MVP implementation, forecasting rather than
infilling.

architecture-primer.html §5-6: the gap concept is gone. A gap-and-length sampler existed to
guarantee a target token's receptive field shared no raw sample with the context prefix's own last
visible token (`winder.jepa.leakage.min_disjoint_gap`, under the overlapping `ResidualCnnEncoder`)
-- under the non-overlapping `PatchEncoder` there is no receptive field wider than one patch to
protect against, so that guarantee holds trivially for ANY target position, gap or no gap. What
remains is exactly the MVP's own "one-step-ahead latent prediction" piece: draw a context cutoff `c`
per record, and the target is always the single token immediately after it (`s = c + 1`). The
predictor is only ever asked to forecast the one token strictly after the cutoff it was given.

Masking is expressed in *token* space, as two boolean masks over `0..n_tokens-1` (`context`,
`target`), not in raw waveform space: `winder.jepa.train`'s single causal encoder pass makes
zeroing raw samples before encoding unnecessary (see that module's docstring) -- a context cutoff
is enforced by which tokens the predictor is allowed to read, not by what the encoder saw.
"""

from dataclasses import dataclass

import torch

from winder.jepa.base import MaskSampler

__all__ = ["CausalMaskPlan", "CausalBlockMaskSamplerConfig", "CausalBlockMaskSampler"]


@dataclass(frozen=True)
class CausalMaskPlan:
    """One batch's sampled context cutoffs, in both mask and per-record scalar form.

    `context`/`target` are `(B, n_tokens)` bool: `context[b, : cutoff[b]+1]` is True and nothing
    else; `target[b, cutoff[b]+1]` is True and nothing else -- exactly one target token per
    record, immediately after its own cutoff (one-step-ahead, architecture-primer.html §5-6).
    `cutoff` is `(B,)` int64, carried alongside the masks so a caller can log its distribution
    without recovering it from the boolean masks.
    """

    context: torch.Tensor
    target: torch.Tensor
    cutoff: torch.Tensor


@dataclass
class CausalBlockMaskSamplerConfig:
    c_min: int = 0


class CausalBlockMaskSampler(MaskSampler):
    """Per record: `c ~ Uniform[c_min, n_tokens - 2]`, target `s = c + 1`. `c`'s upper bound
    (`n_tokens - 2`) is fixed, not per-record (unlike the retired gap/length sampler's own
    conditional upper bound) -- there is nothing left that a record's own draw could make
    infeasible for another record at the same `n_tokens`, so one shared `torch.randint` call
    draws every record's cutoff at once.

    Class/registry-tag name kept as `CausalBlockMaskSampler`/`"causal_block"` despite the target
    no longer being a multi-token block (the sampler simplification cuts it to length 1) -- old
    checkpoints' saved `config.yaml` store the registry tag as a string, and renaming the class
    buys nothing that updating this docstring doesn't already cover.
    """

    def __init__(self, config: CausalBlockMaskSamplerConfig) -> None:
        if config.c_min < 0:
            raise ValueError(f"c_min must be >= 0, got {config.c_min}")
        self.config = config

    def __call__(
        self, batch_size: int, n_tokens: int, *, generator: torch.Generator
    ) -> CausalMaskPlan:
        cfg = self.config
        if n_tokens < cfg.c_min + 2:
            raise ValueError(
                f"n_tokens={n_tokens} cannot fit c_min={cfg.c_min} + 1 target token + 1"
            )
        cutoff = torch.randint(cfg.c_min, n_tokens - 1, (batch_size,), generator=generator)
        positions = torch.arange(n_tokens).unsqueeze(0)  # (1, n_tokens)
        context = positions <= cutoff.unsqueeze(1)  # (B, n_tokens)
        target = positions == (cutoff.unsqueeze(1) + 1)  # (B, n_tokens), exactly one True/row
        return CausalMaskPlan(context=context, target=target, cutoff=cutoff)

"""The MVP `Encoder`: a local 1D residual CNN, exact layer table per the design spec, causal.

Total stride is exactly 4 (a stride-2 stem conv, then a stride-2 average pool): a (B, 12, 1000)
waveform produces exactly (B, 250, 256) tokens at a 40 ms cadence, asserted by
`winder.jepa.model.assemble_jepa`'s dummy forward pass rather than trusted to this arithmetic.

Causal (CM-01): every conv is left-padded only (`F.pad(x, (k-1, 0))`, `padding=0` on the `nn.Conv1d`
itself), never `nn.Conv1d`'s own symmetric `padding=k//2`. `nn.AvgPool1d(2, 2)` is left untouched --
it already reads no sample ahead of its own two-sample window, and padding it would move the
receptive field off the value (113 samples) `notes/methodology/stage-00-contracts.md` (CON-03)
fixes by number. Consequence: token `j` reads raw samples `[4j-110, 4j+2]`
(`winder.jepa.leakage.token_window`), not the `[4j-56, 4j+56]`-ish symmetric window a naive
`padding=k//2` stack would give -- perturbing any sample after `4j+2` never changes token `j`
(`tests/test_stage0_contracts.py::test_cm01_encoder_tokens_do_not_read_future_samples`). The `+2`
comes entirely from `AvgPool1d` being the one layer whose window is not left-anchored
(`last_in(i) = 2i+1`, doubled by the stem's own stride 2); it cancels in every disjointness
argument built on top of this encoder (`winder.jepa.leakage.min_disjoint_gap`), so it never needs
to be threaded through calling code.

No terminal LayerNorm after the token projection: SIGReg (the `Regularizer`) needs to control the
projected distribution's own scale and moments, and a terminal LayerNorm sitting immediately
upstream of that would pre-impose a constraint SIGReg is supposed to be learning to satisfy.
LayerNorm is used freely *inside* the residual blocks (via `ChannelLayerNorm`, below) -- only the
final output is left unnormalized.

This was reasoned from first principles, then independently cross-checked against LeWorldModel
(Maes, Le Lidec, et al., arXiv:2603.19312, Sec 3.1): its ViT encoder's standard terminal LayerNorm
was found, by ablation, to actively prevent SIGReg from being optimized effectively -- their fix is
an added BatchNorm projection stage specifically to undo it. This encoder never introduces that
terminal LayerNorm in the first place (a causal CNN, not a ViT), so no such fix is needed here; the
finding is corroborating evidence for this module's existing design, not a change to make.

This encoder is called exactly once per training step, on the unmasked waveform
(`winder.jepa.train`'s module docstring) -- because it is causal, token `j`'s value already depends
on no sample after its own timestamp, so there is no separate "masked context" pass to run: a
context cutoff is enforced by which tokens the predictor is allowed to read
(`winder.jepa.masking.CausalBlockMaskSampler`), not by zeroing raw samples before encoding.
"""

import math
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from winder.jepa.base import Encoder
from winder.jepa.predictor import RelativePositionBias, TransformerBlock, TransformerPredictorConfig

__all__ = [
    "ChannelLayerNorm",
    "ResidualBlock1d",
    "ResidualCnnEncoderConfig",
    "ResidualCnnEncoder",
    "PatchEncoderConfig",
    "PatchEncoder",
    "DilatedCausalBlock1d",
    "ConvTrunkEncoderConfig",
    "ConvTrunkEncoder",
    "AttnTrunkEncoderConfig",
    "AttnTrunkEncoder",
    "HybridTrunkEncoderConfig",
    "HybridTrunkEncoder",
    "LeadFactorizedEncoderConfig",
    "LeadFactorizedEncoder",
    "DctStemConvTrunkEncoderConfig",
    "DctStemConvTrunkEncoder",
    "MultiscaleStemConvTrunkEncoderConfig",
    "MultiscaleStemConvTrunkEncoder",
    "WindowedHybridTrunkEncoderConfig",
    "WindowedHybridTrunkEncoder",
    "GdnTrunkEncoderConfig",
    "GdnTrunkEncoder",
]


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a `(B, C, T)` tensor -- never batch, never time.

    Implemented as transpose -> `nn.LayerNorm(C)` -> transpose back, so eps-handling and autograd
    exactly match PyTorch's own LayerNorm rather than a hand-rolled mean/var reduction.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.norm(x.transpose(1, 2)).transpose(1, 2))


class ResidualBlock1d(nn.Module):
    """`x + Conv[GELU(CLN(Conv[GELU(CLN(x))]))]` -- two same-length (stride-1) convolutions,
    each preceded by channel-norm and GELU, added back onto the input."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=0, bias=False)
        self.norm2 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=0, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-only pad (CM-01): a stride-1 same-length conv with no future-reading window.
        left_pad = self.conv1.kernel_size[0] - 1
        v = self.conv1(F.pad(self.act(self.norm1(x)), (left_pad, 0)))
        v = self.conv2(F.pad(self.act(self.norm2(v)), (left_pad, 0)))
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, x + v)


@dataclass
class ResidualCnnEncoderConfig:
    n_leads: int = 12
    stem_width: int = 64
    stem_kernel: int = 15
    stem_stride: int = 2
    stage_widths: list[int] = field(default_factory=lambda: [64, 128])
    blocks_per_stage: list[int] = field(default_factory=lambda: [2, 2])
    block_kernel: int = 5
    latent_width: int = 256


class ResidualCnnEncoder(nn.Module, Encoder):
    def __init__(self, config: ResidualCnnEncoderConfig) -> None:
        super().__init__()
        if len(config.stage_widths) != 2 or len(config.blocks_per_stage) != 2:
            raise ValueError(
                f"this encoder is a fixed two-stage design (stem -> stage1 -> downsample -> "
                f"stage2 -> token projection); got stage_widths={config.stage_widths}, "
                f"blocks_per_stage={config.blocks_per_stage}, both must have length 2"
            )
        self.config = config
        self.stem_conv = nn.Conv1d(
            config.n_leads,
            config.stem_width,
            config.stem_kernel,
            stride=config.stem_stride,
            padding=0,
            bias=False,
        )
        self.stem_norm = ChannelLayerNorm(config.stem_width)
        self.stem_act = nn.GELU()

        stage1_width, stage2_width = config.stage_widths
        n_blocks_1, n_blocks_2 = config.blocks_per_stage
        self.stage1 = nn.ModuleList(
            [ResidualBlock1d(stage1_width, config.block_kernel) for _ in range(n_blocks_1)]
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.width_change = nn.Conv1d(stage1_width, stage2_width, kernel_size=1, bias=False)
        self.stage2 = nn.ModuleList(
            [ResidualBlock1d(stage2_width, config.block_kernel) for _ in range(n_blocks_2)]
        )
        self.token_proj = nn.Conv1d(stage2_width, config.latent_width, kernel_size=1, bias=True)

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != self.config.n_leads:
            raise ValueError(
                f"waveform must be (B, {self.config.n_leads}, n_samples), got "
                f"{tuple(waveform.shape)}"
            )
        stem_left_pad = self.stem_conv.kernel_size[0] - 1
        x = self.stem_act(self.stem_norm(self.stem_conv(F.pad(waveform, (stem_left_pad, 0)))))
        for block in self.stage1:
            x = block(x)
        x = self.downsample(x)
        x = self.width_change(x)
        for block in self.stage2:
            x = block(x)
        x = self.token_proj(x)
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, x.transpose(1, 2))


@dataclass
class PatchEncoderConfig:
    n_leads: int = 12
    patch_width: int = 8  # P samples/patch (80 ms at 100 Hz, CON-02's Δt); non-overlapping
    hidden_width: int = 512
    latent_width: int = 256


class PatchEncoder(nn.Module, Encoder):
    """The replacement encoder: non-overlapping `patch_width`-sample patches, each
    mapped to a token by one shared 2-layer MLP (mirroring `MlpProjectionHead`'s own
    `input -> hidden -> output` shape, applied per patch instead of per token) -- no run-in, no
    receptive field wider than the patch itself, causal by construction: token `j` reads raw
    samples `[j*P, (j+1)*P - 1]` and nothing else, so it depends on no sample after its own patch
    and shares no raw sample with any other token. This removes the "copy solution" degenerate
    attractor of the overlapping `ResidualCnnEncoder` (adjacent tokens there share ~96% of their
    receptive field) and the gap-based mask sampler it required -- there is no run-in for
    `winder.jepa.leakage`'s floor arithmetic to protect against, so none of it applies here.

    Like `ResidualCnnEncoder`/`MlpProjectionHead`, no normalization anywhere in this module: the
    output feeds SIGReg (via the projector), which needs to control this distribution's own scale
    and moments -- a LayerNorm on the output would pre-impose part of what SIGReg is meant to
    learn (see `encoder.py`'s own module docstring, LeWorldModel Sec 3.1).
    """

    def __init__(self, config: PatchEncoderConfig) -> None:
        super().__init__()
        if config.patch_width <= 0:
            raise ValueError(f"patch_width must be positive, got {config.patch_width}")
        self.config = config
        input_width = config.n_leads * config.patch_width
        self.fc1 = nn.Linear(input_width, config.hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_width, config.latent_width)

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != self.config.n_leads:
            raise ValueError(
                f"waveform must be (B, {self.config.n_leads}, n_samples), got "
                f"{tuple(waveform.shape)}"
            )
        b, n_leads, n_samples = waveform.shape
        p = self.config.patch_width
        if n_samples % p != 0:
            raise ValueError(
                f"n_samples={n_samples} must be exactly divisible by patch_width={p} -- "
                f"non-overlapping patches require the record to tile evenly, not a ragged "
                f"final patch"
            )
        n_patches = n_samples // p
        # (B, n_leads, n_samples) -> (B, n_leads, n_patches, p) -> (B, n_patches, n_leads, p) --
        # groups every p CONSECUTIVE raw samples into one patch, in time order: patch j covers
        # samples [j*p, (j+1)*p - 1]. The permute (not a transpose of the flattened result) is
        # what keeps lead and within-patch-time axes together before the final flatten below.
        patches = waveform.reshape(b, n_leads, n_patches, p).permute(0, 2, 1, 3)
        flat = patches.reshape(b, n_patches, n_leads * p)
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.fc2(self.act(self.fc1(flat))))


# --------------------------------------------------------------------------------- SCOUT trunks
#
# Encoder-architecture SCOUT variants (Amendment 12, `artifacts/campaign_x2x2/
# pre_launch_addendum.md`): four registered encoders that share `PatchEncoder`'s own patch stage
# (one 2-layer MLP per non-overlapping 8-sample patch, 96 -> 512 -> 256) and differ only in the
# trunk applied to the resulting (B, 125, 256) token sequence. Token count and latent width never
# change: every variant produces exactly the grid `assemble_jepa` validates against
# `config.n_tokens` (125 tokens x 80 ms -- theta alignment untouched), at K = 256.
#
# Doctrine, binding on every class below (Amendment 12 (B)):
#
# - CM-01 strict causality: token `j` depends on no raw sample with index >= 8*(j+1). The patch
#   stage is causal by construction (patch-local, `PatchEncoder` above); the conv trunks left-pad
#   only (`F.pad(x, (left_pad, 0))`, never `nn.Conv1d`'s symmetric `padding=`), mirroring
#   `ResidualBlock1d`; the attention trunks add the exact lower-triangular causal mask
#   `TransformerPredictor.forward` builds (CM-02's own construction, reused not re-derived).
#   `tests/test_encoder_scout.py` proves this per encoder by future-sample perturbation,
#   mirroring `tests/test_stage0_contracts.py`'s CM-01 test.
# - NO terminal normalization before the projector: SIGReg must control the output distribution's
#   own scale and moments (this module's own docstring, LeWorldModel arXiv:2603.19312 Sec 3.1).
#   Norms appear only *inside* residual blocks (pre-norm), so every trunk ends on a residual add,
#   never a LayerNorm.
# - The Amendment 8(B) derivation gate: a context trunk satisfies the phase-transport prior iff
#   the extra context it encodes is beat-stationary. `conv_trunk`'s 12-token lookback (0.96 s at
#   80 ms/token) stays inside one beat, where the prior binds token-to-token; the attention/hybrid
#   trunks step outside it deliberately -- that regime split is exactly what the scout measures.
#
# These are scout arms, non-claim-bearing, quarantined from every manuscript claim until a full
# pre-registered campaign promotes one (Amendment 12 (E)'s report-first rule).


class DilatedCausalBlock1d(nn.Module):
    """`x + Conv[GELU(CLN(Conv[GELU(CLN(x))]))]` -- `ResidualBlock1d`'s exact shape with one
    addition: a per-block dilation shared by both convolutions. Left-only pad of
    `(kernel_size - 1) * dilation` (CM-01): a dilated stride-1 same-length conv whose window
    reads `dilation * (kernel_size - 1)` positions back and zero positions ahead, so one block's
    lookback is `2 * (kernel_size - 1) * dilation` tokens."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=0, dilation=dilation, bias=False
        )
        self.norm2 = ChannelLayerNorm(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=0, dilation=dilation, bias=False
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-only pad (CM-01), scaled by the dilation: a dilated kernel's leftmost tap sits
        # (k-1)*d positions back, and its rightmost tap must stay on the current position.
        left_pad = (self.conv1.kernel_size[0] - 1) * self.conv1.dilation[0]
        v = self.conv1(F.pad(self.act(self.norm1(x)), (left_pad, 0)))
        v = self.conv2(F.pad(self.act(self.norm2(v)), (left_pad, 0)))
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, x + v)


class _CausalTokenAttention(nn.Module):
    """N pre-LN causal transformer blocks over the token sequence, with one SHARED
    `RelativePositionBias` and the exact lower-triangular additive mask
    `TransformerPredictor.forward` builds (CM-02's construction) -- the one genuinely shared
    component between `AttnTrunkEncoder` and `HybridTrunkEncoder`, factored so the mask/bias
    arithmetic exists once. Reuses `TransformerBlock` itself rather than mirroring it: the block
    reads only `width`/`n_heads`/`feedforward_width`/`dropout`/`layernorm_eps` from its config,
    all supplied here; at the scout's `dropout=0.0` its `SeededDropout`s are exact identities
    that never draw from their generators (that module's own `p == 0.0` early return)."""

    def __init__(
        self,
        n_layers: int,
        *,
        width: int,
        n_heads: int,
        feedforward_width: int,
        dropout: float,
        rel_pos_max_distance: int,
        layernorm_eps: float,
    ) -> None:
        super().__init__()
        block_config = TransformerPredictorConfig(
            width=width,
            n_layers=n_layers,
            n_heads=n_heads,
            feedforward_width=feedforward_width,
            dropout=dropout,
            rel_pos_max_distance=rel_pos_max_distance,
            layernorm_eps=layernorm_eps,
        )
        self.rel_bias = RelativePositionBias(n_heads, rel_pos_max_distance)
        # dropout_seed mirrors TransformerPredictor's own 1000*i convention; inert at the scout's
        # dropout=0.0 (each SeededDropout instance owns an independent generator regardless).
        self.blocks = nn.ModuleList(
            [TransformerBlock(block_config, dropout_seed=1000 * i) for i in range(n_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        s = tokens.shape[1]
        rel_bias = self.rel_bias.forward(s, device=tokens.device)
        # Causal (CM-01 here, via CM-02's exact construction in TransformerPredictor.forward):
        # `causal[i, j]` is True iff `j <= i`; future positions get an additive -inf so softmax
        # zeroes them exactly.
        causal = torch.ones(s, s, dtype=torch.bool, device=tokens.device).tril()
        attn_mask = rel_bias.masked_fill(~causal, float("-inf"))
        for block in self.blocks:
            tokens = block(tokens, attn_mask)
        return tokens


@dataclass
class ConvTrunkEncoderConfig:
    # Patch-stage fields, named identically to PatchEncoderConfig's so
    # scripts/s2_pretrain_jepa.py's --latent-width width-family overrides compose unchanged.
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    # Trunk: one DilatedCausalBlock1d per entry of `dilations`, kernel `kernel_size` (arm C1's
    # defaults: dilations (1, 2) -> lookback 2*(3-1)*(1+2) = 12 tokens = 0.96 s).
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2])


class ConvTrunkEncoder(nn.Module, Encoder):
    """Arm C1: `PatchEncoder`'s patch stage, then causal residual conv blocks on the TOKEN axis.

    Key args: `dilations` (one block per entry, both convs in a block share its dilation),
    `kernel_size`. Usage: `ConvTrunkEncoder(ConvTrunkEncoderConfig())` -> (B, 12, 1000) ->
    (B, 125, 256). No terminal norm (module docstring); output is the last block's residual add.
    """

    def __init__(self, config: ConvTrunkEncoderConfig) -> None:
        super().__init__()
        if not config.dilations:
            raise ValueError("dilations must name at least one block, got an empty list")
        self.config = config
        self.patch = PatchEncoder(
            PatchEncoderConfig(
                n_leads=config.n_leads,
                patch_width=config.patch_width,
                hidden_width=config.hidden_width,
                latent_width=config.latent_width,
            )
        )
        self.blocks = nn.ModuleList(
            [
                DilatedCausalBlock1d(config.latent_width, config.kernel_size, dilation)
                for dilation in config.dilations
            ]
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.patch.forward(waveform)  # (B, n_tokens, K); validates shape itself
        x = tokens.transpose(1, 2)  # (B, K, n_tokens): Conv1d's channel-first layout
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)


@dataclass
class AttnTrunkEncoderConfig:
    # Patch-stage fields (see ConvTrunkEncoderConfig's naming note).
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    # Trunk: n_layers pre-LN causal transformer blocks (arms C2/C3/C7 at 2/4/6). Amendment 12
    # pins rel_pos_max_distance=124 (the 125-token grid's own maximum offset), NOT the
    # predictor's 249 default -- its param-target table appears to have been computed with the
    # 249-entry-per-head table (1,996 params vs this 996), hence the exact 1,000-param gap
    # tests/test_encoder_scout.py documents against those targets.
    n_layers: int = 2
    n_heads: int = 4
    feedforward_width: int = 1024
    dropout: float = 0.0
    rel_pos_max_distance: int = 124
    layernorm_eps: float = 1e-5


class AttnTrunkEncoder(nn.Module, Encoder):
    """Arms C2/C3/C7: `PatchEncoder`'s patch stage, then `n_layers` pre-LN causal transformer
    blocks (width 256, 4 heads, ff 1024, dropout 0.0, one shared rel-pos bias).

    Key args: `n_layers` (2/4/6 for C2/C3/C7). Usage:
    `AttnTrunkEncoder(AttnTrunkEncoderConfig(n_layers=4))` -> (B, 12, 1000) -> (B, 125, 256).
    No terminal norm (module docstring); output is the last block's residual add.
    """

    def __init__(self, config: AttnTrunkEncoderConfig) -> None:
        super().__init__()
        if config.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {config.n_layers}")
        self.config = config
        self.patch = PatchEncoder(
            PatchEncoderConfig(
                n_leads=config.n_leads,
                patch_width=config.patch_width,
                hidden_width=config.hidden_width,
                latent_width=config.latent_width,
            )
        )
        self.attn = _CausalTokenAttention(
            config.n_layers,
            width=config.latent_width,
            n_heads=config.n_heads,
            feedforward_width=config.feedforward_width,
            dropout=config.dropout,
            rel_pos_max_distance=config.rel_pos_max_distance,
            layernorm_eps=config.layernorm_eps,
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.patch.forward(waveform)  # (B, n_tokens, K); validates shape itself
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.attn(tokens))


@dataclass
class HybridTrunkEncoderConfig:
    # Patch-stage fields (see ConvTrunkEncoderConfig's naming note).
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    # Conv half: ConvTrunkEncoderConfig's exact defaults (C1's blocks).
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2])
    # Attention half: AttnTrunkEncoderConfig's exact defaults at n_attn_layers=2.
    n_attn_layers: int = 2
    n_heads: int = 4
    feedforward_width: int = 1024
    dropout: float = 0.0
    rel_pos_max_distance: int = 124
    layernorm_eps: float = 1e-5


class HybridTrunkEncoder(nn.Module, Encoder):
    """Arm C6: `PatchEncoder`'s patch stage, then C1's conv blocks, THEN 2 attention blocks --
    local causal smoothing feeding a global causal mixer, in that order.

    Usage: `HybridTrunkEncoder(HybridTrunkEncoderConfig())` -> (B, 12, 1000) -> (B, 125, 256).
    No terminal norm (module docstring); output is the last attention block's residual add.
    """

    def __init__(self, config: HybridTrunkEncoderConfig) -> None:
        super().__init__()
        if not config.dilations:
            raise ValueError("dilations must name at least one block, got an empty list")
        if config.n_attn_layers < 1:
            raise ValueError(f"n_attn_layers must be >= 1, got {config.n_attn_layers}")
        self.config = config
        self.patch = PatchEncoder(
            PatchEncoderConfig(
                n_leads=config.n_leads,
                patch_width=config.patch_width,
                hidden_width=config.hidden_width,
                latent_width=config.latent_width,
            )
        )
        self.conv_blocks = nn.ModuleList(
            [
                DilatedCausalBlock1d(config.latent_width, config.kernel_size, dilation)
                for dilation in config.dilations
            ]
        )
        self.attn = _CausalTokenAttention(
            config.n_attn_layers,
            width=config.latent_width,
            n_heads=config.n_heads,
            feedforward_width=config.feedforward_width,
            dropout=config.dropout,
            rel_pos_max_distance=config.rel_pos_max_distance,
            layernorm_eps=config.layernorm_eps,
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.patch.forward(waveform)  # (B, n_tokens, K); validates shape itself
        x = tokens.transpose(1, 2)  # (B, K, n_tokens): Conv1d's channel-first layout
        for block in self.conv_blocks:
            x = block(x)
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.attn(x.transpose(1, 2)))


@dataclass
class LeadFactorizedEncoderConfig:
    n_leads: int = 12
    patch_width: int = 8
    lead_width: int = 64  # per-lead feature width: shared MLP patch_width -> lead_width -> width
    latent_width: int = 256


class LeadFactorizedEncoder(nn.Module, Encoder):
    """Arm C4 (no trunk): per-lead patching. Each lead's `patch_width`-sample patch goes through
    ONE shared MLP (`patch_width -> lead_width -> lead_width`, GELU between), a learned per-lead
    embedding (`n_leads x lead_width`) is added to that lead's features, and the `n_leads`
    feature vectors are concatenated (`n_leads * lead_width`) into a single linear mix to
    `latent_width`. Causal by construction, exactly like `PatchEncoder`: token `j` reads raw
    samples `[j*P, (j+1)*P - 1]` and nothing else (patch-local, no trunk).

    Usage: `LeadFactorizedEncoder(LeadFactorizedEncoderConfig())` -> (B, 12, 1000) ->
    (B, 125, 256). No normalization anywhere (module docstring).
    """

    def __init__(self, config: LeadFactorizedEncoderConfig) -> None:
        super().__init__()
        if config.patch_width <= 0:
            raise ValueError(f"patch_width must be positive, got {config.patch_width}")
        self.config = config
        self.lead_fc1 = nn.Linear(config.patch_width, config.lead_width)
        self.act = nn.GELU()
        self.lead_fc2 = nn.Linear(config.lead_width, config.lead_width)
        # Zero-init, matching RelativePositionBias's own raw-parameter convention (learned
        # offsets from zero) -- reset_parameters_deterministic below reproduces it.
        self.lead_embedding = nn.Parameter(torch.zeros(config.n_leads, config.lead_width))
        self.mix = nn.Linear(config.n_leads * config.lead_width, config.latent_width)

    def reset_parameters_deterministic(self, gen: torch.Generator) -> None:
        """Zero-init, matching construction -- see `winder.determinism.init_parameters`, whose
        closed-vocabulary check this hook satisfies for a raw `nn.Parameter` no standard layer
        type covers (the `RelativePositionBias` pattern)."""
        nn.init.zeros_(self.lead_embedding)

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != self.config.n_leads:
            raise ValueError(
                f"waveform must be (B, {self.config.n_leads}, n_samples), got "
                f"{tuple(waveform.shape)}"
            )
        b, n_leads, n_samples = waveform.shape
        p = self.config.patch_width
        if n_samples % p != 0:
            raise ValueError(
                f"n_samples={n_samples} must be exactly divisible by patch_width={p} -- "
                f"non-overlapping patches require the record to tile evenly, not a ragged "
                f"final patch"
            )
        n_patches = n_samples // p
        # (B, n_leads, n_samples) -> (B, n_leads, n_patches, p): patch j of every lead covers
        # samples [j*p, (j+1)*p - 1], the same time-ordered grouping as PatchEncoder -- but the
        # shared MLP here reads ONE lead's patch at a time, not the 12-lead flattened patch.
        patches = waveform.reshape(b, n_leads, n_patches, p)
        feats = self.lead_fc2(self.act(self.lead_fc1(patches)))  # (B, n_leads, n_patches, W)
        feats = feats + self.lead_embedding[None, :, None, :]  # per-lead offset, broadcast
        # (B, n_leads, n_patches, W) -> (B, n_patches, n_leads, W) -> (B, n_patches, n_leads*W):
        # lead-major concatenation, mirroring PatchEncoder's own lead-major flatten.
        concat = feats.permute(0, 2, 1, 3).reshape(b, n_patches, n_leads * self.config.lead_width)
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.mix(concat))


# STATUS: PROTOTYPE - not tested on real data (Stage-2 push variants below; the SCOUT section
# above is unchanged and carries its own campaign-validated status.)
# ------------------------------------------------- STAGE-2 push variants (Amendment 14, eighth
#                                                    addendum -- "CTO's three-stage endgame")
#
# Last-optimization-push encoder variants (`artifacts/campaign_x2x2/pre_launch_addendum.md`,
# AMENDMENT 14, EIGHTH ADDENDUM, stage 2): each is COMPOSABLE with the crowned conv-trunk base --
# D3 (dilations (1, 2, 4)) or CE1 (dilations (1, 2)) -- selected via `dilations`; the operator
# spectrum rides on orthogonal CLI flags and never appears here. Same binding doctrine as the
# SCOUT section above: CM-01 strict causality, NO terminal normalization before the projector,
# token grid fixed at (B, 125, 256). Every variant is a NEW registry tag backed by a NEW class;
# the SCOUT classes above are structurally untouched (the paired-comparison doctrine: incumbent
# arms stay bit-identical AND code-identical). `tests/test_encoder_stage2_variants.py` carries
# the param-count, CM-01 bitwise, handshake, and determinism proofs.


def _dct_ii_orthonormal_basis(n: int) -> torch.Tensor:
    """The orthonormal DCT-II analysis matrix `C` (scipy's `dct(..., norm="ortho")` convention):
    `C[k, m] = s_k * cos(pi * (2m + 1) * k / (2n))`, `s_0 = sqrt(1/n)`, `s_k = sqrt(2/n)` for
    `k >= 1`, so `C @ C.T = I` in exact arithmetic. Computed in float64 then cast to float32: a
    closed-form constant with no RNG anywhere, identical on every construction by construction.
    """
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    m = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * (2.0 * m + 1.0) * k / (2.0 * n)) * (2.0 / n) ** 0.5
    basis[0] = basis[0] / 2.0**0.5
    # cast: this torch stub types Tensor.to as Any-returning; this is a real torch.Tensor.
    return cast(torch.Tensor, basis.to(torch.float32))


@dataclass
class DctStemConvTrunkEncoderConfig:
    # Same fields as ConvTrunkEncoderConfig, by the eighth addendum's V1 spec: --latent-width /
    # --encoder-json compositions written for conv_trunk carry over unchanged.
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2])


class DctStemConvTrunkEncoder(nn.Module, Encoder):
    """V1 (spectral stem): `ConvTrunkEncoder` with a FROZEN orthonormal DCT-II pre-rotation
    inserted before the patch embedder's first Linear -- flattened 96-dim patch -> fixed 96x96
    DCT-II -> learnable 96->512 -> GELU -> 512->256 -> the conv trunk, exactly as before.

    The basis is a registered BUFFER (`dct_basis`), not a parameter: it adds zero learnable
    params (`sum(p.numel())` equals the matching `conv_trunk`'s exactly), never requires grad,
    is untouched by `winder.determinism.init_parameters` (which iterates parameters, not
    buffers), and rides in `state_dict` so checkpoints carry the exact matrix. Motivation on
    record (eighth addendum V1): ttl-phase's frozen-DCT stem precedent -- the fixed oscillatory
    basis constrains the stem toward spectral structure at init/optimization level; because the
    following 96->512 Linear is full, expressivity is unchanged (an orthonormal change of basis),
    only the optimization geometry moves. Note, pinned by the addendum: the DCT acts on the FULL
    lead-major flattened 96-vector (12 leads x 8 samples), not per-lead 8-point blocks.

    Key args: `dilations` (the crowned-base selector: (1, 2) = CE1, (1, 2, 4) = D3),
    `kernel_size`. Usage: `DctStemConvTrunkEncoder(DctStemConvTrunkEncoderConfig())` ->
    (B, 12, 1000) -> (B, 125, 256). CM-01 and no-terminal-norm doctrine as the section header.
    """

    def __init__(self, config: DctStemConvTrunkEncoderConfig) -> None:
        super().__init__()
        if config.patch_width <= 0:
            raise ValueError(f"patch_width must be positive, got {config.patch_width}")
        if not config.dilations:
            raise ValueError("dilations must name at least one block, got an empty list")
        self.config = config
        input_width = config.n_leads * config.patch_width
        self.register_buffer("dct_basis", _dct_ii_orthonormal_basis(input_width))
        self.dct_basis: torch.Tensor
        self.fc1 = nn.Linear(input_width, config.hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_width, config.latent_width)
        self.blocks = nn.ModuleList(
            [
                DilatedCausalBlock1d(config.latent_width, config.kernel_size, dilation)
                for dilation in config.dilations
            ]
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != self.config.n_leads:
            raise ValueError(
                f"waveform must be (B, {self.config.n_leads}, n_samples), got "
                f"{tuple(waveform.shape)}"
            )
        b, n_leads, n_samples = waveform.shape
        p = self.config.patch_width
        if n_samples % p != 0:
            raise ValueError(
                f"n_samples={n_samples} must be exactly divisible by patch_width={p} -- "
                f"non-overlapping patches require the record to tile evenly"
            )
        n_patches = n_samples // p
        # PatchEncoder's exact lead-major flatten (see its forward): patch j covers raw samples
        # [j*p, (j+1)*p - 1] of every lead, causal by construction.
        patches = waveform.reshape(b, n_leads, n_patches, p).permute(0, 2, 1, 3)
        flat = patches.reshape(b, n_patches, n_leads * p)
        # The frozen pre-rotation: y = C x per patch (F.linear computes x @ C.T, i.e. rows of C
        # are the analysis basis functions).
        rotated = F.linear(flat, self.dct_basis)
        tokens = self.fc2(self.act(self.fc1(rotated)))
        x = tokens.transpose(1, 2)  # (B, K, n_tokens): Conv1d's channel-first layout
        for block in self.blocks:
            x = block(x)
        return cast(torch.Tensor, x.transpose(1, 2))


# The eighth addendum's V2 scales, pinned: each token also sees x2/x4/x8 trailing views.
_MULTISCALE_STRIDES: tuple[int, ...] = (1, 2, 4, 8)


@dataclass
class MultiscaleStemConvTrunkEncoderConfig:
    # Same fields as ConvTrunkEncoderConfig (scales are pinned by the addendum, not a knob).
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2])


class MultiscaleStemConvTrunkEncoder(nn.Module, Encoder):
    """V2 (multiscale stem): per token `j`, in addition to its own 8-sample patch, LEFT-ALIGNED
    trailing downsampled views ending at the token's own last sample -- scale `s` in {2, 4, 8}
    summarizes the trailing `8s` raw samples per lead (window `[8(j+1) - 8s, 8(j+1) - 1]`) as 8
    values, so the token input is 12 leads x 8 samples x 4 scales = 384 -> hidden 512 -> 256,
    then the conv trunk as usual. Early tokens with insufficient history are LEFT-PADDED with
    zeros. STRICT CM-01: every view ends at raw sample `8(j+1) - 1`; token `j` reads samples in
    `[8(j+1) - 64, 8(j+1) - 1]` and NOTHING at or after `8(j+1)` -- receptive field widened to
    640 ms, sub-beat, DELIBERATE (eighth addendum V2: Reverso's multi-scale input adapted
    causally, "each token also sees x2/x4/x8 left-aligned downsampled views of its trailing
    <= 640 ms").

    Summarization choice, on record: each scale-`s` view is the per-block MEAN of `s`
    consecutive samples (average pooling, kernel = stride = `s`), not `x[::s]` decimation. The
    addendum pins two properties simultaneously -- every view ends at the token's own last
    sample, and the receptive field spans the full trailing 640 ms -- and strided decimation can
    satisfy at most one of them (either its last tap misses sample `8(j+1) - 1` or its first tap
    misses the window start); the block mean satisfies both, and is anti-aliased. The flatten
    order is the addendum's literal nesting: (lead, sample, scale), scales (x1, x2, x4, x8).

    Key args: `dilations` (CE1 = (1, 2), D3 = (1, 2, 4)). Usage:
    `MultiscaleStemConvTrunkEncoder(MultiscaleStemConvTrunkEncoderConfig())` ->
    (B, 12, 1000) -> (B, 125, 256). No terminal norm (section header doctrine).
    """

    def __init__(self, config: MultiscaleStemConvTrunkEncoderConfig) -> None:
        super().__init__()
        if config.patch_width <= 0:
            raise ValueError(f"patch_width must be positive, got {config.patch_width}")
        if not config.dilations:
            raise ValueError("dilations must name at least one block, got an empty list")
        self.config = config
        input_width = config.n_leads * config.patch_width * len(_MULTISCALE_STRIDES)
        self.fc1 = nn.Linear(input_width, config.hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_width, config.latent_width)
        self.blocks = nn.ModuleList(
            [
                DilatedCausalBlock1d(config.latent_width, config.kernel_size, dilation)
                for dilation in config.dilations
            ]
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != self.config.n_leads:
            raise ValueError(
                f"waveform must be (B, {self.config.n_leads}, n_samples), got "
                f"{tuple(waveform.shape)}"
            )
        b, n_leads, n_samples = waveform.shape
        p = self.config.patch_width
        if n_samples % p != 0:
            raise ValueError(
                f"n_samples={n_samples} must be exactly divisible by patch_width={p} -- "
                f"non-overlapping patches require the record to tile evenly"
            )
        n_patches = n_samples // p
        views = []
        for s in _MULTISCALE_STRIDES:
            win = p * s
            # Left-pad by (win - p) zeros so window j of the unfold covers, in ORIGINAL sample
            # coordinates, exactly [p*(j+1) - win, p*(j+1) - 1]: trailing, ending at the token's
            # own last raw sample, zero-history for early tokens (CM-01: nothing at or after
            # p*(j+1) is ever inside any window).
            padded = F.pad(waveform, (win - p, 0))
            u = padded.unfold(2, win, p)  # (B, n_leads, n_patches, win)
            # Per-block mean: s consecutive samples -> 1 summary value (see class docstring).
            views.append(u.reshape(b, n_leads, n_patches, p, s).mean(dim=-1))
        stack = torch.stack(views, dim=-1)  # (B, n_leads, n_patches, p, n_scales)
        flat = stack.permute(0, 2, 1, 3, 4).reshape(b, n_patches, self.fc1.in_features)
        tokens = self.fc2(self.act(self.fc1(flat)))
        x = tokens.transpose(1, 2)  # (B, K, n_tokens): Conv1d's channel-first layout
        for block in self.blocks:
            x = block(x)
        return cast(torch.Tensor, x.transpose(1, 2))


class _WindowedCausalTokenAttention(_CausalTokenAttention):
    """`_CausalTokenAttention` with the causal mask additionally BANDED: position `i` attends
    only to `j` in `[i - window, i]`. A mask-only change -- `__init__` (blocks, shared rel-pos
    bias) is inherited untouched, so parameters are identical to the unwindowed stack by
    construction. The diagonal (`j = i`) is always inside the band, so no row is all `-inf`
    (softmax never sees an empty support)."""

    def __init__(
        self,
        n_layers: int,
        *,
        window: int,
        width: int,
        n_heads: int,
        feedforward_width: int,
        dropout: float,
        rel_pos_max_distance: int,
        layernorm_eps: float,
    ) -> None:
        super().__init__(
            n_layers,
            width=width,
            n_heads=n_heads,
            feedforward_width=feedforward_width,
            dropout=dropout,
            rel_pos_max_distance=rel_pos_max_distance,
            layernorm_eps=layernorm_eps,
        )
        self.window = window

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        s = tokens.shape[1]
        rel_bias = self.rel_bias.forward(s, device=tokens.device)
        idx = torch.arange(s, device=tokens.device)
        offset = idx.unsqueeze(1) - idx.unsqueeze(0)  # (S, S), i - j
        # Banded causal: keep j <= i (CM-01, the parent's tril) AND i - j <= window (the band).
        banded = (offset >= 0) & (offset <= self.window)
        attn_mask = rel_bias.masked_fill(~banded, float("-inf"))
        for block in self.blocks:
            tokens = block(tokens, attn_mask)
        return tokens


@dataclass
class WindowedHybridTrunkEncoderConfig:
    # HybridTrunkEncoderConfig's exact fields, plus the band half-width `attn_window`.
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2])
    n_attn_layers: int = 2
    n_heads: int = 4
    feedforward_width: int = 1024
    dropout: float = 0.0
    rel_pos_max_distance: int = 124
    layernorm_eps: float = 1e-5
    attn_window: int = 12  # tokens of past each position may attend to (~1 beat at 80 ms/token)


class WindowedHybridTrunkEncoder(nn.Module, Encoder):
    """V3 (windowed attention): `HybridTrunkEncoder` (patch stage -> conv blocks -> attention
    blocks) with the attention mask additionally banded to `j` in `[i - attn_window, i]`,
    default 12 tokens (~1 beat at 80 ms/token). The eighth addendum's untested attention cell:
    C6-like mixing CONFINED inside the beat-stationarity window, where the Amendment 8(B)
    phase-transport prior binds token-to-token. Mask-only change: parameters are identical to
    the matching unwindowed `hybrid_trunk` (asserted in tests).

    Key args: `attn_window` (band width, >= 1), `dilations` (CE1 = (1, 2), D3 = (1, 2, 4)).
    Usage: `WindowedHybridTrunkEncoder(WindowedHybridTrunkEncoderConfig())` ->
    (B, 12, 1000) -> (B, 125, 256). No terminal norm (section header doctrine); output is the
    last attention block's residual add.
    """

    def __init__(self, config: WindowedHybridTrunkEncoderConfig) -> None:
        super().__init__()
        if not config.dilations:
            raise ValueError("dilations must name at least one block, got an empty list")
        if config.n_attn_layers < 1:
            raise ValueError(f"n_attn_layers must be >= 1, got {config.n_attn_layers}")
        if config.attn_window < 1:
            raise ValueError(f"attn_window must be >= 1, got {config.attn_window}")
        self.config = config
        self.patch = PatchEncoder(
            PatchEncoderConfig(
                n_leads=config.n_leads,
                patch_width=config.patch_width,
                hidden_width=config.hidden_width,
                latent_width=config.latent_width,
            )
        )
        self.conv_blocks = nn.ModuleList(
            [
                DilatedCausalBlock1d(config.latent_width, config.kernel_size, dilation)
                for dilation in config.dilations
            ]
        )
        self.attn = _WindowedCausalTokenAttention(
            config.n_attn_layers,
            window=config.attn_window,
            width=config.latent_width,
            n_heads=config.n_heads,
            feedforward_width=config.feedforward_width,
            dropout=config.dropout,
            rel_pos_max_distance=config.rel_pos_max_distance,
            layernorm_eps=config.layernorm_eps,
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.patch.forward(waveform)  # (B, n_tokens, K); validates shape itself
        x = tokens.transpose(1, 2)  # (B, K, n_tokens): Conv1d's channel-first layout
        for block in self.conv_blocks:
            x = block(x)
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.attn(x.transpose(1, 2)))


class _GatedDeltaNetLayer(nn.Module):
    """One Gated DeltaNet layer over the token stream, plain sequential recurrence in pure
    torch (L = 125 tokens: a fused kernel is deliberately NOT used, per the eighth addendum's
    simplest-correct-form instruction). Per head, state `S` is (head_dim_v x head_dim_k):

        S_i = S_{i-1} (alpha_i (I - beta_i k_i k_i^T)) + beta_i v_i k_i^T,   y_i = S_i q_i

    with `q/k/v` linear projections of the pre-LN token stream, `beta_i`/`alpha_i` per-token,
    per-head sigmoid scalars from their own linear projections (Gated DeltaNet: Yang, Kautz,
    Hatamizadeh, arXiv:2412.06464; alpha is the forget gate, beta the delta-rule write
    strength). Output: `x + concat_heads(y)` -- LN-free residual (no output norm, no output
    projection; heads mix through the next layer's projections). Causal by construction: the
    recurrence reads only positions <= i.

    Deviation from the literal brief, on record: `k` is L2-normalized per head (the paper's own
    convention). Unit keys make `(I - beta k k^T)`'s spectrum `{1 - beta, 1}` with
    `beta = sigmoid(.) in (0, 1)` -- a contraction; with raw keys the eigenvalue along `k` is
    `1 - beta * ||k||^2`, expansive whenever `beta ||k||^2 > 2`, and the recurrence diverges
    over the 125-token roll. Silently unstable math is not shipped.
    """

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width={width} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_width = width // n_heads
        self.norm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.beta_proj = nn.Linear(width, n_heads)
        self.alpha_proj = nn.Linear(width, n_heads)

    def alpha_gates(self, x: torch.Tensor) -> torch.Tensor:
        """The forget gates this layer would apply to `x`: (B, T, n_heads), each in (0, 1) --
        computed exactly as `forward` computes them (same pre-LN input)."""
        return torch.sigmoid(self.alpha_proj(self.norm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_width)
        k = F.normalize(self.k_proj(h).view(b, t, self.n_heads, self.head_width), dim=-1)
        v = self.v_proj(h).view(b, t, self.n_heads, self.head_width)
        beta = torch.sigmoid(self.beta_proj(h))  # (B, T, H)
        alpha = torch.sigmoid(self.alpha_proj(h))  # (B, T, H)
        state = x.new_zeros(b, self.n_heads, self.head_width, self.head_width)  # (B,H,Dv,Dk)
        outs = []
        for i in range(t):
            k_i = k[:, i]  # (B, H, Dk)
            v_i = v[:, i]  # (B, H, Dv)
            beta_i = beta[:, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            alpha_i = alpha[:, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            # S (alpha (I - beta k k^T)) = alpha (S - beta (S k) k^T), then + beta v k^T.
            state_k = torch.einsum("bhvk,bhk->bhv", state, k_i)  # (B, H, Dv)
            state = alpha_i * (
                state - beta_i * state_k.unsqueeze(-1) * k_i.unsqueeze(-2)
            ) + beta_i * v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
            y_i = torch.einsum("bhvk,bhk->bhv", state, q[:, i])  # (B, H, Dv)
            outs.append(y_i.reshape(b, w))
        return x + torch.stack(outs, dim=1)


@dataclass
class GdnTrunkEncoderConfig:
    # ConvTrunkEncoderConfig's exact fields, plus the GDN stack's own two.
    n_leads: int = 12
    patch_width: int = 8
    hidden_width: int = 512
    latent_width: int = 256
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2])
    n_gdn_layers: int = 2
    n_heads: int = 4  # head dim = latent_width / n_heads = 64 at the defaults


class GdnTrunkEncoder(nn.Module, Encoder):
    """V6 (wildcard): `ConvTrunkEncoder`'s composition (patch stage -> conv blocks) followed by
    `n_gdn_layers` Gated DeltaNet layers (`_GatedDeltaNetLayer`, arXiv:2412.06464) on the token
    stream. No terminal norm (section header doctrine); output is the last GDN residual add.

    On record (eighth addendum scope extension): EXPECTED TO UNDERPERFORM on probes if the
    forget gate fails to learn a short memory -- an unbounded-context recurrence is exactly the
    poison the scout campaign measured -- and this arm's PRIMARY readout is the learned memory
    length itself (`effective_memory_tokens`), i.e. whether the trained gate converges to ~the
    beat period (~12 tokens at 80 ms/token).

    Key args: `dilations` (CE1 = (1, 2), D3 = (1, 2, 4)), `n_gdn_layers` (default 2), `n_heads`
    (default 4, head dim 64). Usage: `GdnTrunkEncoder(GdnTrunkEncoderConfig())` ->
    (B, 12, 1000) -> (B, 125, 256).
    """

    def __init__(self, config: GdnTrunkEncoderConfig) -> None:
        super().__init__()
        if not config.dilations:
            raise ValueError("dilations must name at least one block, got an empty list")
        if config.n_gdn_layers < 1:
            raise ValueError(f"n_gdn_layers must be >= 1, got {config.n_gdn_layers}")
        self.config = config
        self.patch = PatchEncoder(
            PatchEncoderConfig(
                n_leads=config.n_leads,
                patch_width=config.patch_width,
                hidden_width=config.hidden_width,
                latent_width=config.latent_width,
            )
        )
        self.conv_blocks = nn.ModuleList(
            [
                DilatedCausalBlock1d(config.latent_width, config.kernel_size, dilation)
                for dilation in config.dilations
            ]
        )
        self.gdn_layers = nn.ModuleList(
            [
                _GatedDeltaNetLayer(config.latent_width, config.n_heads)
                for _ in range(config.n_gdn_layers)
            ]
        )

    @property
    def latent_width(self) -> int:
        return self.config.latent_width

    def _tokens_before_gdn(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.patch.forward(waveform)  # (B, n_tokens, K); validates shape itself
        x = tokens.transpose(1, 2)  # (B, K, n_tokens): Conv1d's channel-first layout
        for block in self.conv_blocks:
            x = block(x)
        return x.transpose(1, 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._tokens_before_gdn(waveform)
        for layer in self.gdn_layers:
            x = layer(x)
        return x

    def effective_memory_tokens(self, waveform: torch.Tensor) -> torch.Tensor:
        """Per-(layer, head) decay length of the learned forget gate, in TOKENS: the
        (n_gdn_layers, n_heads) tensor `1 / (1 - mean(alpha))`, the mean taken over batch and
        token positions at each layer's ACTUAL input (the stream the gate really sees).

        Estimator definition: the state obeys `S_i ~ alpha_i S_{i-1} + write_i`, so under a
        constant gate `alpha` a write's influence decays as `alpha^n` -- geometric with time
        constant `1/(1 - alpha)` tokens (the AR(1) mean lifetime). Replacing the token-varying
        gate by its empirical mean gives the reported figure; `~12` means the head remembers
        about one beat at 80 ms/token. Unbounded (`inf`) if the mean gate saturates at 1 --
        reported raw, not clamped: saturation IS the unbounded-context finding.
        """
        with torch.no_grad():
            x = self._tokens_before_gdn(waveform)
            rows = []
            for module in self.gdn_layers:
                # cast: ModuleList iteration is typed as bare nn.Module; these are the
                # _GatedDeltaNetLayer instances __init__ put there.
                layer = cast(_GatedDeltaNetLayer, module)
                alpha = layer.alpha_gates(x)  # (B, T, H)
                rows.append(1.0 / (1.0 - alpha.mean(dim=(0, 1))))
                x = layer.forward(x)
        return torch.stack(rows)

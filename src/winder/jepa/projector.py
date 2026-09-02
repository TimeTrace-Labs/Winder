"""The MVP `ProjectionHead`: a small per-token MLP, no normalization anywhere in it.

`256 -> 512 -> 256` (an earlier design pass had `256 -> 512 -> 128`; the design spec fixes
`output_width == input_width == 256`, matching the encoder's `latent_width` so SIGReg operates in
the same width as the encoder's own embedding space). Applied identically, with the same weights,
to both the full-waveform and the raw-waveform-masked encoder outputs -- `nn.Linear` broadcasts
over the leading `(B, n_tokens)` dims automatically, so no manual per-token loop is needed.

No BatchNorm, LayerNorm, dropout, or output normalization anywhere in this module -- same reason
as the encoder's missing terminal LayerNorm: this is precisely the tensor SIGReg constrains, and
imposing a normalization immediately upstream would pre-satisfy part of what SIGReg is meant to
learn (cross-checked against LeWorldModel, arXiv:2603.19312 Sec 3.1 -- see `encoder.py`'s module
docstring for the finding). Discarded entirely at evaluation
(see `winder.jepa.model.JepaModel.embed`).

Depth is 2 linear layers (`fc1`, `fc2`), one hidden layer between them. LeJEPA's own projector-depth
ablation (arXiv:2511.08544 Table 4, ImageNet linear-probe accuracy) shows a real, fairly consistent
gain from 1- to 2- to 3-layer projectors across three backbones (e.g. ResNet50: 79.71 -> 82.44 ->
83.93). Not adopted here: that ablation is on a different domain (ImageNet classification, ViT/
ResNet backbones, K in the hundreds-to-thousands) with no pretrained checkpoint on this domain yet
to test a deeper projector against, and this MVP's own first real-data pretraining run should not
also be changing the architecture it's validating. Recorded as a candidate for the future
calibration sweep (DIAG-02/E1-05) once a real checkpoint exists to test it against, not silently
dropped.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from winder.jepa.base import ProjectionHead

__all__ = [
    "MlpProjectionHeadConfig",
    "MlpProjectionHead",
    "Mlp3ProjectionHeadConfig",
    "Mlp3ProjectionHead",
]


@dataclass
class MlpProjectionHeadConfig:
    input_width: int = 256
    hidden_width: int = 512
    output_width: int = 256


class MlpProjectionHead(nn.Module, ProjectionHead):
    def __init__(self, config: MlpProjectionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_width, config.hidden_width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_width, config.output_width)

    @property
    def input_width(self) -> int:
        return self.config.input_width

    @property
    def output_width(self) -> int:
        return self.config.output_width

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.config.input_width:
            raise ValueError(
                f"tokens last dim {tokens.shape[-1]} != input_width {self.config.input_width}"
            )
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.fc2(self.act(self.fc1(tokens))))


@dataclass
class Mlp3ProjectionHeadConfig:
    """`256 -> 512 -> 512 -> 256` -- one extra `hidden_width`-wide layer relative to
    `MlpProjectionHeadConfig`'s `256 -> 512 -> 256`. A fixed 3-layer shape (`fc1`/`fc_mid`/`fc2`,
    not a `n_hidden_layers` depth knob) so its parameter count derives term-for-term from three
    named `nn.Linear` shapes, matching Amendment 12's own arm C5 accounting."""

    input_width: int = 256
    hidden_width: int = 512
    output_width: int = 256


class Mlp3ProjectionHead(nn.Module, ProjectionHead):
    """Amendment 12's arm C5: `MlpProjectionHead`'s own module docstring records a 3-layer
    projector as a candidate once a real checkpoint exists to calibrate against (LeJEPA's
    projector-depth ablation, arXiv:2511.08544 Table 4) -- this is that candidate, registered as
    a NEW class under a NEW registry tag (`"mlp3"`) rather than a config knob on the existing
    `MlpProjectionHead`/`"mlp"` entry, so the pre-existing class's architecture, state_dict keys,
    and every checkpoint/config built against it are untouched by this addition.

    `fc1 -> GELU -> fc_mid -> GELU -> fc2`, no normalization or dropout anywhere in it -- same
    reasoning as `MlpProjectionHead`'s own module docstring (SIGReg must control this tensor's own
    scale and moments). Applied identically, per token, to both the full-waveform and the
    raw-waveform-masked encoder outputs, exactly like `MlpProjectionHead`.

    Usage: `Mlp3ProjectionHead(Mlp3ProjectionHeadConfig())` applied to `(B, n_tokens, 256)` ->
    `(B, n_tokens, 256)`.
    """

    def __init__(self, config: Mlp3ProjectionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_width, config.hidden_width)
        self.act = nn.GELU()
        self.fc_mid = nn.Linear(config.hidden_width, config.hidden_width)
        self.fc2 = nn.Linear(config.hidden_width, config.output_width)

    @property
    def input_width(self) -> int:
        return self.config.input_width

    @property
    def output_width(self) -> int:
        return self.config.output_width

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.config.input_width:
            raise ValueError(
                f"tokens last dim {tokens.shape[-1]} != input_width {self.config.input_width}"
            )
        h = self.act(self.fc1(tokens))
        h = self.act(self.fc_mid(h))
        # cast: nn.Module.__call__ is typed to return Any; this is a real torch.Tensor.
        return cast(torch.Tensor, self.fc2(h))

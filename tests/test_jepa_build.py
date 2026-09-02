"""The composition root's handshake, tested against hand-rolled doubles rather than the real
neural primitives (which land in later commits) -- `assemble_jepa` takes already-constructed
instances precisely so this is possible without touching any registry."""

import pytest
import torch
from torch import nn

from winder.jepa.base import (
    Encoder,
    MaskSampler,
    PredictionLoss,
    Predictor,
    ProjectionHead,
    Regularizer,
)
from winder.jepa.losses import MsePredictionLoss, MsePredictionLossConfig
from winder.jepa.masking import CausalBlockMaskSampler, CausalBlockMaskSamplerConfig
from winder.jepa.model import JepaConfig, assemble_jepa
from winder.jepa.registry import (
    ENCODER_REGISTRY,
    PREDICTOR_REGISTRY,
    PROJECTION_HEAD_REGISTRY,
    build_from_registry,
    resolve_sub_config,
)


class _FakeEncoder(nn.Module, Encoder):
    def __init__(self, latent_width: int, n_tokens: int) -> None:
        super().__init__()
        self._latent_width = latent_width
        self._n_tokens = n_tokens

    @property
    def latent_width(self) -> int:
        return self._latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        b = waveform.shape[0]
        return torch.zeros(b, self._n_tokens, self._latent_width)


class _WrongTokenCountEncoder(nn.Module, Encoder):
    def __init__(self, latent_width: int) -> None:
        super().__init__()
        self._latent_width = latent_width

    @property
    def latent_width(self) -> int:
        return self._latent_width

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        b = waveform.shape[0]
        return torch.zeros(b, 1, self._latent_width)  # deliberately wrong token count


class _FakeProjectionHead(nn.Module, ProjectionHead):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self._input_width = input_width
        self._output_width = output_width

    @property
    def input_width(self) -> int:
        return self._input_width

    @property
    def output_width(self) -> int:
        return self._output_width

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s, _ = tokens.shape
        return torch.zeros(b, s, self._output_width)


class _FakePredictor(nn.Module, Predictor):
    def __init__(self, width: int) -> None:
        super().__init__()
        self._width = width

    @property
    def width(self) -> int:
        return self._width

    def forward(self, z_ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(z_ctx)


class _WrongShapePredictor(nn.Module, Predictor):
    """Returns a shape inconsistent with its own declared width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self._width = width

    @property
    def width(self) -> int:
        return self._width

    def forward(self, z_ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = z_ctx.shape
        return torch.zeros(b, s, self._width + 1)


class _FiniteRegularizer(Regularizer):
    def __call__(self, z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        return z.square().mean()


class _NonFiniteRegularizer(Regularizer):
    def __call__(self, z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        return torch.tensor(float("nan"))


class _WrongShapeRegularizer(Regularizer):
    def __call__(self, z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        return torch.zeros(3)


def _real_mask_sampler() -> MaskSampler:
    return CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())


def _real_prediction_loss() -> PredictionLoss:
    return MsePredictionLoss(MsePredictionLossConfig())


def _config(n_tokens: int = 250) -> JepaConfig:
    return JepaConfig(n_leads=12, n_samples=1000, n_tokens=n_tokens)


def test_assemble_jepa_succeeds_with_consistent_doubles() -> None:
    gen = torch.Generator().manual_seed(0)
    model = assemble_jepa(
        _config(),
        _FakeEncoder(latent_width=32, n_tokens=250),
        _FakeProjectionHead(input_width=32, output_width=16),
        _FakePredictor(width=16),
        _real_mask_sampler(),
        _real_prediction_loss(),
        _FiniteRegularizer(),
        generator=gen,
    )
    assert model.encoder.latent_width == 32


def test_encoder_projector_width_mismatch_raises() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="latent_width=32.*input_width=99"):
        assemble_jepa(
            _config(),
            _FakeEncoder(latent_width=32, n_tokens=250),
            _FakeProjectionHead(input_width=99, output_width=16),
            _FakePredictor(width=16),
            _real_mask_sampler(),
            _real_prediction_loss(),
            _FiniteRegularizer(),
            generator=gen,
        )


def test_projector_predictor_width_mismatch_raises() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="output_width=16.*width=7"):
        assemble_jepa(
            _config(),
            _FakeEncoder(latent_width=32, n_tokens=250),
            _FakeProjectionHead(input_width=32, output_width=16),
            _FakePredictor(width=7),
            _real_mask_sampler(),
            _real_prediction_loss(),
            _FiniteRegularizer(),
            generator=gen,
        )


def test_wrong_token_count_raises() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match=r"expected \(1, 250,"):
        assemble_jepa(
            _config(n_tokens=250),
            _WrongTokenCountEncoder(latent_width=32),
            _FakeProjectionHead(input_width=32, output_width=16),
            _FakePredictor(width=16),
            _real_mask_sampler(),
            _real_prediction_loss(),
            _FiniteRegularizer(),
            generator=gen,
        )


def test_predictor_wrong_output_shape_raises() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="predictor produced shape"):
        assemble_jepa(
            _config(),
            _FakeEncoder(latent_width=32, n_tokens=250),
            _FakeProjectionHead(input_width=32, output_width=16),
            _WrongShapePredictor(width=16),
            _real_mask_sampler(),
            _real_prediction_loss(),
            _FiniteRegularizer(),
            generator=gen,
        )


def test_non_finite_regularizer_raises() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="regularizer returned"):
        assemble_jepa(
            _config(),
            _FakeEncoder(latent_width=32, n_tokens=250),
            _FakeProjectionHead(input_width=32, output_width=16),
            _FakePredictor(width=16),
            _real_mask_sampler(),
            _real_prediction_loss(),
            _NonFiniteRegularizer(),
            generator=gen,
        )


def test_wrong_shape_regularizer_raises() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="regularizer returned"):
        assemble_jepa(
            _config(),
            _FakeEncoder(latent_width=32, n_tokens=250),
            _FakeProjectionHead(input_width=32, output_width=16),
            _FakePredictor(width=16),
            _real_mask_sampler(),
            _real_prediction_loss(),
            _WrongShapeRegularizer(),
            generator=gen,
        )


def test_lambda_zero_does_not_block_assembly() -> None:
    """Reversed from an earlier handshake rule that rejected lambda_sig=0 paired with a real
    (non-'none') regularizer tag, on the theory it was a config error. The spec's
    collapse-control ablation *requires* exactly this configuration (lambda_sig=0, the real
    regularizer still registered and still logged) -- assemble_jepa performs no such rejection;
    lambda_sig is not even read by the handshake."""
    config = _config()
    config.lambda_sig = 0.0
    gen = torch.Generator().manual_seed(0)
    model = assemble_jepa(
        config,
        _FakeEncoder(latent_width=32, n_tokens=250),
        _FakeProjectionHead(input_width=32, output_width=16),
        _FakePredictor(width=16),
        _real_mask_sampler(),
        _real_prediction_loss(),
        _FiniteRegularizer(),
        generator=gen,
    )
    assert model.config.lambda_sig == 0.0


def test_assemble_jepa_with_real_registered_encoder_projector_predictor() -> None:
    """Encoder (residual_cnn), ProjectionHead (mlp), and Predictor (transformer) are all real,
    registry-built primitives here -- only Regularizer is still a double, since SIGReg lands in
    a later commit. Confirms the real widths (256 -> 256 -> 256) are mutually compatible per the
    design spec, not just the hand-picked widths used in the double-only tests above."""
    encoder_cfg = resolve_sub_config(ENCODER_REGISTRY, "residual_cnn", {})
    encoder = build_from_registry(ENCODER_REGISTRY, "residual_cnn", encoder_cfg)
    projector_cfg = resolve_sub_config(PROJECTION_HEAD_REGISTRY, "mlp", {})
    projector = build_from_registry(PROJECTION_HEAD_REGISTRY, "mlp", projector_cfg)
    predictor_cfg = resolve_sub_config(PREDICTOR_REGISTRY, "transformer", {})
    predictor = build_from_registry(PREDICTOR_REGISTRY, "transformer", predictor_cfg)

    gen = torch.Generator().manual_seed(0)
    model = assemble_jepa(
        _config(),
        encoder,
        projector,
        predictor,
        _real_mask_sampler(),
        _real_prediction_loss(),
        _FiniteRegularizer(),
        generator=gen,
    )
    assert model.encoder.latent_width == 256

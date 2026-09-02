import pytest
import torch

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
from winder.jepa.registry import (
    ENCODER_REGISTRY,
    MASK_SAMPLER_REGISTRY,
    PREDICTION_LOSS_REGISTRY,
    PREDICTOR_REGISTRY,
    PROJECTION_HEAD_REGISTRY,
    REGULARIZER_REGISTRY,
    build_from_registry,
    resolve_sub_config,
)


@pytest.mark.parametrize(
    "abc_cls",
    [Encoder, ProjectionHead, Predictor, MaskSampler, PredictionLoss, Regularizer],
)
def test_abcs_cannot_be_instantiated_directly(abc_cls: type) -> None:
    with pytest.raises(TypeError):
        abc_cls()  # type: ignore[abstract,call-arg]


@pytest.mark.parametrize("name", list(MASK_SAMPLER_REGISTRY))
def test_registered_mask_samplers_instantiate(name: str) -> None:
    schema_cls, ctor = MASK_SAMPLER_REGISTRY[name]
    sampler = ctor(schema_cls())
    assert isinstance(sampler, MaskSampler)


@pytest.mark.parametrize("name", list(PREDICTION_LOSS_REGISTRY))
def test_registered_prediction_losses_instantiate(name: str) -> None:
    schema_cls, ctor = PREDICTION_LOSS_REGISTRY[name]
    loss = ctor(schema_cls())
    assert isinstance(loss, PredictionLoss)


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_registered_encoders_instantiate(name: str) -> None:
    schema_cls, ctor = ENCODER_REGISTRY[name]
    encoder = ctor(schema_cls())
    assert isinstance(encoder, Encoder)


@pytest.mark.parametrize("name", list(PROJECTION_HEAD_REGISTRY))
def test_registered_projection_heads_instantiate(name: str) -> None:
    schema_cls, ctor = PROJECTION_HEAD_REGISTRY[name]
    head = ctor(schema_cls())
    assert isinstance(head, ProjectionHead)


@pytest.mark.parametrize("name", list(PREDICTOR_REGISTRY))
def test_registered_predictors_instantiate(name: str) -> None:
    schema_cls, ctor = PREDICTOR_REGISTRY[name]
    predictor = ctor(schema_cls())
    assert isinstance(predictor, Predictor)


@pytest.mark.parametrize("name", list(REGULARIZER_REGISTRY))
def test_registered_regularizers_instantiate(name: str) -> None:
    schema_cls, ctor = REGULARIZER_REGISTRY[name]
    regularizer = ctor(schema_cls())
    assert isinstance(regularizer, Regularizer)


def test_causal_block_mask_sampler_registered_under_causal_block_tag() -> None:
    schema_cls, ctor = MASK_SAMPLER_REGISTRY["causal_block"]
    assert schema_cls is CausalBlockMaskSamplerConfig
    assert ctor is CausalBlockMaskSampler


def test_mse_prediction_loss_registered_under_mse_tag() -> None:
    schema_cls, ctor = PREDICTION_LOSS_REGISTRY["mse"]
    assert schema_cls is MsePredictionLossConfig
    assert ctor is MsePredictionLoss


def test_resolve_sub_config_round_trips_override_and_preserves_defaults() -> None:
    resolved = resolve_sub_config(MASK_SAMPLER_REGISTRY, "causal_block", {"c_min": 40})
    assert resolved.c_min == 40


def test_resolve_sub_config_preserves_the_default_when_not_overridden() -> None:
    resolved = resolve_sub_config(MASK_SAMPLER_REGISTRY, "causal_block", {})
    assert resolved.c_min == 0  # default preserved


def test_build_from_registry_constructs_the_tagged_primitive() -> None:
    resolved = resolve_sub_config(MASK_SAMPLER_REGISTRY, "causal_block", {})
    sampler = build_from_registry(MASK_SAMPLER_REGISTRY, "causal_block", resolved)
    assert isinstance(sampler, CausalBlockMaskSampler)


def test_unknown_tag_raises_key_error() -> None:
    with pytest.raises(KeyError):
        resolve_sub_config(MASK_SAMPLER_REGISTRY, "not_a_real_sampler", {})
    with pytest.raises(KeyError):
        build_from_registry(
            MASK_SAMPLER_REGISTRY, "not_a_real_sampler", CausalBlockMaskSamplerConfig()
        )


def test_mask_sampler_call_signature_is_batched() -> None:
    """Reversed from an earlier per-step-shared-mask simplification: MaskSampler takes an
    explicit batch_size and returns one independent draw per record."""
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen = torch.Generator().manual_seed(0)
    plan = sampler(4, 250, generator=gen)
    assert plan.context.shape == (4, 250)
    assert plan.context.dtype == torch.bool

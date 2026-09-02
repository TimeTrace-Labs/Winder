"""Tag+registry for the six single-concern JEPA primitives.

Each primitive gets its own `name -> (config schema, constructor)` registry, exactly the pattern
in `winder.operators.registry.OPERATOR_REGISTRY` / `winder.data.normalization.NORM_REGISTRY`:
OmegaConf structured configs can't express a `Union` of concrete schemas directly, so a config
carries a string tag resolved through the matching registry instead. Plain module-level dict
literals, no decorator-based registration -- new entries are added by editing the literal, which
forces the schema and class imports to the top of this file.

`resolve_sub_config`/`build_from_registry` generalize `resolve_operator_config`/`build_operator`
into one dispatch pair used six times, rather than six copies of the same four-line body
(CLAUDE.md: don't duplicate a second code path for something that should be polymorphic -- here,
the *dispatch logic* is the thing that was about to be duplicated, not the registries themselves,
which are genuinely different data for six different kinds of primitive). `resolve_operator_config`
and `resolve_norm_config` are NOT refactored onto these by this change; they predate it, already
work, and absorbing them is a separate, deliberate edit.
"""

from collections.abc import Callable
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

from winder.jepa.base import (
    Encoder,
    MaskSampler,
    PredictionLoss,
    Predictor,
    ProjectionHead,
    Regularizer,
)
from winder.jepa.encoder import (
    AttnTrunkEncoder,
    AttnTrunkEncoderConfig,
    ConvTrunkEncoder,
    ConvTrunkEncoderConfig,
    DctStemConvTrunkEncoder,
    DctStemConvTrunkEncoderConfig,
    GdnTrunkEncoder,
    GdnTrunkEncoderConfig,
    HybridTrunkEncoder,
    HybridTrunkEncoderConfig,
    LeadFactorizedEncoder,
    LeadFactorizedEncoderConfig,
    MultiscaleStemConvTrunkEncoder,
    MultiscaleStemConvTrunkEncoderConfig,
    PatchEncoder,
    PatchEncoderConfig,
    ResidualCnnEncoder,
    ResidualCnnEncoderConfig,
    WindowedHybridTrunkEncoder,
    WindowedHybridTrunkEncoderConfig,
)
from winder.jepa.losses import MsePredictionLoss, MsePredictionLossConfig
from winder.jepa.masking import CausalBlockMaskSampler, CausalBlockMaskSamplerConfig
from winder.jepa.predictor import TransformerPredictor, TransformerPredictorConfig
from winder.jepa.projector import (
    Mlp3ProjectionHead,
    Mlp3ProjectionHeadConfig,
    MlpProjectionHead,
    MlpProjectionHeadConfig,
)
from winder.jepa.regularizers import NoRegularizer, NoRegularizerConfig, SigReg, SigRegConfig

__all__ = [
    "ENCODER_REGISTRY",
    "PROJECTION_HEAD_REGISTRY",
    "PREDICTOR_REGISTRY",
    "MASK_SAMPLER_REGISTRY",
    "PREDICTION_LOSS_REGISTRY",
    "REGULARIZER_REGISTRY",
    "resolve_sub_config",
    "build_from_registry",
]

_T = TypeVar("_T")


def resolve_sub_config(
    registry: dict[str, tuple[type, Callable[[Any], Any]]], name: str, params: Any
) -> DictConfig:
    """Merge `params` onto `registry[name]`'s schema. An unknown `name` raises `KeyError`."""
    schema_cls, _ = registry[name]
    return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(schema_cls), params))


def build_from_registry(
    registry: dict[str, tuple[type, Callable[[Any], _T]]], name: str, config: object
) -> _T:
    """Construct `registry[name]`'s primitive from an already-resolved config. An unknown `name`
    raises `KeyError`."""
    _, ctor = registry[name]
    return ctor(config)


REGULARIZER_REGISTRY: dict[str, tuple[type, Callable[[Any], Regularizer]]] = {
    "sigreg": (SigRegConfig, SigReg),
    "none": (NoRegularizerConfig, NoRegularizer),
}

ENCODER_REGISTRY: dict[str, tuple[type, Callable[[Any], Encoder]]] = {
    "residual_cnn": (ResidualCnnEncoderConfig, ResidualCnnEncoder),
    "patch": (PatchEncoderConfig, PatchEncoder),
    # Amendment 12's encoder-architecture SCOUT arms (winder.jepa.encoder's "SCOUT trunks"
    # section): the shared patch stage plus a conv (C1) / attention (C2/C3/C7) /
    # conv-then-attention (C6) trunk, and the trunk-free per-lead factorization (C4).
    "conv_trunk": (ConvTrunkEncoderConfig, ConvTrunkEncoder),
    "attn_trunk": (AttnTrunkEncoderConfig, AttnTrunkEncoder),
    "hybrid_trunk": (HybridTrunkEncoderConfig, HybridTrunkEncoder),
    "lead_factorized": (LeadFactorizedEncoderConfig, LeadFactorizedEncoder),
    # Amendment 14, eighth addendum's stage-2 push variants (winder.jepa.encoder's "STAGE-2 push
    # variants" section), each composable with the crowned conv-trunk base via `dilations`
    # (CE1 = [1, 2], D3 = [1, 2, 4]). gdn_trunk, on record: expected to underperform on probes
    # if the gate fails to learn a short memory (unbounded-context poison, measured); that arm's
    # primary readout is the learned memory length (GdnTrunkEncoder.effective_memory_tokens).
    "dct_stem_conv_trunk": (DctStemConvTrunkEncoderConfig, DctStemConvTrunkEncoder),
    "multiscale_stem_conv_trunk": (
        MultiscaleStemConvTrunkEncoderConfig,
        MultiscaleStemConvTrunkEncoder,
    ),
    "windowed_hybrid_trunk": (WindowedHybridTrunkEncoderConfig, WindowedHybridTrunkEncoder),
    "gdn_trunk": (GdnTrunkEncoderConfig, GdnTrunkEncoder),
}
PROJECTION_HEAD_REGISTRY: dict[str, tuple[type, Callable[[Any], ProjectionHead]]] = {
    "mlp": (MlpProjectionHeadConfig, MlpProjectionHead),
    # Amendment 12's arm C5: a NEW class (winder.jepa.projector.Mlp3ProjectionHead), not a config
    # knob on "mlp" -- see that class's own docstring.
    "mlp3": (Mlp3ProjectionHeadConfig, Mlp3ProjectionHead),
}
PREDICTOR_REGISTRY: dict[str, tuple[type, Callable[[Any], Predictor]]] = {
    "transformer": (TransformerPredictorConfig, TransformerPredictor),
}

MASK_SAMPLER_REGISTRY: dict[str, tuple[type, Callable[[Any], MaskSampler]]] = {
    "causal_block": (CausalBlockMaskSamplerConfig, CausalBlockMaskSampler),
}
PREDICTION_LOSS_REGISTRY: dict[str, tuple[type, Callable[[Any], PredictionLoss]]] = {
    "mse": (MsePredictionLossConfig, MsePredictionLoss),
}

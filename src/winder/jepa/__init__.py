"""JEPA pretraining: six single-concern primitives composed by `build_jepa` into a `JepaModel`.

See `base.py` for the six contracts (encoder, projector, predictor, mask sampler, prediction
loss, regularizer) and `model.py` for the composition root that validates and wires them.
"""

from winder.jepa import checkpoint as checkpoint
from winder.jepa.base import (
    Encoder,
    MaskSampler,
    PredictionLoss,
    Predictor,
    ProjectionHead,
    Regularizer,
)
from winder.jepa.dataset import EcgWindowDataset, EcgWindowItem
from winder.jepa.encoder import (
    AttnTrunkEncoder,
    AttnTrunkEncoderConfig,
    ChannelLayerNorm,
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
    ResidualBlock1d,
    ResidualCnnEncoder,
    ResidualCnnEncoderConfig,
    WindowedHybridTrunkEncoder,
    WindowedHybridTrunkEncoderConfig,
)
from winder.jepa.losses import MsePredictionLoss, MsePredictionLossConfig
from winder.jepa.masking import CausalBlockMaskSampler, CausalBlockMaskSamplerConfig, CausalMaskPlan
from winder.jepa.model import JepaConfig, JepaModel, assemble_jepa, build_jepa, load_jepa_config
from winder.jepa.predictor import (
    RelativePositionBias,
    TransformerPredictor,
    TransformerPredictorConfig,
)
from winder.jepa.projector import (
    Mlp3ProjectionHead,
    Mlp3ProjectionHeadConfig,
    MlpProjectionHead,
    MlpProjectionHeadConfig,
)
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
from winder.jepa.regularizers import NoRegularizer, NoRegularizerConfig, SigReg, SigRegConfig
from winder.jepa.seeded_dropout import SeededDropout

__all__ = [
    "checkpoint",
    "EcgWindowDataset",
    "EcgWindowItem",
    "Encoder",
    "ProjectionHead",
    "Predictor",
    "MaskSampler",
    "PredictionLoss",
    "Regularizer",
    "ChannelLayerNorm",
    "ResidualBlock1d",
    "ResidualCnnEncoder",
    "ResidualCnnEncoderConfig",
    "PatchEncoder",
    "PatchEncoderConfig",
    "ConvTrunkEncoder",
    "ConvTrunkEncoderConfig",
    "AttnTrunkEncoder",
    "AttnTrunkEncoderConfig",
    "HybridTrunkEncoder",
    "HybridTrunkEncoderConfig",
    "LeadFactorizedEncoder",
    "LeadFactorizedEncoderConfig",
    "DctStemConvTrunkEncoder",
    "DctStemConvTrunkEncoderConfig",
    "MultiscaleStemConvTrunkEncoder",
    "MultiscaleStemConvTrunkEncoderConfig",
    "WindowedHybridTrunkEncoder",
    "WindowedHybridTrunkEncoderConfig",
    "GdnTrunkEncoder",
    "GdnTrunkEncoderConfig",
    "MlpProjectionHead",
    "MlpProjectionHeadConfig",
    "Mlp3ProjectionHead",
    "Mlp3ProjectionHeadConfig",
    "RelativePositionBias",
    "TransformerPredictor",
    "TransformerPredictorConfig",
    "SeededDropout",
    "SigReg",
    "SigRegConfig",
    "NoRegularizer",
    "NoRegularizerConfig",
    "MsePredictionLoss",
    "MsePredictionLossConfig",
    "CausalBlockMaskSampler",
    "CausalBlockMaskSamplerConfig",
    "CausalMaskPlan",
    "ENCODER_REGISTRY",
    "PROJECTION_HEAD_REGISTRY",
    "PREDICTOR_REGISTRY",
    "MASK_SAMPLER_REGISTRY",
    "PREDICTION_LOSS_REGISTRY",
    "REGULARIZER_REGISTRY",
    "build_from_registry",
    "resolve_sub_config",
    "JepaConfig",
    "load_jepa_config",
    "JepaModel",
    "assemble_jepa",
    "build_jepa",
]

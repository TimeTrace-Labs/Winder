"""`JepaConfig` and the composition root: `assemble_jepa` (validate + wire six already-constructed
primitives) and `build_jepa` (resolve a config through the six registries, then delegate to
`assemble_jepa`).

No target encoder, no stop-gradient, no EMA. This MVP's six primitives, taken together, imply one
causal forward pass through a single shared `Encoder`, called once on the unmasked waveform
(`winder.jepa.train`'s module docstring explains why a second, raw-waveform-masked pass is
unnecessary once the encoder is causal), followed by a single shared `ProjectionHead`;
`PredictionLoss` scores the `Predictor`'s guess against the un-detached projected target. That is
the honest LeJEPA reading (its whole pitch is heuristic-free), and it has a direct consequence: the
prediction loss alone admits the trivial constant-embedding collapse solution, so the
`Regularizer` is the *only* thing standing between this MVP and that collapse. This is intentional,
not an oversight -- see the plan's "Remaining risks".

`assemble_jepa` takes six already-constructed primitive instances rather than a config to
resolve, so a test can hand it a deliberately-mismatched double (e.g. a fake `Encoder` with the
wrong `latent_width`) without touching any registry. `build_jepa` is the registry-resolving
convenience wrapper actual configs use; it does no validation of its own; all of that lives in
`assemble_jepa`.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from torch import nn

from winder.jepa.base import (
    Encoder,
    MaskSampler,
    PredictionLoss,
    Predictor,
    ProjectionHead,
    Regularizer,
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

__all__ = ["JepaConfig", "load_jepa_config", "JepaModel", "assemble_jepa", "build_jepa"]


@dataclass
class JepaConfig:
    """Root config: fixed dimensions plus one tag+params pair per primitive, mirroring
    `winder.config.ArmConfig`'s `operator_name`/`operator` pattern six times over.

    `lambda_sig` is not `MISSING`: unlike a hyperparameter with no defensible default, the design
    spec gives both a working value (0.1) and a concrete sweep+selection procedure, so leaving it
    un-defaultable would just force every caller to repeat the same number. `lambda_sig=0` with
    `regularizer_name="sigreg"` is a legal, *required* configuration (the collapse-control
    ablation) -- `assemble_jepa` does not reject it.
    """

    n_leads: int = 12
    n_samples: int = 1000
    n_tokens: int = 250

    lambda_sig: float = 0.1
    seed_pretrain: int = 0
    seed_probe: int = 0

    encoder_name: str = MISSING
    encoder: dict[str, Any] = field(default_factory=dict)
    projector_name: str = MISSING
    projector: dict[str, Any] = field(default_factory=dict)
    predictor_name: str = MISSING
    predictor: dict[str, Any] = field(default_factory=dict)
    mask_sampler_name: str = MISSING
    mask_sampler: dict[str, Any] = field(default_factory=dict)
    prediction_loss_name: str = MISSING
    prediction_loss: dict[str, Any] = field(default_factory=dict)
    regularizer_name: str = MISSING
    regularizer: dict[str, Any] = field(default_factory=dict)


def load_jepa_config(*sources: str | dict[str, Any]) -> DictConfig:
    """Merge YAML paths / dicts, in order, onto `JepaConfig`'s schema. Mirrors
    `winder.config.load_arm_config`."""
    merged = OmegaConf.structured(JepaConfig)
    for source in sources:
        overlay = OmegaConf.load(source) if isinstance(source, str) else OmegaConf.create(source)
        merged = OmegaConf.merge(merged, overlay)
    return cast(DictConfig, merged)


class JepaModel(nn.Module):
    """The assembled JEPA. Not itself swappable -- only its six ingredients are, so this class
    carries no ABC of its own. Construct via `assemble_jepa`/`build_jepa`, not this
    `__init__` directly, so the handshake always runs first."""

    def __init__(
        self,
        config: "JepaConfig | DictConfig",
        encoder: Encoder,
        projector: ProjectionHead,
        predictor: Predictor,
        mask_sampler: MaskSampler,
        prediction_loss: PredictionLoss,
        regularizer: Regularizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.mask_sampler = mask_sampler
        self.prediction_loss = prediction_loss
        self.regularizer = regularizer

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, n_leads, n_samples) -> (B, n_tokens, latent_width): the frozen-encoder eval
        surface -- a LOCAL, context-free descriptor (the probe repointing: under `PatchEncoder` this
        carries no temporal context at all, unlike the retired `ResidualCnnEncoder`'s 1.13s
        window). Projector and predictor are both discarded here -- this method never touches
        either. See `predictor_hidden_states` for the eval surface that DOES carry context.
        """
        return self.encoder.forward(waveform)

    def predictor_hidden_states(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, n_leads, n_samples) -> (B, n_tokens, width): the predictor's own causal hidden
        state at every position, from a single UNMASKED forward pass -- no learned `mask_token`
        substitution anywhere, valid because the predictor's own causal attention mask already
        guarantees position `t`'s output depends only on tokens `<= t` (CM-02), so no masking is
        needed to keep this causal. Shared by the probe repointing and the anomaly score, since
        neither existed before this MVP needed to read the predictor's hidden state directly
        rather than only its masked-target-block output: the probe's own pooling target moved
        here (from `embed`'s local, context-free encoder output), and MVP piece 4's
        prediction-residual anomaly score reads this at every position instead of only the
        training-time target token.
        """
        projected = self.projector.forward(self.encoder.forward(waveform))
        b, n_tokens, _ = projected.shape
        no_mask = torch.zeros(b, n_tokens, dtype=torch.bool, device=projected.device)
        return self.predictor.forward(projected, no_mask)


def assemble_jepa(
    config: "JepaConfig | DictConfig",
    encoder: Encoder,
    projector: ProjectionHead,
    predictor: Predictor,
    mask_sampler: MaskSampler,
    prediction_loss: PredictionLoss,
    regularizer: Regularizer,
    *,
    generator: torch.Generator,
) -> JepaModel:
    """Validate that six already-constructed primitives agree with each other and with `config`,
    then wire them into a `JepaModel`. Raises immediately with an actionable, named-field message
    on any mismatch -- the same style as `winder.data.folds._check_seal_invariant` -- rather than
    failing three layers deep in a forward pass with a bare shape error.

    Every check below exercises the primitives' *actual* behaviour on a dummy input (a dummy
    forward pass, a dummy mask draw, a dummy loss/regularizer call), not a re-derived formula: an
    encoder whose real stride arithmetic doesn't match `config.n_tokens` is caught here, not
    three layers into a training run.
    """
    if encoder.latent_width != projector.input_width:
        raise ValueError(
            f"encoder.latent_width={encoder.latent_width} != projector.input_width="
            f"{projector.input_width}. Fix the projector's input_width (or the encoder's "
            f"latent_width) so the two agree."
        )
    if projector.output_width != predictor.width:
        raise ValueError(
            f"projector.output_width={projector.output_width} != predictor.width="
            f"{predictor.width}. Fix one of the two configs so they agree."
        )

    n_leads = int(config.n_leads)
    n_samples = int(config.n_samples)
    n_tokens = int(config.n_tokens)

    dummy_waveform = torch.zeros(1, n_leads, n_samples)
    tokens = encoder.forward(dummy_waveform)
    if tokens.ndim != 3 or tokens.shape[1] != n_tokens or tokens.shape[2] != encoder.latent_width:
        raise ValueError(
            f"encoder produced tokens of shape {tuple(tokens.shape)} from a dummy "
            f"(1, {n_leads}, {n_samples}) waveform; expected (1, {n_tokens}, "
            f"{encoder.latent_width}). This checks the encoder's actual stride arithmetic "
            f"against config.n_tokens={n_tokens}, not a re-derived formula -- fix the encoder's "
            f"stride/kernel configuration or config.n_tokens so they agree."
        )

    projected = projector.forward(tokens)
    if tuple(projected.shape) != (1, n_tokens, projector.output_width):
        raise ValueError(
            f"projector produced shape {tuple(projected.shape)} from {tuple(tokens.shape)} "
            f"tokens; expected (1, {n_tokens}, {projector.output_width})."
        )

    plan = mask_sampler(1, n_tokens, generator=generator)
    if tuple(plan.context.shape) != (1, n_tokens) or plan.context.dtype != torch.bool:
        raise ValueError(
            f"mask_sampler produced a context mask of shape {tuple(plan.context.shape)} dtype "
            f"{plan.context.dtype}; expected (1, {n_tokens}) bool."
        )
    if tuple(plan.target.shape) != (1, n_tokens) or plan.target.dtype != torch.bool:
        raise ValueError(
            f"mask_sampler produced a target mask of shape {tuple(plan.target.shape)} dtype "
            f"{plan.target.dtype}; expected (1, {n_tokens}) bool."
        )

    predicted = predictor.forward(projected, ~plan.context)
    if tuple(predicted.shape) != (1, n_tokens, predictor.width):
        raise ValueError(
            f"predictor produced shape {tuple(predicted.shape)} from context of shape "
            f"{tuple(projected.shape)}; expected (1, {n_tokens}, {predictor.width})."
        )

    loss = prediction_loss(predicted, projected, plan.target)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise ValueError(
            f"prediction_loss returned a non-scalar or non-finite value ({loss!r}) on the "
            f"dummy forward pass."
        )

    dummy_z = torch.randn(64, projector.output_width, generator=generator)
    reg = regularizer(dummy_z, generator=generator)
    if reg.ndim != 0 or not bool(torch.isfinite(reg)):
        raise ValueError(
            f"regularizer returned a non-scalar or non-finite value ({reg!r}) on a dummy "
            f"(64, {projector.output_width}) input."
        )

    return JepaModel(
        config, encoder, projector, predictor, mask_sampler, prediction_loss, regularizer
    )


def build_jepa(config: "JepaConfig | DictConfig", *, generator: torch.Generator) -> JepaModel:
    """Resolve each of `config`'s six tagged sub-configs through its registry, construct the six
    primitives, then hand them to `assemble_jepa` for validation and wiring. An unregistered tag
    raises `KeyError` (from the registry lookup, not from this function)."""
    encoder_cfg = resolve_sub_config(ENCODER_REGISTRY, config.encoder_name, config.encoder)
    encoder = build_from_registry(ENCODER_REGISTRY, config.encoder_name, encoder_cfg)

    projector_cfg = resolve_sub_config(
        PROJECTION_HEAD_REGISTRY, config.projector_name, config.projector
    )
    projector = build_from_registry(PROJECTION_HEAD_REGISTRY, config.projector_name, projector_cfg)

    predictor_cfg = resolve_sub_config(PREDICTOR_REGISTRY, config.predictor_name, config.predictor)
    predictor = build_from_registry(PREDICTOR_REGISTRY, config.predictor_name, predictor_cfg)

    mask_cfg = resolve_sub_config(
        MASK_SAMPLER_REGISTRY, config.mask_sampler_name, config.mask_sampler
    )
    mask_sampler = build_from_registry(MASK_SAMPLER_REGISTRY, config.mask_sampler_name, mask_cfg)

    loss_cfg = resolve_sub_config(
        PREDICTION_LOSS_REGISTRY, config.prediction_loss_name, config.prediction_loss
    )
    prediction_loss = build_from_registry(
        PREDICTION_LOSS_REGISTRY, config.prediction_loss_name, loss_cfg
    )

    reg_cfg = resolve_sub_config(REGULARIZER_REGISTRY, config.regularizer_name, config.regularizer)
    regularizer = build_from_registry(REGULARIZER_REGISTRY, config.regularizer_name, reg_cfg)

    return assemble_jepa(
        config,
        encoder,
        projector,
        predictor,
        mask_sampler,
        prediction_loss,
        regularizer,
        generator=generator,
    )

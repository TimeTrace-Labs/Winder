"""Prediction losses: `MsePredictionLoss`, `PredictionLoss`'s only MVP implementation.

SIGReg (the `Regularizer` implementation) lives in `regularizers.py`, not here -- the two are
deliberately separate registries so either can be swapped independently, per this project's
"single concern, composable" primitives.
"""

from dataclasses import dataclass

import torch

from winder.jepa.base import PredictionLoss

__all__ = ["MsePredictionLossConfig", "MsePredictionLoss"]


@dataclass
class MsePredictionLossConfig:
    """No-op: MSE has no hyperparameters of its own."""


class MsePredictionLoss(PredictionLoss):
    """Masked mean-squared error between predicted and target token embeddings (spec Sec 10).

    The mean is taken over the embedding dimension first, then over masked positions only:
    `((predicted - target) ** 2).mean(dim=-1)[mask].mean()`. `target` is never detached by this
    loss -- see `winder.jepa.model`'s module docstring for why that is deliberate, not an
    oversight: there is no target encoder or stop-gradient anywhere in this MVP, so SIGReg is the
    only thing preventing the trivial constant-embedding solution this loss alone would admit.
    """

    def __init__(self, config: MsePredictionLossConfig) -> None:
        self.config = config

    def __call__(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted shape {tuple(predicted.shape)} != target shape {tuple(target.shape)}"
            )
        if mask.shape != predicted.shape[:-1]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match predicted's leading dims "
                f"{tuple(predicted.shape[:-1])}"
            )
        if not bool(mask.any()):
            raise ValueError(
                "mask has no masked positions; nothing to compute a prediction loss over "
                "(spec: an empty prediction mask is an invalid-run condition)"
            )
        per_token = (predicted - target).square().mean(dim=-1)
        # `mask` comes from the mask sampler, which has no device concept of its own (CPU
        # tensors throughout) -- move it onto predicted's device before the boolean-index below,
        # same reasoning as winder.jepa.predictor.TransformerPredictor.forward's own fix.
        return per_token[mask.to(per_token.device)].mean()

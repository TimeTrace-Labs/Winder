import pytest
import torch

from winder.jepa.losses import MsePredictionLoss, MsePredictionLossConfig


def test_perfect_prediction_gives_zero_loss() -> None:
    loss_fn = MsePredictionLoss(MsePredictionLossConfig())
    target = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, :2] = True
    loss = loss_fn(target.clone(), target, mask)
    assert torch.allclose(loss, torch.zeros(()))


def test_loss_only_scores_masked_positions() -> None:
    loss_fn = MsePredictionLoss(MsePredictionLossConfig())
    target = torch.zeros(1, 4, 2)
    predicted = target.clone()
    predicted[0, 3] = 100.0  # a huge error at an UNMASKED position
    mask = torch.tensor([[True, True, False, False]])
    loss = loss_fn(predicted, target, mask)
    assert torch.allclose(loss, torch.zeros(()))


def test_shape_mismatch_raises() -> None:
    loss_fn = MsePredictionLoss(MsePredictionLossConfig())
    with pytest.raises(ValueError, match="!="):
        loss_fn(torch.zeros(1, 4, 2), torch.zeros(1, 4, 3), torch.zeros(1, 4, dtype=torch.bool))


def test_mask_shape_mismatch_raises() -> None:
    loss_fn = MsePredictionLoss(MsePredictionLossConfig())
    with pytest.raises(ValueError, match="mask shape"):
        loss_fn(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2), torch.zeros(1, 3, dtype=torch.bool))


def test_empty_mask_raises() -> None:
    loss_fn = MsePredictionLoss(MsePredictionLossConfig())
    with pytest.raises(ValueError, match="no masked positions"):
        loss_fn(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2), torch.zeros(1, 4, dtype=torch.bool))


def test_gradients_reach_predicted() -> None:
    loss_fn = MsePredictionLoss(MsePredictionLossConfig())
    predicted = torch.randn(1, 4, 2, requires_grad=True)
    target = torch.randn(1, 4, 2)
    mask = torch.tensor([[True, False, True, False]])
    loss = loss_fn(predicted, target, mask)
    loss.backward()
    assert predicted.grad is not None
    assert torch.any(predicted.grad[0, 0] != 0)
    assert torch.all(predicted.grad[0, 1] == 0)  # unmasked position gets no gradient

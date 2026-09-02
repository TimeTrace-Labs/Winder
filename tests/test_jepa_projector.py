import pytest
import torch
from torch import nn

from winder.jepa.base import ProjectionHead
from winder.jepa.projector import MlpProjectionHead, MlpProjectionHeadConfig


def test_output_shape() -> None:
    proj = MlpProjectionHead(MlpProjectionHeadConfig())
    out = proj.forward(torch.randn(4, 250, 256))
    assert out.shape == (4, 250, 256)


def test_widths_configurable_and_independent_of_the_default() -> None:
    proj = MlpProjectionHead(
        MlpProjectionHeadConfig(input_width=128, hidden_width=64, output_width=32)
    )
    assert proj.input_width == 128
    assert proj.output_width == 32
    out = proj.forward(torch.randn(2, 10, 128))
    assert out.shape == (2, 10, 32)


def test_satisfies_the_projection_head_contract() -> None:
    proj = MlpProjectionHead(MlpProjectionHeadConfig())
    assert isinstance(proj, ProjectionHead)


def test_no_normalization_or_dropout_anywhere() -> None:
    proj = MlpProjectionHead(MlpProjectionHeadConfig())
    modules = list(proj.modules())
    assert not any(isinstance(m, nn.LayerNorm | nn.BatchNorm1d) for m in modules)
    assert not any(isinstance(m, nn.Dropout) for m in modules)


def test_wrong_input_width_raises() -> None:
    proj = MlpProjectionHead(MlpProjectionHeadConfig(input_width=256))
    with pytest.raises(ValueError, match="input_width"):
        proj.forward(torch.randn(1, 5, 64))


def test_gradients_flow_through() -> None:
    proj = MlpProjectionHead(MlpProjectionHeadConfig())
    tokens = torch.randn(1, 4, 256, requires_grad=True)
    proj.forward(tokens).sum().backward()
    assert tokens.grad is not None
    assert torch.any(tokens.grad != 0)

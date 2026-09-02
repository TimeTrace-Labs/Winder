import pytest
import torch
from torch import nn

from winder.jepa.base import Predictor
from winder.jepa.predictor import (
    RelativePositionBias,
    TransformerPredictor,
    TransformerPredictorConfig,
)
from winder.jepa.seeded_dropout import SeededDropout


def test_relative_bias_clamped_beyond_max_distance() -> None:
    """table starts at all-zero, which would make any clamping bug invisible (every entry would
    trivially equal every other). Fill it with distinct values first so the assertions actually
    exercise the clamp, not the initializer."""
    bias_module = RelativePositionBias(n_heads=2, max_distance=4)
    with torch.no_grad():
        bias_module.table.copy_(torch.arange(2 * 9).reshape(2, 9).float())
    bias = bias_module.forward(20, device=torch.device("cpu"))
    assert bias.shape == (1, 2, 20, 20)
    # i=19, j=0 -> i-j=19, clamped to max_distance=4 -- must equal the i-j=4 entry exactly.
    assert torch.equal(bias[0, :, 19, 0], bias[0, :, 5, 1])  # i-j=19 vs. i-j=5-1=4
    # i=0, j=19 -> i-j=-19, clamped to -4 -- must equal the i-j=-4 entry exactly.
    assert torch.equal(bias[0, :, 0, 19], bias[0, :, 1, 5])  # i-j=-19 vs. i-j=1-5=-4
    # In-range offsets are NOT collapsed together.
    assert not torch.equal(bias[0, :, 5, 4], bias[0, :, 6, 4])  # i-j=1 vs. i-j=2


def test_relative_bias_table_size_matches_two_max_distance_plus_one() -> None:
    bias_module = RelativePositionBias(n_heads=3, max_distance=64)
    assert bias_module.table.shape == (3, 129)


def test_predictor_output_shape() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig())
    z_ctx = torch.randn(2, 250, 256)
    mask = torch.zeros(2, 250, dtype=torch.bool)
    mask[:, 100:120] = True
    out = predictor.forward(z_ctx, mask)
    assert out.shape == (2, 250, 256)


def test_satisfies_the_predictor_contract() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig())
    assert isinstance(predictor, Predictor)
    assert predictor.width == 256


def test_no_absolute_positional_embedding_parameter() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig())
    names = [name for name, _ in predictor.named_parameters()]
    assert not any("pos_emb" in n or "position_embedding" in n for n in names)


def test_uses_seeded_dropout_not_plain_nn_dropout() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig())
    modules = list(predictor.modules())
    assert not any(type(m) is nn.Dropout for m in modules)
    assert any(isinstance(m, SeededDropout) for m in modules)


def test_mask_token_and_bias_table_receive_gradients() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig())
    z_ctx = torch.randn(1, 250, 256, requires_grad=True)
    mask = torch.zeros(1, 250, dtype=torch.bool)
    mask[:, 5:10] = True
    predictor.forward(z_ctx, mask).sum().backward()
    assert predictor.mask_token.grad is not None
    assert torch.any(predictor.mask_token.grad != 0)
    assert predictor.rel_bias.table.grad is not None


def test_wrong_width_raises() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig(width=256))
    with pytest.raises(ValueError, match="must be"):
        predictor.forward(torch.randn(1, 10, 64), torch.zeros(1, 10, dtype=torch.bool))


def test_mask_shape_mismatch_raises() -> None:
    predictor = TransformerPredictor(TransformerPredictorConfig())
    with pytest.raises(ValueError, match="mask shape"):
        predictor.forward(torch.randn(1, 250, 256), torch.zeros(1, 100, dtype=torch.bool))


def test_causal_a_later_position_never_influences_an_earlier_prediction() -> None:
    """CM-02: a masked position at the START of the sequence must be exactly unaffected by a
    perturbation to a LATER unmasked position -- the inverse of the old
    `test_bidirectional_not_causal`, which asserted precisely the leak this predictor now forbids
    (see `tests/test_stage0_contracts.py`'s former `test_cm02_...` xfail marker, now deleted
    because this is exactly what it demanded)."""
    predictor = TransformerPredictor(TransformerPredictorConfig())
    predictor.eval()
    z_ctx = torch.randn(1, 250, 256)
    mask = torch.zeros(1, 250, dtype=torch.bool)
    mask[:, 0] = True
    with torch.no_grad():
        baseline = predictor.forward(z_ctx, mask)
        perturbed = z_ctx.clone()
        perturbed[0, 200] += 10.0
        out = predictor.forward(perturbed, mask)
    torch.testing.assert_close(baseline[0, 0], out[0, 0], rtol=0.0, atol=0.0)


def test_causal_an_earlier_position_does_influence_a_later_prediction() -> None:
    """Sanity check on the test above: the causal mask must not have accidentally zeroed out
    genuine dependence in the allowed direction. The measured effect is small (~1e-7) --
    softmax dilutes one perturbed token's contribution over 250 positions -- but real."""
    predictor = TransformerPredictor(TransformerPredictorConfig())
    predictor.eval()
    z_ctx = torch.randn(1, 250, 256)
    mask = torch.zeros(1, 250, dtype=torch.bool)
    mask[:, 200] = True
    with torch.no_grad():
        baseline = predictor.forward(z_ctx, mask)
        perturbed = z_ctx.clone()
        perturbed[0, 50] += 10.0
        out = predictor.forward(perturbed, mask)
    assert (baseline[0, 200] - out[0, 200]).abs().max() > 1e-9


def test_width_not_divisible_by_heads_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        TransformerPredictor(TransformerPredictorConfig(width=250, n_heads=4))

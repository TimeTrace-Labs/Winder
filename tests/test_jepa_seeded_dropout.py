import pytest
import torch

from winder.jepa.seeded_dropout import SeededDropout


def test_reproducibility_same_seed_gives_identical_output() -> None:
    a = SeededDropout(0.5, seed=42)
    b = SeededDropout(0.5, seed=42)
    a.train()
    b.train()
    x = torch.ones(1000)
    assert torch.equal(a.forward(x), b.forward(x))


def test_consecutive_calls_on_one_instance_differ() -> None:
    d = SeededDropout(0.5, seed=0)
    d.train()
    x = torch.ones(1000)
    first = d.forward(x)
    second = d.forward(x)
    assert not torch.equal(first, second)


def test_expectation_scaling() -> None:
    """Monte Carlo mean over many draws converges to the un-dropped input, confirming the
    1/keep_prob scaling is correct."""
    d = SeededDropout(0.3, seed=0)
    d.train()
    x = torch.ones(10_000)
    total = torch.zeros(10_000)
    n = 200
    for _ in range(n):
        total += d.forward(x)
    mean_over_draws = (total / n).mean()
    assert mean_over_draws.item() == pytest.approx(1.0, abs=0.05)


def test_eval_mode_is_exact_bitwise_identity() -> None:
    d = SeededDropout(0.5, seed=0)
    d.eval()
    x = torch.randn(100)
    assert torch.equal(d.forward(x), x)


def test_zero_probability_is_exact_identity_even_in_training_mode() -> None:
    d = SeededDropout(0.0, seed=0)
    d.train()
    x = torch.randn(100)
    assert torch.equal(d.forward(x), x)


def test_checkpointed_rng_continuation() -> None:
    """Saving and restoring the generator's state across a checkpoint boundary reproduces the
    same subsequent dropout sequence as an uninterrupted run."""
    uninterrupted = SeededDropout(0.5, seed=7)
    uninterrupted.train()
    x = torch.ones(500)
    uninterrupted.forward(x)  # advance state once
    expected_next = uninterrupted.forward(x)  # the draw we want to reproduce after a "restart"

    before_checkpoint = SeededDropout(0.5, seed=7)
    before_checkpoint.train()
    before_checkpoint.forward(x)  # advance state to the same point as `uninterrupted`
    checkpoint = before_checkpoint.state_dict()

    restored = SeededDropout(0.5, seed=999)  # different seed -- load_state_dict must override it
    restored.load_state_dict(checkpoint)
    restored.train()
    actual_next = restored.forward(x)

    assert torch.equal(expected_next, actual_next)


def test_invalid_probability_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        SeededDropout(1.0, seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        SeededDropout(-0.1, seed=0)

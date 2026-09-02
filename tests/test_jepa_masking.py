import pytest
import torch

from winder.jepa.masking import CausalBlockMaskSampler, CausalBlockMaskSamplerConfig


def test_target_index_is_exactly_one_past_the_cutoff() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        plan = sampler(4, 125, generator=gen)
        for b in range(4):
            target_indices = plan.target[b].nonzero().flatten()
            assert target_indices.numel() == 1
            assert int(target_indices[0]) == int(plan.cutoff[b]) + 1


def test_sampled_cutoff_is_within_configured_bounds() -> None:
    cfg = CausalBlockMaskSamplerConfig(c_min=3)
    sampler = CausalBlockMaskSampler(cfg)
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        plan = sampler(8, 125, generator=gen)
        assert bool((plan.cutoff >= cfg.c_min).all())
        assert bool((plan.cutoff <= 125 - 2).all())


def test_context_mask_is_exactly_the_prefix_up_to_cutoff() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen = torch.Generator().manual_seed(0)
    plan = sampler(4, 125, generator=gen)
    for b in range(4):
        c = int(plan.cutoff[b])
        assert bool(plan.context[b, : c + 1].all())
        assert not bool(plan.context[b, c + 1 :].any())


def test_masks_independent_per_record_in_a_batch() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen = torch.Generator().manual_seed(0)
    plan = sampler(8, 125, generator=gen)
    assert plan.context.shape == (8, 125)
    assert plan.target.shape == (8, 125)
    assert not bool((plan.cutoff == plan.cutoff[0]).all())


def test_masks_differ_across_calls_on_the_same_generator() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen = torch.Generator().manual_seed(0)
    first = sampler(4, 125, generator=gen)
    second = sampler(4, 125, generator=gen)
    assert not bool(torch.equal(first.cutoff, second.cutoff))


def test_same_seed_gives_identical_plans() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    plan_a = sampler(4, 125, generator=gen_a)
    plan_b = sampler(4, 125, generator=gen_b)
    assert torch.equal(plan_a.context, plan_b.context)
    assert torch.equal(plan_a.target, plan_b.target)
    assert torch.equal(plan_a.cutoff, plan_b.cutoff)


def test_dtype_and_shape() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig())
    gen = torch.Generator().manual_seed(0)
    plan = sampler(3, 125, generator=gen)
    assert plan.context.dtype == torch.bool
    assert plan.target.dtype == torch.bool
    assert plan.context.shape == (3, 125)
    assert plan.target.shape == (3, 125)
    assert plan.cutoff.dtype == torch.int64


def test_infeasible_n_tokens_raises() -> None:
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig(c_min=99))
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="cannot fit"):
        sampler(1, 100, generator=gen)


def test_negative_c_min_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="c_min"):
        CausalBlockMaskSampler(CausalBlockMaskSamplerConfig(c_min=-1))


def test_c_min_of_zero_allows_the_first_token_as_sole_context() -> None:
    """No receptive-field run-in under PatchEncoder (architecture-primer.html §5-6), so unlike
    the retired gap/length sampler's own CM-07 floor, c_min=0 is a legal default -- token 0 is
    exactly as
    well-supported as any other."""
    sampler = CausalBlockMaskSampler(CausalBlockMaskSamplerConfig(c_min=0))
    gen = torch.Generator().manual_seed(0)
    plan = sampler(4, 125, generator=gen)
    assert bool((plan.cutoff >= 0).all())

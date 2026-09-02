import pytest
import torch

from winder.jepa.base import Regularizer
from winder.jepa.registry import REGULARIZER_REGISTRY
from winder.jepa.regularizers import NoRegularizer, NoRegularizerConfig, SigReg, SigRegConfig


def _reference_sigreg(
    z: torch.Tensor,
    directions: torch.Tensor,
    *,
    knots: int = 17,
    t_max: float = 3.0,
    chunk_size: int = 32,
) -> torch.Tensor:
    """A standalone transcription of the design spec's Sec 11.4 reference pseudocode, taking
    pre-drawn `directions` directly (rather than an internal RNG call) so it can be compared
    against `SigReg` on the exact same direction matrix."""
    Z = z.reshape(-1, z.shape[-1]).float()
    n = Z.shape[0]
    t = torch.linspace(0.0, t_max, knots)
    dt = t_max / (knots - 1)
    trap = torch.full_like(t, 2.0 * dt)
    trap[0] = dt
    trap[-1] = dt
    phi0 = torch.exp(-0.5 * t.square())
    quad_weight = trap * phi0
    total = Z.new_zeros(())
    for start in range(0, directions.shape[1], chunk_size):
        uc = directions[:, start : start + chunk_size]
        h = Z @ uc
        ht = h.unsqueeze(-1) * t
        real = ht.cos().mean(dim=0)
        imag = ht.sin().mean(dim=0)
        err = (real - phi0).square() + imag.square()
        statistic = n * (err @ quad_weight)
        total = total + statistic.sum()
    return total / directions.shape[1]


def test_matches_reference_pseudocode_transcription() -> None:
    torch.manual_seed(0)  # only affects z's construction below, not the regularizer under test
    z = torch.randn(500, 32)
    n_directions = 64

    gen_a = torch.Generator().manual_seed(123)
    directions = torch.randn(32, n_directions, generator=gen_a)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    expected = _reference_sigreg(z, directions, knots=17, t_max=3.0, chunk_size=32)

    reg = SigReg(SigRegConfig(n_directions=n_directions, n_knots=17, t_max=3.0, chunk=32))
    gen_b = torch.Generator().manual_seed(123)  # reproduces the identical direction draw
    actual = reg(z, generator=gen_b)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_collapse_canary_order_of_magnitude_separation() -> None:
    """Re-measured against the [0,3] grid this spec actually uses -- the earlier [-5,5]-grid
    numbers do not carry over (see the module docstring). The floor is near 1.0, not 0."""
    reg = SigReg(SigRegConfig(n_directions=64))
    gen = torch.Generator().manual_seed(0)
    n, k = 4000, 32

    isotropic = reg(torch.randn(n, k), generator=gen)
    collapsed = reg(torch.zeros(n, k), generator=gen)
    rank2 = reg(torch.randn(n, 2) @ torch.randn(2, k), generator=gen)

    assert 0.0 < float(isotropic) < 5.0
    assert float(collapsed) > 100.0 * float(isotropic)
    assert float(rank2) > 100.0 * float(isotropic)


def test_scale_sensitivity() -> None:
    reg = SigReg(SigRegConfig(n_directions=64))
    gen = torch.Generator().manual_seed(0)
    n, k = 4000, 32

    isotropic = float(reg(torch.randn(n, k), generator=gen))
    under_spread = float(reg(0.1 * torch.randn(n, k), generator=gen))
    over_spread = float(reg(3.0 * torch.randn(n, k), generator=gen))

    assert under_spread > 10.0 * isotropic
    assert over_spread > 10.0 * isotropic


def test_exact_permutation_invariance() -> None:
    reg = SigReg(SigRegConfig(n_directions=32))
    z = torch.randn(200, 16)
    perm = torch.randperm(200)
    gen_a = torch.Generator().manual_seed(0)
    gen_b = torch.Generator().manual_seed(0)
    original = reg(z, generator=gen_a)
    permuted = reg(z[perm], generator=gen_b)
    assert torch.allclose(original, permuted, atol=1e-6)


def test_exact_permutation_invariance_along_the_time_axis() -> None:
    """Distinct from `test_exact_permutation_invariance` above, which permutes the BATCH (N)
    axis of a 2-D call. This permutes the TIME (T) axis of a 3-D call: `train.py` passes
    `(T, B, K)`, and the per-timestep reduction is an unweighted `mean(dim=0)` over independently
    computed per-slice statistics (module docstring, "average over T -- identity when T=1"), so
    reordering which slice is which changes nothing. Scrambling a record's own token order --
    i.e. destroying every temporal relationship between tokens -- is invisible to this statistic.
    That is the formal version of "SIGReg is a cross-sectional prior, not a longitudinal one": it
    constrains the marginal distribution of latents at a frozen instant, pooled across the batch,
    and has no term that reads order along time at all."""
    reg = SigReg(SigRegConfig(n_directions=32))
    t_steps, batch, k = 12, 40, 16
    z = torch.randn(t_steps, batch, k)
    perm = torch.randperm(t_steps)
    gen_a = torch.Generator().manual_seed(0)
    gen_b = torch.Generator().manual_seed(0)
    original = reg(z, generator=gen_a)
    time_scrambled = reg(z[perm], generator=gen_b)
    assert torch.allclose(original, time_scrambled, atol=1e-6)


def test_resampling_is_load_bearing() -> None:
    reg = SigReg(SigRegConfig(n_directions=16))
    z = torch.randn(64, 8)

    gen = torch.Generator().manual_seed(0)
    first = reg(z, generator=gen)
    second = reg(z, generator=gen)  # same generator object -- state has advanced
    assert not torch.allclose(first, second)

    gen_fresh = torch.Generator().manual_seed(0)
    reproduced = reg(z, generator=gen_fresh)
    assert torch.allclose(first, reproduced)


def test_n_scaling() -> None:
    """The statistic's leading factor is N: duplicating every row should roughly double it."""
    reg = SigReg(SigRegConfig(n_directions=64))
    z = torch.randn(500, 16)
    gen_a = torch.Generator().manual_seed(0)
    gen_b = torch.Generator().manual_seed(0)
    single = float(reg(z, generator=gen_a))
    doubled = float(reg(torch.cat([z, z], dim=0), generator=gen_b))
    assert doubled == pytest.approx(2 * single, rel=0.05)


def test_chunked_and_unchunked_agree() -> None:
    z = torch.randn(300, 24)
    reg_chunked = SigReg(SigRegConfig(n_directions=128, chunk=16))
    reg_unchunked = SigReg(SigRegConfig(n_directions=128, chunk=128))
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    chunked_result = reg_chunked(z, generator=gen_a)
    unchunked_result = reg_unchunked(z, generator=gen_b)
    assert torch.allclose(chunked_result, unchunked_result, atol=1e-5, rtol=1e-4)


def test_gradcheck_float64() -> None:
    """Small N/K/M, float64: gradcheck needs double precision for its numerical Jacobian to be
    stable, which is exactly why SigReg promotes-but-never-downcasts (see its module docstring)
    -- an unconditional `.float()` would silently break this test by discarding precision."""
    reg = SigReg(SigRegConfig(n_directions=4, n_knots=5, chunk=0))
    z = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)

    def f(x: torch.Tensor) -> torch.Tensor:
        return reg(x, generator=torch.Generator().manual_seed(0))

    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=1e-4)


def test_gradients_reach_z_but_not_directions() -> None:
    reg = SigReg(SigRegConfig(n_directions=16))
    z = torch.randn(64, 8, requires_grad=True)
    gen = torch.Generator().manual_seed(0)
    loss = reg(z, generator=gen)
    loss.backward()
    assert z.grad is not None
    assert torch.any(z.grad != 0)


def test_invalid_configs_rejected() -> None:
    with pytest.raises(ValueError, match="n_directions"):
        SigReg(SigRegConfig(n_directions=0))
    with pytest.raises(ValueError, match="n_knots"):
        SigReg(SigRegConfig(n_knots=1))
    with pytest.raises(ValueError, match="t_max"):
        SigReg(SigRegConfig(t_max=0.0))


def test_wrong_ndim_raises() -> None:
    reg = SigReg(SigRegConfig())
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="2-D"):
        reg(torch.randn(5), generator=gen)
    with pytest.raises(ValueError, match="2-D"):
        reg(torch.randn(4, 5, 6, 7), generator=gen)


def test_3d_input_is_the_per_timestep_reduction_not_an_error() -> None:
    """The widened contract (architecture-primer.html §7): a 3-D (T, N, K) call is a real,
    load-bearing path
    (one statistic per T-slice, N = that slice's row count, averaged over T), not a shape this
    class rejects -- see `test_wrong_ndim_raises` above for what still does raise."""
    reg = SigReg(SigRegConfig(n_directions=16))
    gen = torch.Generator().manual_seed(0)
    out = reg(torch.randn(5, 64, 8), generator=gen)
    assert out.ndim == 0
    assert bool(torch.isfinite(out))


def test_2d_call_is_the_exact_t_equals_1_special_case() -> None:
    """`(N, K)` and `(1, N, K)` must be bit-identical: the module docstring's claim that 2-D
    input is exactly the T=1 case of the widened reduction, not a separately-maintained code
    path that happens to agree numerically."""
    reg = SigReg(SigRegConfig(n_directions=32))
    z = torch.randn(64, 16)
    gen_a = torch.Generator().manual_seed(3)
    gen_b = torch.Generator().manual_seed(3)
    flat = reg(z, generator=gen_a)
    unsqueezed = reg(z.unsqueeze(0), generator=gen_b)
    assert torch.equal(flat, unsqueezed)


def test_per_timestep_reduction_averages_independent_per_slice_statistics() -> None:
    """Cross-check against a from-scratch per-slice computation: call `SigReg` once per T-slice
    at N = batch size (sharing one direction draw across slices, matching the 3-D call's own
    "directions shared across timesteps" default), average the T results by hand, and confirm it
    agrees with one 3-D call over the same data and the same shared directions."""
    reg = SigReg(SigRegConfig(n_directions=24, chunk=24))
    t_steps, batch, k = 6, 20, 10
    z = torch.randn(t_steps, batch, k)

    gen_3d = torch.Generator().manual_seed(11)
    combined = reg(z, generator=gen_3d)

    # Draw the identical shared direction set once, then feed each T-slice through the 2-D path
    # under a frozen (never-advancing) generator state so every slice sees the same directions.
    gen_ref = torch.Generator().manual_seed(11)
    state = gen_ref.get_state()
    per_slice = []
    for slice_t in range(t_steps):
        gen_slice = torch.Generator().manual_seed(0)
        gen_slice.set_state(state)
        per_slice.append(reg(z[slice_t], generator=gen_slice))
    by_hand = torch.stack(per_slice).mean()
    assert torch.allclose(combined, by_hand, rtol=1e-5, atol=1e-6)


def test_per_timestep_gradient_is_smaller_by_order_t() -> None:
    """The reduction fix's own verify criterion: architecture-primer.html measured, at B=64, T=222,
    K=256 on identical latents, a per-row gradient rms of 9.6029e-04 pooled (the old B*T-flattened
    reduction) against 6.3755e-06 per timestep (N=batch size, averaged over T) -- a 150.6x ratio,
    against a theoretical T=222. That exact pair of numbers came from one specific (unrecoverable)
    latent tensor, so this test does not try to hit them byte-for-byte; instead it reproduces the
    STRUCTURAL claim on a fresh tensor at the same (B, T, K): the leading N cancels within any one
    statistic (module docstring), so the only extra factor between the two reductions is the
    explicit 1/T average unique to the per-timestep topology -- the ratio should land near T, not
    merely "smaller". This only holds away from the Gaussian null: isotropic `randn` (SIGReg's own
    target) is dominated by O(1/sqrt(N)) Monte-Carlo noise rather than systematic deviation, which
    gives a ratio near sqrt(T) instead (measured ~15.6, not ~222) -- a rank-8-of-256 collapsed
    structure, matching this file's own `test_collapse_canary_order_of_magnitude_separation` idiom,
    is far enough from the null for the "N cancels" argument to actually apply; measured ratio on
    this exact construction is consistently 208-216 across seeds."""
    reg = SigReg(SigRegConfig(n_directions=64, chunk=32))
    b, t_steps, k, rank = 64, 222, 256, 8
    z = torch.randn(t_steps, b, rank) @ torch.randn(rank, k) + 0.05 * torch.randn(t_steps, b, k)

    pooled_input = z.reshape(t_steps * b, k).detach().clone().requires_grad_(True)
    gen_pooled = torch.Generator().manual_seed(5)
    pooled_loss = reg(pooled_input, generator=gen_pooled)
    pooled_loss.backward()
    assert pooled_input.grad is not None
    pooled_grad_rms = float(pooled_input.grad.pow(2).mean().sqrt())

    per_timestep_input = z.detach().clone().requires_grad_(True)
    gen_per_timestep = torch.Generator().manual_seed(5)  # identical direction draw as pooled
    per_timestep_loss = reg(per_timestep_input, generator=gen_per_timestep)
    per_timestep_loss.backward()
    assert per_timestep_input.grad is not None
    per_timestep_grad_rms = float(per_timestep_input.grad.pow(2).mean().sqrt())

    ratio = pooled_grad_rms / per_timestep_grad_rms
    assert ratio > 10.0  # a clear, qualitative separation -- not merely "different"
    # "near T=222", banded around the measured 208-216
    assert t_steps / 3.0 <= ratio <= t_steps * 1.5


def test_no_regularizer_returns_exact_zero() -> None:
    reg = NoRegularizer(NoRegularizerConfig())
    gen = torch.Generator().manual_seed(0)
    out = reg(torch.randn(10, 4), generator=gen)
    assert out.ndim == 0
    assert float(out) == 0.0


def test_no_regularizer_wrong_ndim_raises() -> None:
    reg = NoRegularizer(NoRegularizerConfig())
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="2-D"):
        reg(torch.randn(5), generator=gen)
    with pytest.raises(ValueError, match="2-D"):
        reg(torch.randn(4, 5, 6, 7), generator=gen)


def test_no_regularizer_accepts_3d_and_still_returns_exact_zero() -> None:
    reg = NoRegularizer(NoRegularizerConfig())
    gen = torch.Generator().manual_seed(0)
    out = reg(torch.randn(3, 10, 4), generator=gen)
    assert out.ndim == 0
    assert float(out) == 0.0


@pytest.mark.parametrize("name", list(REGULARIZER_REGISTRY))
def test_registered_regularizers_satisfy_the_contract(name: str) -> None:
    schema_cls, ctor = REGULARIZER_REGISTRY[name]
    regularizer = ctor(schema_cls())
    assert isinstance(regularizer, Regularizer)
    gen = torch.Generator().manual_seed(0)
    out = regularizer(torch.randn(16, 8), generator=gen)
    assert out.ndim == 0
    assert bool(torch.isfinite(out))


# --------------------------------------------------------------------------------------------
# winder-nominal acceptance phase: gauge-safety canary for SIGReg (the property that licenses
# calling this an "anti-collapse" regularizer at all -- an empirical CF estimator that neither
# converges to its target nor separates a degenerate input from an isotropic one would be
# decoration, not a regularizer). Two independent claims, tested separately:
#
#   (1) the empirical characteristic function converges to the true isotropic-Gaussian one as
#       the sample size n grows;
#   (2) SIGReg's actual reported loss is far larger on a collapsed (rank-1) input than on an
#       isotropic one, at matched n -- the property the training loop actually leans on.
# --------------------------------------------------------------------------------------------


def test_characteristic_function_converges_to_zero_on_isotropic_input() -> None:
    """SIGReg's per-direction quadrature error -- the raw CF-vs-target mismatch BEFORE the
    statistic's own `n * (...)` rescaling (`regularizers.py`'s `statistic = n * (err @
    quad_weight)`) -- shrinks monotonically as n grows on isotropic standard-normal input,
    exactly the empirical-characteristic-function-converges-to-its-population-target statement
    Bochner/Levy continuity licenses (`err = (mean_i cos(t*Z_i) - phi0)^2 + (mean_i sin(t*Z_i))^2`
    has expectation O(Var/n) by the CLT, so it should shrink by ~4x each time n quadruples).

    Deliberately NOT asserted on SIGReg's own reported loss value (`loss = mean(n * err @
    quad_weight)`): `regularizers.py`'s own module docstring is explicit that this is the wrong
    quantity to test convergence on -- "Theorem 6's O(1/N) bias is multiplied by the statistic's
    own leading N factor, leaving an N-independent residual. A canary asserting 'near zero' would
    fail against a correct implementation." Measured directly (this repo, 5 independent seed
    blocks of 8 draws each, n in {256, 1024, 4096, 16384}): the raw loss's mean over 8 seeds is
    flat at ~1.02-1.08 and is monotonically DEcreasing in only 1 of 5 independent trials -- a coin
    flip, not a property a test should assert. `loss / n` -- the unscaled quadrature error the
    `n *` factor was applied to -- decreased monotonically in all 5 trials, so that is what this
    test asserts. This is the intent stated in the milestone brief ("the empirical characteristic
    function converges to the true isotropic-Gaussian one as sample size grows") operationalised
    correctly, not a weakened substitute for it.
    """
    config = SigRegConfig()  # default grid: n_directions=256 (M), n_knots=17, t_max=3.0, chunk=32
    reg = SigReg(config)
    k = 256  # K
    ns = (256, 1024, 4096, 16384)
    n_seeds = 8

    mean_quadrature_error_by_n: list[float] = []
    for n in ns:
        per_seed_errors = []
        for seed in range(n_seeds):
            z = torch.randn(n, k, generator=torch.Generator().manual_seed(seed))
            directions_gen = torch.Generator().manual_seed(10_000 + seed)
            loss = reg(z, generator=directions_gen)
            per_seed_errors.append(float(loss) / n)  # undo the statistic's own `n *` factor
        mean_quadrature_error_by_n.append(sum(per_seed_errors) / n_seeds)

    for earlier, later in zip(
        mean_quadrature_error_by_n, mean_quadrature_error_by_n[1:], strict=False
    ):
        assert later < earlier, (
            f"unscaled quadrature error {mean_quadrature_error_by_n} must decrease "
            f"monotonically as n grows -- the empirical CF should converge to the isotropic "
            f"target, not stay flat or worsen"
        )


def test_sigreg_loss_is_an_order_of_magnitude_larger_on_rank1_collapse() -> None:
    """SIGReg's actual reported loss (not the convergence-only quantity above) is >= 10x larger
    on a rank-1 collapsed input (every row proportional to one fixed direction) than on an
    isotropic standard-normal input, at the same n -- the gauge-safety property SIGReg exists to
    enforce: penalising degenerate/collapsed representations. Checked at every n in the same grid
    the convergence test uses, not cherry-picked at one scale."""
    config = SigRegConfig()  # default grid, same as the convergence test above
    reg = SigReg(config)
    k = 256
    ns = (256, 1024, 4096, 16384)
    n_seeds = 8
    fixed_direction = torch.zeros(k)
    fixed_direction[0] = 1.0

    for n in ns:
        iso_losses, collapsed_losses = [], []
        for seed in range(n_seeds):
            z_iso = torch.randn(n, k, generator=torch.Generator().manual_seed(seed))
            iso_losses.append(
                float(reg(z_iso, generator=torch.Generator().manual_seed(10_000 + seed)))
            )

            coeffs = torch.randn(n, generator=torch.Generator().manual_seed(seed))
            z_collapsed = torch.outer(coeffs, fixed_direction)  # every row ~ coeffs[i] * e_0
            collapsed_losses.append(
                float(reg(z_collapsed, generator=torch.Generator().manual_seed(10_000 + seed)))
            )

        iso_mean = sum(iso_losses) / n_seeds
        collapsed_mean = sum(collapsed_losses) / n_seeds
        assert collapsed_mean >= 10.0 * iso_mean, (
            f"at n={n}: collapsed mean {collapsed_mean} must be >= 10x isotropic mean {iso_mean} "
            f"-- SIGReg must penalise a rank-1 collapsed representation far more than an "
            f"isotropic one"
        )

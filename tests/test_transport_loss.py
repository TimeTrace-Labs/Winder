import math
from typing import cast

import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.transport.loss import transport_loss

_TOY_K0 = 2
_TOY_N_J = [1, 2, 3]
_TOY_K_J = [1, 2, 1]  # K = 2 + 2*4 = 10


def _toy_cyclic() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_TOY_K0, n_j=_TOY_N_J, k_j=_TOY_K_J))


def _toy_free() -> FreeOperator:
    return FreeOperator(FreeOperatorConfig(k0=_TOY_K0, n_j=_TOY_N_J, k_j=_TOY_K_J))


# =================================================================== shape / input validation


def test_rejects_wrong_z_ndim() -> None:
    operator = _toy_cyclic()
    with pytest.raises(ValueError, match="B, L, K"):
        transport_loss(torch.zeros(5, 10), torch.zeros(5), operator)


def test_rejects_theta_shape_mismatch() -> None:
    operator = _toy_cyclic()
    with pytest.raises(ValueError, match="theta shape"):
        transport_loss(torch.zeros(2, 5, 10), torch.zeros(2, 6), operator)


def test_rejects_dimension_mismatch() -> None:
    operator = _toy_cyclic()  # dimension 10
    with pytest.raises(ValueError, match="operator.dimension"):
        transport_loss(torch.zeros(2, 5, 16), torch.zeros(2, 5), operator)


# ==================================================================== M3-A1: within-record only


def test_within_record_only() -> None:
    """Record-uniform averaging over independent records: the loss on a 2-record batch must be
    the plain arithmetic mean of each record's own isolated (B=1) loss -- proving no pair ever
    crosses a record boundary, not merely that the code happens to average correctly."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(0)
    n_tok = 6
    z_a = torch.randn(1, n_tok, 10, generator=gen)
    theta_a = torch.rand(1, n_tok, generator=gen) * 2 * math.pi
    z_b = torch.randn(1, n_tok, 10, generator=gen)
    theta_b = torch.rand(1, n_tok, generator=gen) * 2 * math.pi

    loss_a = transport_loss(z_a, theta_a, operator).loss
    loss_b = transport_loss(z_b, theta_b, operator).loss

    z_batch = torch.cat([z_a, z_b], dim=0)
    theta_batch = torch.cat([theta_a, theta_b], dim=0)
    loss_batch = transport_loss(z_batch, theta_batch, operator).loss

    torch.testing.assert_close(loss_batch, (loss_a + loss_b) / 2, atol=1e-6, rtol=1e-5)


def test_permuting_one_records_tokens_does_not_affect_another_record() -> None:
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(1)
    n_tok = 6
    z = torch.randn(2, n_tok, 10, generator=gen)
    theta = torch.rand(2, n_tok, generator=gen) * 2 * math.pi

    perm = torch.randperm(n_tok, generator=gen)
    z_perturbed = z.clone()
    theta_perturbed = theta.clone()
    z_perturbed[1] = z[1, perm]
    theta_perturbed[1] = theta[1, perm]

    # Record 0 is untouched; record 1's tokens are permuted (a within-record relabelling, which
    # DOES change record 1's own pair set ordering but must not move record 0's contribution).
    loss_record0_before = transport_loss(z[0:1], theta[0:1], operator).loss
    loss_record0_after = transport_loss(z_perturbed[0:1], theta_perturbed[0:1], operator).loss
    torch.testing.assert_close(loss_record0_before, loss_record0_after, atol=1e-6, rtol=1e-5)


# ============================================================= M3-A2: total NaN-token exclusion


def test_nan_theta_token_is_never_read_even_if_its_z_is_nan() -> None:
    """A token whose theta is NaN must be fully excluded -- injecting NaN into ITS z value must
    not change the loss at all, proving that value is never read, not merely down-weighted."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(2)
    n_tok = 6
    z = torch.randn(1, n_tok, 10, generator=gen)
    theta = torch.rand(1, n_tok, generator=gen) * 2 * math.pi
    theta[0, 3] = float("nan")  # token 3 excluded

    baseline = transport_loss(z, theta, operator).loss

    z_poisoned = z.clone()
    z_poisoned[0, 3] = float("nan")
    poisoned = transport_loss(z_poisoned, theta, operator).loss

    torch.testing.assert_close(baseline, poisoned, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(poisoned)


def test_n_valid_pairs_excludes_nan_tokens_and_self_pairs() -> None:
    operator = _toy_cyclic()
    n_tok = 5
    z = torch.randn(1, n_tok, 10)
    theta = torch.tensor([[0.1, 0.2, float("nan"), 0.4, 0.5]])
    out = transport_loss(z, theta, operator)
    # 4 valid tokens -> 4*4 - 4 (diagonal) = 12 ordered non-self pairs
    assert out.n_valid_pairs == 12
    assert out.n_records_with_pairs == 1


# ===================================================================== M3-A5: zero-pair safety


def test_all_nan_theta_gives_zero_finite_loss() -> None:
    operator = _toy_cyclic()
    z = torch.randn(3, 6, 10)
    theta = torch.full((3, 6), float("nan"))
    out = transport_loss(z, theta, operator)
    assert float(out.loss) == 0.0
    assert torch.isfinite(out.loss)
    assert out.n_valid_pairs == 0
    assert out.n_records_with_pairs == 0


def test_single_valid_token_record_contributes_nothing_but_others_still_count() -> None:
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(3)
    n_tok = 6
    z = torch.randn(2, n_tok, 10, generator=gen)
    theta = torch.rand(2, n_tok, generator=gen) * 2 * math.pi
    theta[0, 1:] = float("nan")  # record 0 has exactly one valid token: no pairs possible

    out_batch = transport_loss(z, theta, operator)
    out_record1_alone = transport_loss(z[1:2], theta[1:2], operator)
    torch.testing.assert_close(out_batch.loss, out_record1_alone.loss, atol=1e-6, rtol=1e-5)
    assert out_batch.n_records_with_pairs == 1


def test_zero_pair_loss_composes_safely_into_a_larger_backward_pass() -> None:
    """Matches the real usage pattern (`winder.jepa.train.train_step`): trans_loss is always
    ADDED to other terms that do require grad, so a zero-pair batch's own lack of computational
    dependency on z must not break that outer backward pass, nor poison it with NaN/inf."""
    operator = _toy_cyclic()
    z = torch.randn(2, 6, 10, requires_grad=True)
    theta = torch.full((2, 6), float("nan"))
    out = transport_loss(z, theta, operator)
    assert float(out.loss) == 0.0

    dummy = (z * 2.0).sum()  # stands in for pred_loss + lambda_sig * sigreg_loss
    total = dummy + 0.5 * out.loss
    total.backward()  # must not raise
    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


# ================================================================================ gradcheck


def test_gradcheck_wrt_z_cyclic() -> None:
    operator = _toy_cyclic().double()
    gen = torch.Generator().manual_seed(4)
    z = torch.randn(2, 5, 10, dtype=torch.float64, generator=gen, requires_grad=True)
    theta = (torch.rand(2, 5, dtype=torch.float64, generator=gen) * 2 * math.pi).detach()

    def f(z_: torch.Tensor) -> torch.Tensor:
        return transport_loss(z_, theta, operator).loss

    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=1e-4)


def test_gradcheck_wrt_omega_free_arm() -> None:
    operator = _toy_free().double()
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(2, 5, 10, dtype=torch.float64, generator=gen).detach()
    theta = (torch.rand(2, 5, dtype=torch.float64, generator=gen) * 2 * math.pi).detach()

    def f(_omega: torch.Tensor) -> torch.Tensor:
        # gradcheck perturbs `_omega` (== operator.omega, same tensor identity, passed directly
        # below) in place and re-runs this closure -- reassigning a NEW nn.Parameter here would
        # sever that identity and always report a zero analytical gradient.
        return transport_loss(z, theta, operator).loss

    assert torch.autograd.gradcheck(f, (operator.omega,), eps=1e-6, atol=1e-4)


# ============================================================= M3-A6: floor identity, real shape


def test_floor_matches_a_direct_python_loop_computation() -> None:
    """Independent re-implementation (a plain double loop, not this module's own vectorised
    reduction) of Eq. 16's closed form, on a small case with an irregular valid-token pattern."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(6)
    n_tok = 6
    z = torch.randn(1, n_tok, 10, generator=gen)
    theta = torch.tensor([[0.1, 0.2, float("nan"), 0.4, 0.5, float("nan")]])
    zhat = z / z.norm(dim=-1, keepdim=True)

    valid = [j for j in range(n_tok) if math.isfinite(theta[0, j])]
    total, count = 0.0, 0
    for s in valid:
        for t in valid:
            if s == t:
                continue
            total += 1.0 - float((zhat[0, s, :2] * zhat[0, t, :2]).sum())
            count += 1
    expected_floor = total / count

    out = transport_loss(z, theta, operator)
    assert float(out.floor) == pytest.approx(expected_floor, abs=1e-5)


# ============================================= radial term (campaign_x2x2 pre-launch addendum)


def test_radial_weight_one_realises_the_geometric_mean_normalised_vector_defect() -> None:
    """The pinned identity (artifacts/campaign_x2x2/pre_launch_addendum.md): at radial_weight=1
    the total per-pair term equals ||R_Delta z_s - z_t||^2 / (2ab), a = ||z_s||, b = ||z_t||
    UNnormalised -- the law of cosines: (1 - cos phi) + 0.5*(a/b + b/a - 2) =
    (a^2 + b^2 - 2ab cos phi) / (2ab). Reference: a plain python double loop calling
    operator.transport on the RAW (unnormalised) z per pair. One NaN-theta token exercises the
    shared pair mask (both sides must drop the same pairs). Agreement is to the module's own
    eps-clamped normaliser (~1e-8 relative), far tighter than any wrong formula could produce."""
    operator = _toy_cyclic().double()
    gen = torch.Generator().manual_seed(8)
    n_tok = 5
    scales = 1.0 + 2.0 * torch.rand(2, n_tok, 1, dtype=torch.float64, generator=gen)
    z = torch.randn(2, n_tok, 10, dtype=torch.float64, generator=gen) * scales
    theta = torch.rand(2, n_tok, dtype=torch.float64, generator=gen) * 2 * math.pi
    theta[0, 2] = float("nan")

    out = transport_loss(z, theta, operator, radial_weight=1.0)

    record_means = []
    for rec in range(2):
        valid = [j for j in range(n_tok) if math.isfinite(float(theta[rec, j]))]
        pair_vals = []
        for s in valid:
            for t in valid:
                if s == t:
                    continue
                delta = (theta[rec, t] - theta[rec, s]).reshape(1)
                moved = operator.transport(z[rec, s].unsqueeze(0), delta)[0]
                a = float(z[rec, s].norm())
                b = float(z[rec, t].norm())
                pair_vals.append(float((moved - z[rec, t]).square().sum()) / (2.0 * a * b))
        record_means.append(sum(pair_vals) / len(pair_vals))
    expected = sum(record_means) / len(record_means)

    assert float(out.loss) == pytest.approx(expected, rel=1e-6, abs=1e-8)
    # The separated diagnostic components must reassemble the total at this weight.
    assert float(out.directional_term) + float(out.radial_term) == pytest.approx(
        float(out.loss), rel=1e-12, abs=1e-12
    )


def test_radial_weight_zero_reproduces_the_directional_only_loss_exactly() -> None:
    """Regression pin for eq-28 paired comparability: radial_weight=0 must reproduce the shipped
    formula (eps-clamped normalise, transport, 1 - cos, record-uniform mean). (a) The explicit
    radial_weight=0.0 call is BITWISE equal to the kwarg-omitted default -- the two call forms
    are one code path. (b) A from-scratch python-loop reference of that formula (same 1e-8 eps
    clamp as loss.py's _EPS) agrees at fp64 summation-order tightness -- exact equality up to
    reduction order, not allclose-loose."""
    operator = _toy_cyclic().double()
    gen = torch.Generator().manual_seed(9)
    n_tok = 6
    z = torch.randn(2, n_tok, 10, dtype=torch.float64, generator=gen) * 2.0
    theta = torch.rand(2, n_tok, dtype=torch.float64, generator=gen) * 2 * math.pi
    theta[1, 4] = float("nan")

    out_default = transport_loss(z, theta, operator)
    out_zero = transport_loss(z, theta, operator, radial_weight=0.0)
    assert torch.equal(out_default.loss, out_zero.loss)

    zhat = z / (z.norm(dim=-1, keepdim=True) + 1e-8)  # Eq. 10's clamped form
    record_means = []
    for rec in range(2):
        valid = [j for j in range(n_tok) if math.isfinite(float(theta[rec, j]))]
        pair_vals = []
        for s in valid:
            for t in valid:
                if s == t:
                    continue
                delta = (theta[rec, t] - theta[rec, s]).reshape(1)
                moved = operator.transport(zhat[rec, s].unsqueeze(0), delta)[0]
                pair_vals.append(1.0 - float((moved * zhat[rec, t]).sum()))
        record_means.append(sum(pair_vals) / len(pair_vals))
    expected = sum(record_means) / len(record_means)

    assert float(out_zero.loss) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # Component reporting at the 0.0 default: the directional term IS the loss; the radial term
    # is the "not applicable" NaN sentinel (winder.jepa.train.StepMetrics' own convention), never
    # a silently-wrong 0.0.
    assert torch.equal(out_zero.directional_term, out_zero.loss.detach())
    assert math.isnan(float(out_zero.radial_term))


def test_radial_term_is_scale_free_globally_but_penalises_per_token_scaling() -> None:
    """Norm-constancy is about RELATIVE norms: multiplying every token of every record by one
    global lambda leaves the radial term unchanged (up to the eps clamp), while inflating a
    single token's norm strictly increases it -- the defect reads within-pair norm ratios only.
    The x100 single-token factor guarantees |log(100 r)| > |log r| for every within-record ratio
    r this draw can produce, so the increase is strict pair by pair, not on net."""
    operator = _toy_cyclic().double()
    gen = torch.Generator().manual_seed(10)
    n_tok = 6
    scales = 1.0 + 2.0 * torch.rand(2, n_tok, 1, dtype=torch.float64, generator=gen)
    z = torch.randn(2, n_tok, 10, dtype=torch.float64, generator=gen) * scales
    theta = torch.rand(2, n_tok, dtype=torch.float64, generator=gen) * 2 * math.pi

    base = transport_loss(z, theta, operator, radial_weight=1.0)
    scaled = transport_loss(3.7 * z, theta, operator, radial_weight=1.0)
    assert float(scaled.radial_term) == pytest.approx(float(base.radial_term), rel=1e-6, abs=1e-8)

    z_one = z.clone()
    z_one[0, 1] = 100.0 * z[0, 1]
    one_scaled = transport_loss(z_one, theta, operator, radial_weight=1.0)
    assert float(one_scaled.radial_term) > float(base.radial_term) + 1.0


def test_gradient_geometry_radial_is_parallel_and_directional_is_tangential() -> None:
    """The two components pull on orthogonal degrees of freedom, token by token: the radial term
    reaches z only through ||z|| (its eps clamps enter only through scalar functions of the
    norms), so its per-token gradient is EXACTLY parallel to that token's own z; the directional
    term reads only the eps-clamped zhat, so its gradient's radial projection is O(eps/||z||) --
    small but not exactly zero. fp64 throughout; the radial gradient is isolated as
    grad(radial_weight=1) - grad(radial_weight=0), whose directional parts cancel bitwise on
    CPU (identical op sequences)."""
    operator = _toy_cyclic().double()
    gen = torch.Generator().manual_seed(11)
    n_tok = 5
    scales = torch.linspace(0.8, 2.5, n_tok, dtype=torch.float64).reshape(1, n_tok, 1)
    z = torch.randn(1, n_tok, 10, dtype=torch.float64, generator=gen) * scales
    z.requires_grad_(True)
    theta = (torch.rand(1, n_tok, dtype=torch.float64, generator=gen) * 2 * math.pi).detach()

    (g_dir,) = torch.autograd.grad(transport_loss(z, theta, operator, radial_weight=0.0).loss, z)
    (g_tot,) = torch.autograd.grad(transport_loss(z, theta, operator, radial_weight=1.0).loss, z)
    g_rad = g_tot - g_dir

    for j in range(n_tok):
        zj = z[0, j].detach()
        nj = float(zj.norm())
        gd, gr = g_dir[0, j], g_rad[0, j]
        assert float(gd.norm()) > 1e-12  # a vanishing gradient would make the checks vacuous
        assert float(gr.norm()) > 1e-12
        radial_frac = abs(float((gd * zj).sum())) / (float(gd.norm()) * nj)
        assert radial_frac < 1e-6
        tangential = gr - ((gr * zj).sum() / (nj * nj)) * zj
        tangential_frac = float(tangential.norm()) / float(gr.norm())
        assert tangential_frac < 1e-10


def test_radial_invalid_token_guard_loss_finite_grad_zero_at_excluded_tokens() -> None:
    """Mirrors test_nan_theta_token_is_never_read_even_if_its_z_is_nan, at radial_weight=1: an
    excluded token's z_filled norm is 0, so an unguarded ratio would put inf (and 0 * inf = NaN
    backward poison, module docstring's bug class) into the graph. The norms-replaced-with-1.0
    guard must keep the loss finite and unchanged by a NaN injected into the excluded token's own
    z, and z.grad finite everywhere, EXACTLY zero at the excluded positions."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(12)
    n_tok = 6
    z = torch.randn(2, n_tok, 10, generator=gen)
    theta = torch.rand(2, n_tok, generator=gen) * 2 * math.pi
    theta[0, 3] = float("nan")
    theta[1, 0] = float("nan")

    z_clean = z.clone().requires_grad_(True)
    out_clean = transport_loss(z_clean, theta, operator, radial_weight=1.0)

    z_poisoned = z.clone()
    z_poisoned[0, 3] = float("nan")
    z_poisoned.requires_grad_(True)
    out_poisoned = transport_loss(z_poisoned, theta, operator, radial_weight=1.0)

    assert torch.isfinite(out_poisoned.loss)
    torch.testing.assert_close(out_poisoned.loss, out_clean.loss, atol=1e-6, rtol=1e-5)

    out_poisoned.loss.backward()
    grad = z_poisoned.grad
    assert grad is not None
    assert torch.all(torch.isfinite(grad))
    assert torch.all(grad[0, 3] == 0.0)
    assert torch.all(grad[1, 0] == 0.0)
    assert torch.any(grad[0, 1] != 0.0)  # valid tokens still receive gradient


def test_gradcheck_wrt_z_cyclic_radial_weight_one() -> None:
    operator = _toy_cyclic().double()
    gen = torch.Generator().manual_seed(13)
    z = torch.randn(2, 5, 10, dtype=torch.float64, generator=gen, requires_grad=True)
    theta = (torch.rand(2, 5, dtype=torch.float64, generator=gen) * 2 * math.pi).detach()

    def f(z_: torch.Tensor) -> torch.Tensor:
        return transport_loss(z_, theta, operator, radial_weight=1.0).loss

    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=1e-4)


# =========================================================== M3-A7: gain near zero at random init


def test_stop_gradient_target_leaves_every_forward_value_bitwise_identical() -> None:
    """`stop_gradient_target` is a backward-only intervention: detaching the target branch must
    not move a single forward number -- loss, floor, directional_term, radial_term all
    `torch.equal` against the two-sided run, at radial_weight 0.0 and 1.0 alike. This is the
    property that makes X4-vs-W3 a clean paired contrast: the two arms optimise the SAME loss
    surface and differ only in which directions the optimiser is allowed to move along it."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(7)
    z = torch.randn(2, 5, 10, generator=gen)
    theta = torch.rand(2, 5, generator=gen) * (2 * math.pi)
    theta[1, 3] = float("nan")
    for radial_weight in (0.0, 1.0):
        two_sided = transport_loss(z, theta, operator, radial_weight=radial_weight)
        one_sided = transport_loss(
            z, theta, operator, radial_weight=radial_weight, stop_gradient_target=True
        )
        assert torch.equal(two_sided.loss, one_sided.loss)
        assert torch.equal(two_sided.floor, one_sided.floor)
        assert torch.equal(two_sided.directional_term, one_sided.directional_term)
        if radial_weight == 0.0:
            assert math.isnan(float(one_sided.radial_term))
        else:
            assert torch.equal(two_sided.radial_term, one_sided.radial_term)


def test_stop_gradient_target_grads_match_a_detached_target_loop_reference() -> None:
    """fp64: with the flag on, z.grad must equal an independent per-ordered-pair loop that
    detaches the target branch explicitly -- source token s receives gradient from pair (s, t),
    target token t receives none from that pair (it still earns gradient as the SOURCE of the
    reverse pair (t, s), the symmetrised stop-gradient convention). Also asserts the grads
    genuinely differ from the two-sided run's."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(11)
    z0 = torch.randn(1, 4, 10, generator=gen, dtype=torch.float64)
    theta = torch.rand(1, 4, generator=gen, dtype=torch.float64) * (2 * math.pi)
    radial_weight = 1.0

    z_flag = z0.clone().requires_grad_(True)
    out = transport_loss(
        z_flag, theta, operator, radial_weight=radial_weight, stop_gradient_target=True
    )
    out.loss.backward()
    assert z_flag.grad is not None

    z_ref = z0.clone().requires_grad_(True)
    n_tok = z_ref.shape[1]
    defects = []
    for s in range(n_tok):
        for t in range(n_tok):
            if s == t:
                continue
            zs, zt = z_ref[0, s], z_ref[0, t].detach()
            zs_hat = zs / (zs.norm() + 1e-8)
            zt_hat = zt / (zt.norm() + 1e-8)
            delta = theta[0, t] - theta[0, s]
            transported = operator.transport(zs_hat, delta)
            directional = 1.0 - (transported * zt_hat).sum()
            a, b = zs.norm(), zt.norm()
            radial = 0.5 * (a / (b + 1e-8) + b / (a + 1e-8) - 2.0)
            defects.append(directional + radial_weight * radial)
    torch.stack(defects).mean().backward()
    assert z_ref.grad is not None
    assert torch.allclose(z_flag.grad, z_ref.grad, rtol=1e-9, atol=1e-12)

    z_two = z0.clone().requires_grad_(True)
    transport_loss(z_two, theta, operator, radial_weight=radial_weight).loss.backward()
    assert z_two.grad is not None
    assert not torch.allclose(z_flag.grad, z_two.grad, rtol=1e-3, atol=1e-6)


def test_stop_gradient_target_invalid_token_grad_still_exactly_zero() -> None:
    """The flag must not reopen the 0*NaN backward class: with radial on and the flag on, an
    invalid-theta token's z.grad stays exactly zero and every gradient entry stays finite."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(13)
    z = torch.randn(2, 5, 10, generator=gen).requires_grad_(True)
    theta = torch.rand(2, 5, generator=gen) * (2 * math.pi)
    theta[0, 2] = float("nan")
    theta[1, 4] = float("nan")
    out = transport_loss(z, theta, operator, radial_weight=1.0, stop_gradient_target=True)
    out.loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.equal(z.grad[0, 2], torch.zeros_like(z.grad[0, 2]))
    assert torch.equal(z.grad[1, 4], torch.zeros_like(z.grad[1, 4]))


def _random_orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    """QR of an n x n standard-Gaussian draw -- an orthogonal Q, float64. Not claimed Haar-exact
    (no sign correction on R's diagonal), but "some" orthogonal matrix is all this test needs,
    both for the block-commuting construction and the generic negative control."""
    a = torch.randn(n, n, dtype=torch.float64, generator=gen)
    q, _ = torch.linalg.qr(a)
    return cast(torch.Tensor, q.contiguous())


def _rotation_2d(angle: torch.Tensor) -> torch.Tensor:
    """The 2x2 rotation matrix R(angle), float64."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([c, -s, s, c]).reshape(2, 2).contiguous()


def test_gauge_invariance_under_an_arbitrary_orthogonal_q() -> None:
    """R_Delta = I_k0 (+) direct-sum_j kron(I_{k_j}, R(n_j*Delta)): `HarmonicTransport.transport`
    lays out z as [invariant block (k0)] followed by m = sum(k_j) planes of shape (m, 2),
    row-major (plane index slower than the (x, y) pair within a plane). `_omega_per_plane`'s
    `repeat_interleave(omega, k_j)` groups harmonic j's k_j planes contiguously, in ascending n_j
    order, and every one of those k_j planes rotates by the SAME angle n_j*Delta -- i.e. harmonic
    j's block acts on its flat 2*k_j coordinates as exactly kron(I_{k_j}, R(n_j*Delta)): the
    Kronecker product of the k_j-dim "which-copy" identity with the 2x2 rotation.

    Because kron(A, B) @ kron(C, D) == kron(A@C, B@D), for ANY orthogonal B_j (k_j x k_j, mixing
    across harmonic j's k_j multiplicity copies, identically on the x part and the y part):
        kron(B_j, I_2) @ kron(I_{k_j}, R) == kron(B_j @ I_{k_j}, I_2 @ R) == kron(B_j, R)
        kron(I_{k_j}, R) @ kron(B_j, I_2) == kron(I_{k_j} @ B_j, R @ I_2) == kron(B_j, R)
    so kron(B_j, I_2) commutes with kron(I_{k_j}, R(n_j*Delta)) for every Delta -- purely
    algebraic, holding per-harmonic-block regardless of what happens in any other block. Any
    orthogonal Q0 on the k0 invariant block trivially commutes too (R_Delta acts as identity
    there for every Delta). Assembling Q = block_diag(Q0, kron(B_1, I_2), ..., kron(B_m, I_2))
    therefore commutes with R_Delta for every Delta, block by block.

    transport_loss reads only cos<.,.> between L2-normalised vectors; cos<.,.> is invariant under
    applying the SAME orthogonal Q to both arguments, and R_Delta(Q zhat_s) == Q R_Delta(zhat_s)
    exactly because Q commutes with R_Delta -- so replacing z with z @ Q.T must leave the loss
    unchanged (to float64 rounding).

    Negative control: a GENERIC (no block structure, full K x K) orthogonal Q_generic does NOT
    commute with R_Delta in general, and DOES change the loss -- without this, a test that only
    checks the positive case cannot distinguish "gauge invariance holds" from "this test always
    passes regardless of Q".
    """
    k0 = 4
    n_j = list(range(1, 11))
    k_j = [24, 24, 20, 16, 12, 10, 8, 6, 4, 2]  # the crowned recipe's operator spectrum
    operator = CyclicOperator(CyclicOperatorConfig(k0=k0, n_j=n_j, k_j=k_j)).double()
    k_dim = operator.dimension
    assert k_dim == 256

    gen = torch.Generator().manual_seed(100)

    # --- step 1: verify the pure Kronecker-commutation identity in isolation, on a random B and
    # a random (non-integer, operator-unrelated) frequency and delta -- BEFORE relying on it to
    # build the full operator-sized Q below.
    k_example = 5
    b_example = _random_orthogonal(k_example, gen)
    delta_example = torch.rand((), dtype=torch.float64, generator=gen) * 2 * math.pi
    n_example = 3.7  # arbitrary non-integer frequency: the identity is purely algebraic and does
    # not depend on n_j being an integer or on any operator-specific constraint.
    r_example = _rotation_2d(n_example * delta_example)
    eye_k = torch.eye(k_example, dtype=torch.float64)
    eye_2 = torch.eye(2, dtype=torch.float64)
    lhs = torch.kron(b_example, eye_2) @ torch.kron(eye_k, r_example)
    mid = torch.kron(b_example, r_example)
    rhs = torch.kron(eye_k, r_example) @ torch.kron(b_example, eye_2)
    torch.testing.assert_close(lhs, mid, atol=1e-12, rtol=0)
    torch.testing.assert_close(mid, rhs, atol=1e-12, rtol=0)

    # --- step 2: build the full K x K commuting Q -- block-diagonal, one independent random
    # orthogonal block per harmonic (plus the invariant block), in the same contiguous ascending
    # order `_omega_per_plane`'s `repeat_interleave` produces.
    q0 = _random_orthogonal(k0, gen)
    blocks = [q0]
    for kj in k_j:
        b_j = _random_orthogonal(kj, gen)
        blocks.append(torch.kron(b_j, eye_2))
    q = torch.block_diag(*blocks)
    assert tuple(q.shape) == (k_dim, k_dim)
    torch.testing.assert_close(q @ q.T, torch.eye(k_dim, dtype=torch.float64), atol=1e-10, rtol=0)

    # --- step 3: confirm Q commutes with R_Delta directly, as an explicit K x K matrix, for a
    # random Delta -- BEFORE using Q in the actual transport_loss comparison below.
    delta_scalar = torch.rand((), dtype=torch.float64, generator=gen) * 2 * math.pi
    basis = torch.eye(k_dim, dtype=torch.float64)
    delta_broadcast = delta_scalar.expand(k_dim)
    # row i of `transported` is R_delta @ e_i (transport applied to the i-th standard basis
    # vector); its transpose therefore has R_delta @ e_i as COLUMN i -- R_delta's own matrix,
    # built column by column via one batched call.
    transported = operator.transport(basis, delta_broadcast)
    r_delta_matrix = transported.T
    torch.testing.assert_close(q @ r_delta_matrix, r_delta_matrix @ q, atol=1e-10, rtol=0)

    # --- step 4: the actual gauge-invariance check on transport_loss itself.
    b, length = 2, 6
    z = torch.randn(b, length, k_dim, dtype=torch.float64, generator=gen)
    theta = torch.rand(b, length, dtype=torch.float64, generator=gen) * 2 * math.pi  # all finite

    loss_a = transport_loss(z, theta, operator).loss
    loss_b = transport_loss(z @ q.T, theta, operator).loss
    assert torch.allclose(loss_a, loss_b, atol=1e-10, rtol=0)

    # --- negative control: a generic (no block structure, unrelated to R_Delta) orthogonal
    # Q_generic does not commute with R_Delta and DOES change the loss -- this keeps the positive
    # check above from being vacuous.
    q_generic = _random_orthogonal(k_dim, gen)
    eye_full = torch.eye(k_dim, dtype=torch.float64)
    torch.testing.assert_close(q_generic @ q_generic.T, eye_full, atol=1e-10, rtol=0)
    assert not torch.allclose(q_generic @ r_delta_matrix, r_delta_matrix @ q_generic, atol=1e-6)
    loss_c = transport_loss(z @ q_generic.T, theta, operator).loss
    assert not torch.allclose(loss_a, loss_c, atol=1e-6)


def test_gain_is_small_at_random_init_full_scale_operator() -> None:
    """At a random (untrained) encoder, the operator has no learned equivariant structure to
    exploit -- floor and loss should both be close to their common high-dimensional-random
    value (~1), so their difference (the gain) should be small, not because either is near 0."""
    operator = CyclicOperator(CyclicOperatorConfig())  # the real, M0-calibrated K=256 default
    gen = torch.Generator().manual_seed(7)
    b, n_tok = 8, 40
    z = torch.randn(b, n_tok, operator.dimension, generator=gen)
    theta = torch.rand(b, n_tok, generator=gen) * 2 * math.pi

    out = transport_loss(z, theta, operator)
    gain = float(out.floor) - float(out.loss)
    assert abs(gain) < 0.05
    assert float(out.loss) == pytest.approx(1.0, abs=0.15)
    assert float(out.floor) == pytest.approx(1.0, abs=0.15)

import math

import pytest
import torch
from torch import nn

from winder.jepa.regularizers import SigReg, SigRegConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.transport.diagnostics import (
    block_energy_participation_ratio,
    k0_energy_fraction,
    ln_gamma_cv,
    omega_summary,
)
from winder.transport.loss import transport_loss

_K0, _N_J, _K_J = 4, [1, 2, 3], [4, 4, 4]  # K = 4 + 2*12 = 28


def _toy_cyclic() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


def _toy_free() -> FreeOperator:
    return FreeOperator(FreeOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


# ======================================================================== k0_energy_fraction


def test_k0_energy_fraction_of_a_pure_invariant_latent_is_one() -> None:
    z = torch.zeros(5, 28)
    z[:, :_K0] = torch.randn(5, _K0)
    assert k0_energy_fraction(z, _K0) == pytest.approx(1.0)


def test_k0_energy_fraction_of_a_zero_invariant_block_is_zero() -> None:
    z = torch.randn(5, 28)
    z[:, :_K0] = 0.0
    assert k0_energy_fraction(z, _K0) == pytest.approx(0.0, abs=1e-6)


def test_k0_energy_fraction_of_isotropic_random_matches_k0_over_k() -> None:
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(20_000, 28, generator=gen)
    assert k0_energy_fraction(z, _K0) == pytest.approx(_K0 / 28, abs=0.01)


def test_k0_energy_fraction_of_zero_k0_is_zero() -> None:
    z = torch.randn(5, 28)
    assert k0_energy_fraction(z, 0) == 0.0


# ============================================================ block_energy_participation_ratio


def test_participation_ratio_of_pure_invariant_energy_is_one() -> None:
    operator = _toy_cyclic()
    z = torch.zeros(5, 28)
    z[:, :_K0] = torch.randn(5, _K0)
    pr = block_energy_participation_ratio(z, operator)
    assert pr == pytest.approx(1.0, abs=1e-4)


def test_participation_ratio_of_evenly_spread_energy_is_n_blocks() -> None:
    """4 blocks (1 invariant + 3 harmonics), each given EXACTLY equal energy by construction."""
    operator = _toy_cyclic()
    z = torch.zeros(1, 28)
    z[0, :_K0] = 1.0 / math.sqrt(_K0)  # k0 block: unit energy
    offset = _K0
    for k_j in _K_J:
        width = 2 * k_j
        z[0, offset : offset + width] = 1.0 / math.sqrt(width)  # each harmonic: unit energy
        offset += width
    pr = block_energy_participation_ratio(z, operator)
    assert pr == pytest.approx(4.0, abs=1e-4)  # 1 invariant + 3 harmonics, perfectly even


# =========================================================================== omega_summary


def test_omega_summary_of_cyclic_operator_is_trivial() -> None:
    operator = _toy_cyclic()
    summary = omega_summary(operator)
    assert summary.omega == pytest.approx([1.0, 2.0, 3.0])
    assert summary.min_abs_omega == pytest.approx(1.0)
    assert summary.max_int_dist == pytest.approx(0.0, abs=1e-6)


def test_omega_summary_catches_the_omega_to_zero_collapse() -> None:
    """This is the failure closure_residual CANNOT catch (module docstring): omega -> 0 gives
    closure_residual == 0, the SAME value a correctly-closed operator has."""
    operator = _toy_free()
    with torch.no_grad():
        operator.omega.zero_()
    assert float(operator.closure_residual().detach()) == pytest.approx(0.0, abs=1e-6)
    summary = omega_summary(operator)
    assert summary.min_abs_omega == pytest.approx(0.0, abs=1e-6)


def test_omega_summary_max_int_dist_reflects_a_non_integer_drift() -> None:
    operator = _toy_free()
    with torch.no_grad():
        operator.omega[0] = 1.3
    summary = omega_summary(operator)
    assert summary.max_int_dist == pytest.approx(0.3, abs=1e-6)


# ================================================================================= ln_gamma_cv


def test_ln_gamma_cv_of_fresh_layernorms_is_zero() -> None:
    module = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16), nn.LayerNorm(16))
    assert ln_gamma_cv(module) == pytest.approx(0.0, abs=1e-6)


def test_ln_gamma_cv_of_anisotropic_gamma_is_positive() -> None:
    module = nn.LayerNorm(16)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.1, 3.0, 16))
    assert ln_gamma_cv(module) > 0.3


def test_ln_gamma_cv_is_nan_with_no_layernorm() -> None:
    module = nn.Linear(4, 4)
    assert math.isnan(ln_gamma_cv(module))


# ===================================== M4 math gates: the collapse canaries have real teeth


def test_m4_a1_invariant_collapse_gives_exactly_zero_gain() -> None:
    """A latent with ALL energy in the invariant block satisfies the transport loss EXACTLY as
    well as the non-equivariant floor -- gain == 0 exactly, proving the gain statistic is the
    collapse canary claimed, not merely correlated with it."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(1)
    b, n_tok = 4, 10
    z = torch.zeros(b, n_tok, 28)
    z[..., :_K0] = torch.randn(b, n_tok, _K0, generator=gen)
    theta = torch.rand(b, n_tok, generator=gen) * 2 * math.pi

    out = transport_loss(z, theta, operator)
    gain = float(out.floor) - float(out.loss)
    assert gain == pytest.approx(0.0, abs=1e-6)
    assert k0_energy_fraction(z, _K0) == pytest.approx(1.0)


def test_m4_a2_omega_to_zero_gives_exactly_zero_closure_residual() -> None:
    """closure_residual cannot distinguish this from a correctly-closed operator -- proven, not
    just claimed: both give exactly 0."""
    collapsed = _toy_free()
    with torch.no_grad():
        collapsed.omega.zero_()
    closed = _toy_cyclic()

    assert float(collapsed.closure_residual().detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(closed.closure_residual().detach()) == pytest.approx(0.0, abs=1e-5)
    # omega_summary is what tells them apart:
    assert omega_summary(collapsed).min_abs_omega == pytest.approx(0.0, abs=1e-6)
    assert omega_summary(closed).min_abs_omega == pytest.approx(1.0)


def test_m4_a3_sigreg_penalizes_invariant_collapse_far_more_than_isotropic() -> None:
    """SIGReg is the structural guard against invariant collapse (module docstring): per-block
    energy sits inside SIGReg's own commutant (winder.transport.dataset's own note on this),
    so a rank-k0 latent -- all energy in the k0 invariant coordinates, zero everywhere else --
    reads as a severe anisotropic collapse to SIGReg, quantified here rather than assumed."""
    gen = torch.Generator().manual_seed(2)
    reg = SigReg(SigRegConfig(n_directions=256))
    k, n = 28, 2000

    isotropic = torch.randn(n, k, generator=gen)
    isotropic_loss = float(reg(isotropic, generator=torch.Generator().manual_seed(3)))

    collapsed = torch.zeros(n, k)
    collapsed[:, :_K0] = torch.randn(n, _K0, generator=gen)
    collapsed_loss = float(reg(collapsed, generator=torch.Generator().manual_seed(3)))

    assert collapsed_loss > 100.0 * isotropic_loss

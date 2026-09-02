"""M1: the transport operator, offline -- every gate here is a closed-form identity checkable
with no data (notes/internal/phase_equivariance_notes_v13.pdf's own "what is proven" list, §8.1).
"""

import math

import pytest
import torch

from winder.config import ArmConfig, load_arm_config, resolve_operator_config
from winder.operators import (
    CyclicOperator,
    CyclicOperatorConfig,
    FreeOperator,
    FreeOperatorConfig,
    HarmonicTransport,
    TransportOperator,
    build_operator,
)
from winder.operators.registry import OPERATOR_REGISTRY

# A small, tractable spectrum shared by the identity tests below -- distinct from the production
# (M0-calibrated) default, chosen only so dense K x K matrices and commutant-rank computations
# stay cheap. K = k0 + 2*sum(k_j) = 2 + 2*(1+2+1) = 10.
_TOY_K0 = 2
_TOY_N_J = [1, 2, 3]
_TOY_K_J = [1, 2, 1]


def _toy_cyclic() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_TOY_K0, n_j=_TOY_N_J, k_j=_TOY_K_J))


def _toy_free() -> FreeOperator:
    return FreeOperator(FreeOperatorConfig(k0=_TOY_K0, n_j=_TOY_N_J, k_j=_TOY_K_J))


def _dense_matrix(operator: HarmonicTransport, delta: float) -> torch.Tensor:
    """R_delta as an explicit (K, K) matrix, in float64: `transport` applied to every standard
    basis vector at once. Row i of `transport(eye, delta_vec)` is `R_delta @ e_i` -- the i-th
    COLUMN of R_delta -- so the returned matrix (that stack, transposed) has column i equal to
    `R_delta @ e_i`, i.e. is R_delta itself."""
    k = operator.dimension
    eye = torch.eye(k, dtype=torch.float64)
    delta_vec = torch.full((k,), float(delta), dtype=torch.float64)
    rows = operator.transport(eye, delta_vec)  # rows[i] = R_delta @ e_i
    return rows.T


# ============================================================== construction / validation


def test_transport_operator_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        TransportOperator()  # type: ignore[abstract]


def test_dimension_property() -> None:
    assert _toy_cyclic().dimension == 10
    assert _toy_free().dimension == 10


@pytest.mark.parametrize(
    "n_j,k_j,match",
    [
        ([2, 3], [1, 1], "contiguous"),  # gap at 1
        ([1, 1, 2], [1, 1, 1], "contiguous"),  # repeat
        ([2, 1], [1, 1], "contiguous"),  # descending
        ([1, 2], [1, 1, 1], "equal length"),  # length mismatch
        ([], [], "non-empty"),  # empty spectrum
        ([1, 2], [1, 0], "positive"),  # non-positive multiplicity
    ],
)
def test_invalid_spectrum_raises(n_j: list[int], k_j: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CyclicOperator(CyclicOperatorConfig(k0=2, n_j=n_j, k_j=k_j))


def test_negative_k0_raises() -> None:
    with pytest.raises(ValueError, match="k0"):
        CyclicOperator(CyclicOperatorConfig(k0=-1, n_j=[1], k_j=[1]))


# ============================================================================= registry/config


@pytest.mark.parametrize("name", list(OPERATOR_REGISTRY))
def test_registered_operators_instantiate_at_their_own_defaults(name: str) -> None:
    schema_cls, operator_cls = OPERATOR_REGISTRY[name]
    operator = operator_cls(schema_cls())
    assert isinstance(operator, TransportOperator)
    assert isinstance(operator, HarmonicTransport)
    assert operator.dimension == 256  # the M0-calibrated production default


def test_arm_config_round_trips_operator_override() -> None:
    arm = load_arm_config(
        {"name": "day3_c11_cyclic", "operator_name": "cyclic", "operator": {"k0": 6}}
    )
    assert arm.name == "day3_c11_cyclic"

    operator_config = resolve_operator_config(arm)
    assert operator_config.k0 == 6
    assert list(operator_config.n_j) == [1, 2, 3, 4, 5, 6]  # default preserved

    operator = build_operator(arm.operator_name, operator_config)
    assert isinstance(operator, CyclicOperator)


def test_free_arm_resolves_to_free_operator() -> None:
    arm = load_arm_config({"name": "day3_c10", "operator_name": "free"})
    operator_config = resolve_operator_config(arm)
    operator = build_operator(arm.operator_name, operator_config)
    assert isinstance(operator, FreeOperator)


def test_unknown_operator_name_is_missing_from_registry() -> None:
    arm = ArmConfig(name="bogus", operator_name="not_a_real_operator")
    with pytest.raises(KeyError):
        resolve_operator_config(arm)


# =========================================================== M1-A1..A6: the pure group identities


@pytest.mark.parametrize("build", [_toy_cyclic, _toy_free])
def test_r0_is_identity(build: object) -> None:
    operator = build()  # type: ignore[operator]
    m0 = _dense_matrix(operator, 0.0)
    torch.testing.assert_close(m0, torch.eye(10, dtype=torch.float64), atol=1e-14, rtol=0)


@pytest.mark.parametrize("build", [_toy_cyclic, _toy_free])
def test_composition(build: object) -> None:
    operator = build()  # type: ignore[operator]
    gen = torch.Generator().manual_seed(0)
    for _ in range(8):
        d1 = float(torch.rand(1, generator=gen) * 4 * math.pi - 2 * math.pi)
        d2 = float(torch.rand(1, generator=gen) * 4 * math.pi - 2 * math.pi)
        lhs = _dense_matrix(operator, d1) @ _dense_matrix(operator, d2)
        rhs = _dense_matrix(operator, d1 + d2)
        torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=0)


@pytest.mark.parametrize("build", [_toy_cyclic, _toy_free])
def test_orthogonality(build: object) -> None:
    operator = build()  # type: ignore[operator]
    gen = torch.Generator().manual_seed(1)
    for _ in range(8):
        d = float(torch.rand(1, generator=gen) * 4 * math.pi)
        r = _dense_matrix(operator, d)
        torch.testing.assert_close(r.T @ r, torch.eye(10, dtype=torch.float64), atol=1e-12, rtol=0)
        # norm-preservation on random vectors: the property `transport` is actually exercised
        # for (never a dense matmul at the real K=256), tested independently of `_dense_matrix`.
        z = torch.randn(5, 10, dtype=torch.float64, generator=gen)
        delta = torch.full((5,), d, dtype=torch.float64)
        transported = operator.transport(z, delta)
        torch.testing.assert_close(transported.norm(dim=-1), z.norm(dim=-1), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("build", [_toy_cyclic, _toy_free])
def test_block_diagonal_exact_structure(build: object) -> None:
    """The invariant block is literally I_k0 for every delta; each harmonic plane's own 2x2
    submatrix is exactly R(omega_j * delta); every off-block entry is exactly 0."""
    operator = build()  # type: ignore[operator]
    d = 0.73
    r = _dense_matrix(operator, d)
    torch.testing.assert_close(
        r[:_TOY_K0, :_TOY_K0], torch.eye(_TOY_K0, dtype=torch.float64), atol=1e-14, rtol=0
    )
    plane_index = 0
    for n_j, k_j in zip(_TOY_N_J, _TOY_K_J, strict=True):
        cos_p, sin_p = math.cos(n_j * d), math.sin(n_j * d)
        expected_plane = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]], dtype=torch.float64)
        for _ in range(k_j):
            lo = _TOY_K0 + 2 * plane_index
            torch.testing.assert_close(
                r[lo : lo + 2, lo : lo + 2], expected_plane, atol=1e-12, rtol=0
            )
            plane_index += 1
    # off-block-diagonal: zero out every block found above and assert nothing is left.
    mask = torch.zeros_like(r, dtype=torch.bool)
    mask[:_TOY_K0, :_TOY_K0] = True
    for p in range(sum(_TOY_K_J)):
        lo = _TOY_K0 + 2 * p
        mask[lo : lo + 2, lo : lo + 2] = True
    assert torch.all(r[~mask].abs() < 1e-12)


# ================================================================= M1-A3/A4: closure, at Delta=2pi


def test_cyclic_closure_residual_is_exactly_zero() -> None:
    operator = _toy_cyclic()
    assert float(operator.closure_residual()) < 1e-6  # fp32 buffer; see agreement test below


def test_free_at_init_closure_residual_is_exactly_zero() -> None:
    """The free arm is INITIALISED at the same integers the cyclic arm freezes at -- so at
    construction (before any training step moves omega) its closure residual is also ~0. This is
    the free arm's own t=0 baseline for the empirical closure test M6 runs on trained
    checkpoints."""
    operator = _toy_free()
    assert float(operator.closure_residual().detach()) < 1e-6


@pytest.mark.parametrize("build", [_toy_cyclic, _toy_free])
def test_closure_residual_closed_form_matches_dense_r_2pi(build: object) -> None:
    """The closed form (from parameters, at Delta=2*pi exactly) must agree with ||R_{2*pi} -
    I||_F computed by materialising the dense matrix -- proving `closure_residual` measures the
    right Delta, unlike a grid-argmin statistic (which is what reversed the predecessor
    prototype's own architecture decision -- see the module docstring)."""
    operator = build()  # type: ignore[operator]
    dense = _dense_matrix(operator, 2 * math.pi)
    direct = torch.linalg.matrix_norm(dense - torch.eye(10, dtype=torch.float64))
    closed_form = operator.closure_residual().double()
    torch.testing.assert_close(closed_form, direct, atol=1e-6, rtol=0)


def test_closure_residual_is_nonzero_for_a_non_integer_spectrum() -> None:
    """Negative control: closure is a property of INTEGER omega, not of this class's plumbing --
    perturbing omega off-integer must move the residual away from 0."""
    operator = _toy_free()
    with torch.no_grad():
        operator.omega += 0.3
    assert float(operator.closure_residual().detach()) > 0.5


# ========================================================================== M1-A7: Eq 14 Parseval


def test_parseval_decomposition() -> None:
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(2)
    z_src = torch.randn(20, 10, dtype=torch.float64, generator=gen)
    z_tgt = torch.randn(20, 10, dtype=torch.float64, generator=gen)
    delta = torch.rand(20, dtype=torch.float64, generator=gen) * 2 * math.pi

    transported = operator.transport(z_src, delta)
    whole = (transported - z_tgt).square().sum(dim=-1)

    invariant = (z_src[:, :_TOY_K0] - z_tgt[:, :_TOY_K0]).square().sum(dim=-1)
    per_block = torch.zeros(20, dtype=torch.float64)
    plane_index = 0
    for n_j, k_j in zip(_TOY_N_J, _TOY_K_J, strict=True):
        for _ in range(k_j):
            lo = _TOY_K0 + 2 * plane_index
            u = z_src[:, lo : lo + 2]
            v = z_tgt[:, lo : lo + 2]
            cos_p, sin_p = torch.cos(n_j * delta), torch.sin(n_j * delta)
            ru = torch.stack(
                [cos_p * u[:, 0] - sin_p * u[:, 1], sin_p * u[:, 0] + cos_p * u[:, 1]], dim=-1
            )
            per_block = per_block + (ru - v).square().sum(dim=-1)
            plane_index += 1

    torch.testing.assert_close(whole, invariant + per_block, atol=1e-12, rtol=1e-10)


# ============================================================== M1-A8: Eq 16 non-equivariant floor


def test_non_equivariant_floor_identity_by_monte_carlo() -> None:
    """E_delta[1 - cos<R_delta zhat_src, zhat_tgt>] -> 1 - <zhat_src[:k0], zhat_tgt[:k0]> as delta
    ~ U[0, 2*pi) -- the note's Eq. 16, specialised to normalised arguments (a floor equal to what
    an operator-free, K0-only comparison would give). Convergence is at the Monte Carlo rate
    (O(1/sqrt(N))), so the tolerance here is generous by construction, not by omission."""
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(3)
    z_src = torch.randn(10, dtype=torch.float64, generator=gen)
    z_tgt = torch.randn(10, dtype=torch.float64, generator=gen)
    zhat_src = z_src / z_src.norm()
    zhat_tgt = z_tgt / z_tgt.norm()

    n = 200_000
    delta = torch.rand(n, dtype=torch.float64, generator=gen) * 2 * math.pi
    src_batch = zhat_src.unsqueeze(0).expand(n, -1)
    transported = operator.transport(src_batch, delta)
    cos_sim = (transported * zhat_tgt).sum(dim=-1)
    empirical_floor = float((1 - cos_sim).mean())

    invariant_term = float(zhat_src[:_TOY_K0] @ zhat_tgt[:_TOY_K0])
    closed_form_floor = 1 - invariant_term

    assert abs(empirical_floor - closed_form_floor) < 3e-3


# ============================================ M1-A9/A10: Prop 4.1 harmonic annihilation, Prop 4.2


def _equivariant_trajectory(
    operator: HarmonicTransport, t: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equally-spaced theta in [0, 2*pi) and the exactly-equivariant trajectory z(theta) =
    R_theta @ z0 for a random z0 -- returns (theta, z), both (T,) / (T, K)."""
    theta = torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)
    z0 = torch.randn(operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.unsqueeze(0).expand(t, -1), theta)
    return theta, z


def test_prop_4_1_mean_pooling_annihilates_harmonics() -> None:
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(4)
    t = 125  # > 2*n_max, comfortably in the exact-annihilation regime this identity requires
    theta, z = _equivariant_trajectory(operator, t, gen)
    mean_pooled = z.mean(dim=0)
    # z(0) is z[0] directly (theta[0] == 0 by _equivariant_trajectory's construction); Prop 4.1
    # says the mean equals Pi_0 @ z(0), i.e. z(0)'s own invariant-block coordinates with every
    # harmonic block zeroed.
    expected = torch.zeros(operator.dimension, dtype=torch.float64)
    expected[:_TOY_K0] = z[0, :_TOY_K0]
    torch.testing.assert_close(mean_pooled, expected, atol=1e-12, rtol=0)


def test_prop_4_2_demodulated_pooling_is_exact() -> None:
    operator = _toy_cyclic()
    gen = torch.Generator().manual_seed(5)
    t = 37  # any T > 0 works for Prop 4.2 (unlike 4.1, no equidistribution requirement)
    theta, z = _equivariant_trajectory(operator, t, gen)
    demodulated = operator.transport(z, -theta)
    z_at_zero = z[0]  # theta[0] == 0 by construction
    for row in demodulated:
        torch.testing.assert_close(row, z_at_zero, atol=1e-11, rtol=0)
    # Retains all K dims, not just K0 -- contrast directly against Prop 4.1's mean pool on the
    # SAME trajectory: mean pooling recovers only z_at_zero's invariant-block coordinates
    # (everywhere else averages to ~0), while demodulation recovers the full K-dim vector,
    # including the (generically nonzero, since z0 was drawn from an isotropic Gaussian)
    # harmonic-block coordinates mean pooling erased.
    mean_pooled = z.mean(dim=0)
    harmonic_energy_recovered = demodulated[0, _TOY_K0:].abs().sum()
    harmonic_energy_pooled = mean_pooled[_TOY_K0:].abs().sum()
    assert float(harmonic_energy_recovered) > 0.1  # z0 was drawn generic; this is not near 0
    assert float(harmonic_energy_pooled) < 1e-10  # Prop 4.1: exactly annihilated


# ================================================================ M1-A11: commutant dimension


def test_commutant_dimension_matches_corrected_formula() -> None:
    """dim(C \\cap Sym) = k0*(k0+1)/2 + sum_j k_j^2 (the multiplicity-corrected form -- SO(2)
    becomes U(k_j) at multiplicity k_j > 1, notes §2.4.1/A.1.2). Verified by building the exact
    Haar-averaging projector on Sym(K) via a finite quadrature: every entry of
    `R_theta @ M @ R_theta.T` is a trigonometric polynomial in theta of degree <= 2*n_max, so a
    quadrature with more than that many equally-spaced points integrates it exactly -- this is
    the same discrete-orthogonality mechanism Prop 4.1 uses, applied to Sym(K) instead of R^K."""
    operator = _toy_cyclic()
    k = operator.dimension
    n_max = max(_TOY_N_J)
    quad_points = 4 * n_max + 3  # comfortably above the required 2*n_max + 1

    theta = torch.arange(quad_points, dtype=torch.float64) * (2 * math.pi / quad_points)
    dense = torch.stack([_dense_matrix(operator, float(d)) for d in theta])  # (Q, K, K)

    basis_dim = k * (k + 1) // 2
    iu = torch.triu_indices(k, k)
    projector_cols = []
    for idx in range(basis_dim):
        i, j = int(iu[0, idx]), int(iu[1, idx])
        m = torch.zeros(k, k, dtype=torch.float64)
        m[i, j] = 1.0
        m[j, i] = 1.0
        averaged = torch.einsum("qab,bc,qdc->ad", dense, m, dense) / quad_points
        projector_cols.append(averaged[iu[0], iu[1]])
    projector = torch.stack(projector_cols, dim=1)  # (basis_dim, basis_dim)

    rank = int(torch.linalg.matrix_rank(projector, atol=1e-9, rtol=0))
    expected = _TOY_K0 * (_TOY_K0 + 1) // 2 + sum(k * k for k in _TOY_K_J)
    assert rank == expected


# ============================================================ M1-A13: demodulation closed form


def test_pairwise_transport_agrees_with_demodulated_closed_form() -> None:
    """<R_{theta_t - theta_s} zhat_s, zhat_t> summed over every (s, t) pair (including s == t)
    equals ||sum_t demod_t||^2 where demod_t = R_{-theta_t} zhat_t -- a consequence of ordinary
    2-D rotation composition (R(a)^T R(b) = R(b-a)), true for ANY real omega, not only integers.
    This gives the all-pairs transport statistic an O(T*K) closed form via demodulation instead
    of an O(T^2*K) pairwise loop -- used as a diagnostic identity here, not as the training-time
    implementation (M3 computes pairs directly, so a bug in one path cannot silently validate
    itself against the other)."""
    gen = torch.Generator().manual_seed(6)
    t = 11
    theta = torch.rand(t, dtype=torch.float64, generator=gen) * 2 * math.pi

    for operator in (_toy_cyclic(), _toy_free()):  # holds regardless of integrality
        z = torch.randn(t, operator.dimension, dtype=torch.float64, generator=gen)
        zhat = z / z.norm(dim=-1, keepdim=True)
        # (T*T,); flat index [s,t] carries theta_t - theta_s
        delta = (theta.unsqueeze(0) - theta.unsqueeze(1)).reshape(-1)
        src = zhat.unsqueeze(1).expand(t, t, -1).reshape(t * t, -1)
        tgt = zhat.unsqueeze(0).expand(t, t, -1).reshape(t * t, -1)
        pairwise_sum = (operator.transport(src, delta) * tgt).sum(dim=-1).sum()
        closed_form_sum = operator.transport(zhat, -theta).sum(dim=0).square().sum()
        torch.testing.assert_close(pairwise_sum, closed_form_sum, atol=1e-8, rtol=1e-6)


def test_wrapping_delta_mod_2pi_is_harmless_iff_integer_omega() -> None:
    """Composition alone (tested above) does not need integer omega -- closure does. Wrapping
    Delta into [0, 2*pi) before transporting changes nothing for the cyclic arm (each plane's
    rotation is then genuinely 2*pi-periodic, since R(n*(delta + 2*pi*k)) == R(n*delta) exactly
    when n*k is an integer) but DOES change the free arm's prediction at non-integer omega.

    This is a deliberate negative control against the predecessor prototype's own S3 defect: its
    closure statistic was computed on a Delta binned/wrapped mod 2*pi, which is only a
    theory-neutral operation for a closing operator -- for a non-closing one it silently measures
    a different Delta than the one requested. `winder/transport/loss.py` must never wrap Delta
    for this reason; this test is what would fail if it did, for the free arm specifically."""
    delta_raw = torch.tensor(
        [2 * math.pi + 0.4, -2 * math.pi - 1.1, 4 * math.pi + 0.7], dtype=torch.float64
    )
    delta_wrapped = torch.remainder(delta_raw, 2 * math.pi)

    cyclic = _toy_cyclic()
    z = torch.randn(3, cyclic.dimension, dtype=torch.float64)
    r_raw = cyclic.transport(z, delta_raw)
    r_wrapped = cyclic.transport(z, delta_wrapped)
    torch.testing.assert_close(r_raw, r_wrapped, atol=1e-10, rtol=0)

    free = _toy_free()
    with torch.no_grad():
        free.omega += 0.3  # off-integer: this is the arm under test, not a control here
    f_raw = free.transport(z, delta_raw)
    f_wrapped = free.transport(z, delta_wrapped)
    assert float((f_raw - f_wrapped).abs().max().detach()) > 0.05

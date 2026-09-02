import math

import numpy as np
import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.transport.delta_gain import (
    cluster_bootstrap_mean,
    delta_stratified_gain,
    source_phase_stratified_gain,
)
from winder.transport.loss import transport_loss

_K0, _N_J, _K_J = 2, [1, 2, 3], [1, 2, 1]  # K = 2 + 2*4 = 10


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


def _uniform_theta(n: int, t: int) -> torch.Tensor:
    return (torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)).expand(n, t).contiguous()


# ============================================================== exact identities (math gates)


def test_gain_is_exactly_zero_when_every_token_shares_one_phase() -> None:
    """Delta == 0 for every pair => R_0 == I => transported cosine IS the identity cosine.
    Exact, not approximate: the same tensor is subtracted from itself."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(6, 12, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.full((6, 12), 1.234, dtype=torch.float64)

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    assert result.overall_mean_gain == 0.0
    assert all(g == 0.0 for g in result.per_record_mean_gain)


def test_phase_invariant_latent_gives_identically_zero_gain_in_every_stratum() -> None:
    """The trivial-optimum signature (module docstring). All energy in the K0 block, which
    R_Delta fixes pointwise, so transport is a no-op and the gain is EXACTLY 0 everywhere --
    which is why a flat-at-zero curve diagnoses collapse rather than merely weak learning."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(1)
    n, t = 5, 20
    z = torch.zeros(n, t, operator.dimension, dtype=torch.float64)
    z[..., : operator.k0] = torch.randn(n, t, operator.k0, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    populated = [g for g, c in zip(result.mean_gain, result.n_pairs, strict=True) if c > 0]
    assert len(populated) == 8
    assert all(g == 0.0 for g in populated)


def test_exactly_equivariant_latent_reaches_the_maximum_possible_gain() -> None:
    """Positive control. If z_t = R_{theta_t} z_0 exactly, then R_Delta zhat_s == zhat_t, so the
    transported cosine is 1 in every stratum, the gain is (1 - identity cosine), and the
    ceiling-normalised `gain_fraction` is exactly 1.0.

    Tolerance is 1e-7, not machine epsilon, and the shortfall is deterministic rather than noise:
    the normaliser is Eq. 10's eps-CLAMPED form `z / (||z|| + 1e-8)`, so `||zhat|| < 1` by
    `1e-8 / ||z||` and the best achievable cosine is `||zhat_s|| * ||zhat_t||`, short of 1 by
    `~2e-8 / ||z||`. At ||z|| ~ sqrt(K) that is ~7e-9 -- present in the real training loss too,
    and correctly reproduced here rather than normalised away."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    n, t = 8, 24
    theta = _uniform_theta(n, t)
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    for pred, ident, gain, frac, count in zip(
        result.mean_transported_cos,
        result.mean_identity_cos,
        result.mean_gain,
        result.gain_fraction,
        result.n_pairs,
        strict=True,
    ):
        if count == 0:
            continue
        assert pred == pytest.approx(1.0, abs=1e-7)
        assert gain == pytest.approx(1.0 - ident, abs=1e-7)
        assert frac == pytest.approx(1.0, abs=1e-7)
    assert result.overall_gain_fraction == pytest.approx(1.0, abs=1e-7)


def test_gain_fraction_is_zero_under_invariant_collapse() -> None:
    """The other end of the `gain_fraction` scale: a phase-invariant latent achieves exactly none
    of the available gain, in every stratum."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(11)
    n, t = 5, 20
    z = torch.zeros(n, t, operator.dimension, dtype=torch.float64)
    z[..., : operator.k0] = torch.randn(n, t, operator.k0, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    assert result.overall_gain_fraction == 0.0
    assert all(f == 0.0 for f, c in zip(result.gain_fraction, result.n_pairs, strict=True) if c)


def test_raw_gain_rises_with_delta_even_when_the_operator_is_perfect() -> None:
    """The confound `gain_fraction` exists to remove, made explicit: on EXACTLY equivariant data
    the raw gain still climbs steeply from the near-zero-Delta stratum to the half-cycle stratum,
    purely because the achievable ceiling `1 - <zhat_s, zhat_t>` does. Anyone reading a raw-gain
    curve as "the operator works better at large Delta" is reading this artefact."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(12)
    n, t = 8, 32
    theta = _uniform_theta(n, t)
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    raw, frac = np.asarray(result.mean_gain), np.asarray(result.gain_fraction)
    assert raw.max() / raw.min() > 2.0  # measured 2.66x on this seed: entirely ceiling, not skill
    assert frac.max() - frac.min() < 1e-7  # the same data, ceiling divided out: dead flat at 1.0


def test_transported_cosine_agrees_with_transport_loss_on_the_same_input() -> None:
    """`transport_loss` computes `1 - <R_Delta zhat_s, zhat_t>` over the same all-pairs set.
    Its record-uniform mean must therefore equal `1 -` this module's own record-uniform mean of
    the transported cosine -- pinning the two implementations to one number rather than trusting
    that they were written from the same formula."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(3)
    n, t = 4, 16
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    loss_out = transport_loss(z, theta, operator)
    # Reproduce the record-uniform convention: per-record mean of (1 - transported cos).
    per_record = []
    for i in range(n):
        single = delta_stratified_gain(
            z[i : i + 1], theta[i : i + 1], operator, n_strata=1, record_chunk=1
        )
        per_record.append(1.0 - single.mean_transported_cos[0])
    assert float(loss_out.loss) == pytest.approx(float(np.mean(per_record)), abs=1e-9)


# ============================================================================ stratification


def test_strata_partition_the_full_cycle_and_account_for_every_valid_pair() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(4)
    n, t, n_strata = 6, 15, 12
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    result = delta_stratified_gain(z, theta, operator, n_strata=n_strata)
    assert sum(result.n_pairs) == n * t * (t - 1)  # every ordered, non-self pair
    width = 2 * math.pi / n_strata
    assert result.delta_centers[0] == pytest.approx(width / 2)
    assert result.delta_centers[-1] == pytest.approx(2 * math.pi - width / 2)


def test_result_is_independent_of_record_chunk_size() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(7, 10, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(7, 10, dtype=torch.float64, generator=gen) * 2 * math.pi

    a = delta_stratified_gain(z, theta, operator, n_strata=6, record_chunk=1)
    b = delta_stratified_gain(z, theta, operator, n_strata=6, record_chunk=7)
    assert a.n_pairs == b.n_pairs
    np.testing.assert_allclose(a.mean_gain, b.mean_gain, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.per_record_mean_gain, b.per_record_mean_gain, rtol=0, atol=1e-12)


def test_empty_stratum_is_nan_not_zero() -> None:
    """An unpopulated stratum has no mean. Reporting 0.0 there would read as 'transport does not
    help at this Delta', which is a measurement; NaN correctly reads as 'not measured'."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(6)
    n, t = 3, 8
    theta = torch.zeros(n, t, dtype=torch.float64)
    theta[:, 1:] = 0.05  # every Delta is 0 or +-0.05 -> only the first/last strata populate
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)

    result = delta_stratified_gain(z, theta, operator, n_strata=16)
    assert any(math.isnan(g) for g in result.mean_gain)
    for gain, count in zip(result.mean_gain, result.n_pairs, strict=True):
        assert math.isnan(gain) == (count == 0)


# ================================================================== NaN theta and edge cases


def test_nan_theta_tokens_contribute_no_pairs() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(7)
    n, t = 4, 10
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    theta[:, :4] = float("nan")  # 6 valid tokens per record

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    assert sum(result.n_pairs) == n * 6 * 5
    assert result.n_records_with_pairs == n


def test_record_with_no_valid_theta_is_excluded_not_zeroed() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(8)
    n, t = 5, 10
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    theta[2] = float("nan")

    result = delta_stratified_gain(z, theta, operator, n_strata=8)
    assert result.n_records_with_pairs == n - 1
    assert 2 not in result.record_index


def test_a_z_row_that_is_nan_where_theta_is_nan_does_not_poison_the_result() -> None:
    """Mirrors winder.transport.loss's own NaN discipline: an excluded token must be unreadable,
    not merely down-weighted."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(9)
    n, t = 3, 10
    z = torch.randn(n, t, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi
    clean = delta_stratified_gain(z.clone(), theta.clone(), operator, n_strata=8)

    theta[:, 0] = float("nan")
    z[:, 0, :] = float("nan")
    poisoned = delta_stratified_gain(z, theta, operator, n_strata=8)
    assert math.isfinite(poisoned.overall_mean_gain)
    assert poisoned.overall_mean_gain != clean.overall_mean_gain  # a token really was dropped


def test_mismatched_shapes_and_bad_strata_raise() -> None:
    operator = _toy_operator()
    z = torch.randn(2, 5, operator.dimension)
    with pytest.raises(ValueError, match="theta shape"):
        delta_stratified_gain(z, torch.zeros(2, 4), operator)
    with pytest.raises(ValueError, match="operator.dimension"):
        delta_stratified_gain(torch.randn(2, 5, 7), torch.zeros(2, 5), operator)
    with pytest.raises(ValueError, match="n_strata"):
        delta_stratified_gain(z, torch.zeros(2, 5), operator, n_strata=0)


# ================================================== the free arm's non-closure dilutes the gain


def test_a_free_operator_off_the_integers_loses_gain_at_large_delta() -> None:
    """The measurement the module docstring promises: build data that is exactly equivariant for
    integer omega, then evaluate it with an operator whose omega has drifted. The per-plane
    rotation ERROR is `(omega_error) * Delta`, so it grows with Delta -- small-Delta strata barely
    notice, half-cycle strata do. This is the Delta-DEPENDENT failure a pooled mean gain cannot
    see, and it must be read on `gain_fraction`, not raw gain, whose own Delta slope would
    otherwise mask it (see test_raw_gain_rises_with_delta_even_when_the_operator_is_perfect)."""
    truth = _toy_operator()
    gen = torch.Generator().manual_seed(10)
    n, t = 12, 32
    theta = _uniform_theta(n, t)
    z0 = torch.randn(n, 1, truth.dimension, dtype=torch.float64, generator=gen)
    z = truth.transport(z0.expand(n, t, -1), theta)

    drifted = FreeOperator(FreeOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))
    with torch.no_grad():
        drifted.omega.mul_(0.85)  # off the integers: closure is broken by construction

    exact = delta_stratified_gain(z, theta, truth, n_strata=8)
    off = delta_stratified_gain(z, theta, drifted, n_strata=8)
    assert exact.overall_gain_fraction == pytest.approx(1.0, abs=1e-7)
    assert off.overall_gain_fraction < 0.9
    # the smallest-|Delta| strata (0 and 7, either side of Delta = 0 mod 2*pi) keep far more of
    # the achievable gain than the half-cycle stratum
    assert off.gain_fraction[0] > off.gain_fraction[4]
    assert off.gain_fraction[7] > off.gain_fraction[4]
    assert (
        float(drifted.closure_residual().detach()) > 1.0
    )  # and the closed form agrees it is broken


# ========================================================================= cluster_bootstrap_mean


def test_cluster_bootstrap_point_estimate_is_the_plain_mean() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    clusters = np.array([1, 1, 2, 2])
    out = cluster_bootstrap_mean(values, clusters, n_replicates=200, seed=0)
    assert out["mean"] == pytest.approx(values.mean())
    assert out["n_clusters"] == 2
    assert out["lo"] <= out["mean"] <= out["hi"]


def test_cluster_bootstrap_interval_is_wider_than_ignoring_clustering() -> None:
    """Two patients, each contributing many near-identical records: the honest interval must
    reflect n=2 independent units, not n=200. Resampling records instead would understate it."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.0, 0.01, 100), rng.normal(1.0, 0.01, 100)])
    clustered = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    naive = np.arange(200)  # each record its own "cluster" == ignoring the patient structure

    wide = cluster_bootstrap_mean(values, clustered, n_replicates=1000, seed=0)
    narrow = cluster_bootstrap_mean(values, naive, n_replicates=1000, seed=0)
    assert (wide["hi"] - wide["lo"]) > 5 * (narrow["hi"] - narrow["lo"])


def test_cluster_bootstrap_rejects_mismatched_shapes_and_handles_empty() -> None:
    with pytest.raises(ValueError, match="must match"):
        cluster_bootstrap_mean(np.zeros(4), np.zeros(3))
    out = cluster_bootstrap_mean(np.array([]), np.array([]))
    assert math.isnan(out["mean"]) and out["n_clusters"] == 0


# ============================================================= source_phase_stratified_gain


def test_source_phase_gain_is_flat_and_maximal_under_exact_equivariance() -> None:
    """A globally equivariant latent transports equally well from EVERY starting phase, so the
    ceiling-normalised gain is 1.0 in every source-phase bin -- the flat curve the figure is
    read against."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(20)
    n, t = 10, 32
    theta = _uniform_theta(n, t)
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta)

    res = source_phase_stratified_gain(z, theta, operator, n_bins=8)
    frac = np.asarray(res["gain_fraction"])
    np.testing.assert_allclose(frac, 1.0, atol=1e-7)
    assert sum(res["n_pairs"]) == n * t * (t - 1)


def test_source_phase_gain_localises_a_phase_restricted_defect() -> None:
    """The measurement the function exists for. Corrupt the latent in ONE half of the cycle only:
    the gain must drop in the affected source-phase bins and hold in the others. A Delta-stratified
    curve mixes the two halves together in every stratum and cannot show this."""
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(21)
    n, t = 16, 32
    theta = _uniform_theta(n, t)
    z0 = torch.randn(n, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n, t, -1), theta).clone()
    corrupt = theta < math.pi  # the first half of the cycle only
    z[corrupt] = torch.randn(
        int(corrupt.sum()), operator.dimension, dtype=torch.float64, generator=gen
    )

    res = source_phase_stratified_gain(z, theta, operator, n_bins=8)
    frac = np.asarray(res["gain_fraction"])
    assert frac[:4].max() < frac[4:].min()  # damaged half strictly below the intact half
    np.testing.assert_allclose(frac[4:], frac[4], atol=0.35)  # intact half stays high


def test_source_phase_gain_is_zero_under_invariant_collapse() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(22)
    n, t = 5, 20
    z = torch.zeros(n, t, operator.dimension, dtype=torch.float64)
    z[..., : operator.k0] = torch.randn(n, t, operator.k0, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    res = source_phase_stratified_gain(z, theta, operator, n_bins=8)
    assert all(g == 0.0 for g, c in zip(res["mean_gain"], res["n_pairs"], strict=True) if c)


def test_source_phase_gain_chunk_independent_and_validates_shapes() -> None:
    operator = _toy_operator()
    gen = torch.Generator().manual_seed(23)
    z = torch.randn(7, 12, operator.dimension, dtype=torch.float64, generator=gen)
    theta = torch.rand(7, 12, dtype=torch.float64, generator=gen) * 2 * math.pi
    a = source_phase_stratified_gain(z, theta, operator, n_bins=6, record_chunk=1)
    b = source_phase_stratified_gain(z, theta, operator, n_bins=6, record_chunk=7)
    np.testing.assert_allclose(a["mean_gain"], b["mean_gain"], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="n_bins"):
        source_phase_stratified_gain(z, theta, operator, n_bins=0)

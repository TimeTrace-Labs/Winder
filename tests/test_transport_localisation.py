import math

import numpy as np
import pytest
import torch

from winder.data.phase import phase_from_rpeaks
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.transport.localisation import (
    causal_phase_from_rpeaks,
    detection_latency,
    deviation_scores,
    identity_residual_scores,
    localisation_error,
    radial_scores,
    transport_residual_scores,
    within_record_auroc,
)

_K0, _N_J, _K_J = 2, [1, 2, 3], [2, 2, 2]  # K = 2 + 2*6 = 14


def _operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


def _equivariant(n: int, t: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, CyclicOperator]:
    """A record whose tokens are EXACTLY the phase-transport of one another."""
    op = _operator()
    gen = torch.Generator().manual_seed(seed)
    theta = (torch.arange(t, dtype=torch.float64) * (2 * math.pi / t)).expand(n, t).contiguous()
    z0 = torch.randn(n, 1, op.dimension, dtype=torch.float64, generator=gen)
    return op.transport(z0.expand(n, t, -1), theta), theta, op


# ======================================================= the identity the detector rests on


def test_residual_is_zero_on_an_exactly_equivariant_record() -> None:
    """No lesion, perfect equivariance => nothing to report, at every token. Tolerance is 1e-7,
    not machine epsilon: the eps-clamped normaliser puts ||zhat|| just under 1 (see
    tests/test_transport_delta_gain.py's own note on the same constant)."""
    z, theta, op = _equivariant(4, 24, seed=0)
    scores = transport_residual_scores(z, theta, op)
    assert torch.isfinite(scores).all()
    np.testing.assert_allclose(scores.numpy(), 0.0, atol=1e-7)


def test_a_single_corrupted_token_is_ranked_top_in_its_own_record() -> None:
    """The core localisation claim, in its simplest form."""
    z, theta, op = _equivariant(6, 24, seed=1)
    gen = torch.Generator().manual_seed(2)
    z = z.clone()
    lesion = 11
    z[:, lesion, :] = torch.randn(6, op.dimension, dtype=torch.float64, generator=gen)

    scores = transport_residual_scores(z, theta, op)
    assert (scores.argmax(dim=1) == lesion).all()


def test_a_lesion_raises_its_own_token_far_above_the_others() -> None:
    """Not merely ranked first: separated by orders of magnitude. A lesioned token disagrees with
    every reference, while each clean token disagrees with only the one lesioned reference and so
    is diluted by 1/(T-1)."""
    z, theta, op = _equivariant(5, 30, seed=3)
    gen = torch.Generator().manual_seed(4)
    z = z.clone()
    z[:, 15, :] = torch.randn(5, op.dimension, dtype=torch.float64, generator=gen)

    scores = transport_residual_scores(z, theta, op)
    peak = scores[:, 15]
    others = torch.cat([scores[:, :15], scores[:, 16:]], dim=1)
    assert (peak > 10 * others.max(dim=1).values).all()


# ============================================== what the ROTATION buys, held everything else fixed


def test_transport_beats_identity_when_the_latent_really_rotates_with_phase() -> None:
    """The decisive control. On an equivariant record the identity-transport statistic sees
    ordinary phase motion as anomaly everywhere, so its lesion contrast collapses; the
    phase-aware statistic sees only the lesion. Same tokens, same references, same averaging --
    the ONLY difference is whether the reference is rotated into the query's phase."""
    z, theta, op = _equivariant(8, 24, seed=5)
    gen = torch.Generator().manual_seed(6)
    z = z.clone()
    mask = torch.zeros(8, 24, dtype=torch.bool)
    mask[:, 10:13] = True
    z[mask] = torch.randn(int(mask.sum()), op.dimension, dtype=torch.float64, generator=gen)

    aware = within_record_auroc(transport_residual_scores(z, theta, op), mask)
    blind = within_record_auroc(identity_residual_scores(z, theta), mask)
    assert aware["mean_auroc"] > 0.99
    assert aware["mean_auroc"] > blind["mean_auroc"]


def test_identity_and_transport_agree_when_the_latent_does_not_move_with_phase() -> None:
    """The converse, which stops the previous test from being read as 'transport is always
    better': if the record's latent is phase-INVARIANT, R_Delta acts trivially on it and the two
    statistics are the same number. The advantage is contingent on there being phase structure,
    not automatic."""
    op = _operator()
    gen = torch.Generator().manual_seed(7)
    n, t = 4, 20
    z = torch.zeros(n, t, op.dimension, dtype=torch.float64)
    z[..., : op.k0] = torch.randn(n, t, op.k0, dtype=torch.float64, generator=gen)
    theta = torch.rand(n, t, dtype=torch.float64, generator=gen) * 2 * math.pi

    a = transport_residual_scores(z, theta, op)
    b = identity_residual_scores(z, theta)
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-12)


# ==================================================================================== causality


def test_causal_scores_use_only_the_past() -> None:
    """Changing a LATER token must not change an earlier token's causal score -- the executable
    definition of 'this could have been emitted online'."""
    z, theta, op = _equivariant(3, 20, seed=8)
    gen = torch.Generator().manual_seed(9)
    before = transport_residual_scores(z, theta, op, causal=True)
    z2 = z.clone()
    z2[:, 17, :] = torch.randn(3, op.dimension, dtype=torch.float64, generator=gen)
    after = transport_residual_scores(z2, theta, op, causal=True)
    np.testing.assert_allclose(before[:, :17].numpy(), after[:, :17].numpy(), atol=1e-12)
    assert (after[:, 17] > before[:, 17] + 0.1).all()


def test_causal_first_token_has_no_reference_and_is_nan() -> None:
    z, theta, op = _equivariant(2, 12, seed=10)
    scores = transport_residual_scores(z, theta, op, causal=True)
    assert torch.isnan(scores[:, 0]).all()
    assert torch.isfinite(scores[:, 1:]).all()


def test_window_bounds_the_reference_set() -> None:
    """A bounded-memory online detector only looks back `window` tokens; a lesion outside that
    window must therefore stop affecting the score."""
    z, theta, op = _equivariant(3, 30, seed=11)
    gen = torch.Generator().manual_seed(12)
    z = z.clone()
    z[:, 2, :] = torch.randn(3, op.dimension, dtype=torch.float64, generator=gen)
    scores = transport_residual_scores(z, theta, op, causal=True, window=4)
    # token 25 is far past the lesion and cannot see it
    np.testing.assert_allclose(scores[:, 25].numpy(), 0.0, atol=1e-6)
    assert (scores[:, 3] > 0.1).all()  # token 3 is inside the window and does see it


# ==================================================================== causal theta surrogate


def test_causal_theta_matches_offline_theta_at_constant_heart_rate() -> None:
    """With a constant RR the previous interval IS the current one, so the online surrogate is
    exact -- all of its error comes from RR variability, nothing else."""
    rpeaks = np.arange(0, 1000, 80, dtype=np.float64)
    offline = phase_from_rpeaks(rpeaks, 1000)[:, 0]
    causal = causal_phase_from_rpeaks(rpeaks, 1000)
    both = np.isfinite(offline) & np.isfinite(causal)
    assert both.sum() > 800
    np.testing.assert_allclose(causal[both], offline[both], atol=1e-9)


def test_causal_theta_errs_in_proportion_to_rr_change() -> None:
    """A beat 25% longer than its predecessor: the surrogate runs fast, and by the end of the beat
    it has advanced a quarter of a cycle too far. This is the quantity that bounds any real-time
    claim, so it is pinned rather than described.

    The error must be measured as a CIRCULAR distance. The surrogate divides by the shorter
    previous interval, so late in a lengthened beat it wraps past 2*pi back to a small angle; a
    plain subtraction then reports ~4.8 rad for what is really a 1.5 rad phase error, and would
    make the surrogate look far worse than it is."""
    rpeaks = np.array([0.0, 80.0, 160.0, 240.0, 340.0, 420.0], dtype=np.float64)  # 4th beat +25%
    offline = phase_from_rpeaks(rpeaks, 500)[:, 0]
    causal = causal_phase_from_rpeaks(rpeaks, 500)
    beat = np.arange(240, 340)
    diff = causal[beat] - offline[beat]
    err = np.abs((diff + math.pi) % (2 * math.pi) - math.pi)  # circular
    assert err[0] < 1e-9  # exact at the R-peak
    assert 0.2 * 2 * math.pi < err[-5:].max() < 0.3 * 2 * math.pi


def test_causal_theta_is_undefined_before_the_second_rpeak() -> None:
    rpeaks = np.array([50.0, 130.0, 210.0], dtype=np.float64)
    causal = causal_phase_from_rpeaks(rpeaks, 300)
    assert np.all(np.isnan(causal[:130]))
    assert np.all(np.isfinite(causal[130:210]))


def test_causal_theta_needs_three_peaks() -> None:
    assert np.all(np.isnan(causal_phase_from_rpeaks(np.array([10.0, 90.0]), 200)))


# ==================================================================================== metrics


def test_within_record_auroc_ignores_between_record_level_shifts() -> None:
    """The property the docstring claims: adding a big per-record constant, which would dominate a
    pooled AUROC, leaves the within-record answer untouched."""
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.1, 0.9, 0.2]])
    mask = torch.tensor([[False, True, False], [False, True, False]])
    base = within_record_auroc(scores, mask)
    shifted = within_record_auroc(scores + torch.tensor([[0.0], [100.0]]), mask)
    assert base["mean_auroc"] == pytest.approx(1.0)
    assert shifted["mean_auroc"] == pytest.approx(base["mean_auroc"])


def test_within_record_auroc_is_half_for_a_useless_detector() -> None:
    gen = torch.Generator().manual_seed(13)
    scores = torch.rand(200, 40, generator=gen)
    mask = torch.zeros(200, 40, dtype=torch.bool)
    mask[:, 10:15] = True
    assert within_record_auroc(scores, mask)["mean_auroc"] == pytest.approx(0.5, abs=0.03)


def test_within_record_auroc_skips_records_with_no_lesion() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.3, 0.4, 0.5]])
    mask = torch.tensor([[False, True, False], [False, False, False]])
    assert within_record_auroc(scores, mask)["n_records"] == 1


def test_localisation_error_is_zero_when_the_peak_is_inside_the_lesion() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.9, 0.3]])
    mask = torch.tensor([[False, False, True, False]])
    out = localisation_error(scores, mask, patch_ms=80.0)
    assert out["median_ms"] == 0.0
    assert out["hit_rate"] == 1.0


def test_localisation_error_counts_tokens_to_the_nearest_lesion_edge() -> None:
    scores = torch.tensor([[0.9, 0.1, 0.1, 0.2]])
    mask = torch.tensor([[False, False, True, True]])
    out = localisation_error(scores, mask, patch_ms=80.0)
    assert out["median_ms"] == pytest.approx(160.0)  # peak at 0, nearest lesion token at 2
    assert out["hit_rate"] == 0.0


def test_detection_latency_is_zero_for_an_instant_detector() -> None:
    scores = torch.zeros(1, 20)
    scores[0, 10:] = 5.0
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, 10:] = True
    out = detection_latency(scores, mask, patch_ms=80.0)
    assert out["median_ms"] == 0.0
    assert out["miss_rate"] == 0.0


def test_detection_latency_reports_a_miss_rather_than_a_fake_number() -> None:
    """A detector that never fires must not be scored as if it fired late.

    The post-onset scores are put strictly BELOW the pre-onset range. Pure noise would not do:
    the threshold is a pre-onset quantile, so an i.i.d. post-onset segment crosses it with
    probability ~1 by construction and would test nothing."""
    gen = torch.Generator().manual_seed(14)
    scores = torch.empty(1, 40)
    scores[0, :20] = 0.5 + torch.rand(20, generator=gen) * 0.5
    scores[0, 20:] = torch.rand(20, generator=gen) * 0.1
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[0, 20:] = True
    out = detection_latency(scores, mask, patch_ms=80.0)
    assert out["miss_rate"] == 1.0
    assert math.isnan(out["median_ms"])


def test_detection_latency_threshold_adapts_to_each_records_own_noise() -> None:
    """Two records with the same lesion contrast but very different baseline scales must get the
    same latency -- an absolute shared threshold would not."""
    quiet = torch.zeros(1, 40)
    quiet[0, 20:] = 1.0
    loud = torch.zeros(1, 40) + 50.0
    loud[0, 20:] = 51.0
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[0, 20:] = True
    a = detection_latency(quiet, mask, patch_ms=80.0)
    b = detection_latency(loud, mask, patch_ms=80.0)
    assert a["median_ms"] == b["median_ms"] == 0.0


# =================================================================================== baselines


def test_deviation_scores_flag_an_outlier_token() -> None:
    gen = torch.Generator().manual_seed(15)
    z = torch.randn(1, 1, 8, generator=gen).expand(1, 20, 8).clone()
    theta = torch.rand(1, 20, generator=gen) * 2 * math.pi
    z[0, 7] = -z[0, 7]
    assert int(deviation_scores(z, theta).argmax(dim=1)) == 7


def test_all_scorers_propagate_nan_theta_as_nan_and_never_crash() -> None:
    z, theta, op = _equivariant(3, 16, seed=16)
    theta = theta.clone()
    theta[:, :4] = float("nan")
    for scores in (
        transport_residual_scores(z, theta, op),
        identity_residual_scores(z, theta),
        deviation_scores(z, theta),
        transport_residual_scores(z, theta, op, causal=True),
    ):
        assert torch.isnan(scores[:, :4]).all()
        assert torch.isfinite(scores[:, 5:]).all()


def test_shape_validation() -> None:
    op = _operator()
    with pytest.raises(ValueError, match=r"\(N, T, K\)"):
        transport_residual_scores(torch.randn(4, op.dimension), torch.zeros(4), op)
    with pytest.raises(ValueError, match="theta shape"):
        transport_residual_scores(torch.randn(2, 5, op.dimension), torch.zeros(2, 4), op)


# ================================== the RADIAL detector: the norm channel the others discard


def test_radial_flags_a_norm_step_with_perfect_auroc_and_zero_localisation_error() -> None:
    """T1 sensitivity. A pure amplitude step (direction untouched) at a known onset: every other
    detector here is computed on unit-normalised latents and cannot see it; the radial score must
    put every lesioned token strictly above every clean one, giving AUROC 1, localisation error 0
    and latency 0 through the script's own aggregation path."""
    z, theta, _op = _equivariant(6, 24, seed=17)
    z = z.clone()
    onset = 16
    z[:, onset:, :] *= 3.0  # norm step: log-contrast is log 3 ~ 1.1
    mask = torch.zeros(6, 24, dtype=torch.bool)
    mask[:, onset:] = True

    for causal in (False, True):
        scores = radial_scores(z, theta, causal=causal)
        lesioned, clean = scores[mask], scores[~mask]
        clean = clean[torch.isfinite(clean)]  # causal token 0 has no reference
        assert (lesioned.min() > clean.max() + 0.5).item()  # peak claim, robust to ties
        assert within_record_auroc(scores, mask)["mean_auroc"] == pytest.approx(1.0)
        loc = localisation_error(scores, mask, patch_ms=80.0)
        assert loc["median_ms"] == 0.0
        assert loc["hit_rate"] == 1.0
        lat = detection_latency(scores, mask, patch_ms=80.0)
        assert lat["median_ms"] == 0.0
        assert lat["miss_rate"] == 0.0


def test_radial_is_numerically_silent_on_a_pure_rotation() -> None:
    """T2 specificity, part 1. Constant norm, direction advancing with theta: the radial channel
    carries nothing, so the score must be zero to roundoff at every valid token -- no threshold
    at any operating point could fire on it."""
    z, theta, _op = _equivariant(4, 24, seed=18)
    for causal in (False, True):
        scores = radial_scores(z, theta, causal=causal)
        finite = scores[torch.isfinite(scores)]
        np.testing.assert_allclose(finite.numpy(), 0.0, atol=1e-9)


def test_radial_is_at_chance_on_a_pure_rotation_under_a_random_mask() -> None:
    """T2 specificity, part 1 in AUROC form. The scores on a pure rotation are roundoff noise
    (~1e-15) which may correlate with token position, so the mask is randomised per record to
    make chance the exact expectation; 200 records pin the mean to 0.5 +/- ~0.012 (1 sigma)."""
    z, theta, _op = _equivariant(200, 24, seed=19)
    gen = torch.Generator().manual_seed(20)
    mask = torch.zeros(200, 24, dtype=torch.bool)
    lesion_at = torch.multinomial(torch.ones(200, 24), num_samples=3, generator=gen)
    mask.scatter_(1, lesion_at, True)
    for causal in (False, True):
        auroc = within_record_auroc(radial_scores(z, theta, causal=causal), mask)
        assert auroc["mean_auroc"] == pytest.approx(0.5, abs=0.06)


def test_radial_is_blind_to_the_directional_lesion_the_transport_detector_catches() -> None:
    """T2 specificity, part 2 -- orthogonal failure modes. A lesion that destroys the DIRECTION
    but preserves the norm exactly: the transport detector must catch it, the radial one must
    stay at roundoff (asserted on score magnitude, not rank -- ranking 1e-15 noise is
    meaningless)."""
    z, theta, op = _equivariant(6, 24, seed=21)
    z = z.clone()
    gen = torch.Generator().manual_seed(22)
    mask = torch.zeros(6, 24, dtype=torch.bool)
    mask[:, 11] = True
    keep = z[:, 11, :].norm(dim=-1, keepdim=True)
    rand = torch.randn(6, op.dimension, dtype=torch.float64, generator=gen)
    z[:, 11, :] = rand / (rand.norm(dim=-1, keepdim=True)) * keep

    aware = within_record_auroc(transport_residual_scores(z, theta, op), mask)
    assert aware["mean_auroc"] > 0.99
    for causal in (False, True):
        scores = radial_scores(z, theta, causal=causal)
        finite = scores[torch.isfinite(scores)]
        np.testing.assert_allclose(finite.numpy(), 0.0, atol=1e-9)


def test_radial_excludes_invalid_theta_tokens_from_reference_and_output() -> None:
    """T3 NaN hygiene, in its executable form: mutating the latents of invalid-theta tokens must
    change NOTHING -- not the scores at valid tokens (so invalid tokens never enter a reference)
    and not the NaN pattern. A huge norm is parked behind the invalid thetas to make any leak
    loud."""
    z, theta, _op = _equivariant(3, 16, seed=23)
    theta = theta.clone()
    theta[:, :4] = float("nan")
    spiked = z.clone()
    spiked[:, :4, :] *= 100.0
    for causal in (False, True):
        base = radial_scores(z, theta, causal=causal)
        after = radial_scores(spiked, theta, causal=causal)
        np.testing.assert_allclose(base.numpy(), after.numpy(), atol=1e-12)  # equal_nan default
        assert torch.isnan(base[:, :4]).all()
        # causal token 4 has no valid past and is NaN; offline token 4 has a reference
        assert torch.isfinite(base[:, 5:]).all()
        if not causal:
            assert torch.isfinite(base[:, 4]).all()


def test_radial_causal_uses_only_the_past() -> None:
    """Same executable definition of causality as the transport detector's own test: a LATER norm
    spike must not change any earlier score, and must raise its own."""
    z, theta, _op = _equivariant(3, 20, seed=24)
    before = radial_scores(z, theta, causal=True)
    z2 = z.clone()
    z2[:, 17, :] *= 4.0
    after = radial_scores(z2, theta, causal=True)
    np.testing.assert_allclose(before[:, :17].numpy(), after[:, :17].numpy(), atol=1e-12)
    assert (after[:, 17] > before[:, 17] + 0.1).all()


def test_radial_causal_window_bounds_the_reference_median() -> None:
    """A bounded look-back must forget an early norm regime. Tokens 0..9 sit at 5x norm: at token
    15 the EXPANDING past median still lives on the old regime (10 of 15 past tokens), so the
    clean token reads as anomalous; with window=4 the reference is the 4 clean neighbours and the
    score is zero."""
    z, theta, _op = _equivariant(3, 30, seed=25)
    z = z.clone()
    z[:, :10, :] *= 5.0
    expanding = radial_scores(z, theta, causal=True)
    bounded = radial_scores(z, theta, causal=True, window=4)
    assert (expanding[:, 15] > 1.0).all()  # |log 1 - log 5| ~ 1.61
    np.testing.assert_allclose(bounded[:, 15].numpy(), 0.0, atol=1e-9)


def test_radial_rejects_a_window_without_causal() -> None:
    """The offline reference is the record-level median BY DEFINITION; a windowed offline radial
    is not a config the battery emits, so it is refused rather than given invented semantics."""
    z, theta, _op = _equivariant(2, 12, seed=26)
    with pytest.raises(ValueError, match="causal"):
        radial_scores(z, theta, window=4)


def test_radial_shape_validation() -> None:
    with pytest.raises(ValueError, match=r"\(N, T, K\)"):
        radial_scores(torch.randn(4, 8), torch.zeros(4))
    with pytest.raises(ValueError, match="theta shape"):
        radial_scores(torch.randn(2, 5, 8), torch.zeros(2, 4))

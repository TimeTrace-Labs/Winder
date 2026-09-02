"""Tests for scripts/build_phase_tokens.py's own logic: the feasible-spectrum rounding rule,
the histogram/independence statistics, and the negative control's teeth -- all pure functions,
no PTB-XL data dependency. Adapted from ttl-phase's tests/test_m0_phase_calibration.py.
"""

import importlib.util
import math
import os
import types

import numpy as np
import pytest

from winder.data.phase import TWO_PI

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "build_phase_tokens.py",
)


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("build_phase_tokens", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bpt = _load_script()


# ============================================================== the pre-registered rounding rule


def test_feasible_spectrum_exact_division_no_remainder() -> None:
    n_max, k_j = bpt._feasible_spectrum(n_max_raw=6.0, k0=4, k_total=256)
    assert n_max == 6
    assert k_j == [21, 21, 21, 21, 21, 21]
    assert 4 + 2 * sum(k_j) == 256


def test_feasible_spectrum_remainder_goes_to_lowest_harmonics() -> None:
    # budget = (256-4)//2 = 126; n_max=5 -> 126/5 = 25 remainder 1 -> one harmonic gets 26.
    n_max, k_j = bpt._feasible_spectrum(n_max_raw=5.0, k0=4, k_total=256)
    assert n_max == 5
    assert k_j == [26, 25, 25, 25, 25]  # remainder (1) added to n_j=1, the lowest harmonic
    assert 4 + 2 * sum(k_j) == 256


def test_feasible_spectrum_rounds_to_nearest_integer() -> None:
    n_max, _ = bpt._feasible_spectrum(n_max_raw=5.81, k0=4, k_total=256)
    assert n_max == 6  # round(5.81) == 6, not floor or ceil


def test_feasible_spectrum_clamps_to_the_budget() -> None:
    # budget = 126; requesting far more than that must clamp, not divide by an oversized n_max.
    n_max, k_j = bpt._feasible_spectrum(n_max_raw=500.0, k0=4, k_total=256)
    assert n_max == 126
    assert k_j == [1] * 126


def test_feasible_spectrum_clamps_to_at_least_one() -> None:
    n_max, k_j = bpt._feasible_spectrum(n_max_raw=0.0, k0=4, k_total=256)
    assert n_max == 1
    assert k_j == [126]


# ========================================================================== histogram statistics


def test_uniformity_ratio_of_a_flat_histogram_is_one() -> None:
    hist = np.full(36, 100, dtype=np.int64)
    assert bpt._uniformity_ratio(hist) == pytest.approx(1.0)


def test_uniformity_ratio_of_a_spiked_histogram_exceeds_one() -> None:
    hist = np.ones(36, dtype=np.int64)
    hist[0] = 36 * 10  # one bin holds 10x the flat share
    assert bpt._uniformity_ratio(hist) > 5.0


def test_cramers_v_of_an_independent_table_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    row_marginal = rng.multinomial(200_000, np.full(12, 1 / 12))
    col_marginal = rng.multinomial(200_000, np.full(36, 1 / 36))
    # An exactly independent table (outer product of marginals, scaled) -- Cramer's V must be 0
    # up to the rounding these integer counts introduce.
    joint = np.outer(row_marginal, col_marginal) / 200_000
    assert bpt._cramers_v(joint) < 0.01


def test_cramers_v_of_a_deterministic_table_is_near_one() -> None:
    # row i always co-occurs with column i (i < 12): a perfect association.
    joint = np.zeros((12, 36), dtype=np.int64)
    for i in range(12):
        joint[i, i] = 1000
    assert bpt._cramers_v(joint) == pytest.approx(1.0, abs=1e-6)


# ================================================================= the negative control has teeth


def test_fixed_lag_sampler_is_more_non_uniform_than_all_pairs() -> None:
    """Reproduces the predecessor prototype's own defect on a small synthetic corpus: a
    fixed-token-index-lag pair sampler correlates Delta with local RR/heart-rate, while
    all-pairs-within-record does not. This is the check the real-corpus run is pre-registered
    against (`negative_control_has_teeth`), exercised here on data whose ground truth we set
    ourselves rather than trusting the real corpus to happen to demonstrate it."""
    n_records, n_tokens = 40, bpt.N_TOKENS
    rng = np.random.default_rng(1)
    theta = np.full((n_records, n_tokens), np.nan, dtype=np.float32)
    for r in range(n_records):
        # Each record has its OWN, constant RR (a fixed angular increment per token) -- so a
        # fixed-lag pair always returns close to the SAME Delta (that record's own per-token
        # angular rate times the lag), while all-pairs mixes contributions from every record's
        # differing rate, uniformly averaging out.
        rate = rng.uniform(0.05, 0.5)  # rad/token, deliberately varies record to record
        start = rng.uniform(0, TWO_PI)
        theta[r] = np.mod(start + rate * np.arange(n_tokens), TWO_PI)

    hist = bpt._accumulate_pair_histograms(theta)
    all_pairs_ratio = bpt._uniformity_ratio(hist["all_pairs_delta_hist"])
    fixed_lag_ratio = bpt._uniformity_ratio(hist["fixed_lag_hist"])
    assert fixed_lag_ratio > all_pairs_ratio


# ============================================================================ finite-T leakage


def test_finite_t_leakage_of_equidistributed_theta_is_near_zero() -> None:
    """Prop 4.1's exact-annihilation limit, approached at finite T: equally spaced theta over
    exactly one or more full cycles drives |phi_hat_T(n)| to (near) machine-precision 0 for every
    integer n < T, the discrete-orthogonality mechanism the harmonic-annihilation identity
    relies on."""
    t = 125
    theta = np.tile(np.arange(t, dtype=np.float32) * (TWO_PI / t), (3, 1))
    leakage = bpt._finite_t_leakage(theta, n_max_probe=6)
    for n in range(1, 7):
        assert leakage[n]["median"] < 1e-5


def test_finite_t_leakage_of_a_single_repeated_phase_is_near_one() -> None:
    """Negative control: if every token in a record sits at the SAME phase, the empirical
    characteristic function does not average anything away -- |phi_hat_T(n)| == 1 exactly for
    every n, regardless of T."""
    theta = np.full((2, bpt.N_TOKENS), 1.23, dtype=np.float32)
    leakage = bpt._finite_t_leakage(theta, n_max_probe=3)
    for n in range(1, 4):
        assert leakage[n]["median"] == pytest.approx(1.0, abs=1e-5)


def test_finite_t_leakage_ignores_nan_tokens() -> None:
    theta = np.full((1, bpt.N_TOKENS), np.nan, dtype=np.float32)
    theta[0, :10] = np.arange(10, dtype=np.float32) * (TWO_PI / 125)  # a short equidistributed run
    leakage = bpt._finite_t_leakage(theta, n_max_probe=1)
    assert math.isfinite(leakage[1]["median"])  # must not be NaN-poisoned by the missing tokens


# ================================================================================= theta building


def test_build_theta_tokens_restricts_to_included_records() -> None:
    import pandas as pd

    manifest = pd.DataFrame({"ecg_id": [1, 2, 3], "status": ["included", "excluded", "included"]})
    rpeaks_by_id = {
        1: np.array([0.0, 200.0, 400.0, 600.0]),
        2: np.array([0.0, 200.0, 400.0, 600.0]),
        3: np.array([0.0, 200.0, 400.0, 600.0]),
    }
    ecg_ids, theta = bpt._build_theta_tokens(manifest, rpeaks_by_id)
    assert list(ecg_ids) == [1, 3]
    assert theta.shape == (2, bpt.N_TOKENS)


def test_build_theta_tokens_handles_a_missing_or_degenerate_rpeaks_entry() -> None:
    import pandas as pd

    manifest = pd.DataFrame({"ecg_id": [1, 2], "status": ["included", "included"]})
    rpeaks_by_id = {1: np.array([100.0])}  # ecg_id=2 missing entirely; ecg_id=1 has < 2 peaks
    ecg_ids, theta = bpt._build_theta_tokens(manifest, rpeaks_by_id)
    assert list(ecg_ids) == [1, 2]
    assert np.all(np.isnan(theta))  # neither record has an enclosing R-R interval

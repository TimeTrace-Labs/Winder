"""Tests for winder.eval.descriptors -- the subset ported for build_phase_tokens.py (M0): token/
R-peak timestamp arithmetic. Adapted from ttl-phase's tests/test_eval_descriptors.py, restricted
to the 5 functions this repo actually ports (see descriptors.py's own module docstring for what
was deliberately left out and why).
"""

import math
import os

import numpy as np
import pytest

from winder.data.phase import phase_from_rpeaks
from winder.eval.descriptors import (
    load_rpeaks_by_ecg_id,
    rpeaks_at_output_rate,
    theta_at_tokens,
    token_centre_sample,
    token_last_sample,
)


def test_load_rpeaks_by_ecg_id_unpacks_ragged_archive(tmp_path: object) -> None:
    ecg_ids = np.array([5, 9])
    rpeaks = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    offsets = np.array([0, 3, 5])
    npz_path = os.path.join(str(tmp_path), "rpeaks.npz")
    np.savez(npz_path, ecg_ids=ecg_ids, offsets=offsets, rpeaks=rpeaks, fs=500)

    out = load_rpeaks_by_ecg_id(npz_path)
    assert set(out) == {5, 9}
    np.testing.assert_array_equal(out[5], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(out[9], [40.0, 50.0])


def test_rpeaks_at_output_rate_divides_by_decimation_factor() -> None:
    rpeaks_500 = np.array([100.0, 600.0, 1100.0])
    rpeaks_100 = rpeaks_at_output_rate(rpeaks_500, decimation_factor=5.0)
    np.testing.assert_allclose(rpeaks_100, [20.0, 120.0, 220.0])


def test_rpeaks_at_output_rate_leaves_theta_unchanged() -> None:
    """The whole point of the rescale (module docstring): theta at a given instant must not
    depend on which grid it's evaluated on."""
    rpeaks_500 = np.array([100.0, 600.0, 1100.0, 1600.0])
    theta_500 = phase_from_rpeaks(rpeaks_500, n_samples=2000)

    rpeaks_100 = rpeaks_at_output_rate(rpeaks_500, decimation_factor=5.0)
    theta_100 = phase_from_rpeaks(rpeaks_100, n_samples=400)

    # sample t at 100 Hz <-> sample 5t at 500 Hz (decimate_to's own "no timing shift" contract)
    for t100 in range(400):
        t500 = 5 * t100
        if math.isnan(theta_100[t100, 0]):
            assert math.isnan(theta_500[t500, 0])
        else:
            assert theta_100[t100, 0] == pytest.approx(theta_500[t500, 0], abs=1e-9)


def test_theta_at_tokens_matches_a_directly_scaled_computation() -> None:
    rpeaks_500 = np.array([100.0, 600.0, 1100.0, 1600.0])
    n_tokens, n_samples = 50, 1000
    theta_tok = theta_at_tokens(rpeaks_500, n_tokens, n_samples, decimation_factor=5.0)
    assert theta_tok.shape == (n_tokens,)

    rpeaks_100 = rpeaks_500 / 5.0
    theta_full = phase_from_rpeaks(rpeaks_100, n_samples)[:, 0]
    for j in range(n_tokens):
        pos = token_last_sample(j)
        expected = theta_full[pos]
        if math.isnan(expected):
            assert math.isnan(theta_tok[j])
        else:
            assert theta_tok[j] == pytest.approx(expected)


def test_theta_at_tokens_centre_matches_a_directly_scaled_computation() -> None:
    rpeaks_500 = np.array([100.0, 600.0, 1100.0, 1600.0])
    n_tokens, n_samples = 50, 1000
    theta_tok = theta_at_tokens(
        rpeaks_500, n_tokens, n_samples, decimation_factor=5.0, timestamp="centre"
    )
    assert theta_tok.shape == (n_tokens,)

    rpeaks_100 = rpeaks_500 / 5.0
    theta_full = phase_from_rpeaks(rpeaks_100, n_samples)[:, 0]
    for j in range(n_tokens):
        pos = round(token_centre_sample(j))
        expected = theta_full[pos]
        if math.isnan(expected):
            assert math.isnan(theta_tok[j])
        else:
            assert theta_tok[j] == pytest.approx(expected)


def test_theta_at_tokens_last_is_unaffected_by_the_new_kwarg() -> None:
    """Default behaviour must be bit-identical to before `timestamp` existed -- every existing
    caller omits it."""
    rpeaks_500 = np.array([100.0, 600.0, 1100.0, 1600.0])
    n_tokens, n_samples = 50, 1000
    default = theta_at_tokens(rpeaks_500, n_tokens, n_samples, decimation_factor=5.0)
    explicit_last = theta_at_tokens(
        rpeaks_500, n_tokens, n_samples, decimation_factor=5.0, timestamp="last"
    )
    np.testing.assert_array_equal(default, explicit_last)


def test_theta_at_tokens_rejects_unknown_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        theta_at_tokens(
            np.array([100.0, 600.0]),
            10,
            1000,
            timestamp="bogus",  # type: ignore[arg-type]
        )


def test_token_centre_sample_is_the_midpoint_of_its_own_patch() -> None:
    for j in (0, 1, 29, 100, 124):
        first, last = j * 8, j * 8 + 7  # default patch_width=8
        assert token_centre_sample(j) == pytest.approx((first + last) / 2.0)
        assert token_centre_sample(j) == pytest.approx(j * 8 + 3.5)


def test_token_last_sample_is_the_last_sample_of_its_own_patch() -> None:
    for j in (0, 1, 29, 100, 124):
        assert token_last_sample(j) == (j + 1) * 8 - 1  # default patch_width=8

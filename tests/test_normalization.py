"""normalization.py tests."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from winder.data.normalization import (
    NORM_REGISTRY,
    CorpusStatsNormConfig,
    NormConfig,
    PerBeatNormConfig,
    RawNormConfig,
    apply_corpus_stats,
    apply_perbeat,
    apply_raw,
    beat_rms,
    normalize,
    resolve_norm_config,
)


def test_raw_is_a_no_op() -> None:
    sig = np.random.default_rng(0).normal(size=(100, 3))
    out = apply_raw(sig, np.array([10.0, 50.0, 90.0]), RawNormConfig())
    assert np.array_equal(out, sig)


def test_beat_rms_shared_across_leads() -> None:
    # two leads, one twice the amplitude of the other -- RMS must be one scalar per beat,
    # not one per lead, so it doesn't equalise the two leads' amplitudes.
    t = np.arange(200, dtype=np.float64)
    lead_a = np.sin(t / 5.0)
    lead_b = 2.0 * np.sin(t / 5.0)
    sig = np.stack([lead_a, lead_b], axis=1)
    rpeaks = np.array([0.0, 100.0, 200.0])
    rms = beat_rms(sig, rpeaks, np.arange(200, dtype=np.float64))
    out = sig / rms[:, None]
    # lead_b should still be ~2x lead_a after normalisation (ratio preserved)
    ratio = out[50, 1] / out[50, 0]
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_perbeat_preserves_interlead_ratio_but_changes_absolute_scale() -> None:
    t = np.arange(500, dtype=np.float64)
    sig = np.stack([np.sin(t / 8.0), 3.0 * np.sin(t / 8.0)], axis=1) * 5.0  # 5 mV-ish scale
    rpeaks = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    out = apply_perbeat(sig, rpeaks, PerBeatNormConfig())
    # absolute scale changed (normalised to ~unit RMS per beat)...
    assert np.abs(out).max() < np.abs(sig).max()
    # ...but the inter-lead ratio is preserved (both leads divided by the same scalar)
    nonzero = np.abs(sig[:, 0]) > 1e-6
    ratio_before = sig[nonzero, 1] / sig[nonzero, 0]
    ratio_after = out[nonzero, 1] / out[nonzero, 0]
    assert np.allclose(ratio_before, ratio_after, rtol=1e-6)


def test_perbeat_floor_divides_by_one_not_by_the_floor() -> None:
    """Matches build_pool's exact semantics: below scale_floor, divide by 1.0 (a no-op),
    not by the floor value itself.

    Audit-found: the original version of this test used an all-zero signal, whose RMS is
    0 under EITHER the correct (divide-by-1.0) or the wrong (divide-by-floor-value)
    semantics -- 0 divided by anything is still 0, so it could not actually distinguish
    the two implementations. This version uses a non-zero, exactly-known RMS (a constant
    0.5 mV beat) against a floor far above it, where the two semantics give different,
    non-zero answers (0.5 vs. 0.05)."""
    sig = np.full((30, 1), 0.5)  # constant beat -> RMS exactly 0.5
    rpeaks = np.array([0.0, 30.0])
    floor = 10.0  # far above the 0.5 RMS, so the floor branch is taken
    out = apply_perbeat(sig, rpeaks, PerBeatNormConfig(scale_floor=floor))
    assert np.allclose(out, 0.5)  # divide by 1.0: 0.5 / 1.0 == 0.5
    assert not np.allclose(out, 0.05)  # would be 0.5 / 10.0 == 0.05 under the wrong semantics


def test_beat_rms_with_fewer_than_two_rpeaks_falls_back_to_whole_signal_rms() -> None:
    sig = np.array([[1.0], [2.0], [3.0]])
    rms = beat_rms(sig, np.array([5.0]), np.array([0.0, 1.0, 2.0]))
    expected = np.sqrt((sig**2).mean())
    assert np.allclose(rms, expected)


def test_beat_rms_rejects_1d_signal_instead_of_silently_broadcasting() -> None:
    """Audit-found: a 1-D (T,) signal used to silently broadcast against beat_rms's
    (T,1)-shaped divisor into a (T,T) array with no error -- must now raise instead."""
    sig_1d = np.ones(100)
    with pytest.raises(ValueError, match="2-D"):
        beat_rms(sig_1d, np.array([0.0, 50.0, 100.0]), np.arange(100, dtype=np.float64))


def test_apply_perbeat_rejects_1d_signal() -> None:
    with pytest.raises(ValueError, match="2-D"):
        apply_perbeat(np.ones(100), np.array([0.0, 50.0, 100.0]), PerBeatNormConfig())


# ------------------------------------------------------------------------- tag + registry
def test_norm_config_mode_is_required() -> None:
    cfg = OmegaConf.structured(NormConfig)
    assert OmegaConf.is_missing(cfg, "mode")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.mode


def test_resolve_norm_config_round_trips_perbeat_override() -> None:
    norm = NormConfig(mode="perbeat", params={"scale_floor": 1e-6})
    resolved = resolve_norm_config(norm)
    assert resolved.scale_floor == 1e-6


def test_resolve_norm_config_defaults_when_no_params_given() -> None:
    norm = NormConfig(mode="raw", params={})
    resolved = resolve_norm_config(norm)
    assert isinstance(OmegaConf.to_object(resolved), RawNormConfig)


def test_normalize_dispatches_by_mode() -> None:
    sig = np.ones((10, 1))
    rpeaks = np.array([0.0, 10.0])
    out_raw = normalize("raw", sig, rpeaks, RawNormConfig())
    assert np.array_equal(out_raw, sig)
    out_pb = normalize("perbeat", sig, rpeaks, PerBeatNormConfig())
    assert out_pb.shape == sig.shape


def test_norm_registry_has_exactly_raw_perbeat_and_corpus_stats() -> None:
    assert set(NORM_REGISTRY) == {"raw", "perbeat", "corpus_stats"}


def test_corpus_stats_mean_and_std_are_mandatory() -> None:
    norm = NormConfig(mode="corpus_stats", params={})
    resolved = resolve_norm_config(norm)
    with pytest.raises(MissingMandatoryValue):
        _ = resolved.mean_mv
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_object(resolved)


def test_corpus_stats_applies_zscore_with_floor_semantics() -> None:
    sig = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])  # (T=3, leads=2)
    config = CorpusStatsNormConfig(mean_mv=[2.0, 0.0], std_mv=[2.0, 1e-9], scale_floor=1e-6)
    out = apply_corpus_stats(sig, np.array([0.0, 3.0]), config)
    # lead 0: (x - 2) / 2 -> [-1, 0, 1]
    assert np.allclose(out[:, 0], [-1.0, 0.0, 1.0])
    # lead 1: std=1e-9 is below scale_floor -> divisor is 1.0 (no-op), not the tiny std itself
    assert np.allclose(out[:, 1], [10.0, 10.0, 10.0])


def test_corpus_stats_is_corpus_level_not_per_record() -> None:
    """The whole point of this normalizer: a shared per-lead divisor preserves the amplitude
    RATIO between two records, unlike perbeat's per-record RMS which destroys it."""
    config = CorpusStatsNormConfig(mean_mv=[0.0], std_mv=[1.0], scale_floor=1e-9)
    record_a = np.array([[1.0], [2.0], [3.0]])
    record_b = 2.0 * record_a  # a record with exactly double the amplitude
    rpeaks = np.array([0.0, 3.0])

    out_a = apply_corpus_stats(record_a, rpeaks, config)
    out_b = apply_corpus_stats(record_b, rpeaks, config)
    assert np.allclose(out_b, 2.0 * out_a)  # ratio preserved -- shared divisor, not per-record

    perbeat_a = apply_perbeat(record_a, rpeaks, PerBeatNormConfig(scale_floor=1e-9))
    perbeat_b = apply_perbeat(record_b, rpeaks, PerBeatNormConfig(scale_floor=1e-9))
    assert np.allclose(perbeat_a, perbeat_b)  # ratio destroyed -- perbeat rescales each record


def test_corpus_stats_wrong_lead_count_raises() -> None:
    config = CorpusStatsNormConfig(mean_mv=[0.0, 0.0], std_mv=[1.0, 1.0], scale_floor=1e-6)
    sig = np.ones((5, 3))  # 3 leads, but config has stats for 2
    with pytest.raises(ValueError, match="leads"):
        apply_corpus_stats(sig, np.array([0.0, 5.0]), config)


def test_corpus_stats_registered_dispatch() -> None:
    config = CorpusStatsNormConfig(mean_mv=[0.0], std_mv=[1.0])
    sig = np.array([[1.0], [2.0]])
    out = normalize("corpus_stats", sig, np.array([0.0, 2.0]), config)
    assert np.allclose(out, sig)


def test_unregistered_mode_raises_key_error() -> None:
    """Pins the tag+registry failure mode this module was modeled on -- see
    test_operators.py::test_unknown_operator_name_is_missing_from_registry for the
    identical pattern in the operator registry."""
    norm = NormConfig(mode="not_a_real_mode", params={})
    with pytest.raises(KeyError):
        resolve_norm_config(norm)
    with pytest.raises(KeyError):
        normalize("not_a_real_mode", np.ones((10, 1)), np.array([0.0, 10.0]), RawNormConfig())

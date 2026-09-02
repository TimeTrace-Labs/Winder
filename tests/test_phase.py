"""phase.py tests.

Covers: the frozen theta/bin_id contract (adapted from ttl-phase's
`test_frozen_contract_signatures_are_positionally_callable` / `test_frozen_return_shapes`,
updated for extract_phase's B-free signature), bin_phase/phase_from_rpeaks boundary
behaviour, a synthetic-ground-truth detector sanity check, and the Tier 1/2 golden tests
against real PTB-XL records.
"""

import dataclasses
import inspect
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from _synthetic import evaluate_detector, synthetic_ecg

from winder.data.phase import (
    BIN_EXCLUDE,
    FLAG_FLAT_SIGNAL,
    FLAG_HIGH_RR_CV,
    FLAG_LOW_CONFIDENCE,
    FLAG_LOW_YIELD,
    FLAG_NO_BEATS,
    FLAG_TOO_FEW_BEATS,
    TWO_PI,
    DetectorParams,
    PhaseQCConfig,
    bin_phase,
    detect_rpeaks,
    extract_phase,
    phase_from_rpeaks,
)
from winder.data.wfdb_io import read_record

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# ------------------------------------------------------------------- frozen contract shape
def test_extract_phase_is_positionally_callable_as_sig_fs() -> None:
    params = list(inspect.signature(extract_phase).parameters.values())
    assert params[0].name == "sig"
    assert params[1].name == "fs"
    assert params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD)
    assert params[1].kind in (params[1].POSITIONAL_ONLY, params[1].POSITIONAL_OR_KEYWORD)
    # every parameter after the frozen (sig, fs) pair must be optional
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, f"{p.name} must have a default"


def test_bin_phase_is_positionally_callable_as_theta_b() -> None:
    params = list(inspect.signature(bin_phase).parameters.values())
    assert params[0].name == "theta"
    assert params[1].name == "B"


def test_frozen_return_shapes_and_bin_phase_consistency() -> None:
    sig, _true_rpeaks = synthetic_ecg(n_beats=10, fs=500, seed=0)
    res = extract_phase(sig, fs=500)

    assert res.theta.ndim == 2 and res.theta.shape[1] == 1
    assert res.theta.shape[0] == sig.shape[0]
    finite = np.isfinite(res.theta)
    assert np.all((res.theta[finite] >= 0.0) & (res.theta[finite] < TWO_PI))
    assert isinstance(res.quality, dict)
    assert isinstance(res.n_beats, int)

    bin_id = bin_phase(res.theta, 8)
    assert bin_id.shape == (sig.shape[0],)
    assert bin_id.dtype == np.int64
    assert np.array_equal(np.isnan(res.theta[:, 0]), bin_id == BIN_EXCLUDE)


# --------------------------------------------------------------------------- bin_phase
def test_bin_phase_sentinel_is_not_bin_zero() -> None:
    theta = np.array([[np.nan], [0.0], [np.pi]])
    bins = bin_phase(theta, 4)
    assert bins[0] == BIN_EXCLUDE
    assert bins[1] == 0  # a real phase-0 sample must land in bin 0, not be confused with NaN
    assert bins[2] == 2


def test_bin_phase_edges_are_left_closed() -> None:
    width = TWO_PI / 4
    theta = np.array([[0.0], [width - 1e-9], [width], [TWO_PI - 1e-9]])
    bins = bin_phase(theta, 4)
    assert bins.tolist() == [0, 0, 1, 3]


def test_bin_phase_rejects_out_of_range_theta() -> None:
    with pytest.raises(ValueError):
        bin_phase(np.array([[-0.1]]), 4)
    with pytest.raises(ValueError):
        bin_phase(np.array([[TWO_PI + 1.0]]), 4)


def test_bin_phase_tolerates_float_rounding_at_two_pi() -> None:
    # exactly TWO_PI (a plausible float rounding artefact of a wrapped angle) falls inside
    # bin_phase's 1e-12 slack and lands in the last bin rather than raising -- intentional.
    bins = bin_phase(np.array([[TWO_PI]]), 4)
    assert bins[0] == 3


def test_bin_phase_flattens_multi_axis_row_major() -> None:
    # d=2, B=(2, 3): index (1, 2) should flatten to 1*3 + 2 = 5
    theta = np.array([[np.pi * 1.5, TWO_PI * 5 / 6 + 1e-6]])
    bins = bin_phase(theta, (2, 3))
    assert bins[0] == 5


def test_bin_phase_flattens_three_axes_row_major() -> None:
    """Audit-found: the multi-axis flatten was only tested at d=2; theta is explicitly a
    vector precisely to support d>=2 (a future respiratory clock), so d=3 should be
    exercised too. B=(2,3,4), index (1,2,3) -> 1*(3*4) + 2*4 + 3 = 23."""
    theta = np.array([[np.pi * 1.5, TWO_PI * 5 / 6 + 1e-6, TWO_PI * 7 / 8 + 1e-6]])
    bins = bin_phase(theta, (2, 3, 4))
    assert bins[0] == 23


def test_bin_phase_accepts_numpy_integer_scalar_b() -> None:
    """Regression: an early port of isinstance(B, int) (replacing ttl-phase's
    np.isscalar(B)) rejected numpy integer scalars, which are not Python `int` instances,
    falling through to the tuple-unpacking branch and crashing with "not iterable" --
    found independently by three separate audit passes."""
    theta = np.array([[0.5]])
    assert bin_phase(theta, np.int64(4)) == bin_phase(theta, 4)
    assert bin_phase(theta, np.int32(4)) == bin_phase(theta, 4)


# --------------------------------------------------------------------- phase_from_rpeaks
def test_phase_from_rpeaks_boundaries_are_nan() -> None:
    rpeaks = np.array([10.0, 20.0, 30.0])
    theta = phase_from_rpeaks(rpeaks, n_samples=40)
    assert np.isnan(theta[:10, 0]).all()  # before the first R-peak
    assert np.isnan(theta[30:, 0]).all()  # at/after the last R-peak
    assert np.isfinite(theta[10:30, 0]).all()
    assert theta[10, 0] == pytest.approx(0.0)
    assert theta[19, 0] == pytest.approx(TWO_PI * 9 / 10)


def test_phase_from_rpeaks_needs_at_least_two_peaks() -> None:
    theta = phase_from_rpeaks(np.array([5.0]), n_samples=10)
    assert np.isnan(theta).all()


# ---------------------------------------------------------------------- detector sanity
def test_detector_recovers_synthetic_peaks_at_moderate_snr() -> None:
    sig, true_rpeaks = synthetic_ecg(n_beats=20, fs=500, snr_db=15.0, seed=0)
    rpeaks, _coarse, _info = detect_rpeaks(sig, fs=500)
    scores = evaluate_detector(true_rpeaks, rpeaks, fs=500)
    assert scores["sensitivity"] >= 0.95
    assert scores["ppv"] >= 0.95
    assert scores["rms_ms"] < 5.0  # sub-sample refinement should get well under one sample


# ------------------------------------------------------------- determinism (plan-required)
def test_detector_is_independent_of_global_rng_state() -> None:
    """The plan explicitly requires this: the detector must only ever draw randomness
    from an explicitly-seeded np.random.default_rng (jitter_estimate), never the global
    numpy RNG -- so intervening global-RNG draws between two calls must not perturb the
    result. Audit-found: this test never existed."""
    sig, _true = synthetic_ecg(n_beats=15, fs=500, seed=0)

    rpeaks_a, coarse_a, _ = detect_rpeaks(sig, fs=500)
    np.random.seed(12345)
    np.random.rand(10_000)
    np.random.standard_normal(500)
    rpeaks_b, coarse_b, _ = detect_rpeaks(sig, fs=500)

    assert np.array_equal(rpeaks_a, rpeaks_b)
    assert np.array_equal(coarse_a, coarse_b)

    res_a = extract_phase(sig, fs=500, estimate_jitter=True, jitter_seed=7)
    np.random.seed(999)
    np.random.rand(5000)
    res_b = extract_phase(sig, fs=500, estimate_jitter=True, jitter_seed=7)
    assert np.array_equal(res_a.rpeaks, res_b.rpeaks)
    assert res_a.quality["jitter_ms"] == res_b.quality["jitter_ms"]


def test_detector_is_independent_of_omp_num_threads() -> None:
    """The plan explicitly requires this: BLAS/OpenMP thread count must not change
    detection output (sosfiltfilt/uniform_filter1d can dispatch into threaded BLAS on some
    builds). Thread pools for numerical libraries are configured at process/library-load
    time, so this runs the detector in two subprocesses with different OMP_NUM_THREADS
    rather than trying to change it mid-process. Audit-found: this test never existed."""
    sig, _true = synthetic_ecg(n_beats=15, fs=500, seed=0)
    winder_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys\n"
        "import numpy as np\n"
        "from winder.data.phase import detect_rpeaks\n"
        "sig = np.load(sys.argv[1])\n"
        "rpeaks, _coarse, _info = detect_rpeaks(sig, fs=500)\n"
        "np.save(sys.argv[2], rpeaks)\n"
    )
    with tempfile.TemporaryDirectory() as td:
        sig_path = os.path.join(td, "sig.npy")
        script_path = os.path.join(td, "run.py")
        np.save(sig_path, sig)
        with open(script_path, "w") as f:
            f.write(script)

        outputs = []
        for threads in ("1", "4"):
            out_path = os.path.join(td, f"out_{threads}.npy")
            env = dict(
                os.environ,
                OMP_NUM_THREADS=threads,
                OPENBLAS_NUM_THREADS=threads,
                MKL_NUM_THREADS=threads,
            )
            subprocess.run(
                [sys.executable, script_path, sig_path, out_path],
                check=True,
                env=env,
                cwd=winder_root,
            )
            outputs.append(np.load(out_path))

        assert np.array_equal(outputs[0], outputs[1])


def test_extract_phase_flags_too_few_beats() -> None:
    sig, _true = synthetic_ecg(n_beats=3, fs=500, seed=0)
    res = extract_phase(sig, fs=500, qc=PhaseQCConfig(min_beats=5))
    assert FLAG_TOO_FEW_BEATS in res.quality["flags"]
    assert res.quality["ok"] is False


def test_extract_phase_flags_high_rr_cv() -> None:
    # hr_bpm=70 -> mean RR ~857ms; rr_jitter_ms=450 pushes CV comfortably above the 0.35 gate.
    sig, _true = synthetic_ecg(n_beats=15, fs=500, rr_jitter_ms=450.0, seed=1)
    res = extract_phase(sig, fs=500)
    assert res.quality["rr_cv"] > 0.35
    assert FLAG_HIGH_RR_CV in res.quality["flags"]


def test_extract_phase_flags_flat_signal_and_no_beats() -> None:
    """Audit-found: FLAG_FLAT_SIGNAL (and FLAG_NO_BEATS) had zero test coverage anywhere
    in the suite, despite both being real, reachable flags."""
    sig = np.zeros((5000, 12))
    res = extract_phase(sig, fs=500)
    assert res.quality["dead_leads"] == 12
    assert FLAG_FLAT_SIGNAL in res.quality["flags"]
    assert FLAG_NO_BEATS in res.quality["flags"]


def test_extract_phase_flags_low_yield() -> None:
    """Audit-found: FLAG_LOW_YIELD had zero test coverage. Real beats are detected
    (n_beats stays at the true count) but padding the record with trailing silence far
    beyond the last R-peak drops the fraction of samples with a defined phase below the
    0.60 default -- isolated from FLAG_FLAT_SIGNAL/FLAG_NO_BEATS by using a padding ratio
    that keeps the beat-bearing region's robust scale well above the dead-lead floor."""
    sig, _true = synthetic_ecg(n_beats=10, fs=500, seed=0)
    T = sig.shape[0]
    padded = np.zeros((int(T * 1.8), sig.shape[1]))
    padded[:T] = sig
    res = extract_phase(padded, fs=500)
    assert res.n_beats == 10  # beats were genuinely detected, not lost to FLAT_SIGNAL
    assert res.quality["phase_yield"] < 0.60
    assert FLAG_LOW_YIELD in res.quality["flags"]
    assert FLAG_FLAT_SIGNAL not in res.quality["flags"]


def test_extract_phase_flags_low_confidence_when_configured() -> None:
    """Audit-found: FLAG_LOW_CONFIDENCE had zero test coverage (it's inactive by default
    -- min_detector_confidence=None -- so it never fires on real data, but the mechanism
    itself needs a test independent of calibration). A threshold above 1.0 (correlation
    scores are bounded at 1.0) guarantees the flag fires regardless of signal specifics."""
    sig, _true = synthetic_ecg(n_beats=15, fs=500, seed=0)
    res = extract_phase(sig, fs=500, qc=PhaseQCConfig(min_detector_confidence=1.1))
    assert FLAG_LOW_CONFIDENCE in res.quality["flags"]


def test_extract_phase_emits_rr_mean_and_sd_ms() -> None:
    """Bug fix vs. ttl-phase: rr_mean_ms/rr_sd_ms were read by the manifest writer but
    never emitted by extract_phase's quality dict, so they were structurally always NaN."""
    sig, _true = synthetic_ecg(n_beats=10, fs=500, seed=0)
    res = extract_phase(sig, fs=500)
    assert np.isfinite(res.quality["rr_mean_ms"])
    assert np.isfinite(res.quality["rr_sd_ms"])
    assert res.quality["rr_sd_ms"] == pytest.approx(
        res.quality["rr_cv"] * res.quality["rr_mean_ms"], rel=1e-9
    )


def test_detector_params_are_frozen_and_scalar_only() -> None:
    p = DetectorParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.bp_low = 10.0  # type: ignore[misc]


# ---------------------------------------------------------------------------- Tier 1 golden
def test_tier1_detector_matches_ttl_phase_exactly() -> None:
    """Exact match on real records: rpeaks and every quality field except rr_mean_ms/
    rr_sd_ms (excluded because they don't exist in ttl-phase's golden output -- emitting
    them is this port's own bug fix, not a ttl-phase behaviour to match).

    Audit-found gap (independently flagged by three review dimensions): this test used to
    compare only a subset of scalar fields and never checked `flags`/`ok`, even though the
    golden fixture already contains the complete quality dict -- a regression in the
    flag-setting logic (order, thresholds, bug #4's RR_OUTLIERS/IMPLAUSIBLE_RR split) could
    have passed this "exact match" test silently. Now compares every key ttl-phase's golden
    output actually has.

    On exact vs. tolerance: the port plan allows a tolerance here "if the numeric stack
    isn't pinned exactly" (MANIFEST.json records versions but nothing enforces future runs
    match them). Exact equality is used anyway because it was verified empirically, not
    assumed: this exact comparison reproduces bit-for-bit across two genuinely different
    numpy/scipy pairs (2.2.6/1.15.3 at fixture-generation time vs. 2.4.6/1.17.1 in winder's
    own environment) -- see the PR3 commit. If a future dependency bump ever breaks this,
    that is itself the signal worth investigating, not something to silently tolerate away."""
    tier1 = json.load(open(os.path.join(FIXTURES, "tier1_detector", "golden.json")))
    assert len(tier1) >= 5
    scalar_keys = (
        "n_beats",
        "n_intervals",
        "fs",
        "n_samples",
        "rr_median_ms",
        "rr_cv",
        "rr_min_observed_ms",
        "rr_max_observed_ms",
        "frac_rr_implausible",
        "heart_rate_bpm",
        "phase_yield",
        "detector_confidence",
        "peak_snr_db",
        "n_searchback",
        "n_twave_rejected",
        "dead_leads",
        "subsample_shift_median_ms",
        "jitter_ms",
        "jitter_frac_cycle",
    )
    for ecg_id, rec in tier1.items():
        hea = os.path.join(FIXTURES, "wfdb", f"{rec['stem']}.hea")
        sig, _hdr = read_record(hea)
        res = extract_phase(sig, fs=500, estimate_jitter=True, jitter_seed=int(ecg_id))
        golden_rpeaks = np.array(rec["rpeaks"])
        assert np.array_equal(golden_rpeaks, res.rpeaks), f"ecg_id={ecg_id} rpeaks diverge"
        for key in scalar_keys:
            golden_v, new_v = rec["quality"][key], res.quality[key]
            if isinstance(golden_v, float) and np.isnan(golden_v):
                assert np.isnan(new_v), f"ecg_id={ecg_id} key={key}"
            else:
                assert golden_v == pytest.approx(new_v, rel=1e-9, abs=1e-12), (
                    f"ecg_id={ecg_id} key={key}: golden={golden_v} new={new_v}"
                )
        assert list(rec["quality"]["flags"]) == list(res.quality["flags"]), (
            f"ecg_id={ecg_id} flags diverge: golden={rec['quality']['flags']} "
            f"new={res.quality['flags']}"
        )
        assert rec["quality"]["ok"] == res.quality["ok"], f"ecg_id={ecg_id} ok diverges"


# ---------------------------------------------------------------------------- Tier 2 golden
def test_tier2_bin_id_matches_ttl_phase_exactly() -> None:
    """Bit-for-bit: phase_from_rpeaks + bin_phase against ttl-phase's own bin_id, at every
    B in the sweep, for 60 records. bin_id is a discrete label -- exact equality is the
    right bar, and this is the permanent regression test for a respiratory-axis change
    (see phase.py's module docstring on why theta is a vector)."""
    golden = np.load(os.path.join(FIXTURES, "tier2_phase_clock", "golden.npz"))
    ecg_ids = golden["ecg_ids"]
    rpeaks_concat, offsets = golden["rpeaks"], golden["offsets"]
    n_samples = int(golden["n_samples"])
    assert len(ecg_ids) >= 50

    for b in (4, 8, 16, 32):
        golden_bins = golden[f"bin_id_b{b}"]
        for row in range(len(ecg_ids)):
            rp = rpeaks_concat[offsets[row] : offsets[row + 1]]
            theta = phase_from_rpeaks(rp, n_samples)
            bins = bin_phase(theta, b)
            assert np.array_equal(bins, golden_bins[row]), (
                f"ecg_id={ecg_ids[row]} B={b} bin_id diverges from ttl-phase"
            )

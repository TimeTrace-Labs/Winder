"""Cardiac phase clock: R-peak detection -> sub-sample refinement -> theta -> phase bins.

Ported near-verbatim from ttl-phase's `src/data/phase.py` (pinned at
cfe2e60a5592e30a32ef1f1863ee4fb449e80714 -- see tests/fixtures/MANIFEST.json). This module
is the single source of truth for *when in the cardiac cycle* a sample sits; everything
downstream (the transport operator's R_delta, any phase-conditioned statistic) depends on
the conventions fixed here.

Scope narrowed from the source module -- deferred, not ported:
  * `beat_matrix`, `beat_starts_from_bins`, `theta_from_bin_ramp`: day-2 analysis/
    visualisation helpers, not part of the frozen theta/bin_id contract. Add them when a
    concrete winder consumer needs beat-level windows.
  * `desync`, `desync_seconds`: robustness-eval corruptions, share `corrupt.py`'s deferred
    status (no consumer in winder yet).
  * `synthetic_ecg`, `evaluate_detector`: test fixtures that lived in production code by
    mistake -- moved to tests/_synthetic.py, since nothing in the library depends on them.

Interface change from the source: `extract_phase` no longer takes `B`. The storage design's
whole point is that theta/bin_id for any B are derived on demand from the R-peak archive
(`B_sweep` exists precisely because callers need several) -- baking `B` into extraction is
exactly the kind of decision that's painful to unwind once callers depend on it. Call
`bin_phase(result.theta, b)` separately per sweep value. Consequently `PhaseResult` no
longer carries `bin_id`/`B` fields either.

Conventions fixed by this module
---------------------------------
* **Signal layout.** `sig` is time-major, shape (T, n_leads); a 1-D array is treated as a
  single lead. If a 2-D array arrives with more columns than rows it is transposed, on the
  documented assumption that the longer axis is time (PTB-XL: T=5000 >> n_leads=12).
* **R-peak positions are floats in units of samples**, not seconds and not integers.
  Sub-sample refinement is mandatory.
* **theta is a vector**, shape (T, d) with d = 1 today. `theta[:, 0]` is the within-beat
  phase, linear in time between consecutive refined R-peaks:
      theta = 2*pi * (t - R_i) / (R_{i+1} - R_i)   for R_i <= t < R_{i+1}
  so theta = 0 exactly at an R-peak and increases monotonically to 2*pi. This is a
  *piecewise-linear time-warp clock*, not a Hilbert phase -- and, precisely because it is
  piecewise-linear, it is an RR-normalised time coordinate rather than an isochronous map
  of cardiac electromechanical events (systole does not scale linearly with RR): treat
  "phase" here as that coordinate, not as a claim about which physiological event occurs at
  a given theta across different heart rates.
* **Samples outside the R-peak span are NaN**, never zero. A sample before the first
  R-peak or at/after the last one has no defined within-beat phase. They must be excluded
  downstream; `bin_phase` maps them to the sentinel `BIN_EXCLUDE = -1` and never to bin 0.
* **Bins are uniform on [0, 2*pi)** with left-closed edges: bin j covers
  [2*pi*j/B, 2*pi*(j+1)/B). For d > 1 the per-axis bins are flattened row-major
  (C order, last axis fastest), matching `np.ravel_multi_index(..., order="C")`.
* **Quality flags are machine-readable strings** with a `PHASE_` prefix, intended to be
  written verbatim into a manifest's reason-code column (never drop a record silently).

Nothing here smooths, clips, or normalises beyond what is named in `DetectorParams`; every
numerical floor is an explicit, overridable argument. Pure numpy/scipy by construction.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

__all__ = [
    "TWO_PI",
    "BIN_EXCLUDE",
    "FLAG_NO_BEATS",
    "FLAG_TOO_FEW_BEATS",
    "FLAG_IMPLAUSIBLE_RR",
    "FLAG_RR_OUTLIERS",
    "FLAG_HIGH_RR_CV",
    "FLAG_LOW_YIELD",
    "FLAG_FLAT_SIGNAL",
    "FLAG_LOW_CONFIDENCE",
    "ALL_FLAGS",
    "DetectorParams",
    "PhaseQCConfig",
    "PhaseResult",
    "detect_rpeaks",
    "refine_rpeaks",
    "phase_from_rpeaks",
    "bin_phase",
    "extract_phase",
    "jitter_estimate",
]

TWO_PI = 2.0 * np.pi

#: bin_id sentinel meaning "this sample has no defined phase; exclude it".
BIN_EXCLUDE = -1

# ---------------------------------------------------------------- manifest reason codes
FLAG_NO_BEATS = "PHASE_NO_BEATS"  # detector found < 2 R-peaks: no phase at all
FLAG_TOO_FEW_BEATS = "PHASE_TOO_FEW_BEATS"  # fewer than min_beats R-peaks
FLAG_IMPLAUSIBLE_RR = "PHASE_IMPLAUSIBLE_RR"  # median RR outside [rr_min_ms, rr_max_ms]
FLAG_RR_OUTLIERS = "PHASE_RR_OUTLIERS"  # too large a *fraction* of RRs out of bounds
FLAG_HIGH_RR_CV = "PHASE_HIGH_RR_CV"  # RR CV above rr_cv_max: possible AF / failure
FLAG_LOW_YIELD = "PHASE_LOW_YIELD"  # valid-theta fraction below min_phase_yield
FLAG_FLAT_SIGNAL = "PHASE_FLAT_SIGNAL"  # no lead carries measurable variation
FLAG_LOW_CONFIDENCE = "PHASE_LOW_CONFIDENCE"  # beats mutually inconsistent; opt-in, see below

ALL_FLAGS = (
    FLAG_NO_BEATS,
    FLAG_TOO_FEW_BEATS,
    FLAG_IMPLAUSIBLE_RR,
    FLAG_RR_OUTLIERS,
    FLAG_HIGH_RR_CV,
    FLAG_LOW_YIELD,
    FLAG_FLAT_SIGNAL,
    FLAG_LOW_CONFIDENCE,
)


# ===================================================================== detector settings
@dataclass(frozen=True)
class DetectorParams:
    """Every constant of the R-peak detector, named and overridable.

    Defaults are the classical Pan-Tompkins operating point adapted to a multi-lead
    root-sum-square channel at 500 Hz. Nothing in the body of this module hard-codes a
    number that is not listed here.

    Attributes
    ----------
    bp_low, bp_high, bp_order
        Zero-phase Butterworth bandpass (`sosfiltfilt`, so there is **no group delay** to
        bias peak positions) isolating QRS energy.
    lead_scale_floor
        Floor on a lead's robust scale (MAD-based sigma) before division. Leads below it
        are treated as dead and contribute zero, rather than being amplified to noise.
    integ_ms
        Width of the centred moving-average ("integration") window applied to the squared
        envelope. Broad on purpose: it merges the QRS complex into one hump so the
        threshold logic sees one candidate per beat. It also broadens the peak, which is
        why localisation happens on `narrow_ms` instead.
    narrow_ms
        Width of the centred moving-average used for *localisation*. Short enough that its
        maximum tracks the R-peak, long enough to suppress single-sample noise.
    relocate_ms
        Half-width of the search window, around each coarse candidate, in which the narrow
        envelope's maximum is taken as the integer R-peak.
    refractory_ms
        Physiological refractory period; two accepted peaks may not be closer than this.
    twave_ms, twave_slope_frac, slope_win_ms
        T-wave discrimination: a candidate closer than `twave_ms` to the previous accepted
        peak is rejected unless its maximum absolute slope reaches `twave_slope_frac` of
        the previous peak's, measured over +/- `slope_win_ms`.
    thr_frac, thr2_frac, spki_alpha, npki_alpha, searchback_alpha
        Adaptive-threshold constants. thr1 = NPKI + thr_frac*(SPKI - NPKI); the search-back
        threshold is thr2_frac*thr1. Running estimates update as
        SPKI <- (1-spki_alpha)*SPKI + spki_alpha*peak (searchback_alpha for recovered
        peaks, which are less trustworthy so they move SPKI more).
    rr_miss_frac, rr_hist
        A gap longer than rr_miss_frac times the running median of the last `rr_hist`
        RR intervals triggers search-back.
    init_spki_quantile, init_npki_quantile
        Seeds for the running signal/noise levels: SPKI = this quantile of the *candidate*
        envelope heights over the whole record, NPKI = this quantile of the whole envelope.
        Textbook Pan-Tompkins instead seeds SPKI with the **maximum** over a causal 2 s
        warm-up, which is not robust: one outsized artifact in that window puts thr1 above
        every real QRS, and SPKI only updates on acceptance so it never decays. Measured on
        800 PTB-XL records that failure cost 6 records (0.75%) 1-4 detected beats each,
        despite 24-32 dB peak SNR. A quantile over the whole record fixes all 6, leaves the
        beat-count distribution otherwise unchanged. These records are 10 s and processed
        offline, so there is no reason to accept a causal warm-up's handicap -- and this is
        not leakage: it uses only the signal itself, no label, no other record.
    refine_win_ms, refine_max_lag_ms, refine_iters
        Template cross-correlation refinement: half-width of the comparison window,
        maximum admissible lag correction, and number of (template rebuild, re-align)
        passes.
    """

    bp_low: float = 5.0
    bp_high: float = 15.0
    bp_order: int = 2
    lead_scale_floor: float = 1e-6

    integ_ms: float = 120.0
    narrow_ms: float = 20.0
    relocate_ms: float = 100.0
    refractory_ms: float = 200.0

    twave_ms: float = 360.0
    twave_slope_frac: float = 0.5
    slope_win_ms: float = 60.0

    thr_frac: float = 0.25
    thr2_frac: float = 0.5
    spki_alpha: float = 0.125
    npki_alpha: float = 0.125
    searchback_alpha: float = 0.25

    rr_miss_frac: float = 1.66
    rr_hist: int = 8
    init_spki_quantile: float = 0.90
    init_npki_quantile: float = 0.50

    refine_win_ms: float = 100.0
    refine_max_lag_ms: float = 50.0
    refine_iters: int = 2


@dataclass
class PhaseQCConfig:
    """QC thresholds gating `extract_phase`'s flags. OmegaConf-structured config boundary.

    `min_detector_confidence` is `None` (inactive) deliberately, not a stand-in for a real
    number: it is a *threshold*, and this module does not introduce one that hasn't been
    set via calibration. It exists because the other flags have a blind spot -- a record of
    pure noise can yield plausibly-spaced spurious "beats" and pass every RR test. Measured
    separation on real vs. synthetic-noise records: real ECG confidence min 0.977, 1st pct
    0.980, median 0.998; noise median 0.41, max 0.83 -- any value in the wide empty gap
    works, but it must be chosen by calibration, not guessed here.
    """

    min_beats: int = 5
    rr_min_ms: float = 300.0
    rr_max_ms: float = 2000.0
    rr_cv_max: float = 0.35
    #: bounds the fraction of individual RR intervals outside [rr_min_ms, rr_max_ms] before
    #: PHASE_RR_OUTLIERS fires, so a single ectopic beat does not condemn a record while a
    #: broken detector still does. (Had no config key in ttl-phase; fixed here.)
    rr_outlier_frac_max: float = 0.20
    min_phase_yield: float = 0.60
    min_detector_confidence: float | None = None


@dataclass(frozen=True)
class PhaseResult:
    """Output of `extract_phase`.

    `theta` (T, d) in [0, 2*pi) with NaN outside the R-peak span, `quality` (dict,
    JSON-serialisable) and `n_beats` (int) are the frozen contract. The remaining fields
    are diagnostics carried along so a manifest row can be written without re-running the
    detector.

    `n_beats` counts *detected R-peaks*; the number of complete beats (RR intervals) is
    `n_beats - 1`.
    """

    theta: np.ndarray
    quality: dict[str, Any]
    n_beats: int
    rpeaks: np.ndarray = field(default_factory=lambda: np.empty(0))
    rpeaks_coarse: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fs: int = 0


# =============================================================================== helpers
def _as_time_major(sig: np.ndarray) -> np.ndarray:
    """Return `sig` as float64 (T, n_leads).

    Convention: a 1-D input is one lead; a 2-D input whose first axis is shorter than its
    second is transposed (the longer axis is time). Raises on 0-D or >2-D input.
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"sig must be 1-D or 2-D, got shape {x.shape}")
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def _ms_to_samples(ms: float, fs: int, minimum: int = 1) -> int:
    """Convert a duration in ms to a whole number of samples, at least `minimum`."""
    return max(minimum, int(round(ms * fs / 1000.0)))


def _odd_samples(ms: float, fs: int) -> int:
    """Duration in ms -> an ODD number of samples (>= 3).

    Moving-average widths must be odd: `uniform_filter1d` with an even width is
    asymmetric about the sample it writes, which displaces the envelope maximum by half a
    sample (1 ms at 500 Hz) -- a bias larger than the sub-sample precision this module
    exists to deliver.
    """
    n = _ms_to_samples(ms, fs, minimum=3)
    return n if n % 2 == 1 else n + 1


def _robust_scale(x: np.ndarray, floor: float) -> np.ndarray:
    """Per-column MAD-based sigma estimate (1.4826 * MAD), floored at `floor`.

    Robust to the QRS spikes themselves, which is the point: a normal-theory std would be
    inflated by the very events we are trying to detect.
    """
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0)
    return cast(np.ndarray, np.maximum(1.4826 * mad, floor))


def _detector_channels(sig: np.ndarray, fs: int, p: DetectorParams) -> dict[str, Any]:
    """Build the 1-D detection channels from a multi-lead record.

    Steps: zero-phase bandpass (`p.bp_low`-`p.bp_high` Hz) -> per-lead robust rescaling
    (so one high-gain lead cannot dominate) -> root-sum-square across leads (a sign-free
    combination that needs no lead-quality heuristic) -> squared energy -> two centred
    moving averages, one broad (`integ_ms`, for thresholding) and one narrow
    (`narrow_ms`, for localisation).

    Returns a dict with
      'bp'     (T, L) bandpassed, robustly rescaled leads  (used for template alignment)
      'rss'    (T,)   root-sum-square across leads
      'integ'  (T,)   broad moving average of rss**2
      'narrow' (T,)   narrow moving average of rss**2
      'deriv'  (T,)   first difference of rss (for T-wave slope discrimination)
      'dead'   (L,)   bool, leads whose robust scale hit the floor
    All channels are non-negative except 'bp' and 'deriv'.
    """
    x = _as_time_major(sig)
    nyq = 0.5 * fs
    hi = min(p.bp_high, 0.99 * nyq)
    if not (0.0 < p.bp_low < hi):
        raise ValueError(f"invalid band ({p.bp_low}, {hi}) Hz for fs={fs}")
    sos = butter(p.bp_order, [p.bp_low / nyq, hi / nyq], btype="bandpass", output="sos")
    bp = sosfiltfilt(sos, x, axis=0)

    scale = _robust_scale(bp, p.lead_scale_floor)
    dead = scale <= p.lead_scale_floor
    bpn = bp / scale
    bpn[:, dead] = 0.0

    rss = np.sqrt(np.sum(bpn**2, axis=1))
    energy = rss**2
    integ = uniform_filter1d(energy, size=_odd_samples(p.integ_ms, fs), mode="nearest")
    narrow = uniform_filter1d(energy, size=_odd_samples(p.narrow_ms, fs), mode="nearest")
    deriv = np.gradient(rss)
    return {
        "bp": bpn,
        "rss": rss,
        "integ": integ,
        "narrow": narrow,
        "deriv": deriv,
        "dead": dead,
    }


# ======================================================================= peak detection
def _max_abs_slope(deriv: np.ndarray, i: int, half: int) -> float:
    """Maximum |derivative| in a +/- `half`-sample window around index `i`."""
    lo = max(0, i - half)
    hi = min(deriv.size, i + half + 1)
    return float(np.max(np.abs(deriv[lo:hi]))) if hi > lo else 0.0


def _adaptive_scan(
    cand: np.ndarray, ch: dict[str, Any], fs: int, p: DetectorParams
) -> tuple[list[int], float, float, int]:
    """Pan-Tompkins adaptive-threshold pass over candidate maxima of the broad envelope.

    Returns (accepted indices list, spki, npki, n_twave_rejected). Candidates are visited
    in time order; a candidate within the refractory period of the last acceptance
    replaces it only if it is stronger. SPKI/NPKI are running signal/noise level
    estimates, and the acceptance threshold is NPKI + thr_frac*(SPKI - NPKI). Their seeds
    are robust quantiles rather than a causal maximum -- see `DetectorParams`.
    """
    integ, deriv = ch["integ"], ch["deriv"]
    refr = _ms_to_samples(p.refractory_ms, fs)
    twave = _ms_to_samples(p.twave_ms, fs)
    slope_half = _ms_to_samples(p.slope_win_ms, fs)
    if cand.size == 0:
        return [], 0.0, 0.0, 0

    spki = float(np.quantile(integ[cand], p.init_spki_quantile))
    npki = float(np.quantile(integ, p.init_npki_quantile))

    accepted: list[int] = []
    n_twave = 0
    for c in cand:
        v = float(integ[c])
        thr1 = npki + p.thr_frac * (spki - npki)
        if accepted and (c - accepted[-1]) < refr:
            # Refractory collision: keep whichever is stronger, never both.
            if v > float(integ[accepted[-1]]):
                accepted[-1] = c
                spki = (1.0 - p.spki_alpha) * spki + p.spki_alpha * v
            continue
        if v <= thr1:
            npki = (1.0 - p.npki_alpha) * npki + p.npki_alpha * v
            continue
        if accepted and (c - accepted[-1]) < twave:
            s_new = _max_abs_slope(deriv, c, slope_half)
            s_old = _max_abs_slope(deriv, accepted[-1], slope_half)
            if s_new < p.twave_slope_frac * s_old:
                npki = (1.0 - p.npki_alpha) * npki + p.npki_alpha * v
                n_twave += 1
                continue
        accepted.append(int(c))
        spki = (1.0 - p.spki_alpha) * spki + p.spki_alpha * v
    return accepted, spki, npki, n_twave


def _search_back(
    accepted: list[int],
    cand: np.ndarray,
    ch: dict[str, Any],
    fs: int,
    p: DetectorParams,
    thr2: float,
) -> tuple[list[int], int]:
    """Recover missed beats in gaps longer than `rr_miss_frac` x running median RR.

    Applied to interior gaps *and* to the leading/trailing regions, because a missed first
    or last beat directly costs phase yield at the record edges. Insertion is greedy on
    envelope height, must clear the lowered threshold `thr2`, and must respect the
    refractory period on both sides. Repeats until no further insertion is possible, so a
    doubly-missed beat is also recovered. Returns (accepted, n_recovered).
    """
    integ = ch["integ"]
    refr = _ms_to_samples(p.refractory_ms, fs)
    n_rec = 0
    if len(accepted) < 2:
        return accepted, n_rec

    def best_in(lo: int, hi: int) -> int | None:
        """Strongest candidate strictly inside (lo, hi) clearing thr2, else None."""
        m = (cand > lo) & (cand < hi)
        if not np.any(m):
            return None
        sub = cand[m]
        vals = integ[sub]
        k = int(np.argmax(vals))
        return int(sub[k]) if vals[k] > thr2 else None

    changed = True
    while changed:
        changed = False
        acc = np.asarray(accepted, dtype=np.int64)
        rr = np.diff(acc)
        rr_ref = float(np.median(rr[-p.rr_hist :])) if rr.size else np.inf
        gap_max = p.rr_miss_frac * rr_ref
        # Interior gaps, then the two edges (treated as gaps against the record bounds).
        pairs = zip(acc[:-1], acc[1:], strict=True)
        gaps = [(int(a), int(b)) for a, b in pairs if (b - a) > gap_max]
        if acc[0] > gap_max:
            gaps.append((-1, int(acc[0])))
        if (integ.size - 1 - acc[-1]) > gap_max:
            gaps.append((int(acc[-1]), integ.size))
        for lo, hi in gaps:
            c = best_in(lo + refr - 1, hi - refr + 1)
            if c is None:
                continue
            accepted = sorted(accepted + [c])
            n_rec += 1
            changed = True
    return accepted, n_rec


def detect_rpeaks(
    sig: np.ndarray, fs: int, params: DetectorParams | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Detect R-peaks on a (T, n_leads) ECG record and refine them to sub-sample precision.

    Pipeline: 5-15 Hz zero-phase bandpass -> per-lead robust rescaling -> root-sum-square
    across leads -> squared energy -> broad moving-window integration -> adaptive
    threshold with a 200 ms refractory period and T-wave slope check -> search-back for
    missed beats -> relocation onto the narrow energy envelope -> sub-sample refinement
    (parabolic interpolation, then template cross-correlation).

    Returns
    -------
    rpeaks : (n,) float64
        Refined R-peak positions in **samples** (fractional), sorted, strictly increasing.
    coarse : (n,) int64
        Integer positions before sub-sample refinement (the narrow-envelope maxima).
    info : dict
        Detector internals: 'n_candidates', 'n_twave_rejected', 'n_searchback',
        'thr1', 'thr2', 'spki', 'npki', 'peak_energy' (narrow envelope at each coarse
        peak), 'background_energy' (median narrow envelope away from peaks),
        'dead_leads' (int), 'channels' (the dict from `_detector_channels`),
        'subsample_shift_samples' (the applied refinement, for jitter accounting).
    """
    p = params or DetectorParams()
    ch = _detector_channels(sig, fs, p)
    integ, narrow = ch["integ"], ch["narrow"]
    empty = (np.empty(0), np.empty(0, dtype=np.int64))

    if not np.any(np.isfinite(integ)) or float(np.max(integ)) <= 0.0:
        return (
            *empty,
            {
                "n_candidates": 0,
                "n_twave_rejected": 0,
                "n_searchback": 0,
                "thr1": 0.0,
                "thr2": 0.0,
                "spki": 0.0,
                "npki": 0.0,
                "peak_energy": np.empty(0),
                "background_energy": 0.0,
                "dead_leads": int(ch["dead"].sum()),
                "channels": ch,
                "subsample_shift_samples": np.empty(0),
            },
        )

    refr = _ms_to_samples(p.refractory_ms, fs)
    cand, _ = find_peaks(integ, distance=refr)
    accepted, spki, npki, n_twave = _adaptive_scan(cand, ch, fs, p)
    thr1 = npki + p.thr_frac * (spki - npki)
    thr2 = p.thr2_frac * thr1
    accepted, n_rec = _search_back(list(accepted), cand, ch, fs, p, thr2)

    if not accepted:
        return (
            *empty,
            {
                "n_candidates": int(cand.size),
                "n_twave_rejected": n_twave,
                "n_searchback": 0,
                "thr1": thr1,
                "thr2": thr2,
                "spki": spki,
                "npki": npki,
                "peak_energy": np.empty(0),
                "background_energy": float(np.median(narrow)),
                "dead_leads": int(ch["dead"].sum()),
                "channels": ch,
                "subsample_shift_samples": np.empty(0),
            },
        )

    # Relocate onto the narrow envelope: the broad integration window displaces the
    # apparent maximum by up to integ_ms/2, which at 500 Hz is 30 samples of pure bias.
    half = _ms_to_samples(p.relocate_ms, fs)
    coarse_list = []
    for c in accepted:
        lo, hi = max(0, c - half), min(narrow.size, c + half + 1)
        coarse_list.append(lo + int(np.argmax(narrow[lo:hi])))
    coarse = np.unique(np.asarray(coarse_list, dtype=np.int64))

    refined = refine_rpeaks(coarse, ch, fs, p)
    peak_energy = narrow[coarse]
    mask = np.ones(narrow.size, dtype=bool)
    for c in coarse:
        mask[max(0, c - half) : min(narrow.size, c + half + 1)] = False
    background = float(np.median(narrow[mask])) if mask.any() else float(np.median(narrow))

    info: dict[str, Any] = {
        "n_candidates": int(cand.size),
        "n_twave_rejected": int(n_twave),
        "n_searchback": int(n_rec),
        "thr1": float(thr1),
        "thr2": float(thr2),
        "spki": float(spki),
        "npki": float(npki),
        "peak_energy": peak_energy,
        "background_energy": background,
        "dead_leads": int(ch["dead"].sum()),
        "channels": ch,
        "subsample_shift_samples": refined - coarse.astype(np.float64),
    }
    return refined, coarse, info


# ==================================================================== sub-sample refining
def _parabolic_offset(y_prev: np.ndarray, y_0: np.ndarray, y_next: np.ndarray) -> np.ndarray:
    """Sub-sample offset of a discrete maximum by 3-point parabola fit, in (-0.5, 0.5).

    offset = 0.5 * (y_prev - y_next) / (y_prev - 2*y_0 + y_next). The denominator is the
    second difference, which is **negative** at a strict maximum; the offset is set to 0
    wherever it is non-negative (a plateau or an inflection is not a locatable peak).
    Refusing to extrapolate off a non-concave triple is the only defensible fallback:
    anything else invents precision.
    """
    den = np.atleast_1d(y_prev - 2.0 * y_0 + y_next).astype(np.float64)
    off = np.zeros_like(den)
    ok = den < 0
    off[ok] = 0.5 * (np.atleast_1d(y_prev)[ok] - np.atleast_1d(y_next)[ok]) / den[ok]
    return np.clip(off, -0.5, 0.5)


def _interp_patches(x: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Linear-interpolate multi-lead windows at fractional centres.

    x: (T, L); centers: (n,) float; offsets: (w,) int/float relative sample positions.
    Returns (n, w, L). Positions outside [0, T-1] are clamped to the record ends (edge
    hold), which only affects beats at the very start/end of the record.
    """
    T, L = x.shape
    pos = np.clip(centers[:, None] + offsets[None, :], 0.0, T - 1.0)  # (n, w)
    grid = np.arange(T, dtype=np.float64)
    out = np.empty((centers.size, offsets.size, L), dtype=np.float64)
    for lead in range(L):
        out[:, :, lead] = np.interp(pos, grid, x[:, lead])
    return out


def _xcorr_lags_from_patches(
    ext: np.ndarray, template: np.ndarray, w: int, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample lag of each extended patch against a multi-lead `template`.

    `ext` is (n, 2(w+m)+1, L), `template` is (2w+1, L). Each patch is correlated with the
    template over integer lags -m..m and the correlation sequence's maximum is refined by
    parabolic interpolation. Patches and template are mean-removed per lead; the reported
    score is the normalised (Pearson) correlation at the integer optimum.

    Factored out so that `jitter_estimate` can drive the *identical* aligner on synthetic
    patches. Sharing one aligner is the point: a jitter number produced by a different code
    path would not measure the code we ship.

    Returns (lags in samples, correlation score), each (n,).
    """
    tpl = template - template.mean(axis=0, keepdims=True)  # (2w+1, L)
    tpl_norm = np.sqrt(np.sum(tpl**2)) + np.finfo(float).tiny
    n = ext.shape[0]
    n_lag = 2 * m + 1
    corr = np.empty((n, n_lag), dtype=np.float64)
    nrm = np.empty((n, n_lag), dtype=np.float64)
    for k in range(n_lag):
        seg = ext[:, k : k + 2 * w + 1, :]
        seg = seg - seg.mean(axis=1, keepdims=True)
        corr[:, k] = np.einsum("nwl,wl->n", seg, tpl)
        nrm[:, k] = np.sqrt(np.einsum("nwl,nwl->n", seg, seg))
    kbest = np.argmax(corr, axis=1)
    rows = np.arange(n)
    interior = (kbest > 0) & (kbest < n_lag - 1)
    off = np.zeros(n, dtype=np.float64)
    if np.any(interior):
        ii = rows[interior]
        off[ii] = _parabolic_offset(
            corr[ii, kbest[ii] - 1], corr[ii, kbest[ii]], corr[ii, kbest[ii] + 1]
        )
    lags = (kbest - m).astype(np.float64) + off
    score = corr[rows, kbest] / (nrm[rows, kbest] * tpl_norm + np.finfo(float).tiny)
    return lags, score


def _xcorr_lags(
    x: np.ndarray, centers: np.ndarray, template: np.ndarray, w: int, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample lag of each beat of `x` (windowed at fractional `centers`) vs `template`.

    Thin wrapper: interpolates the extended window [-w-m, w+m] at each fractional centre
    and delegates to `_xcorr_lags_from_patches`.
    """
    offs = np.arange(-w - m, w + m + 1, dtype=np.float64)
    return _xcorr_lags_from_patches(_interp_patches(x, centers, offs), template, w, m)


def refine_rpeaks(
    coarse: np.ndarray,
    channels: dict[str, Any],
    fs: int,
    params: DetectorParams | None = None,
    method: str = "both",
) -> np.ndarray:
    """Refine integer R-peak indices to fractional sample positions.

    Two mechanisms, applied in sequence by default (`method="both"`):

    1. ``"parabolic"`` -- 3-point parabola fit to the narrow energy envelope at the peak.
       Cheap, local, and unbiased for a symmetric peak; resolution limited by envelope
       curvature and noise.
    2. ``"template"`` -- multi-lead cross-correlation against the median beat template,
       with the correlation peak itself parabolically interpolated. This aligns whole QRS
       morphology rather than a single hump and is what actually drives jitter down,
       because it pools ~100 ms x 12 leads of evidence per beat.

    `channels` is the dict returned by `_detector_channels` (so the bandpassed leads and
    the narrow envelope are reused rather than recomputed). Positions are guaranteed
    sorted and strictly increasing; a refinement that would reorder two peaks is reverted
    to the coarse positions for that pair.
    """
    p = params or DetectorParams()
    if method not in ("parabolic", "template", "both", "none"):
        raise ValueError(f"unknown refinement method {method!r}")
    r = np.asarray(coarse, dtype=np.float64).copy()
    if r.size == 0 or method == "none":
        return r

    narrow = channels["narrow"]
    if method in ("parabolic", "both"):
        idx = coarse.astype(np.int64)
        interior = (idx > 0) & (idx < narrow.size - 1)
        off = np.zeros(r.size)
        if np.any(interior):
            i = idx[interior]
            off[interior] = _parabolic_offset(narrow[i - 1], narrow[i], narrow[i + 1])
        r = r + off

    if method in ("template", "both") and r.size >= 2:
        x = channels["bp"]
        w = _ms_to_samples(p.refine_win_ms, fs)
        m = _ms_to_samples(p.refine_max_lag_ms, fs)
        offs = np.arange(-w, w + 1, dtype=np.float64)
        for _ in range(max(0, p.refine_iters)):
            template = np.median(_interp_patches(x, r, offs), axis=0)  # (2w+1, L)
            lags, _ = _xcorr_lags(x, r, template, w, m)
            r = r + lags

    order_ok = bool(np.all(np.diff(r) > 0)) if r.size > 1 else True
    if not order_ok:
        # Refinement must never reorder beats; fall back where monotonicity broke.
        r = np.maximum.accumulate(np.where(np.isfinite(r), r, coarse))
        bad = np.diff(r) <= 0
        if np.any(bad):
            r = coarse.astype(np.float64)
    return r


# ================================================================================= theta
def phase_from_rpeaks(rpeaks: np.ndarray, n_samples: int) -> np.ndarray:
    """Within-beat phase for every sample index 0..n_samples-1, from R-peak positions.

    theta[t] = 2*pi * (t - R_i) / (R_{i+1} - R_i) for R_i <= t < R_{i+1}; NaN for
    t < R_0 or t >= R_last (no enclosing beat -> no defined phase; see module docstring).
    Returns shape (n_samples, 1): theta is a vector so a second (respiratory) clock can be
    appended later without a signature change.
    """
    theta = np.full((int(n_samples), 1), np.nan, dtype=np.float64)
    r = np.asarray(rpeaks, dtype=np.float64)
    if r.size < 2:
        return theta
    t = np.arange(n_samples, dtype=np.float64)
    i = np.searchsorted(r, t, side="right") - 1  # index of enclosing R_i
    valid = (i >= 0) & (i <= r.size - 2)
    if np.any(valid):
        ii = i[valid]
        rr = r[ii + 1] - r[ii]
        theta[valid, 0] = np.mod(TWO_PI * (t[valid] - r[ii]) / rr, TWO_PI)
    return theta


def bin_phase(theta: np.ndarray, B: int | np.integer | tuple[int, ...]) -> np.ndarray:
    """Uniform phase bins on [0, 2*pi)^d, row-major flattened for d > 1.

    Bin j of axis k covers [2*pi*j/B_k, 2*pi*(j+1)/B_k) (left-closed). For d > 1 the
    multi-index is flattened in C order (last axis fastest), i.e.
    `np.ravel_multi_index(idx, B, order="C")`.

    A sample whose phase is NaN on *any* axis gets `BIN_EXCLUDE = -1`. This sentinel is
    deliberately not 0: silently folding undefined phase into the first bin would
    contaminate exactly the bin that contains the R-peak.

    Parameters
    ----------
    theta : (T, d) or (T,) array in [0, 2*pi); NaN allowed.
    B : int, or tuple of length d.

    Returns
    -------
    (T,) int64 bin ids in [0, prod(B)) or -1.
    """
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim == 1:
        th = th[:, None]
    if th.ndim != 2:
        raise ValueError(f"theta must be (T, d) or (T,), got {th.shape}")
    d = th.shape[1]
    # isinstance(B, int) alone regressed vs. ttl-phase's np.isscalar(B): a numpy integer
    # scalar (np.int64(8), the kind produced by iterating a numpy array of B values) is
    # not a Python int, so it fell through to `tuple(int(b) for b in B)` and crashed with
    # "not iterable" instead of being treated as scalar. np.integer restores that.
    Bt = (int(B),) * d if isinstance(B, int | np.integer) else tuple(int(b) for b in B)
    if len(Bt) != d:
        raise ValueError(f"B has length {len(Bt)} but theta has d={d}")
    if any(b < 1 for b in Bt):
        raise ValueError(f"all B must be >= 1, got {Bt}")

    finite = np.isfinite(th).all(axis=1)
    idx = np.zeros((th.shape[0], d), dtype=np.int64)
    for k in range(d):
        v = np.where(finite, th[:, k], 0.0)
        if np.any(finite & ((v < 0.0) | (v >= TWO_PI + 1e-12))):
            raise ValueError("theta must lie in [0, 2*pi)")
        idx[:, k] = np.clip((v / (TWO_PI / Bt[k])).astype(np.int64), 0, Bt[k] - 1)
    out = np.ravel_multi_index(tuple(idx[:, k] for k in range(d)), Bt, order="C")
    return np.where(finite, out, BIN_EXCLUDE).astype(np.int64)


# ======================================================================== the public API
def extract_phase(
    sig: np.ndarray,
    fs: int,
    qc: PhaseQCConfig | None = None,
    params: DetectorParams | None = None,
    estimate_jitter: bool = True,
    jitter_seed: int = 0,
) -> PhaseResult:
    """Cardiac phase clock for one record: R-peaks -> theta (T, 1) -> QC flags.

    Frozen contract: returns `PhaseResult` with `theta` (T, d) in [0, 2*pi)^d (NaN outside
    the R-peak span), `quality` (dict) and `n_beats` (int). Deliberately does not compute
    `bin_id`: call `bin_phase(result.theta, b)` separately per B in a sweep, so this
    function is not re-run for every bin count.

    Parameters
    ----------
    sig : (T, n_leads) float array in mV (time-major; see module docstring).
    fs : int, sampling rate in Hz.
    qc : QC thresholds; `PhaseQCConfig()` if None.
    params : detector constants; `DetectorParams()` if None.
    estimate_jitter : measure R-peak localisation error (`jitter_estimate`) and fold
        'jitter_ms' / 'jitter_frac_cycle' into `quality`.
    jitter_seed : seed for the jitter bootstrap (determinism rule: explicit, never global).

    Notes
    -----
    Flags never modify `theta`; they are advisory metadata for a manifest. Exclusion is a
    decision for the caller, which must log the reason code.
    """
    x = _as_time_major(sig)
    T = x.shape[0]
    qc = qc or PhaseQCConfig()
    p = params or DetectorParams()
    rpeaks, coarse, info = detect_rpeaks(x, fs, p)
    n_beats = int(rpeaks.size)

    theta = phase_from_rpeaks(rpeaks, T)
    yield_frac = float(np.mean(np.isfinite(theta).all(axis=1)))

    rr_ms = np.diff(rpeaks) * (1000.0 / fs) if n_beats >= 2 else np.empty(0)
    rr_mean = float(np.mean(rr_ms)) if rr_ms.size else float("nan")
    rr_median = float(np.median(rr_ms)) if rr_ms.size else float("nan")
    rr_sd = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 2 else float("nan")
    rr_cv = (
        float(np.std(rr_ms, ddof=1) / np.mean(rr_ms))
        if rr_ms.size >= 2 and np.mean(rr_ms) > 0
        else float("nan")
    )
    rr_bad = ((rr_ms < qc.rr_min_ms) | (rr_ms > qc.rr_max_ms)) if rr_ms.size else np.empty(0, bool)
    frac_rr_bad = float(np.mean(rr_bad)) if rr_ms.size else float("nan")

    # Detector confidence: median normalised correlation of each beat's multi-lead QRS
    # window against the median template. High <=> beats are mutually consistent, which is
    # exactly the condition under which a phase label is trustworthy.
    conf = float("nan")
    if n_beats >= 2:
        w = _ms_to_samples(p.refine_win_ms, fs)
        offs = np.arange(-w, w + 1, dtype=np.float64)
        tpl = np.median(_interp_patches(info["channels"]["bp"], rpeaks, offs), axis=0)
        _, scores = _xcorr_lags(info["channels"]["bp"], rpeaks, tpl, w, m=1)
        conf = float(np.median(scores))
    peak_snr_db = float("nan")
    if info["peak_energy"].size and info["background_energy"] > 0:
        peak_snr_db = float(
            10.0 * np.log10(np.median(info["peak_energy"]) / info["background_energy"])
        )

    jit: dict[str, Any] | None = None
    if estimate_jitter and n_beats >= 4:
        jit = jitter_estimate(rpeaks, x, fs, params=p, seed=jitter_seed)

    flags: list[str] = []
    if info["dead_leads"] == x.shape[1]:
        flags.append(FLAG_FLAT_SIGNAL)
    if n_beats < 2:
        flags.append(FLAG_NO_BEATS)
    if n_beats < qc.min_beats:
        flags.append(FLAG_TOO_FEW_BEATS)
    if rr_ms.size and not (qc.rr_min_ms <= rr_median <= qc.rr_max_ms):
        flags.append(FLAG_IMPLAUSIBLE_RR)
    if rr_ms.size and frac_rr_bad > qc.rr_outlier_frac_max:
        flags.append(FLAG_RR_OUTLIERS)
    if np.isfinite(rr_cv) and rr_cv > qc.rr_cv_max:
        flags.append(FLAG_HIGH_RR_CV)
    if yield_frac < qc.min_phase_yield:
        flags.append(FLAG_LOW_YIELD)
    if qc.min_detector_confidence is not None and not (conf >= qc.min_detector_confidence):
        flags.append(FLAG_LOW_CONFIDENCE)

    quality: dict[str, Any] = {
        "n_beats": n_beats,
        "n_intervals": max(0, n_beats - 1),
        "fs": int(fs),
        "n_samples": int(T),
        "rr_mean_ms": rr_mean,  # bug fix: never emitted in ttl-phase (writer read a key
        "rr_sd_ms": rr_sd,  # the producer never set); both natural companions to rr_cv.
        "rr_median_ms": rr_median,
        "rr_cv": rr_cv,
        "rr_min_observed_ms": float(np.min(rr_ms)) if rr_ms.size else float("nan"),
        "rr_max_observed_ms": float(np.max(rr_ms)) if rr_ms.size else float("nan"),
        "frac_rr_implausible": frac_rr_bad,
        "heart_rate_bpm": 60000.0 / rr_median if rr_median > 0 else float("nan"),
        "phase_yield": yield_frac,
        "detector_confidence": conf,
        "peak_snr_db": peak_snr_db,
        "n_searchback": info["n_searchback"],
        "n_twave_rejected": info["n_twave_rejected"],
        "dead_leads": info["dead_leads"],
        "subsample_shift_median_ms": (
            float(np.median(np.abs(info["subsample_shift_samples"]))) * 1000.0 / fs
            if n_beats
            else float("nan")
        ),
        "jitter_ms": (jit["jitter_ms_rms"] if jit else float("nan")),
        "jitter_frac_cycle": (jit["jitter_frac_cycle"] if jit else float("nan")),
        "flags": flags,
        "ok": len(flags) == 0,
    }
    return PhaseResult(
        theta=theta,
        quality=quality,
        n_beats=n_beats,
        rpeaks=rpeaks,
        rpeaks_coarse=coarse,
        fs=int(fs),
    )


# ================================================================== jitter (a measurement)
def jitter_estimate(
    rpeaks_refined: np.ndarray,
    sig: np.ndarray,
    fs: int,
    params: DetectorParams | None = None,
    seed: int = 0,
    n_reps: int = 8,
    n_boot: int = 500,
    inject_max_samples: float = 1.0,
) -> dict[str, Any]:
    """Measure R-peak localisation error, in ms and as a fraction of the cycle.

    Method (``template_residual_bootstrap``)
    ----------------------------------------
    For each beat: take the median multi-lead QRS template, shift it by a **known**
    fractional lag `delta` (drawn uniformly from +/- `inject_max_samples`), add a
    *measured* residual (observed patch minus template) drawn from a **different** beat and
    **circularly rolled** in time, then re-run the shipped aligner and record
    `delta_hat - delta`. The RMS of those errors is the localisation error of the actual
    code under the record's actual noise. Using a measured residual rather than resampled
    Gaussian noise keeps the noise *colour*: these patches are 5-15 Hz bandpassed, so their
    noise is strongly autocorrelated and a white surrogate would flatter the estimate.
    Using a rolled residual from another beat matters too -- an unrolled residual is
    `noise_i - median_j(noise_j)`, whose second term is (minus) the noise inside the very
    template it is matched against, and that shared term biases the recovered lag.

    Validation (synthetic records, known R-peaks, SNR 0-25 dB, n_reps=8): this estimator
    returned 0.92 +/- 0.02 of the true localisation std, uniformly across SNR -- accurate
    to about 10% and mildly **optimistic**. No correction factor is applied: tuning one on
    synthetic morphology would not transfer.

    What this number does *not* cover: (a) the definitional offset between "peak of the QRS
    energy envelope" and any other definition of R, which varies with QRS morphology and
    therefore *across* records (measured at ~2ms sd over a synthetic morphology sweep, an
    order of magnitude larger than the within-record jitter this function measures), and
    (b) error in the piecewise-linear RR interpolation between peaks.

    Why not leave-one-out. Realigning each beat against a template built from the other
    beats is **circular** and understates the error by more than an order of magnitude:
    `refine_rpeaks` has already driven every beat to a fixed point of exactly that
    alignment. Injecting a known shift breaks the circularity.

    A **lead-split cross-check** is also reported: aligning with leads 0..L/2-1 and with
    leads L/2..L-1 separately gives two estimates per beat whose disagreement, divided by
    2, estimates the single-estimator sigma. ECG leads are strongly correlated, so this
    cross-check is optimistic; it is a sanity bound, not the headline.

    Parameters
    ----------
    rpeaks_refined : (n,) float R-peak positions in samples (from `detect_rpeaks`).
    sig : (T, n_leads) record the peaks came from.
    fs : sampling rate in Hz.
    params : detector constants; window widths are reused from here.
    seed : explicit seed for the injected shifts and the bootstrap (determinism rule).
    n_reps : injected shifts per beat.
    n_boot : beat-level bootstrap resamples for the CI on the RMS (0 disables).
    inject_max_samples : half-range of the injected shift, in samples.

    Returns
    -------
    dict with 'n_beats', 'method', 'jitter_samples_rms', 'jitter_ms_rms',
    'jitter_ms_mad', 'jitter_ms_p95', 'jitter_ms_ci95' (lo, hi), 'bias_ms',
    'jitter_frac_cycle', 'jitter_pct_cycle', 'rr_median_ms', 'lead_split_ms',
    'residual_rms_frac', 'template_corr_median', 'n_reps', 'n_boot'.
    """
    p = params or DetectorParams()
    x = _as_time_major(sig)
    r = np.asarray(rpeaks_refined, dtype=np.float64)
    n = r.size
    ms = 1000.0 / fs
    out: dict[str, Any] = {
        "n_beats": int(n),
        "method": "template_residual_bootstrap",
        "n_reps": int(n_reps),
        "n_boot": int(n_boot),
        "jitter_samples_rms": float("nan"),
        "jitter_ms_rms": float("nan"),
        "jitter_ms_mad": float("nan"),
        "jitter_ms_p95": float("nan"),
        "jitter_ms_ci95": (float("nan"), float("nan")),
        "bias_ms": float("nan"),
        "jitter_frac_cycle": float("nan"),
        "jitter_pct_cycle": float("nan"),
        "rr_median_ms": float("nan"),
        "lead_split_ms": float("nan"),
        "residual_rms_frac": float("nan"),
        "template_corr_median": float("nan"),
    }
    if n < 4:
        return out

    bpn = _detector_channels(x, fs, p)["bp"]
    L = bpn.shape[1]
    w = _ms_to_samples(p.refine_win_ms, fs)
    m = _ms_to_samples(p.refine_max_lag_ms, fs)
    pad = int(np.ceil(inject_max_samples)) + 2

    off_ext = np.arange(-w - m, w + m + 1, dtype=np.float64)  # aligner's window
    off_pad = np.arange(-w - m - pad, w + m + pad + 1, dtype=np.float64)  # shiftable window
    ext_obs = _interp_patches(bpn, r, off_ext)  # (n, |off_ext|, L)
    pad_obs = _interp_patches(bpn, r, off_pad)
    tpl_pad = np.median(pad_obs, axis=0)  # (|off_pad|, L)
    tpl_ext = np.median(ext_obs, axis=0)
    tpl_cen = tpl_ext[m : m + 2 * w + 1, :]  # (2w+1, L)
    resid = ext_obs - tpl_ext[None, :, :]  # record's own noise

    tpl_rms = float(np.sqrt(np.mean(tpl_cen**2)))
    res_rms = float(np.sqrt(np.mean(resid**2)))
    out["residual_rms_frac"] = res_rms / tpl_rms if tpl_rms > 0 else float("nan")
    out["template_corr_median"] = float(
        np.median(_xcorr_lags_from_patches(ext_obs, tpl_cen, w, m)[1])
    )

    rng = np.random.default_rng(seed)
    deltas = rng.uniform(-inject_max_samples, inject_max_samples, size=(n, int(n_reps)))
    roll_lo, roll_hi = off_ext.size // 5, off_ext.size - off_ext.size // 5
    rolls = rng.integers(roll_lo, roll_hi, size=(n, int(n_reps)))
    synth = np.empty((n * int(n_reps), off_ext.size, L), dtype=np.float64)
    for j in range(int(n_reps)):
        # Build a patch whose true peak sits delta_ij LATER than the patch centre, i.e.
        # patch(u) = T(u - delta), for which the aligner must return lag = +delta (the
        # same convention `refine_rpeaks` relies on when it does r <- r + lag). Then add
        # back beat i's own residual so the noise amplitude and colour are the record's.
        shifted = np.empty((n, off_ext.size, L), dtype=np.float64)
        for lead in range(L):
            shifted[:, :, lead] = np.stack(
                [np.interp(off_ext - deltas[i, j], off_pad, tpl_pad[:, lead]) for i in range(n)]
            )
        noise = np.stack(
            [np.roll(resid[(i + 1 + j) % n], int(rolls[i, j]), axis=0) for i in range(n)]
        )
        synth[j * n : (j + 1) * n] = shifted + noise
    lag_hat, _ = _xcorr_lags_from_patches(synth, tpl_cen, w, m)
    err = lag_hat - deltas.T.reshape(-1)

    rr_med = float(np.median(np.diff(r))) * ms
    rms = float(np.sqrt(np.mean(err**2)))
    out.update(
        jitter_samples_rms=rms,
        jitter_ms_rms=rms * ms,
        jitter_ms_mad=float(1.4826 * np.median(np.abs(err - np.median(err)))) * ms,
        jitter_ms_p95=float(np.percentile(np.abs(err), 95)) * ms,
        bias_ms=float(np.mean(err)) * ms,
        rr_median_ms=rr_med,
        jitter_frac_cycle=(rms * ms / rr_med) if rr_med > 0 else float("nan"),
        jitter_pct_cycle=(100.0 * rms * ms / rr_med) if rr_med > 0 else float("nan"),
    )

    if n_boot > 0:
        e2 = (err**2).reshape(int(n_reps), n).T  # (n_beats, n_reps)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        boot = np.sqrt(e2[idx].mean(axis=(1, 2))) * ms  # resample BEATS, not reps
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["jitter_ms_ci95"] = (float(lo), float(hi))

    if L >= 4:
        half = L // 2
        la, _ = _xcorr_lags_from_patches(ext_obs[:, :, :half], tpl_cen[:, :half], w, m)
        lb, _ = _xcorr_lags_from_patches(ext_obs[:, :, half:], tpl_cen[:, half:], w, m)
        out["lead_split_ms"] = float(np.std(la - lb, ddof=1) / 2.0) * ms
    return out

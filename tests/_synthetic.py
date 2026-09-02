"""Synthetic-ECG test fixtures with known ground truth.

Moved out of ttl-phase's `src/data/phase.py`, where `synthetic_ecg`/`evaluate_detector`
lived as production code despite being test fixtures with no other caller -- nothing in the
library depends on them, only tests do.
"""

from typing import Any

import numpy as np

TWO_PI = 2.0 * np.pi


def synthetic_ecg(
    n_beats: int = 12,
    fs: int = 500,
    hr_bpm: float = 70.0,
    rr_jitter_ms: float = 25.0,
    snr_db: float = 15.0,
    wander_mv: float = 0.3,
    wander_hz: float = 0.3,
    powerline_mv: float = 0.0,
    powerline_hz: float = 50.0,
    n_leads: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a 12-lead ECG with **known** fractional R-peak positions.

    Beats are Gaussian P/Q/R/S/T waves with per-lead amplitude gains and signs (lead 3 is
    given an inverted R, standing in for aVR). RR intervals are jittered, so R-peaks land
    off the sample grid and sub-sample refinement has something real to recover. Baseline
    wander (a low-frequency sinusoid with random phase), white noise at the requested SNR,
    and optional powerline interference are added.

    SNR convention: `snr_db` = 10*log10(var(clean lead)/var(noise)) applied per lead using
    that lead's own clean variance, so every lead sees the same SNR.

    Returns
    -------
    sig : (T, n_leads) float64, mV-scaled.
    rpeaks_true : (n_beats,) float64, the exact R-wave centres in fractional samples.
    """
    rng = np.random.default_rng(seed)
    rr_mean = 60.0 / hr_bpm * fs
    jit = rng.normal(0.0, rr_jitter_ms * fs / 1000.0, size=n_beats)
    centers = 0.6 * fs + np.cumsum(np.r_[0.0, np.full(n_beats - 1, rr_mean)]) + jit
    centers = np.sort(centers)
    T = int(np.ceil(centers[-1] + 0.6 * fs))

    # (offset_ms, amplitude_mV, sigma_ms) for P, Q, R, S, T relative to the R centre.
    waves = [
        (-160.0, 0.12, 25.0),
        (-22.0, -0.15, 6.0),
        (0.0, 1.20, 8.0),
        (24.0, -0.28, 8.0),
        (180.0, 0.30, 40.0),
    ]
    gains = rng.uniform(0.4, 1.3, size=n_leads)
    signs = np.ones(n_leads)
    if n_leads >= 4:
        signs[3] = -1.0  # aVR-like inversion

    t = np.arange(T, dtype=np.float64)
    clean = np.zeros((T, n_leads), dtype=np.float64)
    for c in centers:
        for off_ms, amp, sig_ms in waves:
            mu = c + off_ms * fs / 1000.0
            s = sig_ms * fs / 1000.0
            lo, hi = int(max(0, mu - 6 * s)), int(min(T, mu + 6 * s + 1))
            if hi <= lo:
                continue
            bump = amp * np.exp(-0.5 * ((t[lo:hi] - mu) / s) ** 2)
            clean[lo:hi, :] += bump[:, None] * (gains * signs)[None, :]

    sig = clean.copy()
    if wander_mv > 0:
        ph = rng.uniform(0, TWO_PI, size=n_leads)
        sig += wander_mv * np.sin(TWO_PI * wander_hz * t[:, None] / fs + ph[None, :])
    if powerline_mv > 0:
        sig += powerline_mv * np.sin(TWO_PI * powerline_hz * t[:, None] / fs)
    var_clean = np.var(clean, axis=0)
    noise_sd = np.sqrt(var_clean / (10.0 ** (snr_db / 10.0)))
    sig += rng.normal(0.0, 1.0, size=sig.shape) * noise_sd[None, :]
    return sig, centers


def evaluate_detector(
    true_rpeaks: np.ndarray, det_rpeaks: np.ndarray, fs: int, tol_ms: float = 50.0
) -> dict[str, Any]:
    """Score detected R-peaks against ground truth by one-to-one nearest matching.

    A detection matches a truth peak if it is the closest unused detection within
    `tol_ms`. Matching is greedy in ascending absolute error, which for peaks separated by
    an RR interval is equivalent to the optimal assignment.

    Returns 'n_true', 'n_det', 'tp', 'fp', 'fn', 'sensitivity', 'ppv', 'f1', and (over
    matched pairs) 'bias_ms', 'mae_ms', 'rms_ms', 'std_ms', 'p95_abs_ms'.
    """
    tr = np.asarray(true_rpeaks, dtype=np.float64)
    de = np.asarray(det_rpeaks, dtype=np.float64)
    tol = tol_ms * fs / 1000.0
    pairs = []
    if tr.size and de.size:
        D = np.abs(tr[:, None] - de[None, :])
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        used_t: set[int] = set()
        used_d: set[int] = set()
        for i, j in order:
            if D[i, j] > tol:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(int(i))
            used_d.add(int(j))
            pairs.append((int(i), int(j)))
    tp = len(pairs)
    fn, fp = int(tr.size - tp), int(de.size - tp)
    err_ms = np.array([de[j] - tr[i] for i, j in pairs]) * 1000.0 / fs if tp else np.empty(0)
    return {
        "n_true": int(tr.size),
        "n_det": int(de.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / tr.size if tr.size else float("nan"),
        "ppv": tp / de.size if de.size else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
        "bias_ms": float(np.mean(err_ms)) if tp else float("nan"),
        "mae_ms": float(np.mean(np.abs(err_ms))) if tp else float("nan"),
        "rms_ms": float(np.sqrt(np.mean(err_ms**2))) if tp else float("nan"),
        "std_ms": float(np.std(err_ms, ddof=1)) if tp > 1 else float("nan"),
        "p95_abs_ms": float(np.percentile(np.abs(err_ms), 95)) if tp else float("nan"),
    }

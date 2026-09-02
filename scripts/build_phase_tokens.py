#!/usr/bin/env python3
"""Phase calibration for the equivariant transport arm.

Ported near-verbatim from ttl-phase's `scripts/m0_phase_calibration.py` (pinned commit, see
notes/build.md). Derives the operator's (k0, n_max, k_j) spectrum from THIS cohort's own phase
clock, per notes/internal/phase_equivariance_notes_v13.pdf A.0.4 (n_max ~ 1/sigma_theta, harmonic
n attenuated by exp(-n^2*sigma_theta^2/2)) and Eq. 23 (the spectrum must be the contiguous set
{1, ..., n_max}, no gaps or repeats). Also runs the Delta-marginal / independence checks Eq. 16's
non-equivariant floor assumes (Delta ~ U[0, 2*pi), independent of absolute phase), with a
deliberately naive fixed-lag sampler run alongside as a negative control -- reproducing the
predecessor prototype's own defect (a non-uniform, phase-correlated Delta marginal) to demonstrate
these checks have teeth, not just to assert a clean number.

Reads (run scripts/build_manifest.py first):
  artifacts/manifest.parquet
  artifacts/phase/rpeaks.npz

Writes:
  artifacts/phase/theta_tokens.npz     (ecg_ids, theta (N, n_tokens) float32; patch centre
                                        timestamp -- see winder.eval.descriptors.token_centre_
                                        sample's docstring for why centre, not last-sample)
  artifacts/phase/m0_calibration.json  every statistic below, plus the derived spectrum

PRE-REGISTERED DECISION RULE (fixed before this script was ever run against real numbers, so the
corpus measurement cannot retroactively justify a different rule):
  k0 = 4 -- a DESIGN CHOICE (how much phase-invariant room to leave, notes A.0.4), not derived
  from data.
  n_max = round(1 / sigma_theta), clamped to [1, (K - k0) // 2].
  k_j: distribute floor(((K - k0) // 2) / n_max) to every one of the n_max harmonics, with the
  remainder (if the budget does not divide evenly by n_max) added ONE EACH to the LOWEST-numbered
  harmonics first -- they survive jitter best (exp(-n^2*sigma_theta^2/2) decays fastest at high
  n), so any leftover width is better spent sharpening the harmonics the clock can actually
  resolve than handed to a harmonic that is numerically near-dead already (notes A.0.4, "why you
  cannot simply make it wider").
  Fallback, only if sigma_theta cannot be measured at all (e.g. every record is missing
  jitter_ms/rr_median_ms): k0=4, n_max=7, k_j=[18]*7 -- the note's own worked config.

sigma_det (this cohort's measured R-peak localisation error, from the manifest's own jitter_ms,
itself winder.data.phase.jitter_estimate's RMS) is a LOWER BOUND, not the true clock error --
that function's own docstring states it excludes the ~2ms cross-record definitional offset
between "peak of the QRS energy envelope" and any other definition of R, and the piecewise-linear
RR-interpolation error. Treat every number this script derives from it as "I think", not "I
know".
"""

import argparse
import json
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from winder.data.phase import TWO_PI
from winder.eval.descriptors import load_rpeaks_by_ecg_id, theta_at_tokens

N_TOKENS = 125
N_SAMPLES = 1000
PATCH_WIDTH = 8
DECIMATION_FACTOR = 5.0  # rpeaks.npz's native 500 Hz -> EcgWindowDataset's 100 Hz grid (DATA-04)
K = 256  # production latent width, configs/baseline.yaml
K0 = 4  # design choice (see module docstring) -- not derived from data
DELTA_BINS = 36  # 10-degree bins
THETA_SRC_BINS = 12  # 30-degree bins, for the Delta-vs-theta_src independence check
FIXED_LAGS = list(range(1, 9))  # the negative-control sampler: adjacent-token-index pairs
LEAKAGE_N_MAX_PROBE = 10  # compute finite-T leakage up to this n; sliced to the real n_max below


def _feasible_spectrum(n_max_raw: float, k0: int, k_total: int) -> tuple[int, list[int]]:
    """The pre-registered rounding rule -- see module docstring."""
    budget = (k_total - k0) // 2
    n_max = max(1, min(round(n_max_raw), budget))
    base, remainder = divmod(budget, n_max)
    k_j = [base + 1 if j < remainder else base for j in range(n_max)]  # remainder -> lowest n
    assert k0 + 2 * sum(k_j) == k_total
    return n_max, k_j


def _build_theta_tokens(
    manifest: pd.DataFrame, rpeaks_by_id: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """theta at every included record's own token-centre timestamps, (n_included, N_TOKENS).

    Restricted to `status == "included"` (the phase-QC pool, build_manifest.py) -- the note's own
    scope statement (Ch. 1.1) reads exact equivariance as an inductive bias over a
    RHYTHM-REGULAR subpopulation, and this is that subpopulation. A record excluded here can
    still enter JEPA pretraining (a different, looser integrity criterion) -- the per-record
    phase_ok mask on the training path is what actually enforces this distinction; this script's
    own calibration is deliberately scoped to the population the transport arm will treat as
    phase-trustworthy.
    """
    included = manifest.loc[manifest["status"] == "included"].sort_values("ecg_id")
    ecg_ids = included["ecg_id"].to_numpy()
    theta = np.full((len(ecg_ids), N_TOKENS), np.nan, dtype=np.float32)
    for row_idx, ecg_id in enumerate(ecg_ids):
        rpeaks = rpeaks_by_id.get(int(ecg_id))
        if rpeaks is None or rpeaks.size < 2:
            continue
        theta[row_idx] = theta_at_tokens(
            rpeaks, N_TOKENS, N_SAMPLES, decimation_factor=DECIMATION_FACTOR, timestamp="centre"
        )
    return ecg_ids, theta


def _sigma_theta(manifest: pd.DataFrame, ecg_ids: np.ndarray) -> dict[str, Any]:
    included = manifest.set_index("ecg_id").loc[ecg_ids]
    rr_median_ms = included["rr_median_ms"].to_numpy(dtype=np.float64)
    jitter_ms = included["jitter_ms"].to_numpy(dtype=np.float64)
    valid = np.isfinite(rr_median_ms) & (rr_median_ms > 0) & np.isfinite(jitter_ms)
    if not np.any(valid):
        return {"measurable": False}

    cohort_t_rr_ms = float(np.median(rr_median_ms[valid]))
    sigma_det_rad_per_record = TWO_PI * jitter_ms[valid] / rr_median_ms[valid]
    sigma_det_rad = float(np.median(sigma_det_rad_per_record))
    patch_width_ms = PATCH_WIDTH / 100.0 * 1000.0  # 100 Hz token grid
    sigma_patch_rad = TWO_PI * (patch_width_ms / math.sqrt(12)) / cohort_t_rr_ms
    sigma_theta_rad = math.sqrt(sigma_det_rad**2 + sigma_patch_rad**2)
    return {
        "measurable": True,
        "cohort_t_rr_ms": cohort_t_rr_ms,
        "n_records_used": int(valid.sum()),
        "sigma_det_rad": sigma_det_rad,
        "patch_width_ms": patch_width_ms,
        "sigma_patch_rad": sigma_patch_rad,
        "sigma_theta_rad": sigma_theta_rad,
        "n_max_raw": 1.0 / sigma_theta_rad,
    }


def _bin_index(x: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip((x / (TWO_PI / n_bins)).astype(np.int64), 0, n_bins - 1)


def _accumulate_pair_histograms(theta: np.ndarray) -> dict[str, Any]:
    """One pass over every included record, accumulating histograms rather than materialising
    the corpus's full pair population (which would run into the hundreds of millions of rows) --
    this gives EXACT corpus-level statistics, not a subsample estimate."""
    all_pairs_delta_hist = np.zeros(DELTA_BINS, dtype=np.int64)
    joint_hist = np.zeros((THETA_SRC_BINS, DELTA_BINS), dtype=np.int64)
    fixed_lag_hist = np.zeros(DELTA_BINS, dtype=np.int64)
    n_pairs_total = 0
    n_fixed_lag_pairs_total = 0

    for theta_row in theta:
        valid_mask = np.isfinite(theta_row)
        valid_idx = np.flatnonzero(valid_mask)
        if valid_idx.size >= 2:
            theta_valid = theta_row[valid_idx]
            src_grid, tgt_grid = np.meshgrid(
                np.arange(valid_idx.size), np.arange(valid_idx.size), indexing="ij"
            )
            src, tgt = src_grid.ravel(), tgt_grid.ravel()
            keep = src != tgt
            delta = np.mod(theta_valid[tgt[keep]] - theta_valid[src[keep]], TWO_PI)
            src_bin = _bin_index(theta_valid[src[keep]], THETA_SRC_BINS)
            delta_bin = _bin_index(delta, DELTA_BINS)
            np.add.at(all_pairs_delta_hist, delta_bin, 1)
            np.add.at(joint_hist, (src_bin, delta_bin), 1)
            n_pairs_total += int(keep.sum())

        for lag in FIXED_LAGS:
            if valid_idx.size <= lag:
                continue
            # A fixed TOKEN-INDEX lag (the predecessor prototype's own scheme: adjacent detected
            # beats) -- deliberately naive, correlates Delta with local RR/heart rate. Kept only
            # as the negative control this calibration is pre-registered to fail.
            s = np.arange(0, N_TOKENS - lag)
            t = s + lag
            pair_valid = valid_mask[s] & valid_mask[t]
            if not np.any(pair_valid):
                continue
            delta_fl = np.mod(theta_row[t[pair_valid]] - theta_row[s[pair_valid]], TWO_PI)
            fl_bin = _bin_index(delta_fl, DELTA_BINS)
            np.add.at(fixed_lag_hist, fl_bin, 1)
            n_fixed_lag_pairs_total += int(pair_valid.sum())

    return {
        "all_pairs_delta_hist": all_pairs_delta_hist,
        "joint_hist": joint_hist,
        "fixed_lag_hist": fixed_lag_hist,
        "n_pairs_total": n_pairs_total,
        "n_fixed_lag_pairs_total": n_fixed_lag_pairs_total,
    }


def _uniformity_ratio(hist: np.ndarray) -> float:
    """max(bin count) / (uniform-expectation bin count) -- 1.0 is exactly uniform."""
    if hist.sum() == 0:
        return float("nan")
    expected = hist.sum() / hist.size
    return float(hist.max() / expected)


def _cramers_v(joint: np.ndarray) -> float:
    """Cramer's V on the (theta_src bin) x (Delta bin) contingency table -- 0 is independence,
    1 is a perfect (deterministic) association."""
    n = joint.sum()
    if n == 0:
        return float("nan")
    row_sums = joint.sum(axis=1, keepdims=True)
    col_sums = joint.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / n
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = np.where(expected > 0, (joint - expected) ** 2 / expected, 0.0).sum()
    r, c = joint.shape
    denom = n * (min(r, c) - 1)
    return float(math.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def _finite_t_leakage(theta: np.ndarray, n_max_probe: int) -> dict[int, dict[str, float]]:
    """|phi_hat_T(n)| = |mean_t exp(i*n*theta_t)| per record, for n = 1..n_max_probe -- the
    empirical characteristic function of the actual token phase sample, not the asymptotic
    equidistributed-theta limit Prop 4.1 assumes. Fully vectorised over records."""
    valid = np.isfinite(theta)
    counts = valid.sum(axis=1)
    theta_filled = np.where(valid, theta, 0.0).astype(np.float64)
    out: dict[int, dict[str, float]] = {}
    for n in range(1, n_max_probe + 1):
        cos_sum = np.where(valid, np.cos(n * theta_filled), 0.0).sum(axis=1)
        sin_sum = np.where(valid, np.sin(n * theta_filled), 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mag = np.sqrt(cos_sum**2 + sin_sum**2) / np.where(counts > 0, counts, np.nan)
        mag = mag[counts > 0]
        out[n] = {"median": float(np.median(mag)), "p90": float(np.percentile(mag, 90))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default="artifacts")
    args = ap.parse_args(argv)

    t0 = time.time()
    manifest_path = os.path.join(args.artifacts_dir, "manifest.parquet")
    rpeaks_path = os.path.join(args.artifacts_dir, "phase", "rpeaks.npz")
    manifest = pd.read_parquet(manifest_path)
    rpeaks_by_id = load_rpeaks_by_ecg_id(rpeaks_path)
    print(
        f"[build_phase_tokens] loaded manifest ({len(manifest)} rows) and rpeaks for "
        f"{len(rpeaks_by_id)} ids"
    )

    ecg_ids, theta = _build_theta_tokens(manifest, rpeaks_by_id)
    print(
        f"[build_phase_tokens] built theta_tokens for {len(ecg_ids)} included records, "
        f"{time.time() - t0:.0f}s"
    )

    phase_dir = os.path.join(args.artifacts_dir, "phase")
    os.makedirs(phase_dir, exist_ok=True)
    theta_path = os.path.join(phase_dir, "theta_tokens.npz")
    np.savez(
        theta_path,
        ecg_ids=ecg_ids,
        theta=theta,
        patch_width=PATCH_WIDTH,
        n_tokens=N_TOKENS,
        decimation_factor=DECIMATION_FACTOR,
        timestamp="centre",
    )
    print(f"[build_phase_tokens] wrote {theta_path}")

    token_yield = float(np.isfinite(theta).mean())

    sigma = _sigma_theta(manifest, ecg_ids)
    if sigma.get("measurable"):
        n_max, k_j = _feasible_spectrum(sigma["n_max_raw"], K0, K)
        spectrum_source = "calibrated"
    else:
        n_max, k_j = 7, [18] * 7
        spectrum_source = "fallback (sigma_theta unmeasurable)"
    n_j = list(range(1, n_max + 1))

    print(f"[build_phase_tokens] accumulating pair histograms over {len(ecg_ids)} records...")
    hist = _accumulate_pair_histograms(theta)
    all_pairs_uniformity = _uniformity_ratio(hist["all_pairs_delta_hist"])
    fixed_lag_uniformity = _uniformity_ratio(hist["fixed_lag_hist"])
    cramers_v = _cramers_v(hist["joint_hist"])
    leakage_probe = _finite_t_leakage(theta, LEAKAGE_N_MAX_PROBE)
    leakage_at_n_max = {n: leakage_probe[n] for n in range(1, n_max + 1)}

    halt = all_pairs_uniformity > 1.25 or cramers_v > 0.05
    negative_control_has_teeth = fixed_lag_uniformity > all_pairs_uniformity

    summary: dict[str, Any] = {
        "n_included_records": int(len(ecg_ids)),
        "token_phase_yield": token_yield,
        "sigma_theta": sigma,
        "spectrum": {
            "source": spectrum_source,
            "k0": K0,
            "n_max": n_max,
            "n_j": n_j,
            "k_j": k_j,
            "k_total": K,
            "dimension_check": K0 + 2 * sum(k_j),
        },
        "delta_marginal": {
            "all_pairs_uniformity_ratio": all_pairs_uniformity,
            "all_pairs_n_pairs": hist["n_pairs_total"],
            "fixed_lag_negative_control_uniformity_ratio": fixed_lag_uniformity,
            "fixed_lag_n_pairs": hist["n_fixed_lag_pairs_total"],
            "negative_control_has_teeth": negative_control_has_teeth,
            "cramers_v_delta_vs_theta_src": cramers_v,
            "halt_recommended": bool(halt),
        },
        "finite_t_leakage_at_n_max": leakage_at_n_max,
        "finite_t_leakage_probe_n1_to_10": leakage_probe,
        "elapsed_s": time.time() - t0,
    }

    summary_path = os.path.join(phase_dir, "m0_calibration.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[build_phase_tokens] wrote {summary_path}")

    print("\n===== BUILD_PHASE_TOKENS SUMMARY =====")
    print(f"included records:        {len(ecg_ids)}")
    print(f"token phase yield:        {token_yield:.4f}")
    if sigma.get("measurable"):
        print(f"cohort T_RR (ms):         {sigma['cohort_t_rr_ms']:.1f}")
        print(f"sigma_det (rad):          {sigma['sigma_det_rad']:.4f}  (lower bound, see doc)")
        print(f"sigma_patch (rad):        {sigma['sigma_patch_rad']:.4f}")
        print(f"sigma_theta (rad):        {sigma['sigma_theta_rad']:.4f}")
        print(f"n_max_raw (1/sigma):      {sigma['n_max_raw']:.2f}")
    else:
        print("sigma_theta: NOT MEASURABLE -- using fallback spectrum")
    print(f"spectrum source:          {spectrum_source}")
    print(f"k0={K0}  n_j={n_j}  k_j={k_j}  (K={K0 + 2 * sum(k_j)})")
    print(f"all-pairs Delta uniformity ratio:   {all_pairs_uniformity:.3f}  (1.0 = uniform)")
    print(f"fixed-lag (negative control) ratio: {fixed_lag_uniformity:.3f}")
    print(f"negative control has teeth:          {negative_control_has_teeth}")
    print(f"Cramer's V (Delta vs theta_src):     {cramers_v:.4f}  (0 = independent)")
    print(f"HALT RECOMMENDED: {halt}")
    print(f"elapsed: {summary['elapsed_s']:.0f}s")
    return 1 if halt else 0


if __name__ == "__main__":
    raise SystemExit(main())

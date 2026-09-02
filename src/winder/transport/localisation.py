"""Per-token anomaly scores for TIME-LOCALISED detection, and the online/causal question.

The construction gives a detector for free. Within one record, exact equivariance says every
token is the phase-transport of every other:

    zhat_t  ==  R_{theta_t - theta_s} zhat_s     for all valid s, t

so the extent to which that fails at token `t` is a per-token anomaly score:

    r_t  =  mean over s of [ 1 - <R_{theta_t - theta_s} zhat_s, zhat_t> ]

Read it as: "how far is this token from what the REST of this record says a token at this point in
the cardiac cycle should look like". That is a different question from the usual reconstruction or
prediction residual, which asks "how far is this token from what came before it" and therefore
confounds a genuine abnormality with an ordinary, expected change of cardiac phase.

**Every score here is record-relative and needs no labels, no threshold fitted on a cohort, and
no reference database.** The comparison is a record against itself.

**The causal variant is the product-relevant one.** `causal=True` restricts the reference set to
`s < t`, so `r_t` depends only on the past and can be emitted online, one value per token
(80 ms at this tokenisation). The encoder is causal by construction (`winder.jepa.encoder`), so
nothing else in the path breaks online use.

**BUT theta itself is not causal, and that is the binding constraint on any real-time claim.**
`winder.data.phase.phase_from_rpeaks` defines `theta_t = 2*pi (t - R_i) / (R_{i+1} - R_i)` -- it
needs the NEXT R-peak, i.e. the current beat must finish before its own phase is known. An honest
online system therefore either (a) accepts about one RR of latency, or (b) extrapolates the
current beat's period from the previous one, which `causal_phase_from_rpeaks` below implements and
which costs accuracy in proportion to beat-to-beat RR variability. Both options are measurable and
both should be reported; quoting the offline theta's detection numbers as if they were achievable
online would be wrong by roughly one cardiac cycle.
"""

import math
from typing import Any

import numpy as np
import torch

from winder.operators.harmonic import HarmonicTransport

__all__ = [
    "transport_residual_scores",
    "identity_residual_scores",
    "deviation_scores",
    "radial_scores",
    "causal_phase_from_rpeaks",
    "within_record_auroc",
    "localisation_error",
    "detection_latency",
]

_EPS = 1e-8  # matches winder.transport.loss's clamped normaliser exactly
TWO_PI = 2.0 * math.pi


def _normalise(z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    filled = torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))
    unit: torch.Tensor = filled / (filled.norm(dim=-1, keepdim=True) + _EPS)
    return unit


def _pair_reference_mask(valid: torch.Tensor, *, causal: bool, window: int | None) -> torch.Tensor:
    """`(N, T) -> (N, T_query, T_reference)`: which references each query token may use."""
    n, t = valid.shape
    pair = valid.unsqueeze(1) & valid.unsqueeze(2)  # [n, query, reference]
    idx = torch.arange(t)
    if causal:
        pair = pair & (idx.unsqueeze(0) < idx.unsqueeze(1)).unsqueeze(0)
    else:
        pair = pair & ~torch.eye(t, dtype=torch.bool).unsqueeze(0)
    if window is not None:
        lag = idx.unsqueeze(1) - idx.unsqueeze(0)  # query - reference
        pair = pair & (lag.abs() <= window).unsqueeze(0)
    return pair


def _residual(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport | None,
    *,
    causal: bool,
    window: int | None,
    record_chunk: int = 32,
) -> torch.Tensor:
    """Shared core. `operator=None` gives the identity-transport control (see
    `identity_residual_scores`)."""
    if z.ndim != 3:
        raise ValueError(f"z must be (N, T, K), got {tuple(z.shape)}")
    n, t, k = z.shape
    if theta.shape != (n, t):
        raise ValueError(f"theta shape {tuple(theta.shape)} must equal z's leading dims {(n, t)}")

    dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
    z, theta = z.to(dtype), theta.to(dtype)
    valid = torch.isfinite(theta)
    zhat = _normalise(z, valid)
    theta_filled = torch.where(valid, theta, torch.zeros_like(theta))

    # Chunked over RECORDS. The all-pairs intermediate is (chunk, T, T, K) -- at T=125, K=256 that
    # is ~16 MB per record per tensor and several exist at once, so a 400-record call would need
    # tens of GB in one allocation. Results are chunk-independent (each record's score depends
    # only on that record), so this is purely a memory knob.
    out = torch.empty((n, t), dtype=dtype)
    for start in range(0, n, record_chunk):
        stop = min(start + record_chunk, n)
        b = stop - start
        zc, tc = zhat[start:stop], theta_filled[start:stop]
        pair = _pair_reference_mask(valid[start:stop], causal=causal, window=window)
        # delta[b, q, r] = theta_query - theta_reference: move the REFERENCE to the query's phase.
        delta = tc.unsqueeze(2) - tc.unsqueeze(1)
        ref = zc.unsqueeze(1).expand(b, t, t, k)
        query = zc.unsqueeze(2).expand(b, t, t, k)
        moved = operator.transport(ref, delta) if operator is not None else ref
        defect = 1.0 - (moved * query).sum(dim=-1)  # (b, query, reference)
        counts = pair.sum(dim=2)
        total = (defect * pair).sum(dim=2)
        out[start:stop] = torch.where(
            counts > 0, total / counts.clamp_min(1), torch.full_like(total, float("nan"))
        )
    return torch.where(valid, out, torch.full_like(out, float("nan")))


def transport_residual_scores(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    *,
    causal: bool = False,
    window: int | None = None,
    record_chunk: int = 32,
) -> torch.Tensor:
    """`(N, T, K)`, `(N, T)` -> `(N, T)`: the phase-aware per-token anomaly score (module
    docstring). NaN at tokens with no defined phase or no admissible reference.

    `window` caps the reference set to tokens within that many positions, which is what a
    bounded-memory online implementation would actually do; `None` uses the whole record.
    """
    return _residual(z, theta, operator, causal=causal, window=window, record_chunk=record_chunk)


def identity_residual_scores(
    z: torch.Tensor,
    theta: torch.Tensor,
    *,
    causal: bool = False,
    window: int | None = None,
    record_chunk: int = 32,
) -> torch.Tensor:
    """The SAME statistic with `R` replaced by the identity -- the control that isolates exactly
    what the rotation buys.

    This is the comparison that matters. It holds the token set, the normalisation, the reference
    set, the averaging and the NaN handling all fixed, and changes one thing: whether the
    reference is rotated into the query's cardiac phase before being compared. Any gap between
    this and `transport_residual_scores` is attributable to the phase-equivariant structure and
    to nothing else.
    """
    return _residual(z, theta, None, causal=causal, window=window, record_chunk=record_chunk)


def deviation_scores(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """`1 - <zhat_t, mean_s zhat_s>`: a phase-BLIND baseline -- distance from the record's own
    average direction. No operator, no phase, no pair structure."""
    valid = torch.isfinite(theta)
    zhat = _normalise(z.to(torch.float32), valid)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (zhat * valid.unsqueeze(-1)).sum(dim=1) / counts
    mean = mean / (mean.norm(dim=-1, keepdim=True) + _EPS)
    scores = 1.0 - (zhat * mean.unsqueeze(1)).sum(dim=-1)
    return torch.where(valid, scores, torch.full_like(scores, float("nan")))


def radial_scores(
    z: torch.Tensor,
    theta: torch.Tensor,
    *,
    causal: bool = False,
    window: int | None = None,
) -> torch.Tensor:
    """`|log(||z_t|| + eps) - log(ref_t + eps)|`: the NORM channel every other score here
    discards by construction.

    All the detectors above l2-normalise their tokens (matching the transport loss's own
    normaliser), so a lesion that RESCALES the latent without turning it -- amplitude
    attenuation, lead dropout -- is invisible to them: radial blindness (theory notes sec 8).
    This score reads exactly and only that discarded channel, comparing each token's log-norm
    against a median reference drawn from the record's own norms:

      offline          `ref_t` is the record-level median norm over ALL valid tokens -- robust to
                       the lesion itself while it covers under half the record.
      causal           `ref_t` is the trailing median of the PAST valid tokens' norms (`s < t`,
                       the same all-past reference convention as the other causal detectors);
                       `window`, when given, bounds the look-back to `s >= t - window`, the same
                       horizon the pair-mask applies. An expanding median (window=None) rather
                       than an expanding mean: consistent with the offline reference and robust
                       to earlier lesioned tokens leaking into their successors' references.

    Log-space makes the score invariant to the record's overall latent scale (a global gain
    cancels between token and reference) and symmetric between attenuation and amplification.
    NaN at invalid-theta tokens and at causal tokens with no valid past reference; invalid
    tokens never enter any reference. `window` without `causal=True` is refused: the offline
    reference is the record-level median by definition, and a windowed offline radial is not a
    configuration the battery emits.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (N, T, K), got {tuple(z.shape)}")
    n, t = z.shape[0], z.shape[1]
    if theta.shape != (n, t):
        raise ValueError(f"theta shape {tuple(theta.shape)} must equal z's leading dims {(n, t)}")
    if window is not None and not causal:
        raise ValueError("window is only defined for the causal variant (see docstring)")

    dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
    z, theta = z.to(dtype), theta.to(dtype)
    valid = torch.isfinite(theta)
    norms = z.norm(dim=-1)
    masked = torch.where(valid, norms, torch.full_like(norms, float("nan")))

    if causal:
        # nanmedian over a growing (or window-bounded) past slice; all-NaN slices yield NaN,
        # which is exactly the "no valid reference" convention the other causal scores use.
        ref = torch.full_like(masked, float("nan"))
        for q in range(t):
            lo = 0 if window is None else max(0, q - window)
            if q > lo:
                ref[:, q] = masked[:, lo:q].nanmedian(dim=1).values
    else:
        ref = masked.nanmedian(dim=1, keepdim=True).values.expand(n, t)

    scores = (torch.log(norms + _EPS) - torch.log(ref + _EPS)).abs()
    return torch.where(valid, scores, torch.full_like(scores, float("nan")))


# ================================================================================ causal theta


def causal_phase_from_rpeaks(rpeaks: np.ndarray, n_samples: int) -> np.ndarray:
    """`theta` computed using ONLY information available at each sample -- the online surrogate.

    `winder.data.phase.phase_from_rpeaks` divides by the CURRENT beat's own `R_{i+1} - R_i`, which
    is unknown until the beat ends. Here each beat is instead divided by the PREVIOUS interval
    `R_i - R_{i-1}`, the standard online predictor:

        theta_hat(t) = 2*pi * (t - R_i) / (R_i - R_{i-1})   for R_i <= t < R_{i+1}

    wrapped into `[0, 2*pi)` so a beat that runs longer than predicted continues past 2*pi rather
    than saturating. The error against the offline definition is
    `theta * (RR_prev / RR_cur - 1)`, i.e. it scales with beat-to-beat RR variability and with
    position within the beat -- zero at the R-peak, largest just before the next one. NaN before
    the SECOND R-peak (the first beat has no previous interval) and after the last, so this is
    strictly more conservative in coverage than the offline version.
    """
    theta = np.full(int(n_samples), np.nan, dtype=np.float64)
    r = np.asarray(rpeaks, dtype=np.float64)
    if len(r) < 3:
        return theta
    t = np.arange(n_samples, dtype=np.float64)
    for i in range(1, len(r) - 1):
        prev_rr = r[i] - r[i - 1]
        if prev_rr <= 0:
            continue
        span = (t >= r[i]) & (t < r[i + 1])
        theta[span] = np.mod(TWO_PI * (t[span] - r[i]) / prev_rr, TWO_PI)
    return theta


# ==================================================================================== metrics


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC with mid-ranks for ties (the Mann-Whitney form)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(len(both), dtype=np.float64)
    ranks[order] = np.arange(1, len(both) + 1)
    # mid-rank ties
    sorted_vals = both[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_record_auroc(scores: torch.Tensor, token_mask: torch.Tensor) -> dict[str, Any]:
    """Token-level detection AUROC computed SEPARATELY IN EACH RECORD, then averaged.

    Within-record, not pooled: a pooled AUROC over all tokens of all records rewards a detector
    whose scores merely differ in level between records (say, higher on noisier recordings),
    which is a record-level property and has nothing to do with localising an event in time. The
    within-record version can only be won by ranking the perturbed tokens above the clean tokens
    OF THE SAME RECORD, which is the actual claim.
    """
    s = scores.detach().cpu().numpy()
    m = token_mask.detach().cpu().numpy().astype(bool)
    per_record = []
    record_index = []
    for i in range(s.shape[0]):
        ok = np.isfinite(s[i])
        pos, neg = s[i][ok & m[i]], s[i][ok & ~m[i]]
        if len(pos) and len(neg):
            per_record.append(_auroc(pos, neg))
            record_index.append(i)
    finite = [v for v in per_record if np.isfinite(v)]
    return {
        "mean_auroc": float(np.mean(finite)) if finite else float("nan"),
        "median_auroc": float(np.median(finite)) if finite else float("nan"),
        "n_records": len(finite),
        "per_record": per_record,
        # `record_index[j]` is the row of `scores` that produced `per_record[j]` -- records with no
        # positive or no negative token contribute nothing and are absent from BOTH lists. Callers
        # need this alignment to attach a cluster label (patient id) to each value: a
        # patient-clustered bootstrap over these AUROCs is impossible from `per_record` alone,
        # because its length is data-dependent and its order carries no record identity.
        "record_index": record_index,
    }


def localisation_error(
    scores: torch.Tensor, token_mask: torch.Tensor, *, patch_ms: float
) -> dict[str, float]:
    """Distance in milliseconds from each record's highest-scoring token to the nearest
    perturbed token. Zero when the peak lands inside the lesion."""
    s = scores.detach().cpu().numpy()
    m = token_mask.detach().cpu().numpy().astype(bool)
    errors = []
    for i in range(s.shape[0]):
        ok = np.isfinite(s[i])
        if not ok.any() or not m[i].any():
            continue
        peak = int(np.nanargmax(np.where(ok, s[i], -np.inf)))
        true_idx = np.flatnonzero(m[i])
        errors.append(float(np.min(np.abs(true_idx - peak)) * patch_ms))
    if not errors:
        return {"median_ms": float("nan"), "mean_ms": float("nan"), "hit_rate": float("nan")}
    arr = np.asarray(errors)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "hit_rate": float((arr == 0).mean()),  # peak fell inside the lesion
        "n_records": len(errors),
    }


def detection_latency(
    scores: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    patch_ms: float,
    false_alarms_per_record: float = 1.0,
) -> dict[str, float]:
    """Time from lesion onset to the first token whose score crosses a threshold calibrated on
    that record's OWN pre-onset segment.

    The threshold is the pre-onset quantile that would admit `false_alarms_per_record` crossings
    before onset, so every record is held to the same false-alarm budget regardless of its own
    noise level -- an absolute threshold shared across records would instead trade sensitivity on
    quiet recordings for false alarms on noisy ones and make the reported latency a function of
    cohort composition. Records that never cross afterwards are reported in `miss_rate`, not
    silently dropped or counted as latency zero.
    """
    s = scores.detach().cpu().numpy()
    m = token_mask.detach().cpu().numpy().astype(bool)
    latencies, misses, usable = [], 0, 0
    for i in range(s.shape[0]):
        if not m[i].any():
            continue
        onset = int(np.flatnonzero(m[i])[0])
        pre = s[i][:onset]
        pre = pre[np.isfinite(pre)]
        if len(pre) < 5:
            continue
        usable += 1
        q = max(0.0, 1.0 - false_alarms_per_record / len(pre))
        threshold = float(np.quantile(pre, q))
        post = s[i][onset:]
        crossed = np.flatnonzero(np.isfinite(post) & (post > threshold))
        if len(crossed):
            latencies.append(float(crossed[0] * patch_ms))
        else:
            misses += 1
    if not usable:
        return {"median_ms": float("nan"), "miss_rate": float("nan"), "n_records": 0}
    return {
        "median_ms": float(np.median(latencies)) if latencies else float("nan"),
        "p90_ms": float(np.quantile(latencies, 0.9)) if latencies else float("nan"),
        "miss_rate": misses / usable,
        "n_records": usable,
    }

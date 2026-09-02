r"""Delta-stratified transport gain over identity -- the "does the operator actually move the
latent to the right place, and at which phase separations" diagnostic.

For one within-record token pair `(s, t)` with `Delta = theta_t - theta_s`:

    gain(s, t) = <R_Delta zhat_s, zhat_t>  -  <zhat_s, zhat_t>
                 \_______ transported ____/    \____ identity ___/

i.e. how much better the transported source predicts the target than leaving the source alone.
This is exactly `1 - pair_defect` (from `winder.transport.loss.transport_loss`) minus the
un-transported cosine, so the two modules agree by construction on the first term; nothing is
re-derived here except the identity baseline and the stratification.

**Why stratify by Delta at all.** The pooled mean gain that `winder.transport.diagnostics` already
logs per step cannot distinguish "the operator helps everywhere" from "the operator helps only at
small Delta, where transport is nearly the identity anyway and the gain is nearly free". Those are
very different claims about whether a phase-equivariant structure was learned. Eq. 16's floor is
also a Delta-independent statement, so a Delta-resolved gain is the only place a
Delta-DEPENDENT failure can show up at all.

**RAW GAIN IS NOT COMPARABLE ACROSS STRATA -- use `gain_fraction` for that.** The largest gain
any operator could possibly achieve on a pair is `1 - <zhat_s, zhat_t>`, since the transported
cosine is bounded by 1. That ceiling is itself strongly Delta-dependent: at small Delta the two
tokens are already nearly aligned, the ceiling is near 0, and even a perfect operator scores
almost no raw gain. So a raw-gain curve that rises with Delta says mostly that the CEILING rises
with Delta. `gain_fraction[i] = sum(gain) / sum(1 - identity_cos)` within stratum `i` divides
that ceiling out: it is exactly 1.0 in every stratum under exact equivariance, exactly 0.0 under
invariant collapse, and negative where transport actively hurts. Read the raw curve for effect
size and the fraction curve for whether the structure holds up across the cycle.

**Two things are deliberately NOT the same axis.** The stratum index is `Delta mod 2*pi`, but the
rotation is applied at the RAW signed `Delta`. For the cyclic arm those coincide (integer omega
=> `R` is 2*pi-periodic, so binning modulo 2*pi loses nothing). For the free arm they do not, and
the difference is the point: a free arm whose omega has drifted off the integers gets its
within-stratum pairs rotated by genuinely different amounts, so its gain in that stratum is
diluted by exactly its own closure failure. That dilution is a measurement of non-closure, not an
artefact to correct for -- read it alongside `HarmonicTransport.closure_residual()`, which
measures the same failure in closed form at `Delta = 2*pi` exactly.

**The invariant-collapse signature.** If the encoder takes `L_trans`'s trivial optimum and becomes
phase-invariant (all energy in the `K0` block, which `R_Delta` fixes pointwise), then
`R_Delta zhat_s == zhat_s` for every Delta and the gain is IDENTICALLY ZERO in every stratum --
not small, exactly zero. A flat-at-zero curve is therefore diagnostic of collapse, and is
distinguishable from "no operator was trained" only by looking at the energy spectrum as well
(`winder.transport.geometry.harmonic_energy_spectrum`). This is the failure mode a predecessor
prototype's free arm hit; the test suite pins it as an exact identity, not an approximation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from winder.operators.harmonic import HarmonicTransport

__all__ = [
    "DeltaStratifiedGain",
    "delta_stratified_gain",
    "source_phase_stratified_gain",
    "cluster_bootstrap_mean",
]

_EPS = 1e-8  # identical to winder.transport.loss's own clamp -- Eq. 10's normaliser
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class DeltaStratifiedGain:
    """`delta_centers[i]` is the centre of stratum `i` in radians on `[0, 2*pi)`; `mean_gain[i]`
    is the pair-uniform mean gain within it, NaN for an empty stratum.

    `per_record_mean_gain` is the same quantity averaged within each record first (one number per
    record that has at least one valid pair), aligned with `record_index` -- the input to a
    clustered bootstrap, which must resample whole records rather than pairs (pairs within a
    record are massively dependent: `T^2` of them are built from `T` tokens).
    """

    n_strata: int
    delta_centers: list[float]
    mean_gain: list[float]
    gain_fraction: list[float]  # gain / achievable gain; 1.0 = exactly equivariant, 0.0 = collapsed
    mean_transported_cos: list[float]
    mean_identity_cos: list[float]
    n_pairs: list[int]
    overall_mean_gain: float
    overall_gain_fraction: float
    per_record_mean_gain: list[float]
    record_index: list[int]
    n_records_with_pairs: int


def delta_stratified_gain(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    *,
    n_strata: int = 16,
    record_chunk: int = 8,
) -> DeltaStratifiedGain:
    """`z` is `(N, T, K)` (K == `operator.dimension`), `theta` is `(N, T)` with NaN where a
    token's phase is undefined. Records are processed in chunks of `record_chunk` because the
    all-pairs tensor is `(chunk, T, T, K)` -- at T=125, K=256 that is ~64 MB per record in fp32,
    so the chunk size is a memory knob, not a numerical one (results are chunk-independent).
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (N, T, K), got shape {tuple(z.shape)}")
    n_records, n_tok, k = z.shape
    if theta.shape != (n_records, n_tok):
        raise ValueError(
            f"theta shape {tuple(theta.shape)} must equal z's leading dims {(n_records, n_tok)}"
        )
    if k != operator.dimension:
        raise ValueError(f"z's last dim {k} != operator.dimension {operator.dimension}")
    if n_strata < 1:
        raise ValueError(f"n_strata must be >= 1, got {n_strata}")

    compute_dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
    width = _TWO_PI / n_strata

    gain_sum = torch.zeros(n_strata, dtype=torch.float64)
    pred_sum = torch.zeros(n_strata, dtype=torch.float64)
    ident_sum = torch.zeros(n_strata, dtype=torch.float64)
    pair_count = torch.zeros(n_strata, dtype=torch.float64)
    per_record_gain: list[float] = []
    record_index: list[int] = []

    eye = torch.eye(n_tok, dtype=torch.bool, device=z.device)
    with torch.no_grad():
        for start in range(0, n_records, record_chunk):
            stop = min(start + record_chunk, n_records)
            zc = z[start:stop].to(compute_dtype)
            tc = theta[start:stop].to(compute_dtype)
            b = zc.shape[0]

            valid = torch.isfinite(tc)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye.unsqueeze(0)
            if not bool(pair_valid.any()):
                continue

            # Same NaN discipline as winder.transport.loss: fill z (not only theta) at invalid
            # positions so nothing non-finite ever enters an arithmetic path, even though this
            # function is under no_grad and has no backward pass to poison.
            z_filled = torch.where(valid.unsqueeze(-1), zc, torch.zeros_like(zc))
            zhat = z_filled / (z_filled.norm(dim=-1, keepdim=True) + _EPS)
            theta_filled = torch.where(valid, tc, torch.zeros_like(tc))
            delta = theta_filled.unsqueeze(1) - theta_filled.unsqueeze(2)  # (b, T, T), src s -> t

            src = zhat.unsqueeze(2).expand(b, n_tok, n_tok, k)
            tgt = zhat.unsqueeze(1).expand(b, n_tok, n_tok, k)
            cos_pred = (operator.transport(src, delta) * tgt).sum(dim=-1)
            cos_ident = (src * tgt).sum(dim=-1)
            gain = (cos_pred - cos_ident).to(torch.float64)

            # Stratify on Delta mod 2*pi; the rotation above already used the RAW signed Delta
            # (module docstring: for the free arm those are deliberately different).
            stratum = torch.clamp(
                (torch.remainder(delta.to(torch.float64), _TWO_PI) / width).long(), 0, n_strata - 1
            )
            flat_ok = pair_valid.reshape(-1)
            idx = stratum.reshape(-1)[flat_ok]
            gain_sum.index_add_(0, idx, gain.reshape(-1)[flat_ok])
            pred_sum.index_add_(0, idx, cos_pred.to(torch.float64).reshape(-1)[flat_ok])
            ident_sum.index_add_(0, idx, cos_ident.to(torch.float64).reshape(-1)[flat_ok])
            pair_count.index_add_(0, idx, torch.ones(int(flat_ok.sum()), dtype=torch.float64))

            rec_sum = (gain * pair_valid).sum(dim=(1, 2))
            rec_count = pair_valid.sum(dim=(1, 2))
            for i in range(b):
                if int(rec_count[i]) > 0:
                    per_record_gain.append(float(rec_sum[i] / rec_count[i]))
                    record_index.append(start + i)

    safe = pair_count.clamp_min(1.0)
    empty = pair_count == 0
    total_pairs = float(pair_count.sum())

    def _finalise(sums: torch.Tensor) -> list[float]:
        out = sums / safe
        return [float("nan") if e else float(v) for v, e in zip(out, empty, strict=True)]

    # Ratio of SUMS, not mean of per-pair ratios: the per-pair denominator (1 - identity cos) goes
    # to 0 as Delta -> 0, so a mean of ratios is dominated by whichever near-zero-Delta pair
    # happened to have the smallest denominator. The ratio of sums is the pair-count-weighted
    # quantity, is stable, and is still exactly 1.0 under exact equivariance.
    headroom_sum = pair_count - ident_sum  # sum over pairs of (1 - identity cos), >= 0
    achievable = headroom_sum.clamp_min(1e-12)
    total_headroom = float(headroom_sum.sum())

    return DeltaStratifiedGain(
        n_strata=n_strata,
        delta_centers=[(i + 0.5) * width for i in range(n_strata)],
        mean_gain=_finalise(gain_sum),
        gain_fraction=[
            float("nan") if e else float(v)
            for v, e in zip(gain_sum / achievable, empty, strict=True)
        ],
        mean_transported_cos=_finalise(pred_sum),
        mean_identity_cos=_finalise(ident_sum),
        n_pairs=[int(v) for v in pair_count],
        overall_mean_gain=float(gain_sum.sum() / total_pairs) if total_pairs > 0 else float("nan"),
        overall_gain_fraction=(
            float(gain_sum.sum() / total_headroom) if total_headroom > 0 else float("nan")
        ),
        per_record_mean_gain=per_record_gain,
        record_index=record_index,
        n_records_with_pairs=len(per_record_gain),
    )


def source_phase_stratified_gain(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    *,
    n_bins: int = 8,
    record_chunk: int = 8,
) -> dict[str, Any]:
    """The same gain, stratified by the SOURCE token's ABSOLUTE phase instead of by `Delta`.

    This is the time-localisation axis, and it asks a question `delta_stratified_gain` cannot:
    *where in the cardiac cycle* does the learned equivariant structure actually hold? Transport
    is a single global operator, so nothing in the objective makes it work equally well at every
    phase -- if the QRS bin transports cleanly while the diastolic bins do not, the model has
    learned a locally-valid rotation, not a global one, and any clinical claim about the
    diastolic segment inherits that.

    Interpretation of the two axes together:
      flat in `Delta`, flat in source phase   -- a genuinely global equivariant structure
      flat in `Delta`, peaked in source phase -- transport holds only near certain landmarks
      peaked in `Delta`                       -- closure/rate error (see `closure_residual`)

    Bins are on `[0, 2*pi)` with the same left-edge convention as
    `winder.transport.geometry.phase_resolved_trajectory`, so bin `b` here is bin `b` there and
    both share the landmark labels measured from the cohort's ensemble beat.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (N, T, K), got shape {tuple(z.shape)}")
    n_records, n_tok, k = z.shape
    if theta.shape != (n_records, n_tok):
        raise ValueError(
            f"theta shape {tuple(theta.shape)} must equal z's leading dims {(n_records, n_tok)}"
        )
    if k != operator.dimension:
        raise ValueError(f"z's last dim {k} != operator.dimension {operator.dimension}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    compute_dtype = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
    width = _TWO_PI / n_bins
    gain_sum = torch.zeros(n_bins, dtype=torch.float64)
    ident_sum = torch.zeros(n_bins, dtype=torch.float64)
    counts = torch.zeros(n_bins, dtype=torch.float64)
    eye = torch.eye(n_tok, dtype=torch.bool, device=z.device)

    with torch.no_grad():
        for start in range(0, n_records, record_chunk):
            zc = z[start : start + record_chunk].to(compute_dtype)
            tc = theta[start : start + record_chunk].to(compute_dtype)
            b = zc.shape[0]
            valid = torch.isfinite(tc)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye.unsqueeze(0)
            if not bool(pair_valid.any()):
                continue
            z_filled = torch.where(valid.unsqueeze(-1), zc, torch.zeros_like(zc))
            zhat = z_filled / (z_filled.norm(dim=-1, keepdim=True) + _EPS)
            theta_filled = torch.where(valid, tc, torch.zeros_like(tc))
            delta = theta_filled.unsqueeze(1) - theta_filled.unsqueeze(2)

            src = zhat.unsqueeze(2).expand(b, n_tok, n_tok, k)
            tgt = zhat.unsqueeze(1).expand(b, n_tok, n_tok, k)
            cos_pred = (operator.transport(src, delta) * tgt).sum(dim=-1)
            cos_ident = (src * tgt).sum(dim=-1)

            # `delta[.., s, t]` transports FROM s, so the source phase varies along dim 1.
            src_bin = torch.clamp((theta_filled / width).long(), 0, n_bins - 1)
            idx = src_bin.unsqueeze(2).expand(b, n_tok, n_tok).reshape(-1)
            ok = pair_valid.reshape(-1)
            gain_sum.index_add_(
                0, idx[ok], (cos_pred - cos_ident).to(torch.float64).reshape(-1)[ok]
            )
            ident_sum.index_add_(0, idx[ok], cos_ident.to(torch.float64).reshape(-1)[ok])
            counts.index_add_(0, idx[ok], torch.ones(int(ok.sum()), dtype=torch.float64))

    safe = counts.clamp_min(1.0)
    empty = counts == 0
    headroom = (counts - ident_sum).clamp_min(1e-12)
    return {
        "n_bins": n_bins,
        "bin_centers": [(i + 0.5) * width for i in range(n_bins)],
        "mean_gain": [
            float("nan") if e else float(v) for v, e in zip(gain_sum / safe, empty, strict=True)
        ],
        "gain_fraction": [
            float("nan") if e else float(v) for v, e in zip(gain_sum / headroom, empty, strict=True)
        ],
        "n_pairs": [int(v) for v in counts],
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_replicates: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile CI for the mean of `values`, resampling whole CLUSTERS with replacement.

    `cluster_ids` is the grouping to resample (patient id here, not record id: PTB-XL contains
    repeat patients, and two records from one patient share a heart, a lead placement and a
    recording session). Each replicate draws `n_clusters` clusters with replacement and takes the
    unweighted mean over all values in the drawn clusters -- so a cluster contributing more
    records carries more weight within a replicate, exactly as it does in the point estimate.
    """
    if values.shape != cluster_ids.shape:
        raise ValueError(f"values {values.shape} and cluster_ids {cluster_ids.shape} must match")
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_clusters": 0}

    uniq, inverse = np.unique(cluster_ids, return_inverse=True)
    by_cluster = [values[inverse == c] for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        draws[r] = np.concatenate([by_cluster[c] for c in pick]).mean()
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "n_clusters": int(len(uniq)),
        "n_values": int(values.size),
    }

"""The two acceptance gates on the transport mechanism -- G1 (is the measured transport gain
identified by the cardiac clock, or would any per-record rotation schedule produce it?) and the
patient-clustered detection-gap CI. ("Pre-registered" here refers to this module's own original
design brief, not `notes/fold10_preregistration.md`'s fold-10 event -- `detection_gap_ci` is
explicitly NOT part of that event's own current scope; see `fold10_nominal_eval.py`'s module
docstring.) Promoted from script-local logic in
`scripts/g1_shuffled_theta_gain_null.py` and `scripts/g2_detection_gap_ci.py` into a real,
importable, unit-tested library module.

**Extraction, not verbatim copy -- both gates were script `main()` bodies, not functions.** The
design brief names `g1_shuffled_theta_gain_null` and `detection_gap_ci` as functions to locate in
the reference scripts; neither exists as a standalone function there. `g1_shuffled_theta_gain_null`
is the per-checkpoint block inside `g1_shuffled_theta_gain_null.py::main`'s `for name in names:`
loop; `detection_gap_ci` is the per-(anomaly, untrained-arm) block inside
`g2_detection_gap_ci.py::main`'s nested loop. Both are extracted here as genuine functions, taking
already-loaded tensors/arrays rather than reading files -- the npz-key-string parsing
(`g2_detection_gap_ci.py::cells_for`, which decodes `{ckpt}|{anomaly}|{amp}|{clock}|{detector}
|auroc` keys out of a `localisation_per_record.npz` dump `scripts/p4_localisation_numerics.py`
produces) is left as script-specific I/O glue, per the design brief's own instruction that
promoted library code need not carry a script's file-reading scaffolding -- and that dump format
does not exist anywhere in winder-nominal (localisation's per-record dump script is out of this
port's scope). `permute_theta_within_record` and `corrected_by_record` ARE real, standalone
functions in the reference scripts and are ported here verbatim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from winder.operators.harmonic import HarmonicTransport
from winder.transport.delta_gain import cluster_bootstrap_mean, delta_stratified_gain

__all__ = [
    "G1_SHUFFLED_FRACTION_BAND",
    "g1_accept",
    "permute_theta_within_record",
    "g1_shuffled_theta_gain_null",
    "corrected_by_record",
    "detection_gap_ci",
]

#: The declared band a SHUFFLED arm's own overall gain fraction must stay within for G1 to accept
#: (Amendment 6): shuffling theta must destroy the gain down to near-zero, not merely reduce it.
G1_SHUFFLED_FRACTION_BAND = 0.02


def g1_accept(ci_excludes_zero: bool, shuffled_overall_gain_fraction: float) -> bool:
    """The G1 accept criterion, as a named function rather than an inline check scattered at call
    sites: the paired true-minus-shuffled CI excludes 0 AND the shuffled arm's own overall gain
    fraction is within +/-0.02 of zero -- i.e. shuffling theta destroys the gain, confirming it is
    IDENTIFIED by the cardiac clock rather than by any per-record rotation schedule."""
    return ci_excludes_zero and abs(shuffled_overall_gain_fraction) <= G1_SHUFFLED_FRACTION_BAND


def permute_theta_within_record(theta: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute each record's FINITE theta values among that record's finite positions.

    The NaN mask is preserved position-for-position, so `delta_stratified_gain` sees exactly the
    same valid-pair set and the same per-record pair counts -- the only thing destroyed is the
    correspondence between a token's position and its phase. That is the null we want: a phase
    schedule with the right marginal distribution and no relation to the waveform.
    """
    rng = np.random.default_rng(seed)
    out = theta.copy()
    for i in range(out.shape[0]):
        finite = np.isfinite(out[i])
        vals = out[i, finite]
        rng.shuffle(vals)
        out[i, finite] = vals
    return out


def g1_shuffled_theta_gain_null(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    patient_ids: np.ndarray,
    *,
    n_strata: int = 16,
    n_replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The G1 shuffled-theta transport-gain null (Amendment 6): is the measured transport gain
    IDENTIFIED by the cardiac clock, or would any per-record rotation schedule produce it?

    `gain = floor - defect` (`winder.transport.delta_gain.delta_stratified_gain`) is computed
    against a closed-form non-equivariant floor -- on its own that cannot rule out "the operator
    is exploiting token adjacency that happens to correlate with theta". This permutation null
    settles that question:
      - true gain:     `delta_stratified_gain(z, theta, operator)`
      - shuffled gain: identical, with theta PERMUTED WITHIN EACH RECORD over its finite
        positions (`permute_theta_within_record` -- the NaN mask is preserved exactly, so the
        same token pairs stay valid and `n_pairs` is unchanged; only the Delta assigned to each
        pair moves)
      - paired: per-record mean gain difference on the records present in BOTH, cluster-
        bootstrapped over PATIENT id (never record id -- PTB-XL repeats patients)

    `seed` feeds BOTH `permute_theta_within_record` and `cluster_bootstrap_mean`, matching the
    reference script's own convention of one `--seed` argument for the whole gate.

    `z`/`theta` are `(N, T, K)`/`(N, T)`, e.g. a checkpoint's cached eval-split projector output;
    `patient_ids` is `(N,)`, aligned to `z`'s leading (record) dimension.

    Dtype is the CALLER's responsibility -- this function does no casting. The reference repo's
    own `scripts/g1_shuffled_theta_gain_null.py` casts its cached z/theta to float32 before this
    call (never float16 from the cache, never a float64 upcast); a caller reproducing that
    script's published `g1_finale.json` booleans exactly must match that float32 convention, or
    an unrelated dtype choice can masquerade as a port bug.
    """
    true = delta_stratified_gain(z, theta, operator, n_strata=n_strata)
    theta_np = theta.detach().cpu().numpy()
    shuf_theta_np = permute_theta_within_record(theta_np, seed=seed)
    shuf = delta_stratified_gain(
        z, torch.from_numpy(shuf_theta_np).to(dtype=theta.dtype), operator, n_strata=n_strata
    )

    # Pair on record_index: the same records must appear in both (the NaN mask is preserved, so
    # they do), but never assume it -- intersect explicitly.
    t_map = dict(zip(true.record_index, true.per_record_mean_gain, strict=True))
    s_map = dict(zip(shuf.record_index, shuf.per_record_mean_gain, strict=True))
    shared = sorted(set(t_map) & set(s_map))
    diff = np.array([t_map[i] - s_map[i] for i in shared], dtype=np.float64)
    clusters = np.array([patient_ids[i] for i in shared])

    boot = cluster_bootstrap_mean(diff, clusters, n_replicates=n_replicates, seed=seed)
    ci_excludes_zero = bool(boot["lo"] > 0.0 or boot["hi"] < 0.0)
    shuffled_frac_in_band = bool(abs(shuf.overall_gain_fraction) <= G1_SHUFFLED_FRACTION_BAND)

    return {
        "true_overall_mean_gain": true.overall_mean_gain,
        "true_overall_gain_fraction": true.overall_gain_fraction,
        "shuffled_overall_mean_gain": shuf.overall_mean_gain,
        "shuffled_overall_gain_fraction": shuf.overall_gain_fraction,
        "paired_true_minus_shuffled": boot,
        "n_records_paired": len(shared),
        "ci_excludes_zero": ci_excludes_zero,
        "shuffled_fraction_within_pm0.02": shuffled_frac_in_band,
        "n_records_with_pairs_true": true.n_records_with_pairs,
        "n_records_with_pairs_shuffled": shuf.n_records_with_pairs,
        "g1_pass": g1_accept(ci_excludes_zero, shuf.overall_gain_fraction),
    }


#: Severity key of the "no anomaly injected" baseline cell every detection-gap correction is
#: measured against.
SEVERITY_ZERO = "0.0"


def corrected_by_record(
    sev: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, float], str] | None:
    """Peak-severity minus severity-0, per record, for one (clock, detector) cell.

    `sev` maps a severity label (e.g. `"0.0"`, `"0.5"`, `"1.0"`) to `(auroc, record_index)`
    arrays for that severity. Returns `(per_record_corrected, peak_severity_label)`, where the
    peak severity is the one with the largest RAW finite-mean AUROC (not the corrected one) --
    the detector's own best operating point, then baseline-corrected against severity 0 on the
    SAME records. `None` if severity 0 is absent, no non-zero severity exists, or every
    non-zero severity fails to overlap severity 0 on any record.
    """
    if SEVERITY_ZERO not in sev:
        return None
    amps = [a for a in sev if a != SEVERITY_ZERO]
    if not amps:
        return None
    null_a, null_i = sev[SEVERITY_ZERO]
    null_map = dict(zip(null_i.tolist(), null_a.tolist(), strict=True))
    best_amp, best_mean, best_map = None, -np.inf, {}
    for a in amps:
        auroc, idx = sev[a]
        finite = np.isfinite(auroc)
        m = float(np.mean(auroc[finite])) if finite.any() else -np.inf
        if m > best_mean:
            per = {
                i: v - null_map[i]
                for i, v in zip(idx.tolist(), auroc.tolist(), strict=True)
                if i in null_map and np.isfinite(v) and np.isfinite(null_map[i])
            }
            best_amp, best_mean, best_map = a, m, per
    return (best_map, str(best_amp)) if best_map else None


def detection_gap_ci(
    trained_severity: dict[str, tuple[np.ndarray, np.ndarray]],
    untrained_cells: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]],
    patient_ids: np.ndarray,
    *,
    n_replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """C4's patient-clustered detection-gap CI (Amendment 6f), for one anomaly type.

    For record `i`:

        g_i = [A_i(trained, peak) - A_i(trained, sev0)]
              - [A_i(untrained, best) - A_i(untrained, sev0)]

    i.e. each side is baseline-corrected against ITS OWN severity-0 null on the SAME record via
    `corrected_by_record`, then differenced. The trained side is the caller's PRE-SPECIFIED
    (clock, detector) cell (`trained_severity`); the untrained side is chosen HERE, as the
    (clock, detector) cell in `untrained_cells` with the largest mean CORRECTED gap (the
    adversarial choice against the trained arm, per Amendment 6f's declared no-flip convention --
    a sign-flipped detector carries hit-rate ~0 and localises nothing, so it is never selected by
    this rule even though it could inflate a naive AUROC-only comparison).

    `patient_ids` is aligned to the SAME record-index space `trained_severity`/`untrained_cells`
    use (i.e. the panel's own `z_<checkpoint>.npz::patient_ids`). Returns `None` if the trained
    cell or every untrained cell fails to resolve a peak-vs-severity-0 correction at all.
    """
    tr = corrected_by_record(trained_severity)
    if tr is None:
        return None
    tr_map, tr_amp = tr

    best: tuple[float, dict[int, float], str, str] | None = None
    for (clk, det), sev in untrained_cells.items():
        got = corrected_by_record(sev)
        if got is None:
            continue
        per, amp = got
        mean = float(np.mean(list(per.values())))
        if best is None or mean > best[0]:
            best = (mean, per, f"{det}/{clk}", amp)
    if best is None:
        return None
    _, un_map, un_label, un_amp = best

    shared = sorted(set(tr_map) & set(un_map))
    gaps = np.array([tr_map[i] - un_map[i] for i in shared], dtype=np.float64)
    clusters = np.array([patient_ids[i] for i in shared])
    boot = cluster_bootstrap_mean(gaps, clusters, n_replicates=n_replicates, seed=seed)
    excl = bool(boot["lo"] > 0.0 or boot["hi"] < 0.0)
    return {
        "trained_peak_severity": tr_amp,
        "untrained_best_detector": un_label,
        "untrained_peak_severity": un_amp,
        "gap": boot,
        "n_records_paired": len(shared),
        "ci_excludes_zero": excl,
        "passes_0.05_bar": bool(boot["lo"] > 0.05),
    }

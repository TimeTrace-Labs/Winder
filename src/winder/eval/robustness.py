"""The null ladder, matched-filter/jitter sweeps, Debye-Waller decay, lead-dropout robustness, and
heart-rate stratification -- promoted from `scripts/p1_panel_numerics.py`'s own script-local
functions (verified line range: `theta_variants` at line 455 through `heart_rate_strata` ending at
line 723 of the reference repo's copy at commit `a62f794`) into a real, importable, unit-tested
library module.

Every sweep here reuses the ALREADY-ENCODED `z` tensor a caller passes in (`z_by_split`) -- only
the lead-dropout arm needs a second encoder forward pass, since it corrupts the raw waveform
itself, not a pooling of an existing latent.

**One undeclared transitive dependency, resolved by inlining rather than porting a whole module.**
`heart_rate_strata`'s reference implementation imports `winder.eval.descriptors.heart_rate_bucket`.
`descriptors.py` itself is not in this port's scope (it pulls in R-peak-native-rate rescaling and
patch-window arithmetic this module has no other use for) and is not otherwise part of
winder-nominal. `heart_rate_bucket` itself, however, is a small, self-contained pure function
(numpy only, no other dependency) -- per this project's own precedent for a small closure (P3's
`winder.transport.loss`/`winder.eval.pooling` closure decision), it is inlined here directly
rather than standing up all of `descriptors.py` for one three-branch classifier.

**One signature change from the reference implementation.** The reference `heart_rate_strata`
takes a `probes_out: dict[str, Any]` first argument that its own body never reads (verified: no
reference to the name anywhere in the function, and the regression test that pins this function's
behavior in the reference repo, `tests/test_p1_panel_numerics.py`, passes it a placeholder
`{"...": None}` dict, confirming its own author never depended on the value read back). Dropped
here as dead scaffolding, not a behavior change -- see this module's own report for the decision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from winder.data.ptbxl import LEAD_ORDER
from winder.eval.metrics import macro_auroc
from winder.eval.pooling import demodulated_pool, masked_mean_pool
from winder.eval.probe import (
    LinearProbeConfig,
    decision_scores,
    fit_linear_probe,
    patient_bootstrap_ci,
)
from winder.eval.readout import encode_z
from winder.eval.tasks import CLASSES, fit_and_score
from winder.jepa.model import JepaModel
from winder.operators.harmonic import HarmonicTransport
from winder.transport.debye_waller import debye_waller_curve, fit_debye_waller_slope

__all__ = [
    "HEART_RATE_BUCKETS",
    "heart_rate_bucket",
    "theta_variants",
    "sweep_probe",
    "robustness_suite",
    "heart_rate_strata",
]

TWO_PI = 2.0 * np.pi

#: bradycardic / normal / tachycardic, the bands any phase-locked analysis stratifies by;
#: `"unknown"` is not a real band, only `heart_rate_bucket`'s missing-RR sentinel.
HEART_RATE_BUCKETS: tuple[str, ...] = ("bradycardic", "normal", "tachycardic")


def heart_rate_bucket(rr_median_ms: float) -> str:
    """`"bradycardic"` (<60 bpm) / `"normal"` (60-100 bpm) / `"tachycardic"` (>100 bpm).

    `"unknown"` for a non-finite or non-positive `rr_median_ms` (a record with no usable RR) --
    a fourth reportable bucket, not a dropped row. Ported verbatim from
    `winder.eval.descriptors.heart_rate_bucket` (module docstring: inlined rather than porting
    that whole module, since this is the only piece of it this module needs).
    """
    if not np.isfinite(rr_median_ms) or rr_median_ms <= 0:
        return "unknown"
    bpm = 60000.0 / rr_median_ms
    if bpm < 60.0:
        return "bradycardic"
    if bpm > 100.0:
        return "tachycardic"
    return "normal"


def theta_variants(
    theta: torch.Tensor, rr_median_ms: np.ndarray, patch_width: int, *, seed: int
) -> dict[str, torch.Tensor]:
    """The null ladder for "is the cardiac phase clock load-bearing, or is ANY per-token scalar
    enough?" Every variant preserves the valid/NaN MASK exactly, so all four are pooled over an
    identical token set and differ only in the angle demodulation undoes.

      `true`            the measured clock.
      `time_index`      a linear ramp at each record's OWN median RR -- correct period, but blind
                        to where the R-peaks actually fell. Isolates "periodicity at the right
                        rate" from "registration to the beat".
      `record_offset`   the true clock plus one random constant per record. Within-record phase
                        DIFFERENCES are exactly preserved (so Delta, and hence L_trans, is
                        untouched); only the common frame across records is destroyed. This is
                        the sharpest of the four: it isolates cross-record registration alone.
      `shuffled`        the record's own theta values permuted across its tokens. Same marginal
                        distribution, no temporal structure at all.
    """
    rng = np.random.default_rng(seed)
    valid = torch.isfinite(theta)
    n, t = theta.shape
    out = {"true": theta.clone()}

    token_centre_ms = (torch.arange(t, dtype=torch.float64) + 0.5) * patch_width * 10.0
    rr = torch.tensor(np.where(np.isfinite(rr_median_ms) & (rr_median_ms > 0), rr_median_ms, 842.6))
    ramp = torch.remainder(TWO_PI * token_centre_ms.unsqueeze(0) / rr.unsqueeze(1), TWO_PI)
    out["time_index"] = torch.where(valid, ramp.to(theta.dtype), theta)

    offsets = torch.tensor(rng.uniform(0, TWO_PI, size=(n, 1)), dtype=theta.dtype)
    out["record_offset"] = torch.where(valid, torch.remainder(theta + offsets, TWO_PI), theta)

    shuffled = theta.clone()
    for i in range(n):
        idx = valid[i].nonzero(as_tuple=True)[0]
        if len(idx) > 1:
            shuffled[i, idx] = theta[i, idx[torch.from_numpy(rng.permutation(len(idx)))]]
    out["shuffled"] = shuffled
    return out


def _fit_score_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    pid_eval: np.ndarray,
    cfg: LinearProbeConfig,
    *,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """`p1_panel_numerics.py::_fit_score`, ported privately: fits one probe over all of
    `winder.eval.tasks.CLASSES`, scores it on the eval split, and returns the macro-AUROC point
    (+ patient-clustered CI when `n_boot > 0`) plus a per-class AUROC breakdown.

    `n_boot=0` returns a point estimate with NaN bounds -- for the two SWEEPS below, whose
    figures are read as curves and whose per-point CI would cost more compute than the rest of
    this module's own tests combined (each CI is `n_boot` macro-AUROC evaluations, and the sweeps
    refit ~19 probes per checkpoint).

    Not exported: this hardcodes `classes=CLASSES` (all 5 superclasses) the way the reference
    repo's own `_fit_score` did; `winder.eval.tasks.fit_and_score` is the general-class-count
    building block this is built on, and is what a caller wanting a different class set should
    use directly.
    """
    scores_full, ev = fit_and_score(x_train, y_train, x_cal, y_cal, x_eval, y_eval, CLASSES, cfg)
    if n_boot > 0:
        point, lo, hi = patient_bootstrap_ci(
            y_eval[ev], scores_full[ev], pid_eval[ev], n_replicates=n_boot
        )
    else:
        point, lo, hi = macro_auroc(y_eval[ev], scores_full[ev])[0], float("nan"), float("nan")
    per_class = []
    for c in range(y_eval.shape[1]):
        col_y, col_s = y_eval[ev][:, c : c + 1], scores_full[ev][:, c : c + 1]
        if not 0 < col_y.sum() < len(col_y):
            per_class.append({"class": CLASSES[c], "auroc": float("nan")})
        else:
            per_class.append({"class": CLASSES[c], "auroc": macro_auroc(col_y, col_s)[0]})
    return {
        "macro_auroc": point,
        "lo": lo,
        "hi": hi,
        "per_class": per_class,
        "n_eval": int(ev.sum()),
        "n_dropped": int((~ev).sum()),
    }


def sweep_probe(
    make_features: Callable[[Any, str], np.ndarray],
    variants: list[Any],
    labels_by_split: dict[str, np.ndarray],
    eval_pid: np.ndarray,
    cfg: LinearProbeConfig,
    *,
    n_boot: int = 0,
) -> list[dict[str, Any]]:
    """Fit-and-score one probe per variant. `make_features(variant, split) -> (N, K)`.

    The probe is REFIT per variant rather than reusing the true-theta probe's weights: a probe
    fitted on true-theta features and applied to shuffled-theta features would measure
    distribution shift, not whether the shuffled representation carries the label. Refitting asks
    the fair question -- "how much class information is in THIS representation at all".
    """
    out = []
    for v in variants:
        feats = {s: make_features(v, s) for s in ("train", "cal", "eval")}
        res = _fit_score_point(
            feats["train"],
            labels_by_split["train"],
            feats["cal"],
            labels_by_split["cal"],
            feats["eval"],
            labels_by_split["eval"],
            eval_pid,
            cfg,
            n_boot=n_boot,
        )
        out.append(res)
    return out


def robustness_suite(
    model: JepaModel,
    operator: HarmonicTransport,
    z_by_split: dict[str, torch.Tensor],
    thetas: dict[str, torch.Tensor],
    waveforms: dict[str, torch.Tensor],
    labels: dict[str, np.ndarray],
    eval_pid: np.ndarray,
    rr_by_split: dict[str, np.ndarray],
    patch_width: int,
    cfg: LinearProbeConfig,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    """Five sweeps on one checkpoint, all reading the already-encoded `z` so nothing but the
    lead-dropout arm needs a second forward pass."""
    out: dict[str, Any] = {}
    splits = ("train", "cal", "eval")

    # ---- (1) null ladder: is the cardiac clock load-bearing, or would any scalar do?
    variant_theta = {
        s: theta_variants(thetas[s], rr_by_split[s], patch_width, seed=seed) for s in splits
    }
    names = ["true", "time_index", "record_offset", "shuffled"]
    ladder = sweep_probe(
        lambda v, s: demodulated_pool(z_by_split[s], variant_theta[s][v], operator).numpy(),
        names,
        labels,
        eval_pid,
        cfg,
        n_boot=500,
    )
    out["null_ladder"] = {n: r for n, r in zip(names, ladder, strict=True)}
    # masked-mean on the SAME token set: the readout that ignores theta entirely, so the ladder
    # has a theta-blind floor as well as a theta-scrambled one.
    mm = sweep_probe(
        lambda _v, s: masked_mean_pool(z_by_split[s], thetas[s]).numpy(),
        ["mm"],
        labels,
        eval_pid,
        cfg,
        n_boot=500,
    )
    out["null_ladder"]["masked_mean_theta_blind"] = mm[0]

    # ---- (2) matched-filter curve: demodulate with theta + phi and sweep phi over the cycle
    phis = [round(TWO_PI * i / 12, 6) for i in range(13)]
    offset_res = sweep_probe(
        lambda phi, s: demodulated_pool(
            z_by_split[s], torch.remainder(thetas[s] + phi, TWO_PI), operator
        ).numpy(),
        phis,
        labels,
        eval_pid,
        cfg,
    )
    out["theta_offset_sweep"] = {"phi": phis, "macro_auroc": [r["macro_auroc"] for r in offset_res]}

    # ---- (3) clock-jitter robustness: theta + N(0, sigma) on FINITE positions only
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    gen = torch.Generator().manual_seed(seed)
    # Splits OUTER, sigmas INNER: the shared generator `gen` is consumed in exactly this nested
    # order, so changing it (e.g. sigmas-outer/splits-inner) would silently draw different noise
    # -- not a numerically-neutral refactor.
    jitter_theta = {
        s: {
            sg: torch.where(
                torch.isfinite(thetas[s]),
                thetas[s] + torch.randn(thetas[s].shape, generator=gen) * sg,
                thetas[s],
            )
            for sg in sigmas
        }
        for s in splits
    }
    jitter_res = sweep_probe(
        lambda sg, s: demodulated_pool(z_by_split[s], jitter_theta[s][sg], operator).numpy(),
        sigmas,
        labels,
        eval_pid,
        cfg,
    )
    out["theta_jitter_sweep"] = {
        "sigma": sigmas,
        "macro_auroc": [r["macro_auroc"] for r in jitter_res],
    }

    # ---- (4) Debye-Waller on the same sigma grid: theory says slope = -1/2, parameter-free
    dw_gen = torch.Generator().manual_seed(seed)
    curve = debye_waller_curve(
        z_by_split["eval"].double(), thetas["eval"].double(), operator, sigmas, generator=dw_gen
    )
    fit_all = fit_debye_waller_slope(curve)
    out["debye_waller"] = {
        "sigmas": curve.sigmas,
        "n_j": curve.n_j,
        "amplitudes": curve.amplitudes,
        "fit_all_harmonics": {
            "slope": fit_all.slope,
            "intercept": fit_all.intercept,
            "r_squared": fit_all.r_squared,
            "n_points": fit_all.n_points,
        },
    }

    # ---- (5) lead dropout: probe FITTED ON CLEAN data, evaluated on a corrupted eval split --
    # the deployment-realistic protocol (a detached electrode is not something you retrain for).
    clean = {s: demodulated_pool(z_by_split[s], thetas[s], operator).numpy() for s in splits}
    # Computed once from the DEMODULATED clean features and reused for BOTH cells below (not
    # recomputed per-cell): masked_mean_pool and demodulated_pool share the identical
    # isfinite(theta) validity gate (winder.eval.pooling's own `_valid_masked_mean`), so a
    # record's NaN-row status is identical under either pooling -- reusing one mask is a
    # simplification, not an approximation.
    tr, ca = np.isfinite(clean["train"]).all(axis=1), np.isfinite(clean["cal"]).all(axis=1)
    dropout: dict[str, Any] = {}
    for cell_name, pool_fn in (
        ("z/demodulated", lambda zt, th: demodulated_pool(zt, th, operator)),
        ("z/mean", lambda zt, th: masked_mean_pool(zt, th)),
    ):
        base = {s: pool_fn(z_by_split[s], thetas[s]).numpy() for s in splits}
        probe = fit_linear_probe(
            base["train"][tr],
            labels["train"][tr],
            base["cal"][ca],
            labels["cal"][ca],
            cfg,
            classes=CLASSES,
        )
        per_lead = []
        for lead in range(len(LEAD_ORDER)):
            corrupted = waveforms["eval"].clone()
            corrupted[:, lead, :] = 0.0  # (N, 12, T) lead-major
            zc = encode_z(model, corrupted, device)
            xc = pool_fn(zc, thetas["eval"]).numpy()
            ok = np.isfinite(xc).all(axis=1)
            pt, lo, hi = patient_bootstrap_ci(
                labels["eval"][ok],
                decision_scores(probe, xc[ok]),
                eval_pid[ok],
                n_replicates=300,
            )
            per_lead.append({"lead": LEAD_ORDER[lead], "macro_auroc": pt, "lo": lo, "hi": hi})
            del zc, corrupted
        xb = base["eval"]
        okb = np.isfinite(xb).all(axis=1)
        ptb, _, _ = patient_bootstrap_ci(
            labels["eval"][okb], decision_scores(probe, xb[okb]), eval_pid[okb], n_replicates=300
        )
        dropout[cell_name] = {"intact_macro_auroc": ptb, "per_lead": per_lead}
    out["lead_dropout"] = dropout

    return out


def heart_rate_strata(
    scores_store: dict[str, np.ndarray],
    labels_eval: np.ndarray,
    eval_pid: np.ndarray,
    rr_eval: np.ndarray,
) -> dict[str, Any]:
    """Post-hoc stratification of a probe's own eval scores by heart-rate band -- no refit, so
    this asks "does the benefit survive within band", not "can a per-band probe do better".

    The confound this closes: a phase-based readout has an obvious route to spurious gain if the
    label distribution shifts with heart rate (a tachycardic record is more likely abnormal). A
    benefit that vanishes inside every band was rate information all along.

    Requires every value of `scores_store` to hold FULL-LENGTH score arrays (NaN in the dropped
    rows, per `winder.eval.tasks.fit_and_score`'s own convention) and RAISES otherwise -- never
    softened to a warning. A compressed array indexed against `buckets`/`labels_eval`/`eval_pid`
    (all in original record order) is the bug this guard exists to make impossible: on real data
    it reported every band at chance, which reads as a genuine rate confound rather than as a
    re-indexing error (reference repo commit `f8ce270`).
    """
    buckets = np.array([heart_rate_bucket(float(v)) for v in rr_eval])
    out: dict[str, Any] = {"bucket_counts": {}}
    for b in sorted(set(buckets)):
        out["bucket_counts"][b] = int((buckets == b).sum())
    for key, scores in scores_store.items():
        if len(scores) != len(buckets):
            raise AssertionError(
                f"[robustness] {key}: {len(scores)} score rows against {len(buckets)} eval "
                "records -- scores must be full-length with NaN in the dropped rows, so that "
                "scores and record metadata are indexed by the same row"
            )
        per_bucket = {}
        for b in sorted(set(buckets)):
            m = (buckets == b) & np.isfinite(scores).all(axis=1)
            if m.sum() < 40 or labels_eval[m].sum(axis=0).min() < 2:
                continue
            pt, lo, hi = patient_bootstrap_ci(
                labels_eval[m], scores[m], eval_pid[m], n_replicates=1000
            )
            per_bucket[b] = {"macro_auroc": pt, "lo": lo, "hi": hi, "n": int(m.sum())}
        if per_bucket:
            out[key] = per_bucket
    return out

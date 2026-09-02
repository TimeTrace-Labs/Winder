#!/usr/bin/env python3
"""Heart-rate-stratified probe readout -- wires up `winder.eval.robustness.heart_rate_strata`,
which was ported into this repo complete and tested but had no call site anywhere.

**The question this answers.** A phase-equivariant readout keyed to cardiac phase has an obvious
route to spurious gain: if the label distribution shifts with heart rate (a tachycardic record is
more likely abnormal), a model that has merely learned "fast = sick" scores well without having
learned any morphology. `heart_rate_strata` closes that confound by re-scoring the probe's own
eval predictions WITHIN each heart-rate band -- no refit, so it asks "does the benefit survive
inside a band", never "can a per-band probe do better". A benefit that vanishes inside every band
was rate information all along.

Secondarily, it is the axis a clinical reader asks about first: does the model fail on
bradycardic or tachycardic patients? Flatness across bands is a robustness property, and it is
measured here rather than assumed.

**Split status -- read this before quoting any number from this script.** The four real arms were
trained on folds 1-9, and fold 10 is spent (scored exactly once under the pre-registered ceremony
in `notes/fold10_preregistration.md`; its per-record scores were never persisted, so re-scoring
fold 10 by heart-rate band would be NEW MEASUREMENT on sealed data, not re-reporting -- forbidden).
There is therefore no clean held-out split available to this script, and it runs on
`LEGACY_FOLD_CONFIG`'s fold-9 eval split, which IS training data for these checkpoints. Every
report this script writes carries `"split_status": "train_contaminated"` and `"headline": false`,
matching `scripts/eval_suite.py`'s own convention.

What that contamination does and does not spoil:
  - SPOILED: the absolute per-band AUROC. It is inflated, in every band, and is not quotable.
  - USABLE: whether the signal-vs-control benefit SURVIVES within band. Contamination inflates
    all bands together; it does not manufacture a within-band gap where none exists, nor close one
    that does. So this axis can still FAIL informatively -- which is the point of running it.

**What this script does not touch.** Nothing on the fold-10 ceremony's import path is modified:
`heart_rate_strata`/`heart_rate_bucket` are imported from `winder.eval.robustness` unchanged, and
the cohort comes from `scripts/eval_suite.py`'s own `build_p9_cohort` rather than a fourth
independently-maintained copy of the split/decode/RR-lookup logic (the same single-source-of-truth
discipline gate-3 round 3 established for the ceremony's own path).
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from eval_suite import _write_json_atomic, build_p9_cohort
from fold10_nominal_eval import DEFAULT_ARMS, DEFAULT_DATA_ROOT

from winder.data.integrity import git_sha
from winder.eval.comparison import EvalCohort
from winder.eval.pooling import masked_mean_pool
from winder.eval.probe import LinearProbeConfig
from winder.eval.readout import (
    discover_seed_checkpoints,
    encode_z,
    load_model_and_operator,
)
from winder.eval.robustness import HEART_RATE_BUCKETS, heart_rate_strata
from winder.eval.tasks import CLASSES, ci_row, fit_and_score

MILESTONE_ID = "heart-rate-stratified-probe-readout"

#: Same literal constants, same reason, as `scripts/eval_suite.py`: emitted at the TOP level of
#: every report unconditionally, never merely documented. See the module docstring's split-status
#: section for what the contamination does and does not spoil.
SPLIT_STATUS = "train_contaminated"
HEADLINE = False

_CLASS_COLUMNS = list(range(len(CLASSES)))

#: The probe cell this script stratifies. `z/mean` (`masked_mean_pool`) is the cell
#: `scripts/eval_suite.py` itself reports and the one the fold-10 ceremony carries as
#: `probe_superclass5_z_mean`, so a reader can line the bands up against a number they have
#: already seen. The demodulated cell is deliberately NOT added here: it would double the
#: runtime for a second answer to the same question.
CELL_NAME = "z/mean"


def strata_for_checkpoint(
    ckpt_dir: str,
    cohort: EvalCohort,
    *,
    device: torch.device,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    """One checkpoint: encode, fit the `z/mean` probe once, then report BOTH the pooled
    (all-band) CI and the per-band CIs off that one set of predictions.

    The pooled row is computed here, alongside the bands, rather than read from another report --
    so "the benefit survived within band" is a comparison between two numbers produced by the
    same probe fit on the same rows, never across two runs that might differ in split or seed.
    """
    model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    try:
        feats = {
            s: masked_mean_pool(
                encode_z(model, cohort.waveforms[s], device), cohort.thetas[s]
            ).numpy()
            for s in ("train", "cal", "eval")
        }
        scores_full, _ev = fit_and_score(
            feats["train"],
            cohort.labels["train"],
            feats["cal"],
            cohort.labels["cal"],
            feats["eval"],
            cohort.labels["eval"],
            CLASSES,
            LinearProbeConfig(seed_probe=seed),
        )
    finally:
        del model, operator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled = ci_row(
        scores_full,
        cohort.labels["eval"],
        cohort.patient_ids["eval"],
        _CLASS_COLUMNS,
        n_boot,
        seed,
    )
    strata = heart_rate_strata(
        {CELL_NAME: scores_full},
        cohort.labels["eval"],
        cohort.patient_ids["eval"],
        cohort.rr_median_ms["eval"],
    )
    return {"pooled": pooled, "strata": strata}


def within_band_deltas(
    per_arm: dict[str, dict[str, Any]], *, treatment_prefix: str, control_prefix: str
) -> dict[str, Any]:
    """Per-band treatment-minus-control gaps, seed-averaged within each side.

    Deliberately a plain point-estimate difference of band means, NOT a paired bootstrap: the two
    arms are separate models, so their per-record scores are not paired in the sense
    `paired_patient_bootstrap_delta` requires, and each side's own band CI is already reported
    above. This row exists to answer "does the gap survive inside the band", at the precision that
    question needs -- it is not offered as an inferential test, and says so in its own output via
    `"inferential": false`.
    """
    bands: dict[str, Any] = {}
    for band in HEART_RATE_BUCKETS:
        t = [
            v["strata"][CELL_NAME][band]["macro_auroc"]
            for k, v in per_arm.items()
            if k.startswith(treatment_prefix) and band in v["strata"].get(CELL_NAME, {})
        ]
        c = [
            v["strata"][CELL_NAME][band]["macro_auroc"]
            for k, v in per_arm.items()
            if k.startswith(control_prefix) and band in v["strata"].get(CELL_NAME, {})
        ]
        if not t or not c:
            continue
        bands[band] = {
            "treatment_mean": float(np.mean(t)),
            "control_mean": float(np.mean(c)),
            "delta": float(np.mean(t) - np.mean(c)),
            "n_treatment_arms": len(t),
            "n_control_arms": len(c),
        }
    pooled_t = [
        v["pooled"]["macro_auroc"] for k, v in per_arm.items() if k.startswith(treatment_prefix)
    ]
    pooled_c = [
        v["pooled"]["macro_auroc"] for k, v in per_arm.items() if k.startswith(control_prefix)
    ]
    return {
        "inferential": False,
        "per_band": bands,
        "pooled": {
            "treatment_mean": float(np.mean(pooled_t)) if pooled_t else None,
            "control_mean": float(np.mean(pooled_c)) if pooled_c else None,
            "delta": (
                float(np.mean(pooled_t) - np.mean(pooled_c)) if pooled_t and pooled_c else None
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument("--lead-stats-path", default=None)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--step", type=int, default=5000, help="checkpoint step to stratify")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0, help="0 = full pool; >0 for fast tests")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="default <artifacts-dir>/reports/hr_strata.json")
    args = ap.parse_args(argv)

    roster_dir = args.roster_dir or os.path.join(args.artifacts_dir, "roster")
    lead_stats_path = args.lead_stats_path or os.path.join(
        args.artifacts_dir, "lead_stats_f1to9.json"
    )
    out_path = args.out or os.path.join(args.artifacts_dir, "reports", "hr_strata.json")
    arm_names = [a for a in args.arms.split(",") if a]
    device = torch.device(args.device)
    t0 = time.time()

    cohort, bookkeeping = build_p9_cohort(
        args.data_root, args.artifacts_dir, lead_stats_path, train_limit=args.train_limit
    )
    band_counts = {
        str(b): int(c)
        for b, c in heart_rate_strata(
            {}, cohort.labels["eval"], cohort.patient_ids["eval"], cohort.rr_median_ms["eval"]
        )["bucket_counts"].items()
    }
    print(
        f"[hr_strata] cohort built ({time.time() - t0:.0f}s); band counts: {band_counts}",
        flush=True,
    )

    per_arm: dict[str, dict[str, Any]] = {}
    for name in arm_names:
        arm_dir = os.path.join(roster_dir, name)
        steps = discover_seed_checkpoints(arm_dir)
        if args.step not in steps:
            raise SystemExit(
                f"[hr_strata] {name}: step {args.step} not found (have {sorted(steps)})"
            )
        per_arm[name] = strata_for_checkpoint(
            steps[args.step], cohort, device=device, seed=args.seed, n_boot=args.n_boot
        )
        cell = per_arm[name]["strata"].get(CELL_NAME, {})
        pooled = per_arm[name]["pooled"]["macro_auroc"]
        bands = " ".join(f"{b}={cell[b]['macro_auroc']:.4f}" for b in sorted(cell))
        print(f"[hr_strata] {name}: pooled={pooled:.4f}  {bands}", flush=True)

    payload = {
        "status": "PASS",
        "milestone_id": MILESTONE_ID,
        "split_status": SPLIT_STATUS,
        "headline": HEADLINE,
        "metrics": {
            "cell": CELL_NAME,
            "step": args.step,
            "band_counts": band_counts,
            "per_arm": per_arm,
            "within_band_deltas": within_band_deltas(
                per_arm, treatment_prefix="signal", control_prefix="control"
            ),
            "cohort": bookkeeping,
        },
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": {
                "arms": arm_names,
                "step": args.step,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "lead_stats_path": lead_stats_path,
                "device": str(device),
            },
            "seed": args.seed,
        },
        "decisions": [
            "Runs on LEGACY_FOLD_CONFIG's fold-9 eval split, which is TRAINING data for these "
            "checkpoints -- fold 10 is spent and its per-record scores were never persisted, so "
            "re-scoring it by band would be new measurement on sealed data. Absolute per-band "
            "AUROC is inflated and not quotable; the within-band survival of the "
            "signal-vs-control gap is the usable readout.",
            "No refit per band: this asks whether the benefit survives inside a band, not "
            "whether a per-band probe could do better.",
            "within_band_deltas is a point-estimate difference, not an inferential test "
            "(separate models, unpaired scores) -- flagged in its own output as inferential=false.",
        ],
        "questions": [],
    }
    _write_json_atomic(out_path, payload)
    print(f"[hr_strata] wrote {out_path} ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

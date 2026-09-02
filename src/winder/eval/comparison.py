"""Generic N-way arm comparison -- the ablation infrastructure's own comparison-table builder,
never hardcoded to two names. New in this port (not extracted from any reference-repo script,
per the design brief's own pseudocode).

**Signature deviation from the design brief, and why.** The brief's pseudocode gives
`arm_comparison_table(arms: dict[str, str], *, device, n_boot=1000, seed=0) -> dict[str, Any]`,
with no data arguments -- implying the function loads PTB-XL waveforms/manifest/theta-tokens/
lead-stats itself. That is untestable against small synthetic fixtures and couples this module to
a `--data-root` a unit test should not need. Instead, `arm_comparison_table` takes an explicit
`EvalCohort` (pre-loaded waveforms/thetas/labels/patient-ids/RR medians for the train/cal/eval
splits every arm is compared on) as a required second argument -- the brief's own name and return
shape are unchanged, only the "where does the data come from" question is answered by explicit
injection rather than an implicit load. A caller building a real campaign table constructs one
`EvalCohort` from `winder.data.ptbxl`/`winder.transport.dataset` once and passes it to every arm.

**One more shape addition beyond the brief's own pseudocode.** Each arm's row also carries a
`"_scores"` key: the full-length, NaN-padded eval-split decision scores (`winder.eval.tasks.
fit_and_score`'s own convention). `pairwise_deltas` needs these to run a genuinely PAIRED
bootstrap (`winder.eval.probe.paired_patient_bootstrap_delta`) rather than comparing two
independently-bootstrapped CIs by eye -- the same reasoning `winder.eval.robustness.
heart_rate_strata` already enforces elsewhere in this package. The leading underscore marks it as
plumbing for `pairwise_deltas`, not a headline number to report on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from winder.eval.gates import g1_shuffled_theta_gain_null
from winder.eval.pooling import masked_mean_pool
from winder.eval.probe import LinearProbeConfig, paired_patient_bootstrap_delta
from winder.eval.readout import encode_z, load_model_and_operator
from winder.eval.robustness import robustness_suite
from winder.eval.tasks import CLASSES, ci_row, fit_and_score
from winder.transport.report import gain_report

__all__ = ["EvalCohort", "arm_comparison_table", "pairwise_deltas"]


@dataclass(frozen=True)
class EvalCohort:
    """The pre-loaded evaluation data every arm in a comparison table is scored against (module
    docstring's signature-deviation note).

    All dict-valued fields are keyed by split name, `{"train", "cal", "eval"}`; every array/tensor
    for a given split shares that split's own leading (record) dimension.
    """

    waveforms: dict[str, torch.Tensor]  #: `(N, 12, T)`, lead-major (winder.jepa.dataset)
    thetas: dict[str, torch.Tensor]  #: `(N, n_tokens)`, NaN where phase is undefined
    labels: dict[str, np.ndarray]  #: `(N, len(tasks.CLASSES))` float32 superclass multi-hot
    patient_ids: dict[str, np.ndarray]  #: `(N,)`
    rr_median_ms: dict[str, np.ndarray]  #: `(N,)`, for `robustness.robustness_suite`'s null ladder
    patch_width: int
    gain_limit: int = 250  #: records used for the all-pairs gain/G1 statistics (memory-bounded)


def arm_comparison_table(
    arms: dict[str, str],
    cohort: EvalCohort,
    *,
    device: torch.device,
    n_boot: int = 1000,
    n_strata: int = 16,
    n_replicates: int = 2000,
    seed: int = 0,
    probe_cfg: LinearProbeConfig | None = None,
) -> dict[str, Any]:
    """One row per arm: `{"macro_auroc", "lo", "hi", "gain_fraction", "g1_pass",
    "lead_dropout_worst_drop", "_scores"}` (module docstring on the last key).

    `arms` maps an arbitrary display name to a checkpoint directory -- never hardcoded to
    "signal"/"control". Built per arm by calling `readout.load_model_and_operator` +
    `readout.encode_z` + `tasks.fit_and_score` (+ `tasks.ci_row`) +
    `transport.report.gain_report` + `gates.g1_shuffled_theta_gain_null` +
    `robustness.robustness_suite`, exactly as named in the design brief's own pseudocode.

    `lead_dropout_worst_drop` is the largest single-lead AUROC drop on the `"z/demodulated"` cell
    (`intact_macro_auroc - per_lead_macro_auroc`, maximised over the 12 leads) -- the phase-aware
    cell this project's own manuscript claims are about, not the theta-blind `"z/mean"` cell.

    Raises `ValueError` if an arm's checkpoint declares no transport operator: `gain_fraction`,
    `g1_pass`, and `lead_dropout_worst_drop` are all undefined without one.
    """
    cfg = probe_cfg or LinearProbeConfig(seed_probe=seed)
    columns = list(range(len(CLASSES)))
    table: dict[str, Any] = {}
    for name, ckpt_dir in arms.items():
        model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
        if operator is None:
            raise ValueError(
                f"arm {name!r} ({ckpt_dir}): no transport operator -- gain_fraction/g1_pass/"
                "lead_dropout_worst_drop are undefined without one"
            )
        try:
            z_by_split = {s: encode_z(model, wf, device) for s, wf in cohort.waveforms.items()}

            scores_full, _ev = fit_and_score(
                masked_mean_pool(z_by_split["train"], cohort.thetas["train"]).numpy(),
                cohort.labels["train"],
                masked_mean_pool(z_by_split["cal"], cohort.thetas["cal"]).numpy(),
                cohort.labels["cal"],
                masked_mean_pool(z_by_split["eval"], cohort.thetas["eval"]).numpy(),
                cohort.labels["eval"],
                CLASSES,
                cfg,
            )
            row = ci_row(
                scores_full,
                cohort.labels["eval"],
                cohort.patient_ids["eval"],
                columns,
                n_boot,
                seed,
            )

            g_lim = min(cohort.gain_limit, z_by_split["eval"].shape[0])
            op_cpu = operator.to("cpu")
            z_gain = z_by_split["eval"][:g_lim].float()
            th_gain = cohort.thetas["eval"][:g_lim].float()
            pid_gain = cohort.patient_ids["eval"][:g_lim]
            gain = gain_report(z_gain, th_gain, op_cpu, pid_gain, n_strata=n_strata, seed=seed)
            g1 = g1_shuffled_theta_gain_null(
                z_gain,
                th_gain,
                op_cpu,
                pid_gain,
                n_strata=n_strata,
                n_replicates=n_replicates,
                seed=seed,
            )

            robustness = robustness_suite(
                model,
                op_cpu,
                z_by_split,
                cohort.thetas,
                cohort.waveforms,
                cohort.labels,
                cohort.patient_ids["eval"],
                cohort.rr_median_ms,
                cohort.patch_width,
                cfg,
                device,
                seed=seed,
            )
            demod = robustness["lead_dropout"]["z/demodulated"]
            drops = [
                demod["intact_macro_auroc"] - lead["macro_auroc"] for lead in demod["per_lead"]
            ]
            worst_drop = float(max(drops)) if drops else float("nan")

            table[name] = {
                "macro_auroc": row["macro_auroc"],
                "lo": row["lo"],
                "hi": row["hi"],
                "gain_fraction": gain["overall_gain_fraction"],
                "g1_pass": g1["g1_pass"],
                "lead_dropout_worst_drop": worst_drop,
                "_scores": scores_full,
            }
        finally:
            del model, operator
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return table


def pairwise_deltas(
    table: dict[str, Any],
    y_eval: np.ndarray,
    patient_ids_eval: np.ndarray,
    *,
    reference: str,
    n_replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Every other arm's macro-AUROC delta vs `reference`.

    Uses `winder.eval.probe.paired_patient_bootstrap_delta` (shared-resample cancellation)
    wherever both arms carry `"_scores"` of identical shape (i.e. were scored on the same eval
    record set) -- plain CI-overlap between the two arms' own independently-bootstrapped
    intervals otherwise, which is weaker (no shared-resample cancellation) but still informative
    when the two arms' scores cannot be paired record-for-record.
    """
    if reference not in table:
        raise ValueError(f"reference arm {reference!r} not in table (arms: {sorted(table)})")
    ref_scores = table[reference].get("_scores")
    columns = list(range(y_eval.shape[1]))
    out: dict[str, Any] = {}
    for name, row in table.items():
        if name == reference:
            continue
        arm_scores = row.get("_scores")
        if (
            ref_scores is not None
            and arm_scores is not None
            and ref_scores.shape == arm_scores.shape
        ):
            both = np.isfinite(ref_scores).all(axis=1) & np.isfinite(arm_scores).all(axis=1)
            d, lo, hi = paired_patient_bootstrap_delta(
                y_eval[both][:, columns],
                ref_scores[both][:, columns],
                arm_scores[both][:, columns],
                patient_ids_eval[both],
                n_replicates=n_replicates,
                seed_probe=seed,
            )
            out[name] = {
                "vs": reference,
                "delta": d,
                "lo": lo,
                "hi": hi,
                "method": "paired_patient_bootstrap_delta",
                "n_paired": int(both.sum()),
            }
        else:
            a, b = table[reference], row
            overlap = not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
            out[name] = {
                "vs": reference,
                "delta": b["macro_auroc"] - a["macro_auroc"],
                "method": "ci_overlap",
                "cis_overlap": bool(overlap),
            }
    return out

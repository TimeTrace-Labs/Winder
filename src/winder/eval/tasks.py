"""The two PTB-XL classification tasks this pipeline probes -- 5-superclass and 23-subclass label
derivation, probe fit-and-score, and per-checkpoint-ladder selection. Promoted from script-local
functions into a real, importable, unit-tested library module.

**Source, and brief-vs-reality discrepancies worth recording up front.** The design brief that
commissioned this module attributed every function here to `scripts/p6_new_coprimary_readouts.py`.
Verified against the actual reference-repo source (`/home/blaised/winder-theory-exp`):
`MIN_POSITIVES`, `subclass_code_map`, `subclass_multihot`, `surviving_columns`, `fit_and_score`,
`macro_over` are indeed in `p6_new_coprimary_readouts.py` -- but `probe_point`, `ci_row`, and
`select_step` actually live in `scripts/scratch_finale_eval.py`. `superclass_multihot` does not
exist as a named function ANYWHERE in the reference repo: `p6_new_coprimary_readouts.py::main`
computes the 5-superclass multi-hot array inline as
`frame[list(MULTIHOT_COLS)].to_numpy(dtype=np.float32)`. It is defined here as a thin named
wrapper over that exact expression, for API symmetry with `subclass_multihot` -- nothing added or
changed, since promoting script logic into a library means every label-derivation path should be
independently callable rather than one being a named function and the other an inline slice.

**The load-bearing convention this module exists to preserve.** `fit_and_score` returns
FULL-LENGTH, NaN-padded score arrays: a record dropped during scoring (non-finite pooled feature)
stays at its ORIGINAL row index, holding NaN, rather than being compressed out of the array. The
reference repo shipped exactly the opposite bug once (commit `f8ce270`, "a row-alignment bug that
made a real accuracy cost read as a null") in `p1_panel_numerics.py::_fit_score` (this module's
own `fit_and_score` is `_fit_score`'s general-class-count sibling in `p6_new_coprimary_readouts.py`,
which was written correctly from the start -- see that commit's diff for the shape of the bug it
fixed elsewhere). See `tests/test_eval_tasks.py` for the regression guard and a worked
demonstration of why a compressed return is actively dangerous, not merely inconvenient: it
silently re-indexes every downstream consumer that pairs these scores against eval-fold metadata
(labels, patient ids, RR buckets) held in original record order, and for a PAIRED delta it shrinks
both arms toward chance TOGETHER -- a plausible-looking null, not an obviously wrong number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from winder.data.ptbxl import MULTIHOT_COLS, parse_scp_codes
from winder.eval.metrics import auroc_binary
from winder.eval.probe import (
    LinearProbeConfig,
    decision_scores,
    fit_linear_probe,
    patient_bootstrap_ci,
)

__all__ = [
    "CLASSES",
    "MIN_POSITIVES",
    "superclass_multihot",
    "subclass_code_map",
    "subclass_multihot",
    "surviving_columns",
    "fit_and_score",
    "macro_over",
    "probe_point",
    "ci_row",
    "select_step",
]

#: The 5 PTB-XL diagnostic superclasses, in `winder.data.ptbxl.MULTIHOT_COLS`'s own column order
#: (`sc_<name>` -> `<name>`) -- identical to `winder.data.ptbxl.SUPERCLASSES` by construction,
#: kept as this derivation (not a bare alias) to document that CLASSES *is* the multihot columns'
#: own label set, matching the reference repo's own definition in `p1_panel_numerics.py`.
CLASSES: tuple[str, ...] = tuple(s.removeprefix("sc_") for s in MULTIHOT_COLS)

#: Declared survivor rule for the 23-subclass task (Amendment 8, Eval 1): a class with STRICTLY
#: FEWER than this many eval-fold positives is excluded from the macro-average.
MIN_POSITIVES = 20


def superclass_multihot(frame: pd.DataFrame, classes: tuple[str, ...] = CLASSES) -> np.ndarray:
    """`(n_records, len(classes))` float32 multi-hot superclass labels, read directly from
    `frame`'s own `sc_<class>` columns (module docstring: a named wrapper over the reference
    repo's inline `frame[list(MULTIHOT_COLS)].to_numpy(dtype=np.float32)`, not a reimplementation
    of new logic)."""
    cols = [f"sc_{c}" for c in classes]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing superclass columns: {missing}")
    return frame[cols].to_numpy(dtype=np.float32)


def subclass_code_map(scp_statements: pd.DataFrame) -> dict[str, str]:
    """`{scp_code: diagnostic_subclass}` over the statements PTB-XL flags `diagnostic == 1`.

    The eligible-statement rule is `winder.data.ptbxl.assign_superclass`'s R1 with
    `diagnostic_subclass` read instead of `diagnostic_class`; this asserts the two columns are
    non-null on exactly the same codes, so the subclass task cannot silently score a different
    record pool than the superclass task it is a refinement of.
    """
    diag = scp_statements.loc[scp_statements["diagnostic"] == 1.0]
    if "diagnostic_subclass" not in diag.columns:
        raise ValueError("scp_statements is missing the 'diagnostic_subclass' column")
    with_class = set(diag.index[diag["diagnostic_class"].notna()].astype(str))
    with_subclass = set(diag.index[diag["diagnostic_subclass"].notna()].astype(str))
    if with_class != with_subclass:
        raise ValueError(
            "diagnostic_class and diagnostic_subclass are non-null on different code sets "
            f"(class-only={sorted(with_class - with_subclass)}, "
            f"subclass-only={sorted(with_subclass - with_class)}) -- the subclass task would "
            "score a different record pool than the superclass task"
        )
    return {
        str(code): str(row["diagnostic_subclass"]).strip()
        for code, row in diag.iterrows()
        if pd.notna(row["diagnostic_subclass"])
    }


def subclass_multihot(
    frame: pd.DataFrame, code_map: dict[str, str], classes: tuple[str, ...]
) -> np.ndarray:
    """`(n_records, len(classes))` float32 multi-hot subclass labels for `frame`, row-aligned.

    Threshold-free: bit `s` is set iff at least one of the record's asserted SCP codes maps to
    subclass `s`. A likelihood of 0.0 means asserted-but-not-quantified, so it sets the bit like
    any other assertion -- no likelihood cut-off is applied anywhere.
    """
    if "scp_codes" not in frame.columns:
        raise ValueError("frame must carry the raw 'scp_codes' column")
    index = {s: i for i, s in enumerate(classes)}
    out = np.zeros((len(frame), len(classes)), dtype=np.float32)
    for i, cell in enumerate(frame["scp_codes"].to_numpy()):
        for code in parse_scp_codes(cell):
            subclass = code_map.get(code)
            if subclass is not None:
                out[i, index[subclass]] = 1.0
    return out


def surviving_columns(y_eval: np.ndarray, min_positives: int) -> list[int]:
    """Column indices with at least `min_positives` positives -- the declared macro membership.

    Strictly fewer excludes; exactly `min_positives` survives (the inequality direction is
    load-bearing, not cosmetic: PTB-XL's LMI sits exactly on this boundary at 20 in fold 9).
    """
    counts = y_eval.sum(axis=0)
    return [c for c in range(y_eval.shape[1]) if int(counts[c]) >= min_positives]


def fit_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    classes: tuple[str, ...],
    cfg: LinearProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """`(scores_full, eval_mask)`: fit a linear probe on the finite-feature train/cal rows, score
    on the finite-feature eval rows, and return FULL-LENGTH scores with NaN at every dropped
    row's ORIGINAL index -- never compressed to the surviving rows (module docstring's
    load-bearing convention)."""
    tr, ca, ev = (np.isfinite(a).all(axis=1) for a in (x_train, x_cal, x_eval))
    probe = fit_linear_probe(x_train[tr], y_train[tr], x_cal[ca], y_cal[ca], cfg, classes=classes)
    scores = decision_scores(probe, x_eval[ev])
    scores_full = np.full((len(y_eval), scores.shape[1]), np.nan, dtype=float)
    scores_full[ev] = scores
    return scores_full, ev


def macro_over(y: np.ndarray, scores: np.ndarray, columns: list[int]) -> float:
    """Mean per-class AUROC over `columns` only -- the declared survivor-restricted macro."""
    per_class = np.array([auroc_binary(y[:, c], scores[:, c]) for c in columns])
    return float("nan") if np.isnan(per_class).all() else float(np.nanmean(per_class))


def probe_point(scores_full: np.ndarray, y_eval: np.ndarray, columns: list[int]) -> float:
    """Survivor-restricted macro-AUROC point estimate from full-length (NaN-padded) scores."""
    ev = np.isfinite(scores_full).all(axis=1)
    return macro_over(y_eval[ev], scores_full[ev], columns)


def ci_row(
    scores_full: np.ndarray,
    y_eval: np.ndarray,
    pid_eval: np.ndarray,
    columns: list[int],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Point + patient-clustered 95% CI on the survivor-restricted macro, rows NaN-masked."""
    ev = np.isfinite(scores_full).all(axis=1)
    point, lo, hi = patient_bootstrap_ci(
        y_eval[ev][:, columns],
        scores_full[ev][:, columns],
        pid_eval[ev],
        n_replicates=n_boot,
        seed_probe=seed,
    )
    return {
        "macro_auroc": point,
        "lo": lo,
        "hi": hi,
        "n_eval": int(ev.sum()),
        "n_dropped": int((~ev).sum()),
        "n_boot": int(n_boot),
    }


def select_step(curves_by_arm: dict[str, dict[int, float]]) -> dict[str, Any]:
    """One step per task: argmax of the across-arm MEAN curve over steps present (and finite) in
    EVERY arm; ties resolve to the earliest step."""
    if not curves_by_arm:
        raise ValueError("no curves to select from")
    matched = sorted(set.intersection(*(set(c.keys()) for c in curves_by_arm.values())))
    matched = [s for s in matched if all(np.isfinite(c[s]) for c in curves_by_arm.values())]
    if not matched:
        raise ValueError("no step is present (and finite) in every arm's curve")
    mean_curve = {s: float(np.mean([c[s] for c in curves_by_arm.values()])) for s in matched}
    best = max(mean_curve.values())
    selected = min(s for s, v in mean_curve.items() if v == best)  # tie -> earliest step
    return {"selected_step": int(selected), "matched_steps": matched, "mean_curve": mean_curve}

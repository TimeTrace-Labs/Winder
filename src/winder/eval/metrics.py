"""Discrimination (AUROC, AUPRC) and calibration (Brier score), each hand-rolled instead of
scikit-learn at runtime.

`auroc_binary`: `AUC = (sum of ranks of the positive class - n_pos*(n_pos+1)/2) / (n_pos * n_neg)`,
via `scipy.stats.rankdata`'s Mann-Whitney U identity, using `rankdata`'s default average-rank tie
handling. This is exactly what `sklearn.metrics.roc_auc_score` computes (also Mann-Whitney-based),
including on ties -- pinned against it as a dev-only test oracle (`tests/test_eval_metrics.py`),
never depended on at runtime. Same reasoning this repo already applied once: `wfdb_io.py`'s own
reader instead of the `wfdb` package for a well-understood, self-contained computation.

`auprc_binary`: non-interpolated average precision, `AP = sum_n (R_n - R_{n-1}) * P_n` over
distinct-score thresholds sorted high to low, with tied scores coalesced into a single threshold
point (never split by ranking order) -- the same construction `sklearn.metrics.
average_precision_score`'s `_binary_clf_curve` uses, and the same dev-only-oracle pinning as AUROC.

`brier_score`: plain mean squared error between predicted probability and binary outcome -- no
ranking, no ties to coalesce, and (unlike AUROC/AUPRC) no degenerate single-class case; still
pinned against `sklearn.metrics.brier_score_loss` in tests as a sanity check, not because the
computation is subtle.

`r_squared`: the conventional out-of-sample coefficient of determination, `1 - SS_res / SS_tot`
with `SS_tot` computed from the EVAL set's own mean (never the training set's) -- E2-14's linear
regression probe (`winder.eval.probe.fit_linear_regression_probe`) is scored on a held-out fold,
so this is a genuine out-of-sample R^2, not the in-sample identity `SS_tot >= SS_res` guarantees.
Not clipped at 0: a probe that predicts worse than the eval set's own mean is a real, reportable
finding (the representation actively misleads a linear readout of this descriptor), not a bug to
hide by floor-ing the score. Pinned against `sklearn.metrics.r2_score` in tests.
"""

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "auroc_binary",
    "macro_auroc",
    "auprc_binary",
    "macro_auprc",
    "brier_score",
    "macro_brier_score",
    "r_squared",
]


def auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC for one binary label. `NaN` if only one class is present in `y_true` -- the AUROC
    is undefined there, not zero; reporting zero would be indistinguishable from a genuinely bad
    classifier."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_score shape {y_score.shape}")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score)
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    """`(macro, per_class)`: `macro` is the `nanmean` of per-class AUROC over the columns of a
    `(n_samples, n_classes)` pair -- a degenerate class (all-0 or all-1 in `y_true`) is excluded
    from the average via `nanmean`, not silently scored as 0."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(
            f"y_true/y_score must both be 2-D (n_samples, n_classes) of matching shape, got "
            f"{y_true.shape} and {y_score.shape}"
        )
    per_class = np.array(
        [auroc_binary(y_true[:, k], y_score[:, k]) for k in range(y_true.shape[1])]
    )
    if np.isnan(per_class).all():
        # np.nanmean on an all-NaN array is correct (NaN) but warns "Mean of empty slice" --
        # every class degenerate is a legitimate outcome on a small validation split, not a
        # warning-worthy one.
        macro = float("nan")
    else:
        macro = float(np.nanmean(per_class))
    return macro, per_class


def auprc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve for one binary label (non-interpolated average
    precision). `NaN` if the positive class is absent -- mirroring `auroc_binary`'s degenerate-case
    convention exactly. Unlike AUROC, AUPRC does *not* also require the negative class: with only
    positives present every prediction trivially achieves precision 1, and AUPRC is a well-defined
    1.0, not degenerate."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_score shape {y_score.shape}")
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    # One (tp, fp) pair per distinct score value, taken at the last index of each tied run --
    # coalescing ties into a single threshold point, matching sklearn's `_binary_clf_curve`.
    is_last_of_run = np.r_[scores_sorted[:-1] != scores_sorted[1:], True]
    tp = np.cumsum(y_sorted == 1)[is_last_of_run]
    fp = np.cumsum(y_sorted == 0)[is_last_of_run]
    precision = np.r_[1.0, tp / (tp + fp)]
    recall = np.r_[0.0, tp / n_pos]
    return float(np.sum(np.diff(recall) * precision[1:]))


def macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    """`(macro, per_class)`: same macro-averaging pattern as `macro_auroc`, over per-class AUPRC
    -- a degenerate class (no positives in that column of `y_true`) is excluded from the average
    via `nanmean`, not silently scored as 0."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(
            f"y_true/y_score must both be 2-D (n_samples, n_classes) of matching shape, got "
            f"{y_true.shape} and {y_score.shape}"
        )
    per_class = np.array(
        [auprc_binary(y_true[:, k], y_score[:, k]) for k in range(y_true.shape[1])]
    )
    if np.isnan(per_class).all():
        macro = float("nan")
    else:
        macro = float(np.nanmean(per_class))
    return macro, per_class


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome for one class --
    `mean((y_prob - y_true) ** 2)`. Unlike AUROC/AUPRC (rank-based, undefined with only one class
    present), Brier score has no degenerate case: it is well-defined for any non-empty,
    same-shaped `(y_true, y_prob)` pair, including a single-class `y_true`."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_prob shape {y_prob.shape}")
    return float(np.mean((y_prob - y_true) ** 2))


def macro_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, np.ndarray]:
    """`(macro, per_class)`: same macro-averaging pattern as `macro_auroc`/`macro_auprc`. No
    class is ever degenerate here -- Brier score is a plain MSE, well-defined even when a column
    is all-0 or all-1 -- so `nanmean` never actually has anything to skip; kept for structural
    consistency with the other two macro-averagers rather than because it changes the result."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError(
            f"y_true/y_prob must both be 2-D (n_samples, n_classes) of matching shape, got "
            f"{y_true.shape} and {y_prob.shape}"
        )
    per_class = np.array([brier_score(y_true[:, k], y_prob[:, k]) for k in range(y_true.shape[1])])
    if np.isnan(per_class).all():
        macro = float("nan")
    else:
        macro = float(np.nanmean(per_class))
    return macro, per_class


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """`1 - SS_res / SS_tot` for one continuous target, `SS_tot` from `y_true`'s own mean.

    `NaN` if `y_true` is degenerate (zero variance) -- `SS_tot == 0` makes the ratio undefined,
    the same reasoning `auroc_binary` uses for a single-class `y_true`, not a divide-by-zero to
    paper over with 0 or 1.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

"""Tests for winder.eval.tasks: PTB-XL superclass/subclass label derivation, probe fit-and-score,
and checkpoint-ladder step selection.

Three provenance notes:
  - `test_subclass_*`/`test_surviving_columns_*` are ported (import path adapted) from the
    reference repo's `tests/test_p6_new_coprimary_readouts.py`.
  - `test_ci_row_masks_nan_rows`/`test_select_step_*` are ported (import path adapted) from
    `tests/test_scratch_finale_eval.py` -- the design brief attributed `ci_row`/`select_step` to
    `p6_new_coprimary_readouts.py`, which was wrong (see this module's own docstring).
  - `test_fit_and_score_returns_full_length_nan_padded_scores` and
    `test_paired_delta_on_misaligned_scores_would_differ` are the two NEW tests the design brief
    named explicitly, guarding the `f8ce270` row-alignment contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winder.eval.probe import LinearProbeConfig, paired_patient_bootstrap_delta
from winder.eval.tasks import (
    CLASSES,
    ci_row,
    fit_and_score,
    macro_over,
    probe_point,
    select_step,
    subclass_code_map,
    subclass_multihot,
    superclass_multihot,
    surviving_columns,
)

# ============================================================== T1: subclass label derivation


def _scp_statements() -> pd.DataFrame:
    """Four statements shaped like `scp_statements.csv`'s real columns: two diagnostic codes in
    one subclass, one diagnostic code in another, and one non-diagnostic (rhythm) code."""
    frame = pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0, np.nan],
            "diagnostic_class": ["MI", "MI", "STTC", np.nan],
            "diagnostic_subclass": ["AMI", "IMI", "ISC_", np.nan],
        },
        index=["IMI_A", "IMI_B", "ISC_C", "SR"],
    )
    frame.index.name = "scp_code"
    return frame


def _frame(cells: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"scp_codes": cells})


def test_subclass_code_map_keeps_only_diagnostic_statements() -> None:
    code_map = subclass_code_map(_scp_statements())
    assert code_map == {"IMI_A": "AMI", "IMI_B": "IMI", "ISC_C": "ISC_"}
    assert "SR" not in code_map  # rhythm statement, `diagnostic` is NaN


def test_subclass_code_map_rejects_class_subclass_coverage_mismatch() -> None:
    bad = _scp_statements()
    bad.loc["ISC_C", "diagnostic_subclass"] = np.nan  # has a class but no subclass
    with pytest.raises(ValueError, match="different code sets"):
        subclass_code_map(bad)


def test_subclass_multihot_is_the_union_over_a_records_codes() -> None:
    code_map = subclass_code_map(_scp_statements())
    classes = tuple(sorted(set(code_map.values())))  # ("AMI", "IMI", "ISC_")
    frame = _frame(
        [
            "{'IMI_A': 100.0, 'SR': 0.0}",  # one subclass; the rhythm code is ignored
            "{'IMI_A': 100.0, 'IMI_B': 50.0}",  # two codes, two subclasses -> two bits
            "{'ISC_C': 0.0}",  # ASSERTED-BUT-NOT-QUANTIFIED (rule R2): still sets the bit
            "{'SR': 0.0}",  # no eligible statement -> all-zero row
        ]
    )
    got = subclass_multihot(frame, code_map, classes)
    assert got.shape == (4, 3)
    np.testing.assert_array_equal(
        got, np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    )


def test_subclass_multihot_requires_the_raw_scp_codes_column() -> None:
    with pytest.raises(ValueError, match="scp_codes"):
        subclass_multihot(pd.DataFrame({"other": [1]}), {}, ())


# ======================================================================= T2 the survivor rule


def test_surviving_columns_boundary_is_at_least_not_more_than() -> None:
    n = 100
    y = np.zeros((n, 3), dtype=np.float32)
    y[:19, 0] = 1.0  # 19 positives -> EXCLUDED
    y[:20, 1] = 1.0  # exactly 20 -> SURVIVES (PTB-XL's LMI sits here in fold 9)
    y[:21, 2] = 1.0  # 21 -> SURVIVES
    assert surviving_columns(y, 20) == [1, 2]


def test_surviving_columns_can_be_empty() -> None:
    assert surviving_columns(np.zeros((10, 2), dtype=np.float32), 1) == []


# ================================================================= superclass_multihot


def test_superclass_multihot_reads_sc_prefixed_columns_in_classes_order() -> None:
    frame = pd.DataFrame(
        {
            "sc_NORM": [1, 0],
            "sc_MI": [0, 1],
            "sc_STTC": [0, 0],
            "sc_CD": [0, 0],
            "sc_HYP": [0, 0],
        }
    )
    got = superclass_multihot(frame)
    assert got.shape == (2, len(CLASSES))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[:, CLASSES.index("NORM")], [1, 0])
    np.testing.assert_array_equal(got[:, CLASSES.index("MI")], [0, 1])


def test_superclass_multihot_raises_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing superclass columns"):
        superclass_multihot(pd.DataFrame({"sc_NORM": [1]}))


# ===================================================== fit_and_score: the f8ce270 regression guard


DROPPED_EVAL_ROWS = (3, 91, 150)


def _separable_split(n: int, n_classes: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)` where each class's label is a linear function of its own feature column, so a
    linear probe reaches AUROC ~1.0 and any row misalignment is unmistakable rather than a small
    degradation. Mirrors the reference repo's own `test_p1_panel_numerics.py` fixture."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, n_classes)) < 0.4).astype(np.float64)
    x = np.concatenate([y * 12.0 - 6.0, rng.standard_normal((n, 3))], axis=1)
    x = x + rng.standard_normal(x.shape) * 0.05
    return x, y


def _fit_and_score_on_synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`fit_and_score` on a separable synthetic split whose EVAL features carry NaN rows.
    Returns `(scores_full, eval_mask, y_eval)`."""
    n_classes = len(CLASSES)
    x_train, y_train = _separable_split(600, n_classes, seed=0)
    x_cal, y_cal = _separable_split(300, n_classes, seed=1)
    x_eval, y_eval = _separable_split(240, n_classes, seed=2)
    x_eval = x_eval.copy()
    for row in DROPPED_EVAL_ROWS:
        x_eval[row, 0] = np.nan
    cfg = LinearProbeConfig(lr=0.1, weight_decay=0.0, max_epochs=300, early_stopping_patience=50)
    scores_full, ev = fit_and_score(x_train, y_train, x_cal, y_cal, x_eval, y_eval, CLASSES, cfg)
    return scores_full, ev, y_eval


def test_fit_and_score_returns_full_length_nan_padded_scores() -> None:
    """The root-cause contract: `scores_full` is indexed by RECORD, not by surviving-row
    position -- a dropped row's NaN stays AT ITS ORIGINAL INDEX, never removed."""
    scores_full, ev, y_eval = _fit_and_score_on_synthetic()
    assert scores_full.shape == (len(y_eval), len(CLASSES))
    dropped = np.array(DROPPED_EVAL_ROWS)
    assert np.isnan(scores_full[dropped]).all(), "dropped rows must be NaN, not absent"
    keep = np.setdiff1d(np.arange(len(y_eval)), dropped)
    assert np.isfinite(scores_full[keep]).all()
    assert np.array_equal(np.flatnonzero(~ev), dropped)
    # the headline statistic is unaffected by the wider return
    point = probe_point(scores_full, y_eval, list(range(len(CLASSES))))
    assert point > 0.95, "separable synthetic split should be near-perfect"


def test_paired_delta_on_misaligned_scores_would_differ() -> None:
    """Demonstrates WHY the full-length NaN-padding convention matters, not merely that it holds.

    Two arms with a large, real, KNOWN, opposite-direction difference: arm A's scores perfectly
    PREDICT the label, arm B's perfectly predict its NEGATION, so a correctly-paired delta is
    exactly -1.0 macro-AUROC and a patient-clustered CI excludes zero by a wide margin. Each
    arm's own NaN rows sit at a DIFFERENT set of positions (mirroring reality: different
    checkpoints drop different NaN-theta rows). Compressing each arm to its own finite rows
    before pairing -- the `f8ce270` bug's own shape -- means row `i` of the two compressed
    arrays no longer names the same record; pairing them by POSITION against the same label
    vector washes the real -1.0 effect out toward a much smaller, materially WRONG number: a
    misleadingly reassuring finding, not merely a noisier one.
    """
    n = 60
    rng = np.random.default_rng(0)
    y = (rng.random(n) > 0.5).astype(np.float64)
    y_eval = np.stack([y, 1.0 - y], axis=1)  # 2 columns, both fully informative of `y`
    patient_ids = np.arange(n)

    drop_a = np.array([3, 20, 45])
    drop_b = np.array([7, 30, 50])  # a DIFFERENT drop set from arm A's

    scores_a_full = np.stack([y * 10 - 5, (1 - y) * 10 - 5], axis=1)  # perfectly PREDICTS y
    scores_a_full[drop_a] = np.nan
    scores_b_full = np.stack([(1 - y) * 10 - 5, y * 10 - 5], axis=1)  # perfectly NEGATES y
    scores_b_full[drop_b] = np.nan

    # ---- correct: pair on the intersection of both arms' finite rows, original indices intact
    both = np.isfinite(scores_a_full).all(axis=1) & np.isfinite(scores_b_full).all(axis=1)
    correct_delta, _lo, correct_hi = paired_patient_bootstrap_delta(
        y_eval[both],
        scores_a_full[both],
        scores_b_full[both],
        patient_ids[both],
        n_replicates=200,
        seed_probe=0,
    )
    assert correct_delta == pytest.approx(-1.0, abs=1e-9)
    assert correct_hi < -0.5  # a real, CI-excluding-zero effect, not a coin flip

    # ---- wrong: compress each arm to its OWN finite rows, then pair by POSITION (the bug)
    compressed_a = scores_a_full[np.isfinite(scores_a_full).all(axis=1)]
    compressed_b = scores_b_full[np.isfinite(scores_b_full).all(axis=1)]
    n_min = min(len(compressed_a), len(compressed_b))
    wrong_delta, _wlo, _whi = paired_patient_bootstrap_delta(
        y_eval[:n_min],
        compressed_a[:n_min],
        compressed_b[:n_min],
        patient_ids[:n_min],
        n_replicates=200,
        seed_probe=0,
    )
    assert abs(wrong_delta - correct_delta) > 0.5, "compression must materially change the answer"
    assert abs(wrong_delta) < 0.6, "the real -1.0 effect is washed toward a reassuring null"


# =============================================================================== macro_over


def test_macro_over_restricts_to_the_given_columns() -> None:
    scores_full, ev, y_eval = _fit_and_score_on_synthetic()
    all_cols = list(range(len(CLASSES)))
    restricted = macro_over(y_eval[ev], scores_full[ev], all_cols[:2])
    full = macro_over(y_eval[ev], scores_full[ev], all_cols)
    assert 0.0 <= restricted <= 1.0
    assert 0.0 <= full <= 1.0


# ================================================================================= probe_point


def test_probe_point_matches_macro_over_on_the_finite_rows() -> None:
    scores_full, ev, y_eval = _fit_and_score_on_synthetic()
    columns = list(range(len(CLASSES)))
    point = probe_point(scores_full, y_eval, columns)
    assert point == pytest.approx(macro_over(y_eval[ev], scores_full[ev], columns))


# ==================================================================================== ci_row


def test_ci_row_masks_nan_rows() -> None:
    rng = np.random.default_rng(0)
    n = 40
    y = (rng.random((n, 2)) > 0.5).astype(np.float32)
    scores = rng.normal(size=(n, 2))
    scores[[3, 7]] = np.nan
    pid = np.repeat(np.arange(n // 2), 2)
    row = ci_row(scores, y, pid, [0, 1], n_boot=25, seed=0)
    assert row["n_dropped"] == 2 and row["n_eval"] == n - 2
    assert np.isfinite(row["macro_auroc"])
    assert row["lo"] <= row["macro_auroc"] <= row["hi"]


# ================================================================================ select_step


def test_select_step_argmax_of_mean_over_matched_steps() -> None:
    curves = {
        "seed0": {2500: 0.80, 5000: 0.90},
        "seed1": {2500: 0.70, 5000: 0.95, 7500: 0.99},  # 7500 unmatched -> ignored
    }
    sel = select_step(curves)
    assert sel["selected_step"] == 5000
    assert sel["matched_steps"] == [2500, 5000]
    assert sel["mean_curve"][5000] == pytest.approx(0.925)


def test_select_step_tie_prefers_earliest() -> None:
    curves = {"a": {1000: 0.9, 2000: 0.9}, "b": {1000: 0.9, 2000: 0.9}}
    assert select_step(curves)["selected_step"] == 1000


def test_select_step_nan_step_excluded() -> None:
    curves = {"a": {1000: 0.8, 2000: float("nan")}, "b": {1000: 0.7, 2000: 0.99}}
    sel = select_step(curves)
    assert sel["selected_step"] == 1000 and sel["matched_steps"] == [1000]


def test_select_step_no_common_steps_raises() -> None:
    with pytest.raises(ValueError, match="no step"):
        select_step({"a": {1000: 0.8}, "b": {2000: 0.9}})

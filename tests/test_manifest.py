"""manifest.py tests.

Adapted from ttl-phase's test_pipeline_contracts.py Section B, plus new tests for the two
bugs this port fixes (#4: RR_OUTLIERS as its own reason code; #9: quality_flags rejects a
pre-joined string instead of silently iterating it into single characters) and for the
pydantic-specific behaviour (frozen, extra="forbid", validators fire on every construction
path including from_parquet).
"""

from pathlib import Path

import pandas as pd
import pydantic
import pytest

from winder.data.manifest import NO_REASON, REASON_CODES, Manifest, RecordRow, multihot


# --------------------------------------------------------------------------- multihot
def test_multihot_over_superclasses() -> None:
    assert multihot(["MI", "STTC"]) == (0, 1, 1, 0, 0)


def test_multihot_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        multihot(["NOT_A_CLASS"])


# ----------------------------------------------------------------------- RecordRow itself
def test_record_row_is_frozen() -> None:
    row = RecordRow(ecg_id=1, status="included")
    with pytest.raises(pydantic.ValidationError):
        row.ecg_id = 2  # type: ignore[misc]


def test_record_row_rejects_unknown_field_name() -> None:
    """A typo'd field name must not silently create a column that's never written."""
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", n_beetz=3)  # type: ignore[call-arg]


def test_record_row_rejects_excluded_without_legal_reason_code() -> None:
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="excluded")  # NO_REASON is not a legal exclusion reason
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="excluded", reason_code="NOT_A_REAL_CODE")


def test_record_row_rejects_included_with_a_reason_code() -> None:
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", reason_code="TOO_FEW_BEATS")


def test_record_row_accepts_a_legal_excluded_row() -> None:
    row = RecordRow(ecg_id=1, status="excluded", reason_code="TOO_FEW_BEATS")
    assert row.reason_code == "TOO_FEW_BEATS"


def test_record_row_rejects_bad_status_literal() -> None:
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="maybe")  # type: ignore[arg-type]


# ------------------------------------------------------------------------------- bug #9
def test_record_row_rejects_preformatted_quality_flags_string() -> None:
    """Bug #9: ttl-phase's Manifest.add() iterated a pre-joined ';'.join(flags) string
    character-by-character. RecordRow must reject a bare str outright instead."""
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", quality_flags="PHASE_RR_OUTLIERS")  # type: ignore[arg-type]


def test_record_row_accepts_a_real_flags_list() -> None:
    flags = ["PHASE_LOW_YIELD", "PHASE_RR_OUTLIERS"]
    row = RecordRow(ecg_id=1, status="included", quality_flags=flags)  # type: ignore[arg-type]
    assert row.quality_flags == ("PHASE_LOW_YIELD", "PHASE_RR_OUTLIERS")
    assert row.to_dict()["quality_flags"] == "PHASE_LOW_YIELD;PHASE_RR_OUTLIERS"


def test_record_row_rejects_bytes_quality_flags_too() -> None:
    """Audit-found: bytes/bytearray reproduce bug #9's exact failure mode (Python
    iterates them into per-byte ints) with no exception at all -- the original fix only
    checked isinstance(v, str)."""
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", quality_flags=b"AB")  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", quality_flags=bytearray(b"AB"))  # type: ignore[arg-type]


# ------------------------------------------------------------------------------- bug #4
def test_rr_outliers_is_its_own_reason_code_distinct_from_implausible_rr() -> None:
    assert "RR_OUTLIERS" in REASON_CODES
    assert "IMPLAUSIBLE_RR" in REASON_CODES
    assert "RR_OUTLIERS" != "IMPLAUSIBLE_RR"
    row = RecordRow(ecg_id=1, status="excluded", reason_code="RR_OUTLIERS")
    assert row.reason_code == "RR_OUTLIERS"


def test_flat_signal_and_low_confidence_are_reason_codes() -> None:
    """The other half of bug #7: phase.py can emit PHASE_FLAT_SIGNAL/PHASE_LOW_CONFIDENCE,
    and both need somewhere to go in the manifest vocabulary."""
    assert "FLAT_SIGNAL" in REASON_CODES
    assert "LOW_CONFIDENCE" in REASON_CODES


# ---------------------------------------------------------------- audit-found regressions
def test_manifest_add_normalizes_none_reason_code() -> None:
    """ttl-phase's Manifest.add() explicitly normalised reason_code=None to NO_REASON;
    the port's first pass dropped that, so an explicit None started raising instead."""
    man = Manifest()
    row = man.add_included(1, reason_code=None)
    assert row.reason_code == NO_REASON


@pytest.mark.parametrize("bad", [None, 5, 5.0])
def test_record_row_superclasses_non_iterable_input_raises_validation_error(bad: object) -> None:
    """The mode='before' coercion's bare TypeError on a non-iterable input used to escape
    pydantic entirely instead of surfacing as ValidationError like every other bad input
    to this model -- breaking the uniform pydantic-boundary error contract."""
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", superclasses=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, 5])
def test_record_row_quality_flags_non_iterable_input_raises_validation_error(bad: object) -> None:
    with pytest.raises(pydantic.ValidationError):
        RecordRow(ecg_id=1, status="included", quality_flags=bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- Manifest
def test_manifest_rejects_unlisted_reason_codes() -> None:
    man = Manifest()
    with pytest.raises(pydantic.ValidationError):
        man.add_excluded(1, reason_code="NOT_A_REAL_CODE")


def test_manifest_rejects_duplicate_ecg_id() -> None:
    man = Manifest()
    man.add_included(1)
    with pytest.raises(ValueError, match="duplicate ecg_id"):
        man.add_included(1)


def test_manifest_status_reason_consistency() -> None:
    man = Manifest()
    with pytest.raises(pydantic.ValidationError):
        man.add_included(1, reason_code="TOO_FEW_BEATS")
    with pytest.raises(pydantic.ValidationError):
        man.add_excluded(2, reason_code=NO_REASON)


def test_assert_accounts_for_fires_on_a_lost_record() -> None:
    man = Manifest()
    man.add_included(1)
    man.add_excluded(2, reason_code="TOO_FEW_BEATS")
    with pytest.raises(AssertionError):
        man.assert_accounts_for(3)  # a third record was never logged
    man.assert_accounts_for(2)  # must not raise


def test_summary_by_superclass() -> None:
    man = Manifest()
    man.add_included(1, superclass="NORM")
    man.add_included(2, superclass="NORM")
    man.add_excluded(3, reason_code="TOO_FEW_BEATS", superclass="MI")
    s = man.summary(by="superclass")
    assert s.loc["NORM", "n_included"] == 2
    assert s.loc["MI", "n_excluded"] == 1
    assert s.loc["TOTAL", "n_total"] == 3


def test_summary_by_reason_code() -> None:
    man = Manifest()
    man.add_included(1)
    man.add_excluded(2, reason_code="TOO_FEW_BEATS")
    man.add_excluded(3, reason_code="TOO_FEW_BEATS")
    r = man.reason_table()
    assert r.loc["TOO_FEW_BEATS", "n_excluded"] == 2
    assert r.loc["HIGH_RR_CV", "n_excluded"] == 0  # a code that never fired is still present


def test_multilabel_summary_counts_a_record_in_every_superclass() -> None:
    man = Manifest()
    man.add_included(1, superclasses=(0, 1, 1, 0, 0))  # MI and STTC both
    s = man.summary_by_superclass_multilabel()
    assert s.loc["MI", "n_included"] == 1
    assert s.loc["STTC", "n_included"] == 1


def test_empty_manifest_has_the_full_schema() -> None:
    man = Manifest()
    df = man.to_dataframe()
    assert len(df) == 0
    from winder.data.manifest import COLUMNS

    assert list(df.columns) == list(COLUMNS)


def test_manifest_parquet_round_trip_is_exact(tmp_path: Path) -> None:
    man = Manifest()
    man.add_included(
        1,
        superclass="NORM",
        superclasses=(1, 0, 0, 0, 0),
        age=45.0,
        sex=0.0,
        n_beats=12,
        phase_yield=0.98,
        rr_mean_ms=850.0,
        rr_median_ms=848.0,
        rr_sd_ms=12.0,
        rr_cv=0.014,
        jitter_ms=0.3,
        quality_flags=["PHASE_LOW_YIELD"],
    )
    man.add_excluded(2, reason_code="TOO_FEW_BEATS", reason_detail="n_beats=2 < 5")
    path = str(tmp_path / "manifest.parquet")
    man.to_parquet(path)
    reloaded = Manifest.from_parquet(path)
    assert reloaded.n_included == man.n_included
    assert reloaded.n_excluded == man.n_excluded
    pd_orig = man.to_dataframe()
    pd_new = reloaded.to_dataframe()
    assert pd_orig.equals(pd_new)


def test_from_parquet_rejects_a_hand_edited_unlisted_reason_code(tmp_path: Path) -> None:
    man = Manifest()
    man.add_excluded(1, reason_code="TOO_FEW_BEATS")
    path = str(tmp_path / "manifest.parquet")
    man.to_parquet(path)

    df = pd.read_parquet(path)
    df.loc[0, "reason_code"] = "SOMETHING_MADE_UP"
    df.to_parquet(path, index=False)
    with pytest.raises(pydantic.ValidationError):
        Manifest.from_parquet(path)


def test_from_parquet_treats_null_string_cells_as_empty_not_the_literal_nan(
    tmp_path: Path,
) -> None:
    """Audit-found bug: from_parquet did a bare str(cell), so a NULL/NaN value in a
    string column (a hand-edited or version-drifted parquet -- exactly this method's own
    stated use case) round-tripped into the fabricated literal string "nan" instead of
    this module's documented empty-string-for-missing convention. For quality_flags this
    fabricated a bogus flag tag that RecordRow accepted as legitimate data."""
    man = Manifest()
    man.add_included(1, quality_flags=["PHASE_LOW_YIELD"], device="Boston", site="A")
    path = str(tmp_path / "manifest.parquet")
    man.to_parquet(path)

    df = pd.read_parquet(path)
    df.loc[0, "quality_flags"] = None
    df.loc[0, "device"] = None
    df.loc[0, "site"] = None
    df.loc[0, "reason_detail"] = None
    df.to_parquet(path, index=False)

    reloaded = Manifest.from_parquet(path)
    row = reloaded.to_dataframe().iloc[0]
    assert row["quality_flags"] == ""
    assert row["device"] == ""
    assert row["site"] == ""
    assert row["reason_detail"] == ""

"""ptbxl.py tests: SCP-code parsing and superclass assignment (R1-R6), plus a real-corpus
regression check against counts verified directly against ttl-phase's own pandas 2.3.3
output before pandas 3.0 was adopted here (see PR4's commit message)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from winder.data.decimation import decimate_to
from winder.data.ptbxl import (
    LEAD_ORDER,
    SUPERCLASSES,
    UNLABELED,
    assign_superclass,
    load_metadata,
    parse_scp_codes,
    read_and_decimate_500hz,
)
from winder.data.wfdb_io import read_record, write_format16
from winder.paths import default_data_root

# ttl-phase's data checkout -- not part of winder, may not exist on every machine.
PTBXL_ROOT = default_data_root()
_HAS_REAL_CORPUS = os.path.isfile(os.path.join(PTBXL_ROOT, "ptbxl_database.csv"))


# ----------------------------------------------------------------------- parse_scp_codes
def test_parse_scp_codes_dict_literal() -> None:
    assert parse_scp_codes("{'NORM': 100.0, 'LVOLT': 0.0}") == {"NORM": 100.0, "LVOLT": 0.0}


def test_parse_scp_codes_handles_empty_and_nan() -> None:
    assert parse_scp_codes("{}") == {}
    assert parse_scp_codes(None) == {}
    assert parse_scp_codes(float("nan")) == {}
    assert parse_scp_codes("nan") == {}


def test_parse_scp_codes_accepts_a_real_dict() -> None:
    assert parse_scp_codes({"SR": 0.0}) == {"SR": 0.0}


def test_parse_scp_codes_rejects_non_dict_literal() -> None:
    with pytest.raises(ValueError):
        parse_scp_codes("[1, 2, 3]")
    with pytest.raises((SyntaxError, ValueError)):  # ast.literal_eval's own errors on garbage
        parse_scp_codes("not a literal at all {")


# ------------------------------------------------------------------------ assign_superclass
def _scp_statements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, float("nan")],
            "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", "MI", None],
        },
        index=["NORM", "IMI", "NST_", "LAFB", "LVH", "FORM_ONLY", "RHYTHM_ONLY"],
    )


def test_assign_superclass_multihot_and_dominant_label() -> None:
    df = pd.DataFrame({"scp_codes": ["{'NORM': 100.0}", "{'IMI': 80.0, 'NST_': 20.0}"]})
    out = assign_superclass(df, _scp_statements())
    assert out.loc[0, "superclass"] == "NORM"
    assert out.loc[0, "sc_NORM"] == 1
    assert out.loc[1, "superclass"] == "MI"  # 80 > 20
    assert out.loc[1, "sc_MI"] == 1 and out.loc[1, "sc_STTC"] == 1


def test_assign_superclass_r1_ignores_non_diagnostic_and_form_rhythm_statements() -> None:
    # FORM_ONLY (diagnostic=0.0) and RHYTHM_ONLY (diagnostic=NaN) must not contribute.
    df = pd.DataFrame({"scp_codes": ["{'FORM_ONLY': 100.0, 'RHYTHM_ONLY': 100.0}"]})
    out = assign_superclass(df, _scp_statements())
    assert out.loc[0, "superclass"] == UNLABELED
    assert out.loc[0, [f"sc_{s}" for s in SUPERCLASSES]].sum() == 0


def test_assign_superclass_r2_zero_likelihood_is_asserted_not_absent() -> None:
    df = pd.DataFrame({"scp_codes": ["{'NORM': 0.0}"]})
    out = assign_superclass(df, _scp_statements(), zero_likelihood_as=100.0)
    assert out.loc[0, "sc_NORM"] == 1  # bit set despite likelihood 0.0
    assert out.loc[0, "scw_NORM"] == pytest.approx(100.0)  # credited at zero_likelihood_as


def test_assign_superclass_r5_ties_broken_by_dominance_order() -> None:
    # NORM and MI tie at 50/50 -- MI must win under the default dominance_order.
    df = pd.DataFrame({"scp_codes": ["{'NORM': 50.0, 'IMI': 50.0}"]})
    out = assign_superclass(df, _scp_statements())
    assert out.loc[0, "superclass"] == "MI"


def test_assign_superclass_r6_unlabeled_when_no_eligible_statement() -> None:
    df = pd.DataFrame({"scp_codes": ["{}"]})
    out = assign_superclass(df, _scp_statements())
    assert out.loc[0, "superclass"] == UNLABELED
    assert out.loc[0, "n_superclass"] == 0


def test_assign_superclass_rejects_bad_dominance_order() -> None:
    df = pd.DataFrame({"scp_codes": ["{}"]})
    with pytest.raises(ValueError):
        assign_superclass(df, _scp_statements(), dominance_order=("NORM", "MI"))  # too short


def test_assign_superclass_rejects_missing_scp_codes_column() -> None:
    with pytest.raises(ValueError):
        assign_superclass(pd.DataFrame({"x": [1]}), _scp_statements())


# ------------------------------------------------------------- load_metadata validation
# Audit-found: load_metadata's own input-validation branches were only ever exercised via
# the real-corpus happy path, never with synthetic malformed CSVs -- so a regression in
# any of these checks would only surface (if at all) against real data, not in CI.
def _minimal_row(ecg_id: int = 1, patient_id: int = 1, strat_fold: int = 1) -> dict:
    return {
        "ecg_id": ecg_id,
        "patient_id": patient_id,
        "strat_fold": strat_fold,
        "age": 50.0,
        "sex": 0,
        "height": None,
        "weight": None,
        "device": "CS-12",
        "site": 1,
        "recording_date": "2000-01-01 00:00:00",
        "filename_lr": "records100/00000/00001_lr",
        "filename_hr": "records500/00000/00001_hr",
        "scp_codes": "{'NORM': 100.0}",
    }


def _write_metadata_csv(tmp_path: Path, rows: list[dict], drop_cols: tuple[str, ...] = ()) -> str:
    from winder.data.ptbxl import KEEP_COLS

    cols = [c for c in KEEP_COLS if c not in drop_cols]
    path = os.path.join(str(tmp_path), "ptbxl_database.csv")
    pd.DataFrame(rows)[cols].to_csv(path, index=False)
    return str(tmp_path)


def test_load_metadata_rejects_duplicate_ecg_id(tmp_path: Path) -> None:
    root = _write_metadata_csv(tmp_path, [_minimal_row(1, 1, 1), _minimal_row(1, 2, 2)])
    with pytest.raises(ValueError, match="duplicate ecg_id"):
        load_metadata(root)


def test_load_metadata_rejects_null_patient_id(tmp_path: Path) -> None:
    root = _write_metadata_csv(tmp_path, [_minimal_row(1, 1, 1)])
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    df.loc[0, "patient_id"] = None
    df.to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)
    with pytest.raises(ValueError, match="null ecg_id or patient_id"):
        load_metadata(root)


def test_load_metadata_rejects_out_of_range_strat_fold(tmp_path: Path) -> None:
    root = _write_metadata_csv(tmp_path, [_minimal_row(1, 1, 11)])
    with pytest.raises(ValueError, match="strat_fold outside 1..10"):
        load_metadata(root)


def test_load_metadata_rejects_missing_required_column(tmp_path: Path) -> None:
    root = _write_metadata_csv(tmp_path, [_minimal_row(1, 1, 1)], drop_cols=("device",))
    with pytest.raises(ValueError, match="missing expected columns"):
        load_metadata(root)


# --------------------------------------------------------------------- read_and_decimate_500hz
def _real_500hz_signal() -> np.ndarray:
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "wfdb")
    hea_path = sorted(
        os.path.join(fixtures_dir, f) for f in os.listdir(fixtures_dir) if f.endswith(".hea")
    )[0]
    sig, header = read_record(hea_path)
    assert header.fs == 500 and sig.shape == (5000, 12)
    return sig


def _write_records500_fixture(tmp_path: Path, stem_rel: str, sig: np.ndarray, fs: float) -> str:
    stem = tmp_path / stem_rel
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_format16(str(stem), sig.astype(np.float64), fs=fs, sig_name=list(LEAD_ORDER))
    return str(stem) + ".hea"


def test_read_and_decimate_500hz_matches_decimate_to_directly(tmp_path: Path) -> None:
    sig500 = _real_500hz_signal()
    hea_path = _write_records500_fixture(tmp_path, "records500/00000/00001_hr", sig500, fs=500)

    sig = read_and_decimate_500hz(hea_path, expected_sig_name=LEAD_ORDER)
    assert sig.shape == (1000, 12)
    assert np.allclose(sig, decimate_to(sig500, 500, 100))


def test_read_and_decimate_500hz_rejects_wrong_source_fs(tmp_path: Path) -> None:
    hea_path = _write_records500_fixture(
        tmp_path, "records500/00000/00002_hr", np.zeros((5000, 12)), fs=250
    )
    with pytest.raises(ValueError, match="expected 500"):
        read_and_decimate_500hz(hea_path, expected_sig_name=LEAD_ORDER)


def test_read_and_decimate_500hz_rejects_wrong_shape(tmp_path: Path) -> None:
    hea_path = _write_records500_fixture(
        tmp_path, "records500/00000/00003_hr", np.zeros((500, 12)), fs=500
    )
    with pytest.raises(ValueError, match=r"expected \(5000, 12\)"):
        read_and_decimate_500hz(hea_path, expected_sig_name=LEAD_ORDER)


def test_read_and_decimate_500hz_rejects_non_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import winder.data.ptbxl as ptbxl_module

    def _fake_read_record(
        hea_path: str, expected_sig_name: object = None
    ) -> tuple[np.ndarray, object]:
        class _Header:
            fs = 500

        return np.full((5000, 12), np.nan, dtype=np.float32), _Header()

    monkeypatch.setattr(ptbxl_module, "read_record", _fake_read_record)
    with pytest.raises(ValueError, match="non-finite"):
        read_and_decimate_500hz("does/not/matter.hea", expected_sig_name=LEAD_ORDER)


def test_read_and_decimate_500hz_error_message_names_the_hea_path_not_an_ecg_id(
    tmp_path: Path,
) -> None:
    """The helper has no metadata to look up an ecg_id from -- its errors reference
    `hea_path` only; a caller that wants an ecg_id-qualified message wraps the ValueError
    itself (see EcgWindowDataset.__getitem__)."""
    hea_path = _write_records500_fixture(
        tmp_path, "records500/00000/00004_hr", np.zeros((500, 12)), fs=500
    )
    with pytest.raises(ValueError) as exc_info:
        read_and_decimate_500hz(hea_path, expected_sig_name=LEAD_ORDER)
    assert hea_path in str(exc_info.value)


# --------------------------------------------------------------------------- real corpus
@pytest.mark.skipif(not _HAS_REAL_CORPUS, reason=f"ttl-phase corpus not found at {PTBXL_ROOT}")
def test_load_metadata_matches_known_counts_on_the_real_corpus() -> None:
    """Regression check: these counts were verified directly against ttl-phase's own
    pandas 2.3.3 output before winder adopted pandas 3.0.5, specifically to catch a
    silent behaviour change from the major-version jump."""
    df = load_metadata(PTBXL_ROOT)
    assert df.shape == (21799, 28)
    assert not df.ecg_id.duplicated().any()
    counts = df["superclass"].value_counts().to_dict()
    assert counts == {
        "NORM": 9077,
        "STTC": 4252,
        "MI": 4062,
        "CD": 3262,
        "HYP": 735,
        "UNLABELED": 411,
    }

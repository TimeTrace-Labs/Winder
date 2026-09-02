"""wfdb_io tests.

Split out of ttl-phase's single `_selftest_roundtrip` mega-function (which only ran via
`python -m src.data.wfdb_io`, never under pytest) into individually named tests, plus a new
test against the real PTB-XL records committed in tests/fixtures/wfdb/ -- replacing
ttl-phase's `_selftest_real`, which needed the full 2.6 GB corpus on disk.
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from winder.data.wfdb_io import (
    DEFAULT_GAIN,
    FORMAT16_DTYPE,
    INVALID_SENTINEL_16,
    ReadError,
    read_header,
    read_record,
    write_format16,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "wfdb")
PTBXL_LEAD_ORDER = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass
class SyntheticRecord:
    hea: str
    raw: np.ndarray
    gains: np.ndarray
    bases: np.ndarray
    names: list[str]


def _expected_mv(rec: SyntheticRecord) -> np.ndarray:
    """The mV signal `write_format16(x_is_raw=True)` + `read_record` should recover."""
    return cast(np.ndarray, ((rec.raw - rec.bases) / rec.gains).astype(np.float32))


@pytest.fixture
def synthetic_record(tmp_path: Path) -> SyntheticRecord:
    """A synthetic multi-lead format-16 record with values at the int16 extremes."""
    rng = np.random.default_rng(0)
    T, n_sig = 777, 5
    raw = rng.integers(-32000, 32000, size=(T, n_sig), dtype=np.int64)
    raw[0, 0] = -32767  # exercise the extremes without hitting the sentinel
    raw[1, 1] = 32767
    gains = np.array([1000.0, 200.0, 500.0, 1000.0, 123.5])
    bases = np.array([0, -50, 17, 0, -3])
    names = ["I", "II", "III", "AVR", "AVL"]
    stem = str(tmp_path / "synth00")
    hea, _dat = write_format16(
        stem, raw, 500, gain=gains, baseline=bases, sig_name=names, x_is_raw=True
    )
    return SyntheticRecord(hea=hea, raw=raw, gains=gains, bases=bases, names=names)


def test_roundtrip_header_fields(synthetic_record: SyntheticRecord) -> None:
    sig, hdr = read_record(
        synthetic_record.hea, verify_checksum=True, expected_sig_name=synthetic_record.names
    )
    assert hdr.n_sig == 5 and hdr.n_samp == 777 and hdr.fs == 500
    # int, not float: read_header narrows an integral fs to int (`hdr.fs == 500` alone
    # can't distinguish this, since 500 == 500.0 in Python -- audit-found gap).
    assert isinstance(hdr.fs, int)
    assert hdr.sig_name == synthetic_record.names
    assert np.array_equal(hdr.baseline, synthetic_record.bases)
    assert np.allclose(hdr.gain, synthetic_record.gains, rtol=0, atol=0)


def test_roundtrip_is_bitwise_exact(synthetic_record: SyntheticRecord) -> None:
    rec = synthetic_record
    sig, hdr = read_record(rec.hea)
    ref = _expected_mv(rec)
    assert np.array_equal(sig, ref), "float32 mV recovery is not bitwise exact"
    back = np.rint(sig.astype(np.float64) * rec.gains + rec.bases).astype(np.int64)
    assert np.array_equal(back, rec.raw), "int16 ADU round-trip is not exact"


def test_short_dat_raises(synthetic_record: SyntheticRecord, tmp_path: Path) -> None:
    with open(synthetic_record.hea, encoding="utf-8") as fh:
        hdr_text = fh.read()
    short_stem = tmp_path / "short"
    with open(str(short_stem) + ".dat", "wb") as fh:
        fh.write(synthetic_record.raw.astype(FORMAT16_DTYPE).tobytes()[:-40])
    with open(str(short_stem) + ".hea", "w", encoding="utf-8") as fh:
        fh.write(hdr_text.replace("synth00", "short"))
    with pytest.raises(ReadError):
        read_record(str(short_stem) + ".hea")


def test_oversized_dat_raises_under_strict_length_but_relaxed_accepts(
    synthetic_record: SyntheticRecord, tmp_path: Path
) -> None:
    with open(synthetic_record.hea, encoding="utf-8") as fh:
        hdr_text = fh.read()
    long_stem = tmp_path / "long"
    with open(str(long_stem) + ".dat", "wb") as fh:
        fh.write(synthetic_record.raw.astype(FORMAT16_DTYPE).tobytes() + b"\x00" * 40)
    with open(str(long_stem) + ".hea", "w", encoding="utf-8") as fh:
        fh.write(hdr_text.replace("synth00", "long"))
    with pytest.raises(ReadError):
        read_record(str(long_stem) + ".hea")

    ref = _expected_mv(synthetic_record)
    sig_relaxed, _ = read_record(str(long_stem) + ".hea", strict_length=False)
    assert np.array_equal(sig_relaxed, ref), "relaxed-length read must still be exact"


def test_lead_order_mismatch_raises_never_reorders(synthetic_record: SyntheticRecord) -> None:
    with pytest.raises(ReadError):
        read_record(synthetic_record.hea, expected_sig_name=["II", "I", "III", "AVR", "AVL"])


@pytest.mark.parametrize(
    "label,text",
    [
        (
            "bad_format",
            "b 2 500 10\nb.dat 212 1000(0)/mV 16 0 0 0 0 I\nb.dat 212 1000(0)/mV 16 0 0 0 0 II\n",
        ),
        ("multi_segment", "b/3 2 500 10\n"),
        ("bad_units", "b 1 500 10\nb.dat 16 1000(0)/furlong 16 0 0 0 0 I\n"),
        (
            "two_dat_files",
            "b 2 500 10\nb1.dat 16 1000(0)/mV 16 0 0 0 0 I\nb2.dat 16 1000(0)/mV 16 0 0 0 0 II\n",
        ),
        ("missing_signal_lines", "b 3 500 10\nb.dat 16 1000(0)/mV 16 0 0 0 0 I\n"),
        ("empty", "# only a comment\n"),
    ],
)
def test_malformed_headers_raise(tmp_path: Path, label: str, text: str) -> None:
    p = tmp_path / f"bad_{label}.hea"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ReadError):
        read_header(str(p))


def test_read_header_substitutes_default_gain_when_gain_is_zero(tmp_path: Path) -> None:
    """Audit-found: this documented WFDB-spec fallback (gain 0 means "unspecified") had
    zero test coverage."""
    raw = np.array([[100], [200], [300]], dtype=np.int64)
    stem = tmp_path / "zerogain"
    raw.astype(FORMAT16_DTYPE).tofile(str(stem) + ".dat")
    (tmp_path / "zerogain.hea").write_text(
        "zerogain 1 500 3\nzerogain.dat 16 0/mV 16 0 100 0 0 I\n", encoding="utf-8"
    )
    hdr = read_header(str(stem) + ".hea")
    assert hdr.gain[0] == DEFAULT_GAIN


def test_read_header_infers_n_samp_from_dat_size_when_unspecified(tmp_path: Path) -> None:
    """Audit-found: this documented fallback (n_samp=0 means "unspecified"; inferred from
    the .dat file size) had zero test coverage."""
    raw = np.zeros((777, 2), dtype=FORMAT16_DTYPE)
    stem = tmp_path / "inferred"
    raw.tofile(str(stem) + ".dat")
    (tmp_path / "inferred.hea").write_text(
        "inferred 2 500 0\n"
        "inferred.dat 16 1000/mV 16 0 0 0 0 I\n"
        "inferred.dat 16 1000/mV 16 0 0 0 0 II\n",
        encoding="utf-8",
    )
    hdr = read_header(str(stem) + ".hea")
    assert hdr.n_samp == 777
    assert hdr.n_samp_inferred is True


def test_invalid_sentinel_raises(synthetic_record: SyntheticRecord, tmp_path: Path) -> None:
    bad = synthetic_record.raw.copy()
    bad[3, 2] = INVALID_SENTINEL_16
    hea_s, _ = write_format16(
        str(tmp_path / "sent"),
        bad,
        500,
        gain=synthetic_record.gains,
        baseline=synthetic_record.bases,
        sig_name=synthetic_record.names,
        x_is_raw=True,
    )
    with pytest.raises(ReadError):
        read_record(hea_s)


def test_checksum_mismatch_raises(synthetic_record: SyntheticRecord, tmp_path: Path) -> None:
    with open(synthetic_record.hea, encoding="utf-8") as fh:
        txt = fh.read().splitlines()
    f0 = txt[1].split()
    f0[6] = str((int(f0[6]) + 1) % 65536)  # corrupt lead 0's checksum by one
    txt[1] = " ".join(f0)
    p = tmp_path / "ck.hea"  # written into synthetic_record's own tmp_path, next to synth00.dat
    p.write_text("\n".join(txt) + "\n", encoding="utf-8")
    with pytest.raises(ReadError):
        read_record(str(p), verify_checksum=True)


def test_real_ptbxl_records_pass_checksum_and_have_expected_shape() -> None:
    """Replaces ttl-phase's `_selftest_real`, which needed the full corpus on disk --
    this runs against the 10 real records committed in tests/fixtures/wfdb/ instead."""
    heas = sorted(glob.glob(os.path.join(FIXTURES, "*_hr.hea")))
    assert len(heas) >= 5, f"expected the golden wfdb fixtures under {FIXTURES}"
    for hea in heas:
        sig, hdr = read_record(hea, verify_checksum=True, expected_sig_name=PTBXL_LEAD_ORDER)
        assert sig.shape == (5000, 12)
        assert hdr.fs == 500
        assert np.isfinite(sig).all()

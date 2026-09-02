"""decimation tests.

Tier 3 golden test: exact match against ttl-phase's own decimate_to output on real records
(scipy 1.15.3 at generation time vs winder's scipy 1.17.1 -- verified bit-for-bit identical
before writing this test as an exact-equality check rather than a tolerance).
"""

import json
import os

import numpy as np
import pytest

from winder.data.decimation import decimate_to, out_len
from winder.data.wfdb_io import read_record

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WFDB_DIR = os.path.join(FIXTURES, "wfdb")


def test_out_len_matches_resample_poly_output_length() -> None:
    x = np.zeros((5000, 12))
    y = decimate_to(x, source_fs=500, fs_out=100)
    assert y.shape[0] == out_len(5000, 500, 100) == 1000


def test_out_len_uses_ceil_not_floor_for_a_non_multiple_input() -> None:
    """5000 is an exact multiple of the down-factor (5), so the test above can't tell
    out_len's documented ceil() convention apart from a floor() implementation -- both
    give 1000. 4999 is not a multiple: ceil(4999/5)=1000, floor(4999/5)=999, and
    resample_poly's actual output length is confirmed 1000 (audit-found test gap)."""
    x = np.zeros((4999, 12))
    y = decimate_to(x, source_fs=500, fs_out=100)
    assert y.shape[0] == out_len(4999, 500, 100) == 1000


def test_identity_when_rates_match() -> None:
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = decimate_to(x, source_fs=100, fs_out=100)
    assert y.dtype == np.float32
    assert np.array_equal(y, x.astype(np.float32))


def test_identity_aliases_the_caller_buffer_for_float32_c_contiguous_input() -> None:
    """Audit-found gap: the test above uses float64, the one dtype for which
    np.ascontiguousarray(x, dtype=np.float32) always copies (a dtype conversion forces
    one) -- so it can never observe the actual zero-copy behaviour for the production
    dtype (wfdb_io.read_record always returns float32). This is intentional, inherited
    behaviour (the docstring's own "no filtering at all" claim relies on it), not a bug --
    this test makes it explicit so it can't silently change unnoticed."""
    x = np.ascontiguousarray(np.arange(24, dtype=np.float32).reshape(12, 2))
    y = decimate_to(x, source_fs=100, fs_out=100)
    assert np.shares_memory(x, y)


def test_identity_copies_for_a_dtype_conversion() -> None:
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = decimate_to(x, source_fs=100, fs_out=100)
    assert not np.shares_memory(x, y)


def test_invalid_rates_raise() -> None:
    x = np.zeros((10, 2))
    with pytest.raises(ValueError):
        decimate_to(x, source_fs=0, fs_out=100)
    with pytest.raises(ValueError):
        decimate_to(x, source_fs=500, fs_out=-1)
    with pytest.raises(ValueError):
        decimate_to(x, source_fs=float("nan"), fs_out=100)


def test_golden_100hz_signal_matches_ttl_phase_exactly() -> None:
    """Tier 3: exact match against decimate_to run on the same records in ttl-phase's own
    environment (scipy 1.15.3). Catches a regression in decimate_to itself -- not a test of
    whether FIR beats IIR (see decimation.py's module docstring)."""
    golden = np.load(os.path.join(FIXTURES, "tier3_decimation", "golden_100hz.npz"))
    ecg_ids, sig100_golden = golden["ecg_ids"], golden["sig100"]
    tier1 = json.load(open(os.path.join(FIXTURES, "tier1_detector", "golden.json")))

    assert len(ecg_ids) >= 5
    for row, ecg_id in enumerate(ecg_ids):
        stem = tier1[str(int(ecg_id))]["stem"]
        sig, hdr = read_record(os.path.join(WFDB_DIR, f"{stem}.hea"))
        y = decimate_to(sig, source_fs=500, fs_out=100)
        assert y.shape == sig100_golden[row].shape
        assert np.array_equal(y, sig100_golden[row]), f"ecg_id={ecg_id} diverges from golden"

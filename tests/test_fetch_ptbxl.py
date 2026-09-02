"""Tests for scripts/fetch_ptbxl.py: the PTB-XL fetch stage with no network calls actually made.

`_download` is monkeypatched throughout -- these tests check the script's OWN logic (checksum
verification, idempotent skip-on-already-present, URL/path derivation from `filename_hr`,
failure collection and reporting), never that PhysioNet is reachable or returns real bytes.
"""

from __future__ import annotations

import hashlib
from typing import Any

import fetch_ptbxl
import pytest


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================================== _fetch_metadata_csv


def test_fetch_metadata_csv_skips_download_when_already_present_with_correct_hash(
    tmp_path: Any, monkeypatch: Any
) -> None:
    name = "ptbxl_database.csv"
    content = b"whatever bytes this file happens to hold"
    monkeypatch.setitem(fetch_ptbxl._EXPECTED_CSV_SHA256, name, _sha256_bytes(content))
    dest = tmp_path / name
    dest.write_bytes(content)

    def _boom(url: str, dest: str, **kwargs: Any) -> None:
        raise AssertionError("must not download when already present with the correct hash")

    monkeypatch.setattr(fetch_ptbxl, "_download", _boom)
    fetch_ptbxl._fetch_metadata_csv(name, str(tmp_path))  # must not raise


def test_fetch_metadata_csv_redownloads_when_present_with_wrong_hash(
    tmp_path: Any, monkeypatch: Any
) -> None:
    name = "ptbxl_database.csv"
    good = b"the correct bytes"
    monkeypatch.setitem(fetch_ptbxl._EXPECTED_CSV_SHA256, name, _sha256_bytes(good))
    dest = tmp_path / name
    dest.write_bytes(b"stale or corrupted bytes")

    calls: list[str] = []

    def _fake_download(url: str, dest_path: str, **kwargs: Any) -> None:
        calls.append(url)
        with open(dest_path, "wb") as fh:
            fh.write(good)

    monkeypatch.setattr(fetch_ptbxl, "_download", _fake_download)
    fetch_ptbxl._fetch_metadata_csv(name, str(tmp_path))
    assert calls == [f"{fetch_ptbxl._BASE_URL}/{name}"]
    assert dest.read_bytes() == good


def test_fetch_metadata_csv_raises_when_downloaded_bytes_still_dont_match(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The refuse-to-proceed guarantee: a download that completes but hashes wrong must not be
    silently accepted as this repo's PTB-XL 1.0.3."""
    name = "scp_statements.csv"
    monkeypatch.setitem(fetch_ptbxl._EXPECTED_CSV_SHA256, name, "0" * 64)

    def _fake_download(url: str, dest_path: str, **kwargs: Any) -> None:
        with open(dest_path, "wb") as fh:
            fh.write(b"definitely not matching the expected hash")

    monkeypatch.setattr(fetch_ptbxl, "_download", _fake_download)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch_ptbxl._fetch_metadata_csv(name, str(tmp_path))


# ============================================================================== _fetch_one_record


def test_fetch_one_record_skips_when_both_files_already_present_and_nonempty(
    tmp_path: Any, monkeypatch: Any
) -> None:
    stem = "records500/00000/00001_hr"
    hea = tmp_path / (stem + ".hea")
    dat = tmp_path / (stem + ".dat")
    hea.parent.mkdir(parents=True)
    hea.write_bytes(b"header")
    dat.write_bytes(b"signal")

    def _boom(url: str, dest: str, **kwargs: Any) -> None:
        raise AssertionError("must not download when both files already present")

    monkeypatch.setattr(fetch_ptbxl, "_download", _boom)
    assert fetch_ptbxl._fetch_one_record(stem, str(tmp_path)) is None


def test_fetch_one_record_fetches_both_files_at_the_derived_urls(
    tmp_path: Any, monkeypatch: Any
) -> None:
    stem = "records500/00000/00001_hr"
    calls: list[tuple[str, str]] = []

    def _fake_download(url: str, dest: str, **kwargs: Any) -> None:
        calls.append((url, dest))
        with open(dest, "wb") as fh:
            fh.write(b"x")

    monkeypatch.setattr(fetch_ptbxl, "_download", _fake_download)
    result = fetch_ptbxl._fetch_one_record(stem, str(tmp_path))
    assert result is None
    urls = [c[0] for c in calls]
    assert urls == [
        f"{fetch_ptbxl._BASE_URL}/{stem}.hea",
        f"{fetch_ptbxl._BASE_URL}/{stem}.dat",
    ]


def test_fetch_one_record_refetches_when_hea_present_but_dat_missing(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """An interrupted prior run can leave a .hea with no matching .dat -- must not be mistaken
    for a complete record."""
    stem = "records500/00000/00001_hr"
    hea = tmp_path / (stem + ".hea")
    hea.parent.mkdir(parents=True)
    hea.write_bytes(b"header")

    calls: list[str] = []

    def _fake_download(url: str, dest: str, **kwargs: Any) -> None:
        calls.append(url)
        with open(dest, "wb") as fh:
            fh.write(b"x")

    monkeypatch.setattr(fetch_ptbxl, "_download", _fake_download)
    fetch_ptbxl._fetch_one_record(stem, str(tmp_path))
    assert calls == [
        f"{fetch_ptbxl._BASE_URL}/{stem}.hea",
        f"{fetch_ptbxl._BASE_URL}/{stem}.dat",
    ]


def test_fetch_one_record_returns_an_error_string_on_failure_rather_than_raising(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The caller (main's ThreadPoolExecutor loop) collects failures across many concurrent
    records rather than aborting on the first -- this function must return, not raise."""
    stem = "records500/00000/00001_hr"

    def _boom(url: str, dest: str, **kwargs: Any) -> None:
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(fetch_ptbxl, "_download", _boom)
    result = fetch_ptbxl._fetch_one_record(stem, str(tmp_path))
    assert result is not None
    assert stem in result
    assert "ConnectionError" in result


# ============================================================================================ main


def test_main_derives_record_stems_from_filename_hr_and_respects_limit(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """`--limit` truncates the record list main() actually fetches, in `ptbxl_database.csv`'s
    own row order -- not a random or hash-order subset."""
    data_root = tmp_path / "ptbxl"
    data_root.mkdir()
    csv_bytes = (
        b"ecg_id,filename_hr\n1,records500/00000/00001_hr\n2,records500/00000/00002_hr\n"
        b"3,records500/00000/00003_hr\n"
    )
    (data_root / "scp_statements.csv").write_bytes(b"placeholder")
    monkeypatch.setattr(fetch_ptbxl, "_EXPECTED_N_RECORDS", 3)
    monkeypatch.setitem(
        fetch_ptbxl._EXPECTED_CSV_SHA256, "ptbxl_database.csv", _sha256_bytes(csv_bytes)
    )
    monkeypatch.setitem(
        fetch_ptbxl._EXPECTED_CSV_SHA256, "scp_statements.csv", _sha256_bytes(b"placeholder")
    )
    (data_root / "ptbxl_database.csv").write_bytes(csv_bytes)

    fetched_stems: list[str] = []

    def _fake_fetch_one_record(stem: str, root: str) -> str | None:
        fetched_stems.append(stem)
        return None

    monkeypatch.setattr(fetch_ptbxl, "_fetch_one_record", _fake_fetch_one_record)
    rc = fetch_ptbxl.main(["--data-root", str(data_root), "--limit", "2", "--workers", "1"])
    assert rc == 0
    assert sorted(fetched_stems) == ["records500/00000/00001_hr", "records500/00000/00002_hr"]


def test_main_raises_on_row_count_mismatch_even_after_hash_verifies(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """A row-count check independent of the hash check -- the hash confirms the FILE is right;
    this catches a bug in this script's own parsing, not a data problem."""
    data_root = tmp_path / "ptbxl"
    data_root.mkdir()
    csv_bytes = b"ecg_id,filename_hr\n1,records500/00000/00001_hr\n"
    (data_root / "ptbxl_database.csv").write_bytes(csv_bytes)
    (data_root / "scp_statements.csv").write_bytes(b"placeholder")
    monkeypatch.setitem(
        fetch_ptbxl._EXPECTED_CSV_SHA256, "ptbxl_database.csv", _sha256_bytes(csv_bytes)
    )
    monkeypatch.setitem(
        fetch_ptbxl._EXPECTED_CSV_SHA256, "scp_statements.csv", _sha256_bytes(b"placeholder")
    )
    # _EXPECTED_N_RECORDS left at its real value (21799); this 1-row CSV must not match it.
    with pytest.raises(ValueError, match="expected 21799"):
        fetch_ptbxl.main(["--data-root", str(data_root)])


def test_main_returns_1_and_reports_failures_without_raising(
    tmp_path: Any, monkeypatch: Any
) -> None:
    data_root = tmp_path / "ptbxl"
    data_root.mkdir()
    csv_bytes = b"ecg_id,filename_hr\n1,records500/00000/00001_hr\n2,records500/00000/00002_hr\n"
    (data_root / "ptbxl_database.csv").write_bytes(csv_bytes)
    (data_root / "scp_statements.csv").write_bytes(b"placeholder")
    monkeypatch.setattr(fetch_ptbxl, "_EXPECTED_N_RECORDS", 2)
    monkeypatch.setitem(
        fetch_ptbxl._EXPECTED_CSV_SHA256, "ptbxl_database.csv", _sha256_bytes(csv_bytes)
    )
    monkeypatch.setitem(
        fetch_ptbxl._EXPECTED_CSV_SHA256, "scp_statements.csv", _sha256_bytes(b"placeholder")
    )

    def _fake_fetch_one_record(stem: str, root: str) -> str | None:
        return f"{stem}: simulated failure"

    monkeypatch.setattr(fetch_ptbxl, "_fetch_one_record", _fake_fetch_one_record)
    rc = fetch_ptbxl.main(["--data-root", str(data_root), "--workers", "1"])
    assert rc == 1

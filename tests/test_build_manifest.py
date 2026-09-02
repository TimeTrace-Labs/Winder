"""Tests for scripts/build_manifest.py's own logic: flag-coverage completeness and the
flag->reason-code precedence mapping. Adapted from ttl-phase's tests/test_s0_phase_manifest.py.
Driver scripts aren't part of the winder package, so this test imports the script directly by
path (see conftest.py's docstring for the general pattern this project uses for CLI-importability
instead).
"""

import importlib.util
import os
import types

import numpy as np
import pytest

from winder.data.phase import ALL_FLAGS, extract_phase

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_manifest.py"
)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("build_manifest", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load_script()


def test_flag_coverage_is_total_against_the_real_phase_all_flags() -> None:
    """The actual guarantee: fails the moment phase.py gains a flag this script doesn't know
    how to map, rather than letting a record slip through unnoticed."""
    bm._assert_flag_coverage()  # must not raise
    mapped = {flag for flag, _ in bm._FLAG_PRECEDENCE}
    assert mapped == set(ALL_FLAGS)


def test_reason_for_respects_precedence_order() -> None:
    # FLAT_SIGNAL must win over TOO_FEW_BEATS when both fire.
    assert bm._reason_for(["PHASE_FLAT_SIGNAL", "PHASE_TOO_FEW_BEATS"]) == "FLAT_SIGNAL"
    # RR_OUTLIERS and IMPLAUSIBLE_RR are genuinely distinct outcomes.
    assert bm._reason_for(["PHASE_RR_OUTLIERS"]) == "RR_OUTLIERS"
    assert bm._reason_for(["PHASE_IMPLAUSIBLE_RR"]) == "IMPLAUSIBLE_RR"
    assert bm._reason_for(["PHASE_IMPLAUSIBLE_RR", "PHASE_RR_OUTLIERS"]) == "IMPLAUSIBLE_RR"


def test_reason_for_returns_none_when_nothing_mapped_fired() -> None:
    assert bm._reason_for([]) is None


@pytest.mark.parametrize(
    "mapped_flag,reason",
    [
        ("PHASE_NO_BEATS", "TOO_FEW_BEATS"),
        ("PHASE_TOO_FEW_BEATS", "TOO_FEW_BEATS"),
        ("PHASE_HIGH_RR_CV", "HIGH_RR_CV"),
        ("PHASE_LOW_YIELD", "LOW_PHASE_YIELD"),
        ("PHASE_LOW_CONFIDENCE", "LOW_CONFIDENCE"),
    ],
)
def test_every_flag_maps_to_its_expected_reason(mapped_flag: str, reason: str) -> None:
    assert bm._reason_for([mapped_flag]) == reason


def test_process_one_on_a_real_fixture_record() -> None:
    from winder.data.manifest import REASON_CODES
    from winder.data.phase import DetectorParams, PhaseQCConfig

    heas = sorted(f for f in os.listdir(os.path.join(FIXTURES, "wfdb")) if f.endswith("_hr.hea"))
    hea_path = os.path.join(FIXTURES, "wfdb", heas[0])
    ecg_id = int(heas[0].split("_")[0])

    result = bm._process_one((ecg_id, hea_path), PhaseQCConfig(), DetectorParams(), True)
    assert result["ecg_id"] == ecg_id
    assert result["status"] in ("included", "excluded")
    if result["status"] == "included":
        assert result["reason_code"] == ""
        assert isinstance(result["rpeaks"], np.ndarray)
    else:
        assert result["reason_code"] in REASON_CODES


def test_process_one_excludes_the_dead_lead_fixture_record_via_high_rr_cv() -> None:
    """The dead_lead_and_excluded fixture record (ecg_id 19299) is excluded in ttl-phase's own
    manifest for HIGH_RR_CV -- confirm this port's driver reaches the same verdict."""
    from winder.data.phase import DetectorParams, PhaseQCConfig

    hea_path = os.path.join(FIXTURES, "wfdb", "19299_hr.hea")
    result = bm._process_one((19299, hea_path), PhaseQCConfig(), DetectorParams(), True)
    assert result["status"] == "excluded"
    assert result["reason_code"] == "HIGH_RR_CV"


def test_process_one_sex_and_missing_fields_use_nan_not_sentinel() -> None:
    import pandas as pd

    assert np.isnan(bm._float_or_nan(pd.NA))
    assert np.isnan(bm._float_or_nan(float("nan")))
    assert bm._float_or_nan(1.0) == 1.0
    assert bm._str_or_empty(pd.NA) == ""
    assert bm._str_or_empty("Boston") == "Boston"


def test_extract_phase_still_importable_and_used_consistently() -> None:
    """Sanity: the script's extract_phase call signature matches phase.py's current contract
    (qc/params, not the old B-taking signature)."""
    sig = np.random.default_rng(0).normal(size=(5000, 12))
    pr = extract_phase(sig, fs=500)
    assert pr.theta.shape == (5000, 1)


# --------------------------------------------------------------------------- real corpus
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.path.join(_REPO_ROOT, "artifacts", "manifest.parquet")


@pytest.mark.skipif(
    not os.path.isfile(_MANIFEST_PATH),
    reason="run scripts/build_manifest.py against the real corpus first",
)
def test_real_run_matches_ttl_phase_reference_counts() -> None:
    """These are ttl-phase's own numbers (21,577 included / 222 excluded out of 21,799),
    verified against winder-nominal's own artifacts/manifest.parquet, which is currently
    byte-identical to the reference repo's (confirmed by md5sum in the port's acceptance gate)."""
    from winder.data.manifest import Manifest

    man = Manifest.from_parquet(_MANIFEST_PATH)
    man.assert_accounts_for(21799)
    assert man.n_included == 21577
    assert man.n_excluded == 222

    reasons = man.reason_table()["n_excluded"].to_dict()
    assert reasons["TOO_FEW_BEATS"] == 10
    assert reasons["IMPLAUSIBLE_RR"] == 4
    assert reasons["RR_OUTLIERS"] == 20
    assert reasons["HIGH_RR_CV"] == 188

    df = man.to_dataframe()
    assert not (df.sex == -1.0).any()
    flagged = df.loc[df.quality_flags != "", "quality_flags"]
    assert (flagged.str.len() > 1).all()  # no single-character flag-string corruption

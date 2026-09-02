"""Tests for scripts/run_pipeline.py: the fetch -> manifest -> phase tokens -> lead stats ->
train -> eval driver.

Every stage's own `main` is monkeypatched here -- these tests check the driver's OWN logic
(stage sequencing, skip-on-existing-output, --force, arm/seed expansion into run_ablation calls,
and stopping at the first nonzero exit code), never that a real fetch/manifest/train/eval
actually runs (each stage script owns its own tests for that).
"""

from __future__ import annotations

import os
from typing import Any

import build_manifest
import build_phase_tokens
import eval_suite
import fetch_ptbxl
import fit_lead_stats
import pytest
import run_ablation
import run_pipeline


def _patch_all_stages_ok(monkeypatch: Any, calls: list[tuple[str, list[str]]]) -> None:
    """Every stage records `(name, argv)` and returns 0 -- the all-stages-succeed baseline every
    test starts from, overridden per-test to check one stage's own behaviour."""
    for mod, name in (
        (fetch_ptbxl, "fetch_ptbxl"),
        (build_manifest, "build_manifest"),
        (build_phase_tokens, "build_phase_tokens"),
        (fit_lead_stats, "fit_lead_stats"),
        (run_ablation, "run_ablation"),
        (eval_suite, "eval_suite"),
    ):

        def _fake_main(argv: list[str] | None = None, _name: str = name) -> int:
            calls.append((_name, list(argv or [])))
            return 0

        monkeypatch.setattr(mod, "main", _fake_main)


# ================================================================================ arm validation


def test_rejects_an_unknown_arm_before_running_any_stage(monkeypatch: Any, tmp_path: Any) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    with pytest.raises(SystemExit):
        run_pipeline.main(
            ["--arms", "not_a_real_arm", "--data-root", str(tmp_path), "--skip-fetch"]
        )
    assert calls == []


# =================================================================================== sequencing


def test_runs_every_stage_in_order_on_a_fresh_artifacts_dir(
    monkeypatch: Any, tmp_path: Any
) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    data_root = str(tmp_path / "data")
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(data_root)

    exit_code = run_pipeline.main(
        [
            "--arms",
            "signal,control",
            "--seeds",
            "0,1",
            "--data-root",
            data_root,
            "--artifacts-dir",
            artifacts_dir,
        ]
    )
    assert exit_code == 0
    stage_names = [name for name, _argv in calls]
    assert stage_names == [
        "fetch_ptbxl",
        "build_manifest",
        "build_phase_tokens",
        "fit_lead_stats",
        "run_ablation",  # signal_seed0
        "run_ablation",  # signal_seed1
        "run_ablation",  # control_seed0
        "run_ablation",  # control_seed1
        "eval_suite",
    ]


def test_expands_arms_and_seeds_into_one_run_ablation_call_each(
    monkeypatch: Any, tmp_path: Any
) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    artifacts_dir = str(tmp_path / "artifacts")

    run_pipeline.main(
        [
            "--arms",
            "no_sigreg",
            "--seeds",
            "0,1",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-dir",
            artifacts_dir,
            "--skip-fetch",
        ]
    )
    ablation_argvs = [argv for name, argv in calls if name == "run_ablation"]
    assert len(ablation_argvs) == 2
    for argv, seed in zip(ablation_argvs, (0, 1), strict=True):
        assert argv[0] == "no_sigreg"
        assert argv[argv.index("--seed") + 1] == str(seed)
        assert argv[argv.index("--artifacts-dir") + 1] == os.path.join(
            artifacts_dir, "roster", f"no_sigreg_seed{seed}"
        )
        assert argv[argv.index("--artifacts-base") + 1] == artifacts_dir


# ============================================================================== skip-on-existing


def test_skips_manifest_when_manifest_parquet_already_exists(
    monkeypatch: Any, tmp_path: Any
) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir)
    open(os.path.join(artifacts_dir, "manifest.parquet"), "w").close()

    run_pipeline.main(
        [
            "--arms",
            "signal",
            "--seeds",
            "0",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-dir",
            artifacts_dir,
            "--skip-fetch",
        ]
    )
    assert "build_manifest" not in [name for name, _argv in calls]


def test_force_reruns_a_stage_even_when_its_output_already_exists(
    monkeypatch: Any, tmp_path: Any
) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    artifacts_dir = str(tmp_path / "artifacts")
    os.makedirs(artifacts_dir)
    open(os.path.join(artifacts_dir, "manifest.parquet"), "w").close()

    run_pipeline.main(
        [
            "--arms",
            "signal",
            "--seeds",
            "0",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-dir",
            artifacts_dir,
            "--skip-fetch",
            "--force",
        ]
    )
    assert "build_manifest" in [name for name, _argv in calls]


def test_skip_fetch_never_calls_fetch_ptbxl_even_on_an_empty_data_root(
    monkeypatch: Any, tmp_path: Any
) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    run_pipeline.main(
        [
            "--arms",
            "signal",
            "--seeds",
            "0",
            "--data-root",
            str(tmp_path / "empty_data_root"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--skip-fetch",
        ]
    )
    assert "fetch_ptbxl" not in [name for name, _argv in calls]


def test_skip_eval_never_calls_eval_suite(monkeypatch: Any, tmp_path: Any) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)
    run_pipeline.main(
        [
            "--arms",
            "signal",
            "--seeds",
            "0",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--skip-fetch",
            "--skip-eval",
        ]
    )
    assert "eval_suite" not in [name for name, _argv in calls]


# =========================================================================== failure propagation


def test_stops_at_the_first_nonzero_exit_and_returns_it(monkeypatch: Any, tmp_path: Any) -> None:
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)

    def _fake_build_manifest_fails(argv: list[str] | None = None) -> int:
        calls.append(("build_manifest", list(argv or [])))
        return 3

    monkeypatch.setattr(build_manifest, "main", _fake_build_manifest_fails)

    exit_code = run_pipeline.main(
        [
            "--arms",
            "signal,control",
            "--seeds",
            "0,1",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--skip-fetch",
        ]
    )
    assert exit_code == 3
    stage_names = [name for name, _argv in calls]
    # build_manifest ran (and failed); nothing after it -- phase tokens, lead stats, every
    # run_ablation call, and eval must never have been reached.
    assert stage_names == ["build_manifest"]


def test_a_nonzero_build_phase_tokens_exit_stops_the_pipeline(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """build_phase_tokens.py returning 1 means its own halt_recommended calibration check
    fired -- a real finding, not a plumbing error -- and this driver must not train past it."""
    calls: list[tuple[str, list[str]]] = []
    _patch_all_stages_ok(monkeypatch, calls)

    def _fake_build_phase_tokens_halts(argv: list[str] | None = None) -> int:
        calls.append(("build_phase_tokens", list(argv or [])))
        return 1

    monkeypatch.setattr(build_phase_tokens, "main", _fake_build_phase_tokens_halts)

    exit_code = run_pipeline.main(
        [
            "--arms",
            "signal",
            "--seeds",
            "0",
            "--data-root",
            str(tmp_path / "data"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--skip-fetch",
        ]
    )
    assert exit_code == 1
    assert "fit_lead_stats" not in [name for name, _argv in calls]
    assert "run_ablation" not in [name for name, _argv in calls]

"""Tests for scripts/run_ablation.py: the thin `resolve_arm` -> `pretrain.main` driver.

`pretrain.main` itself is monkeypatched here -- these tests check `run_ablation` builds the RIGHT
argv and default `--artifacts-dir` from `resolve_arm`, not that a real training run completes
(that is `tests/test_pretrain.py`'s and the separate smoke's job).
"""

from __future__ import annotations

from typing import Any

import pretrain
import pytest
import run_ablation

from winder.ablations import resolve_arm


def test_run_ablation_calls_pretrain_main_with_resolve_arm_s_own_argv(
    monkeypatch: Any,
) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(pretrain, "main", _fake_main)

    exit_code = run_ablation.main(["signal", "--seed", "0"])

    assert exit_code == 0
    expected = resolve_arm("signal", seed=0, artifacts_dir="artifacts/roster/signal_seed0")
    assert captured["argv"] == expected


def test_run_ablation_default_artifacts_dir_follows_the_roster_naming_convention(
    monkeypatch: Any,
) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        pretrain, "main", lambda argv=None: captured.setdefault("argv", list(argv or [])) or 0
    )

    run_ablation.main(["control", "--seed", "1"])
    argv = captured["argv"]
    assert argv[argv.index("--artifacts-dir") + 1] == "artifacts/roster/control_seed1"


def test_run_ablation_honors_an_explicit_artifacts_dir_override(monkeypatch: Any) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        pretrain, "main", lambda argv=None: captured.setdefault("argv", list(argv or [])) or 0
    )

    run_ablation.main(["signal", "--seed", "0", "--artifacts-dir", "/tmp/custom"])
    argv = captured["argv"]
    assert argv[argv.index("--artifacts-dir") + 1] == "/tmp/custom"


def test_run_ablation_propagates_pretrain_s_exit_code(monkeypatch: Any) -> None:
    monkeypatch.setattr(pretrain, "main", lambda argv=None: 2)
    assert run_ablation.main(["signal", "--seed", "0"]) == 2


def test_run_ablation_rejects_an_unknown_arm_name() -> None:
    with pytest.raises(SystemExit):
        run_ablation.main(["not_a_real_arm", "--seed", "0"])

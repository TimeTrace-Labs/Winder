"""Tests for scripts/eval_suite.py (Phase P9): the post-training eval suite.

Split, as elsewhere in this project, into fast always-run tests on pure logic and small synthetic
fixtures (no PTB-XL, no GPU) and skip-gated integration tests that touch the real roster Phase P8
actually wrote (`artifacts/roster/<arm>`) and real PTB-XL data.

`tests/conftest.py` puts `scripts/` on `sys.path`, so `import eval_suite` here means
`scripts/eval_suite.py`, not the `winder` package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import eval_suite
import numpy as np
import pytest
import torch

from winder.config import ArmConfig
from winder.determinism import generator, init_parameters
from winder.eval.comparison import EvalCohort
from winder.eval.probe import LinearProbeConfig
from winder.eval.tasks import CLASSES
from winder.jepa import checkpoint
from winder.jepa.model import JepaConfig, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.paths import default_data_root

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PTBXL_ROOT = default_data_root()
_HAS_PTBXL_ROOT = os.path.isfile(os.path.join(_PTBXL_ROOT, "ptbxl_database.csv"))
_ARTIFACTS_DIR = os.path.join(_REPO_ROOT, "artifacts")
_LEAD_STATS_PATH = os.path.join(_ARTIFACTS_DIR, "lead_stats_f1to9.json")
_MANIFEST_PATH = os.path.join(_ARTIFACTS_DIR, "manifest.parquet")
_THETA_TOKENS_PATH = os.path.join(_ARTIFACTS_DIR, "phase", "theta_tokens.npz")
_HAS_TOP_LEVEL_ARTIFACTS = (
    os.path.isfile(_LEAD_STATS_PATH)
    and os.path.isfile(_MANIFEST_PATH)
    and os.path.isfile(_THETA_TOKENS_PATH)
)
_ROSTER_DIR = os.path.join(_ARTIFACTS_DIR, "roster")
_HAS_REAL_ROSTER = all(
    os.path.isdir(os.path.join(_ROSTER_DIR, arm)) for arm in ("signal_seed0", "control_seed0")
)

# =========================================================================== tiny synthetic cohort

_N_SAMPLES = 1000
_N_TOKENS = 250
_OP_K0, _OP_N_J, _OP_K_J = 2, [1, 2], [2, 2]
_DIM = _OP_K0 + 2 * sum(_OP_K_J)
_PATCH_WIDTH = 4
_SMALL_CFG = LinearProbeConfig(lr=0.1, weight_decay=0.0, max_epochs=3, early_stopping_patience=1)


def _tiny_jepa_config() -> JepaConfig:
    return JepaConfig(
        n_leads=12,
        n_samples=_N_SAMPLES,
        n_tokens=_N_TOKENS,
        encoder_name="residual_cnn",
        encoder={},
        projector_name="mlp",
        projector={"input_width": 256, "hidden_width": 16, "output_width": _DIM},
        predictor_name="transformer",
        predictor={"width": _DIM, "n_heads": 2, "feedforward_width": 16},
        mask_sampler_name="causal_block",
        mask_sampler={},
        prediction_loss_name="mse",
        prediction_loss={},
        regularizer_name="sigreg",
        regularizer={"n_directions": 4, "chunk": 4},
    )


def _write_tiny_checkpoint(ckpt_dir: Path, *, with_operator: bool, seed: int = 0) -> str:
    jepa_cfg = _tiny_jepa_config()
    model = build_jepa(jepa_cfg, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    operator = None
    arm_cfg = None
    if with_operator:
        operator = CyclicOperator(CyclicOperatorConfig(k0=_OP_K0, n_j=_OP_N_J, k_j=_OP_K_J))
        arm_cfg = ArmConfig(
            name="tiny_cyclic",
            seed=seed,
            operator_name="cyclic",
            operator={"k0": _OP_K0, "n_j": _OP_N_J, "k_j": _OP_K_J},
        )
    config_yaml = checkpoint.resolved_config_yaml(
        jepa_cfg, TrainConfig(n_steps=10, seed_pretrain=seed), arm_config=arm_cfg
    )
    checkpoint.save_checkpoint(
        str(ckpt_dir),
        model=model,
        optimizer=optimizer,
        step=10,
        generators={},
        config_yaml=config_yaml,
        meta={},
        operator=operator,
    )
    return str(ckpt_dir)


def _tiny_cohort(seed: int = 0) -> EvalCohort:
    gen = torch.Generator().manual_seed(seed)
    counts = {"train": 6, "cal": 4, "eval": 4}
    waveforms = {s: torch.randn(n, 12, _N_SAMPLES, generator=gen) for s, n in counts.items()}
    thetas = {
        s: torch.remainder(torch.rand(n, _N_TOKENS, generator=gen) * 6.28318, 6.28318)
        for s, n in counts.items()
    }
    labels = {
        s: (torch.rand(n, len(CLASSES), generator=gen) > 0.5).float().numpy()
        for s, n in counts.items()
    }
    patient_ids = {s: np.arange(n) for s, n in counts.items()}
    rr_median_ms = {s: np.full(n, 800.0) for s, n in counts.items()}
    return EvalCohort(
        waveforms=waveforms,
        thetas=thetas,
        labels=labels,
        patient_ids=patient_ids,
        rr_median_ms=rr_median_ms,
        patch_width=_PATCH_WIDTH,
        gain_limit=4,
    )


# ========================================================================= resolve_focus_steps


def test_resolve_focus_steps_maps_literal_and_final_labels() -> None:
    steps = {2500: "a", 5000: "b", 30000: "c"}
    resolved = eval_suite.resolve_focus_steps(["5000", "final"], steps)
    assert resolved == {"5000": 5000, "final": 30000}


def test_resolve_focus_steps_raises_on_a_missing_literal_step() -> None:
    with pytest.raises(ValueError, match="not found"):
        eval_suite.resolve_focus_steps(["9999"], {2500: "a"})


def test_resolve_focus_steps_final_is_the_max_even_if_listed_first() -> None:
    steps = {30000: "final_dir", 2500: "a", 27500: "b"}
    assert eval_suite.resolve_focus_steps(["final"], steps) == {"final": 30000}


# ===================================================================== envelope / atomic write


def test_envelope_carries_split_status_and_headline_at_top_level() -> None:
    report = eval_suite._envelope("PASS", {"x": 1}, ["a decision"], {"seed": 0}, seed=0)
    assert report["split_status"] == "train_contaminated"
    assert report["headline"] is False
    assert report["status"] == "PASS"
    assert report["milestone_id"] == eval_suite.MILESTONE_ID
    assert "a decision" in report["decisions"]


def test_write_json_atomic_leaves_no_tmp_file_and_round_trips(tmp_path: Path) -> None:
    out = str(tmp_path / "nested" / "report.json")
    eval_suite._write_json_atomic(out, {"a": 1.0})
    assert os.path.isfile(out)
    assert not os.path.isfile(out + ".tmp")
    with open(out, encoding="utf-8") as fh:
        assert json.load(fh) == {"a": 1.0}


# ===================================================================== auroc_curve_for_arm


def test_auroc_curve_for_arm_returns_one_point_per_step(tmp_path: Path) -> None:
    ckpt_a = _write_tiny_checkpoint(tmp_path / "step5" / "checkpoint", with_operator=True, seed=0)
    ckpt_b = _write_tiny_checkpoint(tmp_path / "step10" / "checkpoint", with_operator=True, seed=0)
    cohort = _tiny_cohort()
    curve = eval_suite.auroc_curve_for_arm(
        {5: ckpt_a, 10: ckpt_b}, cohort, device=torch.device("cpu"), seed=0
    )
    assert set(curve) == {"5", "10"}
    for point in curve.values():
        assert point["cell"] == "z/mean"
        assert isinstance(point["macro_auroc"], float)
        assert point["n_eval"] + point["n_dropped"] == 4


def test_auroc_curve_for_arm_records_a_per_step_error_without_raising(tmp_path: Path) -> None:
    good = _write_tiny_checkpoint(tmp_path / "good" / "checkpoint", with_operator=True)
    cohort = _tiny_cohort()
    curve = eval_suite.auroc_curve_for_arm(
        {5: good, 10: str(tmp_path / "does_not_exist")},
        cohort,
        device=torch.device("cpu"),
        seed=0,
    )
    assert "macro_auroc" in curve["5"]
    assert "error" in curve["10"]


# ================================================================= full_battery_for_checkpoint


def test_full_battery_for_checkpoint_returns_every_declared_section(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path / "checkpoint", with_operator=True)
    cohort = _tiny_cohort()
    result = eval_suite.full_battery_for_checkpoint(
        "tiny",
        ckpt_dir,
        cohort,
        device=torch.device("cpu"),
        seed=0,
        n_boot=10,
        n_strata=2,
        gain_limit=4,
        n_replicates=10,
        geometry_limit=4,
    )
    assert set(result) == {"operator", "geometry", "gain", "g1", "probe_z_mean", "robustness"}
    assert result["operator"]["has_operator"] is True
    assert isinstance(result["g1"]["g1_pass"], bool)
    assert "null_ladder" in result["robustness"]


def test_full_battery_for_checkpoint_raises_for_an_arm_with_no_operator(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path / "checkpoint", with_operator=False)
    cohort = _tiny_cohort()
    with pytest.raises(ValueError, match="no transport operator"):
        eval_suite.full_battery_for_checkpoint(
            "tiny",
            ckpt_dir,
            cohort,
            device=torch.device("cpu"),
            seed=0,
            n_boot=10,
            n_strata=2,
            gain_limit=4,
            n_replicates=10,
            geometry_limit=4,
        )


# ========================================================== skip-gated: real artifacts / roster


@pytest.mark.skipif(
    not (_HAS_PTBXL_ROOT and _HAS_TOP_LEVEL_ARTIFACTS),
    reason="real PTB-XL data or winder-nominal's own top-level artifacts absent",
)
def test_build_p9_cohort_uses_the_legacy_split_and_the_f1to9_lead_stats() -> None:
    """The lead-stats trap, from the cohort-builder side: this must load waveforms against
    `lead_stats_f1to9.json` (what the real checkpoints were trained with), not the legacy
    folds-1-8 stats `winder.eval.acceptance.build_acceptance_cohort` deliberately uses."""
    cohort, bookkeeping = eval_suite.build_p9_cohort(
        _PTBXL_ROOT, _ARTIFACTS_DIR, _LEAD_STATS_PATH, train_limit=25
    )
    assert bookkeeping["n_train"] == 25
    assert bookkeeping["n_cal"] == 2563
    assert bookkeeping["n_eval"] == 2146
    assert bookkeeping["lead_stats_path"] == _LEAD_STATS_PATH
    assert cohort.waveforms["train"].shape[0] == 25
    assert cohort.waveforms["eval"].shape[0] == 2146


@pytest.mark.skipif(
    not (_HAS_PTBXL_ROOT and _HAS_TOP_LEVEL_ARTIFACTS and _HAS_REAL_ROSTER),
    reason="real PTB-XL data, top-level artifacts, or the real roster absent",
)
def test_main_runs_end_to_end_on_two_real_arms_at_one_focus_step(tmp_path: Path) -> None:
    """The real end-to-end wiring at drastically reduced scope: 2 real arms (not 4), 1 focus
    step (not 2), a small `--train-limit`, tiny bootstrap/replicate counts -- fast enough for the
    standard gate, genuine compute against the real roster, never a substitute for the full run
    this script's own `main()` performs when launched for the actual Phase P9 report.

    Uses whatever device the real production run would use (CUDA if available) rather than
    forcing CPU: unlike `scripts/pretrain.py`'s own smoke test (a handful of minibatch steps,
    genuinely fast on CPU), this test's own `build_p9_cohort` eagerly decodes the FULL cal+eval
    pools (2563+2146 records) regardless of `--train-limit`, then runs real model forward passes
    over them for 2 arms across a 12-step curve, a full battery, and a comparison table -- CPU
    made that combination too slow for the standard gate (empirically: still running after 14+
    minutes wall-clock in this session), and this machine's own A100 is otherwise idle."""
    out = str(tmp_path / "p9_eval_suite.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exit_code = eval_suite.main(
        [
            "--data-root",
            _PTBXL_ROOT,
            "--artifacts-dir",
            _ARTIFACTS_DIR,
            "--out",
            out,
            "--device",
            device,
            "--arms",
            "signal_seed0,control_seed0",
            "--focus-steps",
            "2500",
            "--train-limit",
            "20",
            "--n-boot",
            "20",
            "--n-strata",
            "2",
            "--gain-limit",
            "20",
            "--n-replicates",
            "20",
            "--geometry-limit",
            "20",
        ]
    )
    assert exit_code == 0
    with open(out, encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["status"] == "PASS"
    assert report["split_status"] == "train_contaminated"
    assert report["headline"] is False
    assert report["metrics"]["lead_stats_hash_check"]["n_mismatched"] == 0
    assert set(report["metrics"]["auroc_curves"]) == {"signal_seed0", "control_seed0"}
    assert len(report["metrics"]["auroc_curves"]["signal_seed0"]) == 12
    assert set(report["metrics"]["full_battery"]["signal_seed0"]) == {"2500"}
    assert set(report["metrics"]["comparison_table"]) == {"2500"}
    assert set(report["metrics"]["comparison_table"]["2500"]) == {"signal_seed0", "control_seed0"}
    for row in report["metrics"]["comparison_table"]["2500"].values():
        assert "_scores" not in row  # popped before writing -- not JSON-serialisable

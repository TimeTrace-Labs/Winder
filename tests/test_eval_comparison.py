"""Tests for winder.eval.comparison: the generic N-way arm comparison table.

Per the design brief: tested only structurally/mechanically in this phase, against a small
synthetic checkpoint compared against ITSELF under two arm names -- confirming the function runs
and returns the right shape/keys, and that `pairwise_deltas` computes sane deltas. Reproducing the
real ~0.087 AUROC control gap needs `FIN_LAM0_seed0`'s checkpoint, not yet copied in; that
reproduction is explicitly P6's job (see `comparison.py`'s own module docstring is silent on this
because the deferral belongs to the test suite, not the library code).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from winder.config import ArmConfig
from winder.determinism import generator, init_parameters
from winder.eval.comparison import EvalCohort, arm_comparison_table, pairwise_deltas
from winder.eval.probe import LinearProbeConfig
from winder.eval.tasks import CLASSES
from winder.jepa import checkpoint
from winder.jepa.model import JepaConfig, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig

_N_SAMPLES = 1000
_N_TOKENS = 250
_PATCH_WIDTH = 4
_OP_K0, _OP_N_J, _OP_K_J = 2, [1, 2], [2, 2]
_DIM = _OP_K0 + 2 * sum(_OP_K_J)


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


def _write_tiny_checkpoint(tmp_path: Path, *, with_operator: bool, seed: int = 0) -> str:
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
    ckpt_dir = str(tmp_path / "checkpoint")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=10,
        generators={},
        config_yaml=config_yaml,
        meta={},
        operator=operator,
    )
    return ckpt_dir


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


_SMALL_CFG = LinearProbeConfig(lr=0.1, weight_decay=0.0, max_epochs=3, early_stopping_patience=1)


def test_arm_comparison_table_self_comparison_has_the_expected_shape(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    cohort = _tiny_cohort()

    table = arm_comparison_table(
        {"signal": ckpt_dir, "control": ckpt_dir},
        cohort,
        device=torch.device("cpu"),
        n_boot=20,
        n_strata=4,
        n_replicates=20,
        seed=0,
        probe_cfg=_SMALL_CFG,
    )
    assert set(table) == {"signal", "control"}
    for name, row in table.items():
        assert set(row) == {
            "macro_auroc",
            "lo",
            "hi",
            "gain_fraction",
            "g1_pass",
            "lead_dropout_worst_drop",
            "_scores",
        }, name
        assert isinstance(row["g1_pass"], bool)
        assert row["_scores"].shape == (4, len(CLASSES))

    # The two rows are the SAME checkpoint scored on the SAME cohort -- must agree exactly.
    assert table["signal"]["macro_auroc"] == pytest.approx(table["control"]["macro_auroc"])
    assert table["signal"]["gain_fraction"] == pytest.approx(table["control"]["gain_fraction"])
    np.testing.assert_array_equal(table["signal"]["_scores"], table["control"]["_scores"])


def test_arm_comparison_table_raises_for_an_arm_with_no_operator(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=False)
    cohort = _tiny_cohort()
    with pytest.raises(ValueError, match="no transport operator"):
        arm_comparison_table(
            {"control": ckpt_dir},
            cohort,
            device=torch.device("cpu"),
            n_boot=10,
            n_strata=4,
            n_replicates=10,
            seed=0,
            probe_cfg=_SMALL_CFG,
        )


def test_pairwise_deltas_self_comparison_is_paired_and_near_zero(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    cohort = _tiny_cohort()
    table = arm_comparison_table(
        {"signal": ckpt_dir, "control": ckpt_dir},
        cohort,
        device=torch.device("cpu"),
        n_boot=20,
        n_strata=4,
        n_replicates=20,
        seed=0,
        probe_cfg=_SMALL_CFG,
    )
    deltas = pairwise_deltas(
        table,
        cohort.labels["eval"],
        cohort.patient_ids["eval"],
        reference="signal",
        n_replicates=50,
        seed=0,
    )
    assert set(deltas) == {"control"}
    row = deltas["control"]
    assert row["method"] == "paired_patient_bootstrap_delta"
    assert row["delta"] == pytest.approx(0.0, abs=1e-9)  # identical scores, identical arm


def test_pairwise_deltas_falls_back_to_ci_overlap_on_mismatched_score_shapes() -> None:
    table = {
        "reference": {"macro_auroc": 0.80, "lo": 0.75, "hi": 0.85, "_scores": np.zeros((10, 5))},
        "other": {"macro_auroc": 0.60, "lo": 0.50, "hi": 0.70, "_scores": np.zeros((7, 5))},
    }
    y_eval = np.zeros((10, 5), dtype=np.float32)
    patient_ids = np.arange(10)
    out = pairwise_deltas(table, y_eval, patient_ids, reference="reference")
    assert out["other"]["method"] == "ci_overlap"
    assert out["other"]["delta"] == pytest.approx(0.60 - 0.80)
    assert out["other"]["cis_overlap"] is False  # [0.75,0.85] and [0.50,0.70] do not overlap


def test_pairwise_deltas_raises_for_an_unknown_reference() -> None:
    table = {"a": {"macro_auroc": 0.5, "lo": 0.4, "hi": 0.6, "_scores": np.zeros((3, 2))}}
    with pytest.raises(ValueError, match="reference arm"):
        pairwise_deltas(table, np.zeros((3, 2)), np.arange(3), reference="does-not-exist")

"""Tests for winder.eval.robustness: the null ladder, matched-filter/jitter sweeps, Debye-Waller
decay, lead-dropout robustness, and heart-rate stratification.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from winder.config import ArmConfig
from winder.determinism import generator, init_parameters
from winder.eval.probe import LinearProbeConfig
from winder.eval.readout import encode_z, load_model_and_operator
from winder.eval.robustness import (
    HEART_RATE_BUCKETS,
    heart_rate_bucket,
    heart_rate_strata,
    robustness_suite,
    sweep_probe,
    theta_variants,
)
from winder.eval.tasks import CLASSES
from winder.jepa import checkpoint
from winder.jepa.model import JepaConfig, JepaModel, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.harmonic import HarmonicTransport

TWO_PI = 2.0 * math.pi

# ================================================================================ theta_variants


def test_theta_variants_preserves_the_valid_mask_for_every_variant() -> None:
    theta = torch.tensor([[0.1, float("nan"), 0.5, 1.0], [float("nan")] * 4])
    rr = np.array([800.0, 900.0])
    out = theta_variants(theta, rr, patch_width=10, seed=0)
    assert set(out) == {"true", "time_index", "record_offset", "shuffled"}
    valid = torch.isfinite(theta)
    for name, variant in out.items():
        assert torch.equal(torch.isfinite(variant), valid), name


def test_theta_variants_true_is_the_input_unchanged() -> None:
    theta = torch.tensor([[0.1, 0.2, float("nan")]])
    out = theta_variants(theta, np.array([800.0]), patch_width=10, seed=0)
    torch.testing.assert_close(out["true"], theta, equal_nan=True)


def test_theta_variants_record_offset_preserves_within_record_deltas() -> None:
    """The offset is a single random constant PER RECORD, so within-record phase DIFFERENCES
    (mod 2*pi) must be exactly preserved -- only the common cross-record frame is destroyed."""
    theta = torch.tensor(
        [[0.1, 0.5, 1.5, float("nan")], [0.2, 1.0, float("nan"), 2.0]], dtype=torch.float64
    )
    rr = np.array([800.0, 900.0])
    out = theta_variants(theta, rr, patch_width=10, seed=1)
    valid = torch.isfinite(theta)
    for i in range(theta.shape[0]):
        idx = valid[i].nonzero(as_tuple=True)[0]
        true_deltas = theta[i, idx] - theta[i, idx[0]]
        offset_deltas = out["record_offset"][i, idx] - out["record_offset"][i, idx[0]]
        # Centred remainder (range (-pi, pi]) rather than torch.remainder's own [0, 2*pi): a
        # true difference of exactly 0 mod 2*pi can land at either 0 or 2*pi-eps depending on
        # floating-point rounding, and the centred form treats both as "close to zero".
        wrapped = torch.remainder(offset_deltas - true_deltas + math.pi, TWO_PI) - math.pi
        torch.testing.assert_close(wrapped, torch.zeros_like(true_deltas), atol=1e-6, rtol=0)


def test_theta_variants_shuffled_is_a_within_record_permutation() -> None:
    gen = torch.Generator().manual_seed(0)
    theta = torch.rand(4, 10, dtype=torch.float64, generator=gen) * 6.0
    theta[0, :3] = float("nan")
    out = theta_variants(theta, np.full(4, 800.0), patch_width=10, seed=2)
    for i in range(theta.shape[0]):
        valid = torch.isfinite(theta[i])
        true_sorted = torch.sort(theta[i][valid]).values
        shuf_sorted = torch.sort(out["shuffled"][i][valid]).values
        torch.testing.assert_close(true_sorted, shuf_sorted)


def test_theta_variants_time_index_uses_the_records_own_median_rr() -> None:
    theta = torch.zeros(1, 4)  # values irrelevant to time_index, only the valid MASK matters
    rr = np.array([1000.0])
    out = theta_variants(theta, rr, patch_width=10, seed=0)
    j = torch.arange(4, dtype=torch.float64)
    expected = torch.remainder(TWO_PI * (j + 0.5) * 10 * 10.0 / 1000.0, TWO_PI)
    torch.testing.assert_close(out["time_index"][0].double(), expected, atol=1e-5, rtol=0)


def test_theta_variants_time_index_falls_back_to_the_default_rr_when_missing() -> None:
    theta = torch.zeros(1, 2)
    out = theta_variants(theta, np.array([float("nan")]), patch_width=10, seed=0)
    j = torch.arange(2, dtype=torch.float64)
    expected = torch.remainder(TWO_PI * (j + 0.5) * 10 * 10.0 / 842.6, TWO_PI)
    torch.testing.assert_close(out["time_index"][0].double(), expected, atol=1e-4, rtol=0)


# ============================================================================= heart_rate_bucket


def test_heart_rate_bucket_boundaries() -> None:
    assert heart_rate_bucket(500.0) == "tachycardic"  # 120 bpm
    assert heart_rate_bucket(600.0) == "normal"  # exactly 100 bpm -- not ">" 100
    assert heart_rate_bucket(599.0) == "tachycardic"  # just over 100 bpm
    assert heart_rate_bucket(1000.0) == "normal"  # exactly 60 bpm -- not "<" 60
    assert heart_rate_bucket(1001.0) == "bradycardic"  # just under 60 bpm
    assert heart_rate_bucket(float("nan")) == "unknown"
    assert heart_rate_bucket(0.0) == "unknown"
    assert heart_rate_bucket(-100.0) == "unknown"
    assert set(HEART_RATE_BUCKETS) == {"bradycardic", "normal", "tachycardic"}


# ============================================================================ heart_rate_strata


def test_heart_rate_strata_raises_on_length_mismatch() -> None:
    """The strictness the design brief calls out: a compressed score array must RAISE, never
    silently stratify against the wrong rows."""
    scores = np.zeros((5, 2))
    labels = np.zeros((5, 2), dtype=np.float32)
    pid = np.arange(5)
    rr = np.full(3, 800.0)  # deliberately the WRONG length vs. scores
    with pytest.raises(AssertionError, match="score rows against"):
        heart_rate_strata({"arm|cell": scores}, labels, pid, rr)


def test_heart_rate_strata_recovers_a_within_band_signal() -> None:
    n = 200
    rng = np.random.default_rng(0)
    y = (rng.random((n, 2)) > 0.5).astype(np.float32)
    scores = np.stack([y[:, 0] * 10 - 5, y[:, 1] * 10 - 5], axis=1)  # perfectly predicts y
    pid = np.arange(n)
    rr = np.concatenate([np.full(80, 1100.0), np.full(80, 850.0), np.full(n - 160, 500.0)])

    out = heart_rate_strata({"arm|cell": scores}, y, pid, rr)
    assert out["bucket_counts"] == {"bradycardic": 80, "normal": 80, "tachycardic": n - 160}
    per_bucket = out["arm|cell"]
    assert set(per_bucket) == {"bradycardic", "normal", "tachycardic"}
    for band, stats in per_bucket.items():
        assert stats["macro_auroc"] > 0.95, f"{band} collapsed to {stats['macro_auroc']:.3f}"
    assert sum(s["n"] for s in per_bucket.values()) == n


def test_heart_rate_strata_skips_small_or_degenerate_bands() -> None:
    """A band with < 40 records, or with fewer than 2 positives in some class, is skipped
    entirely rather than reported on an unreliable sample."""
    n = 30  # below the 40-record floor for every band
    y = np.ones((n, 2), dtype=np.float32)
    scores = np.ones((n, 2))
    pid = np.arange(n)
    rr = np.full(n, 1000.0)  # all "normal"
    out = heart_rate_strata({"arm|cell": scores}, y, pid, rr)
    assert "arm|cell" not in out  # every band skipped -> the key itself is absent


# =================================================================================== sweep_probe


def _separable_split(n: int, n_classes: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random((n, n_classes)) < 0.4).astype(np.float64)
    x = np.concatenate([y * 12.0 - 6.0, rng.standard_normal((n, 3))], axis=1)
    x = x + rng.standard_normal(x.shape) * 0.05
    return x, y


def test_sweep_probe_returns_one_row_per_variant_with_the_expected_keys() -> None:
    n_classes = len(CLASSES)
    x_train, y_train = _separable_split(200, n_classes, seed=0)
    x_cal, y_cal = _separable_split(100, n_classes, seed=1)
    x_eval, y_eval = _separable_split(80, n_classes, seed=2)
    features = {"train": x_train, "cal": x_cal, "eval": x_eval}
    labels = {"train": y_train, "cal": y_cal, "eval": y_eval}
    eval_pid = np.arange(80)
    cfg = LinearProbeConfig(lr=0.1, weight_decay=0.0, max_epochs=20, early_stopping_patience=5)

    out = sweep_probe(lambda _v, s: features[s], ["a", "b"], labels, eval_pid, cfg, n_boot=0)
    assert len(out) == 2
    for row in out:
        assert set(row) == {"macro_auroc", "lo", "hi", "per_class", "n_eval", "n_dropped"}
        assert math.isnan(row["lo"]) and math.isnan(row["hi"])  # n_boot=0 -> point only
        assert row["macro_auroc"] > 0.9


def test_sweep_probe_with_n_boot_returns_a_real_interval() -> None:
    n_classes = len(CLASSES)
    x_train, y_train = _separable_split(200, n_classes, seed=3)
    x_cal, y_cal = _separable_split(100, n_classes, seed=4)
    x_eval, y_eval = _separable_split(80, n_classes, seed=5)
    features = {"train": x_train, "cal": x_cal, "eval": x_eval}
    labels = {"train": y_train, "cal": y_cal, "eval": y_eval}
    eval_pid = np.arange(80)
    cfg = LinearProbeConfig(lr=0.1, weight_decay=0.0, max_epochs=20, early_stopping_patience=5)

    out = sweep_probe(lambda _v, s: features[s], ["only"], labels, eval_pid, cfg, n_boot=50)
    row = out[0]
    assert not math.isnan(row["lo"]) and not math.isnan(row["hi"])
    assert row["lo"] <= row["macro_auroc"] <= row["hi"]


# ============================================================================ robustness_suite


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


def _build_model_and_operator(
    tmp_path: Path, *, seed: int = 0
) -> tuple[JepaModel, HarmonicTransport]:
    jepa_cfg = _tiny_jepa_config()
    model = build_jepa(jepa_cfg, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
    loaded_model, loaded_operator = load_model_and_operator(
        ckpt_dir, seed=seed, device=torch.device("cpu")
    )
    assert loaded_operator is not None
    return loaded_model, loaded_operator


def test_robustness_suite_returns_all_five_sweeps_with_expected_shapes(tmp_path: Path) -> None:
    model, operator = _build_model_and_operator(tmp_path)
    device = torch.device("cpu")
    splits = ("train", "cal", "eval")
    counts = {"train": 4, "cal": 3, "eval": 3}
    gen = torch.Generator().manual_seed(0)

    waveforms = {s: torch.randn(n, 12, _N_SAMPLES, generator=gen) for s, n in counts.items()}
    thetas = {
        s: torch.remainder(torch.rand(n, _N_TOKENS, generator=gen) * TWO_PI, TWO_PI)
        for s, n in counts.items()
    }
    labels = {
        s: (torch.rand(n, len(CLASSES), generator=gen) > 0.5).float().numpy()
        for s, n in counts.items()
    }
    eval_pid = np.arange(counts["eval"])
    rr_by_split = {s: np.full(n, 800.0) for s, n in counts.items()}
    cfg = LinearProbeConfig(lr=0.1, weight_decay=0.0, max_epochs=3, early_stopping_patience=1)

    z_by_split = {s: encode_z(model, waveforms[s], device) for s in splits}

    out = robustness_suite(
        model,
        operator,
        z_by_split,
        thetas,
        waveforms,
        labels,
        eval_pid,
        rr_by_split,
        _PATCH_WIDTH,
        cfg,
        device,
        seed=0,
    )

    assert set(out["null_ladder"]) == {
        "true",
        "time_index",
        "record_offset",
        "shuffled",
        "masked_mean_theta_blind",
    }
    assert out["theta_offset_sweep"]["phi"][0] == 0.0
    assert len(out["theta_offset_sweep"]["macro_auroc"]) == 13
    assert out["theta_jitter_sweep"]["sigma"] == [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    assert len(out["theta_jitter_sweep"]["macro_auroc"]) == 6
    assert set(out["debye_waller"]) == {"sigmas", "n_j", "amplitudes", "fit_all_harmonics"}
    assert set(out["lead_dropout"]) == {"z/demodulated", "z/mean"}
    for cell in out["lead_dropout"].values():
        assert len(cell["per_lead"]) == 12
        assert {row["lead"] for row in cell["per_lead"]} == {
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        }

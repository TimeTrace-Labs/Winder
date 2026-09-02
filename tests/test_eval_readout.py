"""Tests for winder.eval.readout: frozen-checkpoint loading, encoding, and checkpoint-ladder
discovery.

`test_discover_seed_checkpoints_*` are ported (adapted to import from the library module rather
than loading a script by path) from the reference repo's `tests/test_scratch_finale_eval.py`,
which is where `discover_seed_checkpoints`/`final_step_from_config` actually live -- the design
brief attributed them to `p1_panel_numerics.py`/`p3_extras_numerics.py`, which was wrong (see
`readout.py`'s own module docstring).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

import winder.eval.readout as readout
from winder.config import ArmConfig
from winder.determinism import generator, init_parameters
from winder.jepa import checkpoint
from winder.jepa.dataset import EcgWindowItem
from winder.jepa.model import JepaConfig, JepaModel, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF_CKPT_DIR = _REPO_ROOT / "artifacts" / "reference" / "FIN_seed0" / "checkpoint_step5000"
_HAS_REF_CKPT = (_REF_CKPT_DIR / checkpoint.STATE_FILENAME).is_file()


# ============================================================================== tiny checkpoint

_N_SAMPLES = 1000
_N_TOKENS = 250
_OP_K0, _OP_N_J, _OP_K_J = 2, [1, 2], [2, 2]  # dimension = 2 + 2*4 = 10
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
    """A from-scratch, tiny (10-D latent) checkpoint -- with or without a declared transport
    operator -- built entirely from this repo's own config/checkpoint plumbing, no real PTB-XL
    data or reference-repo artifact required."""
    jepa_cfg = _tiny_jepa_config()
    model = build_jepa(jepa_cfg, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_cfg = TrainConfig(n_steps=100, seed_pretrain=seed)

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
    config_yaml = checkpoint.resolved_config_yaml(jepa_cfg, train_cfg, arm_config=arm_cfg)

    ckpt_dir = str(tmp_path / "checkpoint")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=100,
        generators={},
        config_yaml=config_yaml,
        meta={"winder_git_sha": "test"},
        operator=operator,
    )
    return ckpt_dir


# =================================================================== load_model_and_operator


def test_load_model_and_operator_returns_eval_mode_model_with_operator(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    model, operator = readout.load_model_and_operator(ckpt_dir, seed=0, device=torch.device("cpu"))
    assert isinstance(model, JepaModel)
    assert model.training is False  # .eval() was called
    assert operator is not None
    assert operator.k_j.tolist() == _OP_K_J
    assert operator.dimension == _DIM


def test_load_model_and_operator_disables_grad_on_the_operator(tmp_path: Path) -> None:
    """The free arm's omega is an nn.Parameter; without requires_grad_(False) a later .numpy()
    on a transported tensor raises. Pinned here on the FREE arm specifically (the arm whose own
    `operator_name` resolves to `FreeOperator`, `learnable_omega=True` by construction -- see
    `winder.operators.free`'s own docstring: learnability is set by which class is instantiated,
    never a config field)."""
    jepa_cfg = _tiny_jepa_config()
    model = build_jepa(jepa_cfg, generator=generator(0, "handshake"))
    init_parameters(model, generator(0, "init"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    operator = FreeOperator(FreeOperatorConfig(k0=_OP_K0, n_j=_OP_N_J, k_j=_OP_K_J))
    arm_cfg = ArmConfig(
        name="tiny_free",
        operator_name="free",
        operator={"k0": _OP_K0, "n_j": _OP_N_J, "k_j": _OP_K_J},
    )
    config_yaml = checkpoint.resolved_config_yaml(
        jepa_cfg, TrainConfig(n_steps=10), arm_config=arm_cfg
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

    _model, loaded_operator = readout.load_model_and_operator(
        ckpt_dir, seed=0, device=torch.device("cpu")
    )
    assert loaded_operator is not None
    assert loaded_operator.omega.requires_grad is False
    z = torch.randn(2, 3, loaded_operator.dimension, dtype=torch.float64)
    delta = torch.zeros(2, 3, dtype=torch.float64)
    transported = loaded_operator.transport(z, delta)
    transported.numpy()  # would raise RuntimeError if requires_grad were still True


def test_load_model_and_operator_returns_none_operator_for_a_control_checkpoint(
    tmp_path: Path,
) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=False)
    _model, operator = readout.load_model_and_operator(ckpt_dir, seed=0, device=torch.device("cpu"))
    assert operator is None


# ===================================================================== operator_from_checkpoint


def test_operator_from_checkpoint_matches_load_model_and_operator(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    _model, via_model = readout.load_model_and_operator(
        ckpt_dir, seed=0, device=torch.device("cpu")
    )
    operator_only = readout.operator_from_checkpoint(ckpt_dir)
    assert via_model is not None and operator_only is not None
    assert torch.equal(via_model.omega, operator_only.omega)
    assert operator_only.k_j.tolist() == _OP_K_J
    assert operator_only.omega.requires_grad is False


def test_operator_from_checkpoint_returns_none_without_an_arm_section(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=False)
    assert readout.operator_from_checkpoint(ckpt_dir) is None


# ============================================================================= encode_z/hidden


def test_encode_z_and_encode_hidden_shapes(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    model, _operator = readout.load_model_and_operator(ckpt_dir, seed=0, device=torch.device("cpu"))
    waveforms = torch.randn(5, 12, _N_SAMPLES)

    z = readout.encode_z(model, waveforms, torch.device("cpu"), bs=2)
    assert z.shape == (5, _N_TOKENS, _DIM)

    hidden = readout.encode_hidden(model, waveforms, torch.device("cpu"), bs=2)
    assert hidden.shape[0] == 5 and hidden.shape[1] == _N_TOKENS


def test_encode_z_matches_a_direct_forward_pass(tmp_path: Path) -> None:
    """chunked batching must not change the numeric result."""
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    model, _operator = readout.load_model_and_operator(ckpt_dir, seed=0, device=torch.device("cpu"))
    waveforms = torch.randn(4, 12, _N_SAMPLES)

    chunked = readout.encode_z(model, waveforms, torch.device("cpu"), bs=1)
    with torch.no_grad():
        direct = model.projector.forward(model.encoder.forward(waveforms))
    torch.testing.assert_close(chunked, direct)


# ================================================================================ mean_features


def _theta_dict(n: int) -> dict[str, torch.Tensor]:
    return {
        split: torch.rand(n, _N_TOKENS, dtype=torch.float32) * 6.28
        for split, n in {"train": n, "cal": n, "eval": n}.items()
    }


def test_mean_features_returns_one_array_per_split(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    waveforms = {s: torch.randn(3, 12, _N_SAMPLES) for s in ("train", "cal", "eval")}
    thetas = _theta_dict(3)

    feats = readout.mean_features(ckpt_dir, waveforms, thetas, torch.device("cpu"), seed=0)
    assert set(feats) == {"train", "cal", "eval"}
    for arr in feats.values():
        assert arr.shape == (3, _DIM)


# ================================================================================ pooled_cells


def test_pooled_cells_returns_both_readout_cells(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=True)
    waveforms = {s: torch.randn(3, 12, _N_SAMPLES) for s in ("train", "cal", "eval")}
    thetas = _theta_dict(3)

    cells = readout.pooled_cells(ckpt_dir, waveforms, thetas, torch.device("cpu"), seed=0)
    assert set(cells) == {"train", "cal", "eval"}
    for split_cells in cells.values():
        assert set(split_cells) == {"z/mean", "z/demodulated"}
        assert split_cells["z/mean"].shape == (3, _DIM)
        assert split_cells["z/demodulated"].shape == (3, _DIM)


def test_pooled_cells_raises_without_a_transport_operator(tmp_path: Path) -> None:
    ckpt_dir = _write_tiny_checkpoint(tmp_path, with_operator=False)
    waveforms = {s: torch.randn(2, 12, _N_SAMPLES) for s in ("train", "cal", "eval")}
    thetas = _theta_dict(2)
    with pytest.raises(ValueError, match="no transport operator"):
        readout.pooled_cells(ckpt_dir, waveforms, thetas, torch.device("cpu"), seed=0)


# ============================================================================= theta_for_frame


def test_theta_for_frame_looks_up_by_ecg_id_and_nans_missing() -> None:
    frame = pd.DataFrame({"ecg_id": [10, 20, 30]})
    theta_by_id = {10: np.array([0.1, 0.2]), 30: np.array([0.5, 0.6])}
    out = readout.theta_for_frame(frame, theta_by_id, n_tokens=2)
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], torch.tensor([0.1, 0.2]))
    assert torch.isnan(out[1]).all()  # ecg_id 20 has no phase-clock entry
    assert torch.allclose(out[2], torch.tensor([0.5, 0.6]))


# ================================================================================ read_waveforms


class _FakeEcgDataset(Dataset[EcgWindowItem]):
    """A minimal `Dataset[EcgWindowItem]` stand-in for `read_waveforms`'s own tests -- exercises
    only its batching/concatenation logic, independent of `EcgWindowDataset`'s own WFDB-reading
    logic (already covered by `tests/test_jepa_dataset.py`)."""

    def __init__(self, waveforms: torch.Tensor) -> None:
        self._waveforms = waveforms

    def __len__(self) -> int:
        return self._waveforms.shape[0]

    def __getitem__(self, index: int) -> EcgWindowItem:
        return {
            "waveform": self._waveforms[index],
            "ecg_id": index,
            "patient_id": index,
            "strat_fold": 1,
            "labels": torch.zeros(5),
            "has_label": False,
        }


def test_read_waveforms_concatenates_in_dataset_order() -> None:
    waveforms = torch.randn(7, 12, 100)
    dataset = _FakeEcgDataset(waveforms)
    out = readout.read_waveforms(dataset, batch_size=3)
    assert out.shape == waveforms.shape
    torch.testing.assert_close(out, waveforms)


# ======================================================= final_step_from_config / seed ladder


def test_final_step_from_config_reads_train_n_steps(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / checkpoint.CONFIG_FILENAME).write_text("train:\n  n_steps: 7500\n")
    assert readout.final_step_from_config(str(ckpt_dir)) == 7500


def test_final_step_from_config_raises_without_train_n_steps(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / checkpoint.CONFIG_FILENAME).write_text("jepa:\n  n_tokens: 10\n")
    with pytest.raises(ValueError, match="train.n_steps"):
        readout.final_step_from_config(str(ckpt_dir))


def _make_ckpt(root: str, name: str, n_steps: int | None = None) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, checkpoint.STATE_FILENAME), "w", encoding="utf-8") as fh:
        fh.write("marker")
    if n_steps is not None:
        with open(os.path.join(d, checkpoint.CONFIG_FILENAME), "w", encoding="utf-8") as fh:
            fh.write(f"train:\n  n_steps: {n_steps}\n")
    return d


def test_discover_seed_checkpoints_steps_and_final(tmp_path: Path) -> None:
    root = str(tmp_path)
    _make_ckpt(root, "checkpoint_step2500")
    _make_ckpt(root, "checkpoint_step5000")
    final = _make_ckpt(root, "checkpoint", n_steps=7500)
    os.makedirs(os.path.join(root, "not_a_checkpoint"))  # no state.pt -> ignored
    ladder = readout.discover_seed_checkpoints(root)
    assert sorted(ladder) == [2500, 5000, 7500]
    assert ladder[7500] == final


def test_discover_seed_checkpoints_final_collision_raises(tmp_path: Path) -> None:
    root = str(tmp_path)
    _make_ckpt(root, "checkpoint_step5000")
    _make_ckpt(root, "checkpoint", n_steps=5000)
    with pytest.raises(ValueError, match="collides"):
        readout.discover_seed_checkpoints(root)


def test_discover_seed_checkpoints_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no complete checkpoints"):
        readout.discover_seed_checkpoints(str(tmp_path))


# ========================================================= preflight_check_checkpoints (P9)


def test_preflight_check_checkpoints_reports_no_failures_on_good_checkpoints(
    tmp_path: Path,
) -> None:
    good_a = _write_tiny_checkpoint(tmp_path / "a", with_operator=True)
    good_b = _write_tiny_checkpoint(tmp_path / "b", with_operator=False, seed=1)
    failed = readout.preflight_check_checkpoints(
        {"a": good_a, "b": good_b}, seed=0, device=torch.device("cpu")
    )
    assert failed == {}


def test_preflight_check_checkpoints_reports_a_corrupt_checkpoint_without_raising(
    tmp_path: Path,
) -> None:
    """The exact scenario Task 1 exists for: a bad checkpoint dir must be reported, not raise
    out of the preflight pass and not be silently skipped."""
    good = _write_tiny_checkpoint(tmp_path / "good", with_operator=True)
    bad_dir = tmp_path / "bad" / "checkpoint"
    bad_dir.mkdir(parents=True)
    (bad_dir / checkpoint.CONFIG_FILENAME).write_text("not: valid: yaml: [")
    failed = readout.preflight_check_checkpoints(
        {"good": good, "bad": str(bad_dir)}, seed=0, device=torch.device("cpu")
    )
    assert set(failed) == {"bad"}
    assert "good" not in failed


def test_preflight_check_checkpoints_continues_past_a_failure_to_check_the_rest(
    tmp_path: Path,
) -> None:
    """A failing entry must not stop the pass from checking every remaining entry."""
    missing = str(tmp_path / "does_not_exist")
    good = _write_tiny_checkpoint(tmp_path / "good", with_operator=True)
    failed = readout.preflight_check_checkpoints(
        {"missing": missing, "good": good}, seed=0, device=torch.device("cpu")
    )
    assert set(failed) == {"missing"}


# ==================================================== assert_lead_stats_matches_checkpoint (P9)


def test_assert_lead_stats_matches_checkpoint_passes_on_a_matching_hash(tmp_path: Path) -> None:
    lead_stats_path = tmp_path / "lead_stats.json"
    lead_stats_path.write_text('{"fake": "stats"}')
    digest = hashlib_sha256_of(lead_stats_path)

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / checkpoint.META_FILENAME).write_text(f'{{"lead_stats_sha256": "{digest}"}}')
    readout.assert_lead_stats_matches_checkpoint(str(ckpt_dir), str(lead_stats_path))  # no raise


def test_assert_lead_stats_matches_checkpoint_raises_on_a_mismatched_hash(tmp_path: Path) -> None:
    lead_stats_path = tmp_path / "lead_stats.json"
    lead_stats_path.write_text('{"fake": "stats"}')

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / checkpoint.META_FILENAME).write_text('{"lead_stats_sha256": "deadbeef"}')
    with pytest.raises(AssertionError, match="lead-stats mismatch"):
        readout.assert_lead_stats_matches_checkpoint(str(ckpt_dir), str(lead_stats_path))


def test_assert_lead_stats_matches_checkpoint_raises_when_meta_has_no_hash_field(
    tmp_path: Path,
) -> None:
    lead_stats_path = tmp_path / "lead_stats.json"
    lead_stats_path.write_text('{"fake": "stats"}')

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / checkpoint.META_FILENAME).write_text("{}")
    with pytest.raises(AssertionError, match="no 'lead_stats_sha256' field"):
        readout.assert_lead_stats_matches_checkpoint(str(ckpt_dir), str(lead_stats_path))


def hashlib_sha256_of(path: Path) -> str:
    """Test-local helper: the same hex-digest convention `assert_lead_stats_matches_checkpoint`
    itself uses, kept here rather than imported so the test does not merely re-run the function
    under test against itself."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ================================================================== against the real checkpoint


pytestmark_ref = pytest.mark.skipif(
    not _HAS_REF_CKPT,
    reason=f"reference checkpoint not found at {_REF_CKPT_DIR}",
)


@pytestmark_ref
def test_load_model_and_operator_on_the_real_fin_seed0_checkpoint() -> None:
    model, operator = readout.load_model_and_operator(
        str(_REF_CKPT_DIR), seed=0, device=torch.device("cpu")
    )
    assert model.training is False
    assert operator is not None
    assert operator.k_j.tolist() == [24, 24, 20, 16, 12, 10, 8, 6, 4, 2]
    assert operator.dimension == 256


@pytestmark_ref
def test_operator_from_checkpoint_matches_the_real_checkpoints_load_model_path() -> None:
    _model, via_model = readout.load_model_and_operator(
        str(_REF_CKPT_DIR), seed=0, device=torch.device("cpu")
    )
    operator_only = readout.operator_from_checkpoint(str(_REF_CKPT_DIR))
    assert via_model is not None and operator_only is not None
    torch.testing.assert_close(via_model.omega, operator_only.omega)


@pytestmark_ref
def test_final_step_from_config_matches_the_real_checkpoints_declared_total() -> None:
    assert readout.final_step_from_config(str(_REF_CKPT_DIR)) == 30000

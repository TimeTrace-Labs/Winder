"""Phase P6, Tier 0: structural parity between winder-nominal's freshly-built architecture and
the reference repo's (winder-theory-exp) own trained checkpoints.

This is the single most load-bearing check in the winder-nominal port: an "architecturally close
enough" reimplementation is not sufficient here. If a freshly-built model's parameter/buffer names
do not match a reference checkpoint's `state_dict` EXACTLY, this is not a faithful
reimplementation, regardless of how many of winder-nominal's own tests pass.

Coverage note (deviation from the build plan's literal wording): the plan's "Acceptance test"
prose says "config-schema round-trip on all four config.yamls", but the artifact set actually
copied in under `artifacts/reference/` is SEVEN checkpoint directories, not four:
`FIN_seed0/{checkpoint_step5000, checkpoint_step25000, checkpoint}`,
`FIN_seed1/{checkpoint_step5000, checkpoint_step25000, checkpoint}`, and
`FIN_LAM0_seed0/checkpoint_step5000` (the existing control, needed later to validate
`eval/comparison.py` against a known-correct AUROC gap). This file tests all seven, per the
orchestrating brief's explicit instruction to test the actual copied-in set rather than silently
picking a subset matching the plan's older draft count.

Individually skipped (not failed) per-checkpoint when that specific reference directory is
absent -- `artifacts/` is gitignored (`.gitignore`'s `artifacts/*`), so any of these directories
only exists on a machine that has explicitly copied it in. A fresh clone with no such copy stays
green throughout.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, cast

import numpy as np
import pytest
import torch

import winder.eval.readout as readout
from winder.config import resolve_operator_config
from winder.data.folds import LEGACY_FOLD_CONFIG, folds
from winder.data.integrity import sha256_file
from winder.data.norm_stats import LeadStats
from winder.data.ptbxl import MULTIHOT_COLS, load_metadata
from winder.determinism import generator
from winder.jepa import checkpoint
from winder.jepa.dataset import EcgWindowDataset
from winder.jepa.model import JepaModel, build_jepa
from winder.operators.harmonic import HarmonicTransport
from winder.operators.registry import OPERATOR_REGISTRY
from winder.paths import default_data_root
from winder.transport.dataset import load_theta_tokens

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ARTIFACTS_ROOT = _REPO_ROOT / "artifacts" / "reference"

# The checkpoints' own recorded architecture/operator spectrum (read directly from the copied-in
# config.yamls this session, not assumed) -- identical across all seven, per FIN_LAM0_seed0's
# "same operator construction, all else equal" convention (train.lambda_trans differs, nothing
# structural does).
_EXPECTED_N_TOKENS = 125
_EXPECTED_ENCODER_NAME = "conv_trunk"
_EXPECTED_PREDICTOR_NAME = "transformer"
_EXPECTED_LAMBDA_SIG = 0.15
_EXPECTED_LAMBDA_PRED = 1.0
_EXPECTED_OPERATOR_NAME = "cyclic"
_EXPECTED_K0 = 4
_EXPECTED_N_J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_EXPECTED_K_J = [24, 24, 20, 16, 12, 10, 8, 6, 4, 2]
_EXPECTED_N_KEYS = 80

# (label, path relative to _ARTIFACTS_ROOT, expected completed-step count, expected
# train.lambda_trans) -- verified empirically against every copied-in state.pt/config.yaml this
# session (torch.load'd step field, and a diff of all seven config.yamls: FIN_seed1's differ from
# FIN_seed0's only in seed_pretrain/arm.name/arm.seed, FIN_LAM0_seed0 differs only in
# train.lambda_trans).
_CHECKPOINTS: list[tuple[str, str, int, float]] = [
    ("FIN_seed0/checkpoint_step5000", "FIN_seed0/checkpoint_step5000", 5000, 1.0),
    ("FIN_seed0/checkpoint_step25000", "FIN_seed0/checkpoint_step25000", 25000, 1.0),
    ("FIN_seed0/checkpoint", "FIN_seed0/checkpoint", 30000, 1.0),
    ("FIN_seed1/checkpoint_step5000", "FIN_seed1/checkpoint_step5000", 5000, 1.0),
    ("FIN_seed1/checkpoint_step25000", "FIN_seed1/checkpoint_step25000", 25000, 1.0),
    ("FIN_seed1/checkpoint", "FIN_seed1/checkpoint", 30000, 1.0),
    ("FIN_LAM0_seed0/checkpoint_step5000", "FIN_LAM0_seed0/checkpoint_step5000", 5000, 0.0),
]


def _ckpt_param(label: str, rel_path: str, expected_step: int, expected_lambda_trans: float) -> Any:
    ckpt_dir = _ARTIFACTS_ROOT / rel_path
    present = (ckpt_dir / checkpoint.STATE_FILENAME).is_file()
    return pytest.param(
        ckpt_dir,
        expected_step,
        expected_lambda_trans,
        id=label,
        marks=pytest.mark.skipif(
            not present,
            reason=(
                f"reference checkpoint not found at {ckpt_dir} -- copy in "
                f"artifacts/roster/{rel_path}/ from the reference repo to run this case"
            ),
        ),
    )


_CHECKPOINT_PARAMS = [_ckpt_param(*row) for row in _CHECKPOINTS]


def _read_config_yaml(ckpt_dir: pathlib.Path) -> str:
    with open(ckpt_dir / checkpoint.CONFIG_FILENAME, encoding="utf-8") as fh:
        return fh.read()


def _build_model(ckpt_dir: pathlib.Path) -> JepaModel:
    """The reference repo's own construction recipe (scripts/p1_panel_numerics.py's
    load_model_and_operator): resolve the checkpoint's own config.yaml, then build_jepa with a
    "handshake" generator stream -- the freshly-initialised weights this produces are immediately
    overwritten by load_checkpoint; only the resulting module STRUCTURE (parameter/buffer names
    and shapes) matters for this test."""
    jepa_cfg = checkpoint.jepa_config_from_yaml(_read_config_yaml(ckpt_dir))
    return build_jepa(jepa_cfg, generator=generator(0, "handshake"))


# ============================================================================ Tier 0: structure


@pytest.mark.parametrize("ckpt_dir, expected_step, expected_lambda_trans", _CHECKPOINT_PARAMS)
def test_jepa_config_from_yaml_matches_checkpoints_recorded_architecture(
    ckpt_dir: pathlib.Path, expected_step: int, expected_lambda_trans: float
) -> None:
    """jepa_config_from_yaml succeeds (no exception) and reproduces the checkpoint's own recorded
    architecture exactly -- values read directly from the copied config.yaml, not assumed."""
    jepa_cfg = checkpoint.jepa_config_from_yaml(_read_config_yaml(ckpt_dir))
    assert int(jepa_cfg.n_tokens) == _EXPECTED_N_TOKENS
    assert jepa_cfg.encoder_name == _EXPECTED_ENCODER_NAME
    assert jepa_cfg.predictor_name == _EXPECTED_PREDICTOR_NAME


@pytest.mark.parametrize("ckpt_dir, expected_step, expected_lambda_trans", _CHECKPOINT_PARAMS)
def test_train_config_from_yaml_matches_checkpoints_recorded_loss_weights(
    ckpt_dir: pathlib.Path, expected_step: int, expected_lambda_trans: float
) -> None:
    """train_config_from_yaml succeeds and reproduces the checkpoint's own recorded loss
    weights -- lambda_trans distinguishes the signal arm (1.0, transport active) from the
    FIN_LAM0 control (0.0, operator constructed but zero-weighted), per-checkpoint expectation."""
    train_cfg = checkpoint.train_config_from_yaml(_read_config_yaml(ckpt_dir))
    assert float(train_cfg.lambda_sig) == pytest.approx(_EXPECTED_LAMBDA_SIG)
    assert float(train_cfg.lambda_pred) == pytest.approx(_EXPECTED_LAMBDA_PRED)
    assert float(train_cfg.lambda_trans) == pytest.approx(expected_lambda_trans)


@pytest.mark.parametrize("ckpt_dir, expected_step, expected_lambda_trans", _CHECKPOINT_PARAMS)
def test_arm_config_from_yaml_matches_checkpoints_recorded_operator_spectrum(
    ckpt_dir: pathlib.Path, expected_step: int, expected_lambda_trans: float
) -> None:
    """arm_config_from_yaml succeeds and reproduces the checkpoint's own recorded cyclic operator
    spectrum exactly -- k0, n_j, and k_j read directly from the copied config.yaml, identical
    across all seven checkpoints (same operator construction, "all else equal")."""
    arm_cfg = checkpoint.arm_config_from_yaml(_read_config_yaml(ckpt_dir))
    assert arm_cfg is not None
    assert arm_cfg.operator_name == _EXPECTED_OPERATOR_NAME
    assert int(arm_cfg.operator.k0) == _EXPECTED_K0
    assert list(arm_cfg.operator.n_j) == _EXPECTED_N_J
    assert list(arm_cfg.operator.k_j) == _EXPECTED_K_J
    assert 4 + 2 * sum(_EXPECTED_K_J) == 256


@pytest.mark.parametrize("ckpt_dir, expected_step, expected_lambda_trans", _CHECKPOINT_PARAMS)
def test_model_state_dict_keys_match_checkpoint_exactly(
    ckpt_dir: pathlib.Path, expected_step: int, expected_lambda_trans: float
) -> None:
    """THE critical structural check: a freshly-built model's parameter/buffer names match the
    checkpoint's own state_dict EXACTLY, key for key -- proof this port has no naming divergence
    from the reference architecture (fused `qkv`, ChannelLayerNorm's `norm1.norm`/`norm2.norm`
    nesting, SeededDropout's `_extra_state` hooks, `mask_token`, `rel_bias.table`, ...). A
    mismatch here must be fixed structurally, never papered over with `strict=False` or key
    renaming. The key COUNT is verified empirically per-checkpoint (not assumed to always be 80),
    even though every one of the seven happens to be 80 in practice."""
    model = _build_model(ckpt_dir)
    state = torch.load(ckpt_dir / checkpoint.STATE_FILENAME, map_location="cpu", weights_only=False)
    ckpt_keys = set(state["model_state_dict"].keys())
    assert len(ckpt_keys) == _EXPECTED_N_KEYS

    model_keys = set(model.state_dict().keys())
    assert model_keys == ckpt_keys, (
        f"structural mismatch between winder-nominal's freshly-built model and {ckpt_dir} -- "
        f"only in checkpoint: {sorted(ckpt_keys - model_keys)}; only in freshly-built model: "
        f"{sorted(model_keys - ckpt_keys)}"
    )


@pytest.mark.parametrize("ckpt_dir, expected_step, expected_lambda_trans", _CHECKPOINT_PARAMS)
def test_strict_load_checkpoint_succeeds_on_model_alone(
    ckpt_dir: pathlib.Path, expected_step: int, expected_lambda_trans: float
) -> None:
    """checkpoint.load_checkpoint with NO strict= override (torch's own strict=True default)
    succeeds against the reference checkpoint with no exception, and recovers this checkpoint's
    own recorded step count -- the single most important verification in this phase. This phase
    does not proceed, and this test must not be worked around, if this fails."""
    model = _build_model(ckpt_dir)
    loaded = checkpoint.load_checkpoint(str(ckpt_dir), model=model)
    assert loaded.step == expected_step


@pytest.mark.parametrize("ckpt_dir, expected_step, expected_lambda_trans", _CHECKPOINT_PARAMS)
def test_strict_load_checkpoint_succeeds_with_operator_and_recovers_spectrum(
    ckpt_dir: pathlib.Path, expected_step: int, expected_lambda_trans: float
) -> None:
    """The full real-world construction recipe (mirroring scripts/p1_panel_numerics.py's own
    load_model_and_operator): build model + HarmonicTransport operator from the checkpoint's own
    config via the operators registry, strict-load both together, and confirm the operator's
    spectrum survived the round trip."""
    model = _build_model(ckpt_dir)
    arm_cfg = checkpoint.arm_config_from_yaml(_read_config_yaml(ckpt_dir))
    assert arm_cfg is not None
    _schema_cls, operator_ctor = OPERATOR_REGISTRY[arm_cfg.operator_name]
    operator = operator_ctor(resolve_operator_config(arm_cfg))
    assert isinstance(operator, HarmonicTransport)

    checkpoint.load_checkpoint(str(ckpt_dir), model=model, operator=operator)

    assert operator.k_j.tolist() == _EXPECTED_K_J
    assert 4 + 2 * int(operator.k_j.sum()) == operator.dimension == 256


# ======================================================================= Tier 0: z-parity fixture

# ttl-phase's data checkout (PTB-XL raw WFDB) -- not part of winder, may not exist on every
# machine. Reading it does not touch the "never seal fold 10" invariant: a raw WFDB file carries
# no fold membership by itself, only a *computed split* (via LEGACY_FOLD_CONFIG below, folds 1-8
# train / 9 val / 10 sealed) does, and this test never requests `unseal=True`.
_PTBXL_ROOT = default_data_root()
_HAS_PTBXL_ROOT = os.path.isfile(os.path.join(_PTBXL_ROOT, "ptbxl_database.csv"))

_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "z_parity_fin_seed0_step5000.npz"
_LEAD_STATS_PATH = _ARTIFACTS_ROOT / "lead_stats_f1to8_legacy.json"
_THETA_TOKENS_PATH = _ARTIFACTS_ROOT / "phase" / "theta_tokens.npz"
_Z_PARITY_CKPT_DIR = _ARTIFACTS_ROOT / "FIN_seed0" / "checkpoint_step5000"

_HAS_Z_PARITY_INPUTS = (
    _FIXTURE_PATH.is_file()
    and _LEAD_STATS_PATH.is_file()
    and _THETA_TOKENS_PATH.is_file()
    and (_Z_PARITY_CKPT_DIR / checkpoint.STATE_FILENAME).is_file()
    and _HAS_PTBXL_ROOT
)

_Z_PARITY_TOLERANCE = 1e-6


@pytest.mark.skipif(
    not _HAS_Z_PARITY_INPUTS,
    reason=(
        "z-parity fixture, lead_stats/theta_tokens reference artifacts, the reference "
        f"checkpoint, or the shared PTB-XL data root ({_PTBXL_ROOT}) is missing -- this test "
        "needs all four to run"
    ),
)
def test_encode_z_matches_reference_fixture_to_1e_minus_6() -> None:
    """The z-parity gate (Phase P6, Tier 0): winder-nominal's OWN `eval.readout.encode_z`,
    driven by winder-nominal's OWN data layer (`load_metadata` + `LEGACY_FOLD_CONFIG` +
    `EcgWindowDataset` + `load_theta_tokens`), reproduces
    `tests/fixtures/z_parity_fin_seed0_step5000.npz` -- an array generated by RUNNING THE
    REFERENCE REPO'S OWN CODE directly in winder-theory-exp, never regenerated from
    winder-nominal's own functions (see the fixture's own sidecar
    `z_parity_fin_seed0_step5000.meta.json` and this test's own staged assertions below).

    Staged on purpose, in this order, so a failure localises to a specific stage rather than
    reading as an undifferentiated "z doesn't match":
      1. the checkpoint's on-disk state.pt hashes to the sha256 the fixture's own provenance
         recorded (proves this test loads the SAME trained weights the fixture was built from);
      2. the reconstructed input SELECTION (which 8 ecg_ids) matches the fixture's ecg_id array
         (an input-selection bug would otherwise be indistinguishable from an architecture bug);
      3. the reconstructed theta values match (a normalization/phase-lookup bug is distinct from
         either of the above);
      4. only then, the actual z comparison.
    """
    fixture = np.load(_FIXTURE_PATH)

    # Stage 1: same trained weights.
    actual_sha256 = sha256_file(str(_Z_PARITY_CKPT_DIR / checkpoint.STATE_FILENAME))
    expected_sha256 = str(fixture["provenance_checkpoint_sha256"])
    assert actual_sha256 == expected_sha256, (
        f"checkpoint state.pt sha256 mismatch: fixture was built from a checkpoint hashing to "
        f"{expected_sha256}, but {_Z_PARITY_CKPT_DIR} currently hashes to {actual_sha256} -- "
        f"this is not the same trained artifact"
    )

    # Reconstruct the exact 8-record input selection: PTB-XL metadata filtered to at least one
    # MULTIHOT_COLS label set, then the LEGACY (train 1-8 / val 9 / sealed 10) fold split's "val"
    # partition, first 8 rows in ascending ecg_id order -- mirrors
    # scripts/p1_panel_numerics.py's own `labeled = metadata.loc[metadata[MULTIHOT_COLS].sum(
    # axis=1) > 0]` then `folds(labeled, fc)["val"]`.
    metadata = load_metadata(_PTBXL_ROOT)
    labeled = metadata.loc[metadata[list(MULTIHOT_COLS)].sum(axis=1) > 0]
    eval_frame = folds(labeled, LEGACY_FOLD_CONFIG)["val"]
    frame = eval_frame.head(8).reset_index(drop=True)

    # Stage 2: same input selection.
    expected_ecg_ids = fixture["ecg_id"].tolist()
    actual_ecg_ids = frame["ecg_id"].tolist()
    assert actual_ecg_ids == expected_ecg_ids, (
        f"input SELECTION mismatch (not a z/architecture issue): reconstructed ecg_ids "
        f"{actual_ecg_ids} != fixture's {expected_ecg_ids} -- diagnose the fold/labeling "
        f"pipeline before comparing z at all"
    )

    lead_stats = LeadStats.from_json(str(_LEAD_STATS_PATH))
    dataset = EcgWindowDataset(frame, _PTBXL_ROOT, lead_stats=lead_stats)
    device = torch.device("cpu")
    waveforms = readout.read_waveforms(dataset)

    theta_by_id, theta_meta = load_theta_tokens(str(_THETA_TOKENS_PATH))
    n_tokens = cast(int, theta_meta["n_tokens"])
    assert n_tokens == _EXPECTED_N_TOKENS
    theta = readout.theta_for_frame(frame, theta_by_id, n_tokens)

    # Stage 3: same theta (NaN-aware -- theta is NaN wherever phase is undefined for a record).
    expected_theta = fixture["theta"]
    actual_theta = theta.numpy()
    both_nan = np.isnan(expected_theta) & np.isnan(actual_theta)
    assert (np.isnan(expected_theta) == np.isnan(actual_theta)).all(), (
        "theta NaN mask mismatch between reconstructed input and fixture -- phase lookup "
        "diverged from the fixture's own generation"
    )
    theta_delta = np.abs(
        np.where(both_nan, 0.0, actual_theta) - np.where(both_nan, 0.0, expected_theta)
    )
    assert theta_delta.max() <= _Z_PARITY_TOLERANCE, (
        f"theta mismatch (not a z/architecture issue): max abs delta = {theta_delta.max()}"
    )

    # Stage 4: the actual z comparison.
    model, _operator = readout.load_model_and_operator(
        str(_Z_PARITY_CKPT_DIR), seed=0, device=device
    )
    z = readout.encode_z(model, waveforms, device).numpy()
    expected_z = fixture["z"]
    assert z.shape == expected_z.shape
    z_delta = np.abs(z - expected_z)
    assert z_delta.max() <= _Z_PARITY_TOLERANCE, (
        f"z-parity FAILED: max abs delta = {z_delta.max()} > {_Z_PARITY_TOLERANCE}. Inputs "
        f"(checkpoint sha256, ecg_id selection, theta) all verified matching above, so this "
        f"delta localises to the model/architecture/weights path itself -- do not loosen this "
        f"tolerance, escalate instead."
    )

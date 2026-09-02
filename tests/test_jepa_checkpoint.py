"""Tests for winder.jepa.checkpoint: CKPT-01 (state round trip), CKPT-02 (config snapshot)."""

import math
import os
from pathlib import Path

import pytest
import torch

from winder.config import ArmConfig, resolve_operator_config
from winder.determinism import generator, init_parameters
from winder.jepa import checkpoint
from winder.jepa.model import JepaConfig, JepaModel, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.operators.registry import OPERATOR_REGISTRY


def _tiny_config(*, projector_output_width: int = 32) -> JepaConfig:
    return JepaConfig(
        n_leads=12,
        n_samples=1000,
        n_tokens=250,
        encoder_name="residual_cnn",
        encoder={},
        projector_name="mlp",
        projector={
            "input_width": 256,
            "hidden_width": 32,
            "output_width": projector_output_width,
        },
        predictor_name="transformer",
        predictor={
            "width": projector_output_width,
            "n_heads": 4,
            "feedforward_width": 64,
        },
        mask_sampler_name="causal_block",
        mask_sampler={},
        prediction_loss_name="mse",
        prediction_loss={},
        regularizer_name="sigreg",
        regularizer={"n_directions": 8, "chunk": 8},
    )


def _build_and_init(seed: int, config: JepaConfig | None = None) -> JepaModel:
    config = config or _tiny_config()
    model = build_jepa(config, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    return model


def test_save_checkpoint_writes_all_three_files(tmp_path: Path) -> None:
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")

    out = checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=7,
        generators={"mask": generator(0, "mask")},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={"git_sha": "abc123"},
    )
    assert out == ckpt_dir
    assert os.path.isfile(os.path.join(ckpt_dir, checkpoint.STATE_FILENAME))
    assert os.path.isfile(os.path.join(ckpt_dir, checkpoint.CONFIG_FILENAME))
    assert os.path.isfile(os.path.join(ckpt_dir, checkpoint.META_FILENAME))


def test_load_checkpoint_round_trips_model_and_optimizer_state(tmp_path: Path) -> None:
    model_a = _build_and_init(0)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=3e-4)
    # Advance the optimizer's own internal state (per-parameter step/exp_avg/exp_avg_sq) so this
    # test actually exercises optimizer.load_state_dict, not just a freshly-constructed one.
    waveform = torch.zeros(1, 12, 1000)
    z = model_a.projector.forward(model_a.encoder.forward(waveform))
    z.sum().backward()
    optimizer_a.step()

    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model_a,
        optimizer=optimizer_a,
        step=3,
        generators={"mask": generator(0, "mask"), "sigreg": generator(0, "sigreg")},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
    )

    model_b = _build_and_init(1)  # different init seed -- must be fully overwritten below
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=3e-4)
    loaded = checkpoint.load_checkpoint(ckpt_dir, model=model_b, optimizer=optimizer_b)

    assert loaded.step == 3
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)
    state_a = optimizer_a.state_dict()["state"]
    state_b = optimizer_b.state_dict()["state"]
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key]["exp_avg"], state_b[key]["exp_avg"])


def test_load_checkpoint_round_trips_generator_states_bitwise(tmp_path: Path) -> None:
    gen_mask = generator(0, "mask")
    for _ in range(5):  # advance past the fresh-seed state so this isn't a vacuous check
        torch.rand(3, generator=gen_mask)

    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=5,
        generators={"mask": gen_mask},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
    )
    # The draw a continuation of gen_mask would produce next, captured on the SAME live object.
    expected_next = torch.rand(3, generator=gen_mask)

    fresh_model = _build_and_init(1)
    loaded = checkpoint.load_checkpoint(ckpt_dir, model=fresh_model)
    restored = torch.Generator()
    restored.set_state(loaded.generator_states["mask"])
    actual_next = torch.rand(3, generator=restored)

    assert torch.equal(expected_next, actual_next)


def test_load_checkpoint_does_not_change_train_eval_mode(tmp_path: Path) -> None:
    """load_checkpoint must not call model.eval()/model.train() -- see the module docstring: a
    resumed *training* step needs the model to stay exactly however the caller left it."""
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=1,
        generators={},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
    )

    fresh_model = _build_and_init(1)
    fresh_model.eval()
    checkpoint.load_checkpoint(ckpt_dir, model=fresh_model)
    assert not fresh_model.training  # unchanged: still eval, load_checkpoint touched nothing

    other_model = _build_and_init(1)
    assert other_model.training  # unchanged: still train (nn.Module's own default)
    checkpoint.load_checkpoint(ckpt_dir, model=other_model)
    assert other_model.training


def test_load_checkpoint_raises_on_missing_state_file(tmp_path: Path) -> None:
    model = _build_and_init(0)
    with pytest.raises(FileNotFoundError):
        checkpoint.load_checkpoint(str(tmp_path / "does-not-exist"), model=model)


def test_load_checkpoint_fails_loudly_on_architecture_mismatch(tmp_path: Path) -> None:
    """A shape mismatch (here: a different projector output_width) must raise from
    load_state_dict's own strict=True check, never silently load a partial/wrong state."""
    model_a = _build_and_init(0, _tiny_config(projector_output_width=32))
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model_a,
        optimizer=optimizer_a,
        step=1,
        generators={},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
    )

    model_b = _build_and_init(0, _tiny_config(projector_output_width=16))
    with pytest.raises(RuntimeError):
        checkpoint.load_checkpoint(ckpt_dir, model=model_b)


def test_resolved_config_yaml_written_verbatim_is_byte_identical(tmp_path: Path) -> None:
    """CKPT-02's literal acceptance line: 'config.yaml byte-identical to the merge used at train
    start.' save_checkpoint must write resolved_config_yaml's own output verbatim, not
    re-serialize it through any second pass that could reorder keys or reformat floats."""
    config = _tiny_config()
    train_cfg = TrainConfig(n_steps=10, seed_pretrain=3)
    expected = checkpoint.resolved_config_yaml(config, train_cfg)

    model = _build_and_init(0, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=1,
        generators={},
        config_yaml=expected,
        meta={},
    )

    with open(os.path.join(ckpt_dir, checkpoint.CONFIG_FILENAME), encoding="utf-8") as fh:
        on_disk = fh.read()
    assert on_disk == expected

    loaded = checkpoint.load_checkpoint(ckpt_dir, model=model)
    assert loaded.config_yaml == expected


def test_jepa_config_from_yaml_reconstructs_a_model_matching_widths() -> None:
    config = _tiny_config(projector_output_width=48)
    train_cfg = TrainConfig(seed_pretrain=5)
    text = checkpoint.resolved_config_yaml(config, train_cfg)

    jepa_cfg = checkpoint.jepa_config_from_yaml(text)
    model = build_jepa(jepa_cfg, generator=generator(0, "handshake"))
    assert model.projector.output_width == 48
    assert model.predictor.width == 48


def test_train_config_from_yaml_recovers_seed_pretrain() -> None:
    config = _tiny_config()
    train_cfg = TrainConfig(seed_pretrain=9, n_steps=42)
    text = checkpoint.resolved_config_yaml(config, train_cfg)

    train_node = checkpoint.train_config_from_yaml(text)
    assert int(train_node.seed_pretrain) == 9
    assert int(train_node.n_steps) == 42


def test_meta_json_round_trips_a_provenance_dict(tmp_path: Path) -> None:
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")
    meta = {
        "winder_git_sha": "deadbeef",
        "manifest_sha256": "abc",
        "lead_stats_sha256": "def",
        "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=1,
        generators={},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta=meta,
    )
    loaded = checkpoint.load_checkpoint(ckpt_dir, model=model)
    assert loaded.meta == meta
    assert not math.isnan(1.0)  # sentinel: json.dump(default=float) must not fire on plain data


# =============================================================== the transport arm's sibling keys


def test_operator_state_round_trips_under_a_sibling_key(tmp_path: Path) -> None:
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    operator_a = FreeOperator(FreeOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    with torch.no_grad():
        operator_a.omega += 0.05  # move it off its init value so the round trip is non-vacuous

    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=1,
        generators={},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
        operator=operator_a,
    )

    operator_b = FreeOperator(FreeOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    loaded = checkpoint.load_checkpoint(ckpt_dir, model=model, operator=operator_b)
    assert loaded.step == 1
    torch.testing.assert_close(operator_a.omega, operator_b.omega)


def test_loading_operator_from_a_checkpoint_saved_without_one_raises(tmp_path: Path) -> None:
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=1,
        generators={},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
        # operator omitted -- a control-arm-style checkpoint
    )

    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    with pytest.raises(ValueError, match="operator_state_dict"):
        checkpoint.load_checkpoint(ckpt_dir, model=model, operator=operator)


def test_model_state_dict_alone_is_unaffected_by_saving_an_operator(tmp_path: Path) -> None:
    """A transport-arm checkpoint's model_state_dict must still load into a plain model with no
    operator knowledge at all -- e.g. winder.eval.probe's existing frozen-encoder eval path."""
    model_a = _build_and_init(0)
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=3e-4)
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    ckpt_dir = str(tmp_path / "ckpt")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model_a,
        optimizer=optimizer,
        step=1,
        generators={},
        config_yaml="jepa: {}\ntrain: {}\n",
        meta={},
        operator=operator,
    )

    model_b = _build_and_init(1)  # different init -- must be fully overwritten
    checkpoint.load_checkpoint(ckpt_dir, model=model_b)  # no operator= at all
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)


def test_resolved_config_yaml_arm_section_is_optional_and_round_trips() -> None:
    config = _tiny_config()
    train_cfg = TrainConfig(seed_pretrain=3)

    without_arm = checkpoint.resolved_config_yaml(config, train_cfg)
    assert "arm:" not in without_arm
    assert checkpoint.arm_config_from_yaml(without_arm) is None

    arm_cfg = ArmConfig(name="free_arm_run", operator_name="free", operator={"k0": 8})
    with_arm = checkpoint.resolved_config_yaml(config, train_cfg, arm_config=arm_cfg)
    assert "arm:" in with_arm

    arm_node = checkpoint.arm_config_from_yaml(with_arm)
    assert arm_node is not None
    assert arm_node.name == "free_arm_run"
    assert arm_node.operator_name == "free"
    assert arm_node.operator.k0 == 8


def test_arm_operator_full_spectrum_survives_the_config_yaml_round_trip() -> None:
    """S2-STEP3 test #1: a checkpoint's saved `"arm:"` section must carry a CUSTOM (k0, n_j,
    k_j) spectrum through YAML text and back out to a build-ready operator config, reproducing
    every field exactly. `k0` in particular is checked here because -- unlike `n_j`/`k_j`, which
    end up inside `HarmonicTransport`'s own buffers (`k_j`) and are recoverable from a loaded
    `state_dict` -- `k0` has NO tensor representation anywhere in the module (it is a plain
    Python `int` used only to slice `z` in `transport()`): it survives a checkpoint reload
    ENTIRELY through this config.yaml round trip, never through `state.pt`. A taper spectrum
    (non-flat k_j, arithmetic ladder rather than the M0-calibrated flat default) is used
    specifically so a bug that silently fell back to CyclicOperatorConfig's own schema defaults
    would be caught (a flat k_j round-tripping "successfully" back to the flat default would be a
    false pass)."""
    config = _tiny_config(projector_output_width=256)
    train_cfg = TrainConfig(seed_pretrain=0)
    k0, n_j, k_j = 4, [1, 2, 3, 4, 5, 6], [36, 30, 24, 18, 12, 6]  # sums to 126 -> dim 4+2*126=256
    arm_cfg = ArmConfig(
        name="cyclic_taper_seed0",
        operator_name="cyclic",
        seed=0,
        operator={"k0": k0, "n_j": n_j, "k_j": k_j},
    )

    config_yaml = checkpoint.resolved_config_yaml(config, train_cfg, arm_config=arm_cfg)
    arm_node = checkpoint.arm_config_from_yaml(config_yaml)
    assert arm_node is not None

    operator_config = resolve_operator_config(arm_node)
    assert int(operator_config.k0) == k0
    assert list(operator_config.n_j) == n_j
    assert list(operator_config.k_j) == k_j

    _schema_cls, operator_ctor = OPERATOR_REGISTRY[arm_node.operator_name]
    operator = operator_ctor(operator_config)
    assert isinstance(operator, CyclicOperator)
    assert operator.dimension == 256
    assert operator.k0 == k0  # the one field with no tensor representation in state_dict at all
    assert operator.k_j.tolist() == k_j

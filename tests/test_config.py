"""Tests for winder.config: ArmConfig/resolve_operator_config (pre-existing) and the config-diff
drift guard (`flatten_yaml`/`diff_yaml`/`assert_expected_config_diff`) `scripts/pretrain.py`'s own
startup check is built on (Phase P7).

Split the same way as `tests/test_eval_acceptance.py`: fast, always-run unit tests on hand-built
YAML strings (no I/O), plus skip-gated tests against the copied-in reference checkpoints when
present, checking the guard against KNOWN structure (FIN_seed0 vs. FIN_LAM0_seed0 differ in
exactly `train.lambda_trans`) before `scripts/pretrain.py` ever trusts it on a new run.
"""

from __future__ import annotations

import os

import pytest

from winder.config import assert_expected_config_diff, diff_yaml, flatten_yaml

# ================================================================================== flatten_yaml

_SMALL_YAML = """
jepa:
  n_tokens: 125
  encoder_name: conv_trunk
  predictor:
    n_layers: 4
train:
  lambda_trans: 1.0
  betas: [0.9, 0.95]
arm:
  name: cyclic_seed0
  operator:
    k0: 4
    n_j: [1, 2, 3]
"""


def test_flatten_yaml_produces_dotted_leaf_keys() -> None:
    flat = flatten_yaml(_SMALL_YAML)
    assert flat["jepa.n_tokens"] == 125
    assert flat["jepa.encoder_name"] == "conv_trunk"
    assert flat["jepa.predictor.n_layers"] == 4
    assert flat["train.lambda_trans"] == 1.0
    assert flat["arm.name"] == "cyclic_seed0"
    assert flat["arm.operator.k0"] == 4


def test_flatten_yaml_treats_lists_as_opaque_leaves_not_recursed_into() -> None:
    flat = flatten_yaml(_SMALL_YAML)
    assert flat["train.betas"] == [0.9, 0.95]
    assert flat["arm.operator.n_j"] == [1, 2, 3]
    assert "train.betas.0" not in flat
    assert "arm.operator.n_j.0" not in flat


# ====================================================================================== diff_yaml


def test_diff_yaml_is_empty_for_identical_configs() -> None:
    assert diff_yaml(_SMALL_YAML, _SMALL_YAML) == {}


def test_diff_yaml_reports_only_changed_leaves() -> None:
    candidate = _SMALL_YAML.replace("lambda_trans: 1.0", "lambda_trans: 0.0")
    diff = diff_yaml(_SMALL_YAML, candidate)
    assert diff == {"train.lambda_trans": (1.0, 0.0)}


def test_diff_yaml_reports_a_leaf_present_on_only_one_side() -> None:
    candidate = _SMALL_YAML + "  extra_field: 7\n"  # 2-space indent: a new child of "arm:"
    diff = diff_yaml(_SMALL_YAML, candidate)
    assert diff["arm.extra_field"] == (None, 7)


# ==================================================================== assert_expected_config_diff


def test_assert_expected_config_diff_passes_when_diff_matches_exactly() -> None:
    candidate = _SMALL_YAML.replace("lambda_trans: 1.0", "lambda_trans: 0.0")
    result = assert_expected_config_diff(_SMALL_YAML, candidate, {"train.lambda_trans": 0.0})
    assert result == {"train.lambda_trans": (1.0, 0.0)}


def test_assert_expected_config_diff_raises_on_an_unwired_flag() -> None:
    """A caller predicts a change that never reached the resolved config -- e.g. `--lambda-trans
    0.0` failing to reach `TrainConfig` -- must fail loud, not silently pass with a narrower diff
    than expected."""
    with pytest.raises(ValueError, match="UNWIRED"):
        assert_expected_config_diff(
            _SMALL_YAML,
            _SMALL_YAML,
            {"train.lambda_trans": 0.0},  # nothing actually changed
        )


def test_assert_expected_config_diff_raises_on_unexplained_drift() -> None:
    """The resolved config changed somewhere no CLI flag/decision names -- e.g. a hardcoded
    default moved -- must fail loud rather than silently widen what counts as expected."""
    candidate = _SMALL_YAML.replace("encoder_name: conv_trunk", "encoder_name: patch")
    with pytest.raises(ValueError, match="DRIFT"):
        assert_expected_config_diff(_SMALL_YAML, candidate, {})


def test_assert_expected_config_diff_raises_on_a_mismatched_predicted_value() -> None:
    """The predicted key did change, but not to the value the caller's own expected_diff named --
    a caller bug in the expected_diff computation itself, distinct from both other categories."""
    candidate = _SMALL_YAML.replace("lambda_trans: 1.0", "lambda_trans: 0.0")
    with pytest.raises(ValueError, match="MISMATCHED"):
        assert_expected_config_diff(_SMALL_YAML, candidate, {"train.lambda_trans": 0.5})


def test_assert_expected_config_diff_returns_the_diff_dict_on_success() -> None:
    candidate = _SMALL_YAML.replace("n_layers: 4", "n_layers: 1")
    result = assert_expected_config_diff(_SMALL_YAML, candidate, {"jepa.predictor.n_layers": 1})
    assert result == {"jepa.predictor.n_layers": (4, 1)}


# ========================================================== skip-gated: real reference checkpoints

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFERENCE_ROOT = os.path.join(_REPO_ROOT, "artifacts", "reference")
_FIN_SEED0_CONFIG = os.path.join(_REFERENCE_ROOT, "FIN_seed0", "checkpoint_step5000", "config.yaml")
_FIN_LAM0_SEED0_CONFIG = os.path.join(
    _REFERENCE_ROOT, "FIN_LAM0_seed0", "checkpoint_step5000", "config.yaml"
)
_FIN_SEED1_CONFIG = os.path.join(_REFERENCE_ROOT, "FIN_seed1", "checkpoint_step5000", "config.yaml")
_HAS_REFERENCE_CONFIGS = all(
    os.path.isfile(p) for p in (_FIN_SEED0_CONFIG, _FIN_LAM0_SEED0_CONFIG, _FIN_SEED1_CONFIG)
)


@pytest.mark.skipif(not _HAS_REFERENCE_CONFIGS, reason=f"{_REFERENCE_ROOT} checkpoints not found")
def test_fin_lam0_seed0_differs_from_fin_seed0_in_exactly_lambda_trans() -> None:
    """The known-correct control/signal pair at seed 0 (Phase P6 Tier 0's own structural-parity
    fixture): reproduces `scripts/pretrain.py`'s own docstring claim that this pair differs in
    exactly `train.lambda_trans`, read directly from the checkpoints rather than trusted by
    description."""
    with open(_FIN_SEED0_CONFIG, encoding="utf-8") as fh:
        signal_yaml = fh.read()
    with open(_FIN_LAM0_SEED0_CONFIG, encoding="utf-8") as fh:
        control_yaml = fh.read()
    diff = diff_yaml(signal_yaml, control_yaml)
    assert diff == {"train.lambda_trans": (1.0, 0.0)}


@pytest.mark.skipif(not _HAS_REFERENCE_CONFIGS, reason=f"{_REFERENCE_ROOT} checkpoints not found")
def test_fin_seed1_differs_from_fin_seed0_without_touching_jepa_seed_pretrain() -> None:
    """The true (four-field, not the build plan's claimed five) allowed-diff set: seed 1 changes
    `train.seed_pretrain`/`arm.seed`/`arm.name`, but leaves `jepa.seed_pretrain` at the reference's
    own default -- `scripts/pretrain.py`'s docstring names exactly this as the reason its own
    `_expected_config_diff` never predicts a `jepa.seed_pretrain` change."""
    with open(_FIN_SEED0_CONFIG, encoding="utf-8") as fh:
        seed0_yaml = fh.read()
    with open(_FIN_SEED1_CONFIG, encoding="utf-8") as fh:
        seed1_yaml = fh.read()
    diff = diff_yaml(seed0_yaml, seed1_yaml)
    assert diff == {
        "train.seed_pretrain": (0, 1),
        "arm.seed": (0, 1),
        "arm.name": ("cyclic_seed0", "cyclic_seed1"),
    }
    assert "jepa.seed_pretrain" not in diff

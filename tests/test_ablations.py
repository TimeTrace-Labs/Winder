"""Tests for winder.ablations: the named-arm registry (Phase P7).

The load-bearing check is the P7 gate's own stated criterion (build plan's phase table): "ablation
registry resolves signal/control to the same flags as the hardcoded launch spec" -- compared as
normalized `{flag: value}` dicts (`parse_flag_pairs`), never by eyeballing two argv lists, so a
reordering can't hide a real drift and can't manufacture a false failure either.
"""

from __future__ import annotations

import pytest

from winder.ablations import ABLATION_ARMS, parse_flag_pairs, resolve_arm

# The build plan's own hardcoded P8 launch line (verbatim flags, LT/NAME/SEED substituted per
# row of its launch table) -- the ground truth `resolve_arm`'s output is checked against.
_LAUNCH_LINE_COMMON = {
    "--transport-arm": "cyclic",
    "--batch-size": "64",
    "--steps": "30000",
    "--device": "cuda",
    "--train-folds": "1,2,3,4,5,6,7,8,9",
    "--lead-stats-path": "artifacts/lead_stats_f1to9.json",
    "--manifest-path": "artifacts/manifest.parquet",
    "--theta-tokens-path": "artifacts/phase/theta_tokens.npz",
    "--lambda-sig": "0.15",
    "--checkpoint-at": "2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500",
    "--k0": "4",
    "--n-j": "1,2,3,4,5,6,7,8,9,10",
    "--k-j": "24,24,20,16,12,10,8,6,4,2",
    "--encoder-name": "conv_trunk",
    "--predictor-json": '{"n_layers":4}',
    "--augment": "gauss,powerline,wander,ampmod,leaddrop,leadgain",
    "--augment-prob": "0.5",
}


def _launch_line_flags(*, lambda_trans: str, seed: str, artifacts_dir: str) -> dict[str, str]:
    return {
        **_LAUNCH_LINE_COMMON,
        "--lambda-trans": lambda_trans,
        "--seed": seed,
        "--artifacts-dir": artifacts_dir,
    }


# ================================================================================ parse_flag_pairs


def test_parse_flag_pairs_round_trips_a_simple_argv_list() -> None:
    assert parse_flag_pairs(["--a", "1", "--b", "2"]) == {"--a": "1", "--b": "2"}


def test_parse_flag_pairs_raises_on_odd_length() -> None:
    with pytest.raises(ValueError, match="even-length"):
        parse_flag_pairs(["--a", "1", "--b"])


def test_parse_flag_pairs_raises_when_a_flag_position_holds_a_bare_value() -> None:
    with pytest.raises(ValueError, match="expected a --flag"):
        parse_flag_pairs(["--a", "1", "not-a-flag", "2"])


# ===================================================================================== resolve_arm


def test_resolve_arm_signal_matches_the_p8_launch_line_at_seed0() -> None:
    resolved = parse_flag_pairs(resolve_arm("signal", seed=0, artifacts_dir="artifacts/roster/x"))
    expected = _launch_line_flags(lambda_trans="1.0", seed="0", artifacts_dir="artifacts/roster/x")
    assert resolved == expected


def test_resolve_arm_control_matches_the_p8_launch_line_at_seed0() -> None:
    resolved = parse_flag_pairs(resolve_arm("control", seed=0, artifacts_dir="artifacts/roster/y"))
    expected = _launch_line_flags(lambda_trans="0.0", seed="0", artifacts_dir="artifacts/roster/y")
    assert resolved == expected


def test_resolve_arm_matches_the_p8_launch_line_at_seed1_too() -> None:
    resolved = parse_flag_pairs(resolve_arm("signal", seed=1, artifacts_dir="artifacts/roster/z"))
    expected = _launch_line_flags(lambda_trans="1.0", seed="1", artifacts_dir="artifacts/roster/z")
    assert resolved == expected


def test_resolve_arm_signal_and_control_differ_only_in_lambda_trans() -> None:
    signal = parse_flag_pairs(resolve_arm("signal", seed=0, artifacts_dir="a"))
    control = parse_flag_pairs(resolve_arm("control", seed=0, artifacts_dir="a"))
    differing = {k for k in signal if signal[k] != control.get(k)}
    assert differing == {"--lambda-trans"}


def test_resolve_arm_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="unknown ablation arm"):
        resolve_arm("not_a_real_arm", seed=0, artifacts_dir="x")


# ============================================================ documented-not-launched arm entries


def test_documented_arms_override_exactly_their_own_named_factor() -> None:
    """`no_augmentation`/`no_sigreg`/`shallow_predictor` must each change exactly ONE flag
    relative to `signal` -- if the ordering fix in `resolve_arm`'s own docstring ever regressed
    (common flags re-clobbering the arm's override), this would show up as the override silently
    reverting to the common recipe's own value."""
    signal = parse_flag_pairs(resolve_arm("signal", seed=0, artifacts_dir="a"))

    no_aug = parse_flag_pairs(resolve_arm("no_augmentation", seed=0, artifacts_dir="a"))
    assert {k for k in signal if signal[k] != no_aug.get(k)} == {"--augment"}
    assert no_aug["--augment"] == ""

    no_sigreg = parse_flag_pairs(resolve_arm("no_sigreg", seed=0, artifacts_dir="a"))
    assert {k for k in signal if signal[k] != no_sigreg.get(k)} == {"--lambda-sig"}
    assert no_sigreg["--lambda-sig"] == "0.0"

    shallow = parse_flag_pairs(resolve_arm("shallow_predictor", seed=0, artifacts_dir="a"))
    assert {k for k in signal if signal[k] != shallow.get(k)} == {"--predictor-json"}
    assert shallow["--predictor-json"] == '{"n_layers":1}'


def test_ablation_arms_registry_names_exactly_the_five_documented_entries() -> None:
    assert set(ABLATION_ARMS) == {
        "signal",
        "control",
        "no_augmentation",
        "no_sigreg",
        "shallow_predictor",
    }

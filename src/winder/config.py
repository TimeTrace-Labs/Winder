"""Config loading: plain-dataclass schemas merged via OmegaConf.

Pydantic is deliberately not used here -- OmegaConf.structured() introspects plain
dataclasses/attrs into its own container types; a pydantic.BaseModel is not a supported input,
and its eager validation fights OmegaConf's interpolation resolution. Pydantic's place in this
project is validating run artifacts (record.json, calibration thresholds) written by one run and
read by a possibly-much-later one -- a genuinely different boundary, not this one.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from omegaconf import MISSING, DictConfig, OmegaConf

from winder.operators.registry import OPERATOR_REGISTRY


@dataclass
class ArmConfig:
    name: str = MISSING
    seed: int = 0
    operator_name: str = MISSING
    operator: dict[str, Any] = field(default_factory=dict)


def load_arm_config(*sources: str | dict[str, Any]) -> DictConfig:
    """Merge YAML paths / dicts, in order, onto ArmConfig's schema."""
    merged = OmegaConf.structured(ArmConfig)
    for source in sources:
        overlay = OmegaConf.load(source) if isinstance(source, str) else OmegaConf.create(source)
        merged = OmegaConf.merge(merged, overlay)
    return cast(DictConfig, merged)


def resolve_operator_config(arm_config: ArmConfig | DictConfig) -> DictConfig:
    """Merge arm_config.operator onto its tagged schema from OPERATOR_REGISTRY."""
    schema_cls, _ = OPERATOR_REGISTRY[arm_config.operator_name]
    return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(schema_cls), arm_config.operator))


def flatten_yaml(config_yaml: str) -> dict[str, Any]:
    """A `winder.jepa.checkpoint.resolved_config_yaml` string -> `{"jepa.encoder_name": ...,
    "arm.operator.k_j": [24, ...], ...}`: one entry per LEAF value, dotted-path keyed.

    Lists/tuples (e.g. `arm.operator.n_j`, `train.betas`) are leaves in their own right, never
    recursed into (`arm.operator.n_j` compares as one list, not `n_j.0`/`n_j.1`/...) -- the
    quantity a caller cares about is "did this whole spectrum change", not "did element 3 change".
    Only dict-like nodes (`DictConfig`, plain `dict`) are recursed into. Used by `diff_yaml`/
    `assert_expected_config_diff` to compare two resolved configs field-by-field regardless of
    which top-level sections (`jepa:`/`train:`/`arm:`) either one carries."""
    container = OmegaConf.to_container(OmegaConf.create(config_yaml), resolve=True)
    flat: dict[str, Any] = {}

    def _walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{prefix}{key}.")
        else:
            flat[prefix.rstrip(".")] = node

    _walk(container, "")
    return flat


def diff_yaml(reference_yaml: str, candidate_yaml: str) -> dict[str, tuple[Any, Any]]:
    """`{dotted.path: (reference_value, candidate_value)}` for every leaf where the two configs
    disagree, including a leaf present in only one of the two (the other side reads as `None` via
    `dict.get`, matching OmegaConf's own missing-key convention -- a caller wanting to distinguish
    "explicitly None" from "absent" should flatten and diff by hand). Order-independent: keys are
    read from the union of both flattened configs."""
    ref_flat = flatten_yaml(reference_yaml)
    cand_flat = flatten_yaml(candidate_yaml)
    diff: dict[str, tuple[Any, Any]] = {}
    for key in ref_flat.keys() | cand_flat.keys():
        ref_val = ref_flat.get(key)
        cand_val = cand_flat.get(key)
        if ref_val != cand_val:
            diff[key] = (ref_val, cand_val)
    return diff


def assert_expected_config_diff(
    reference_yaml: str, candidate_yaml: str, expected_diff: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Diffs `candidate_yaml` against `reference_yaml` (`diff_yaml`) and asserts the result is
    EXACTLY explained by `expected_diff` (`{dotted.path: candidate's own intended value}`, one
    entry per CLI flag or hardcoded construction choice the caller believes changes the resolved
    config relative to the reference) -- catching two distinct failure modes with one check:

      * an UNWIRED flag: `expected_diff` predicts a change that never reached the resolved
        config (e.g. `--lambda-trans 0.0` failing to reach `TrainConfig` would leave
        `train.lambda_trans` unchanged, so a "control" run would silently train as a "signal" one);
      * DRIFT: the resolved config differs from the reference somewhere `expected_diff` does not
        name at all (e.g. a hardcoded default changing without a matching CLI flag to explain it).

    A mismatch (the candidate side of an expected key doesn't match what was actually resolved)
    is reported as its own category, distinct from both of the above, since it usually means the
    caller's own `expected_diff` computation used a different value than the code path that
    actually built `candidate_yaml`.

    Returns the actual diff dict (for logging/provenance) when it matches exactly; raises
    `ValueError` naming every offending key by category otherwise -- never silently widens
    `expected_diff` to force a pass."""
    actual = diff_yaml(reference_yaml, candidate_yaml)
    actual_keys = set(actual)
    expected_keys = set(expected_diff)

    unwired = expected_keys - actual_keys
    drift = actual_keys - expected_keys
    mismatched = {
        key: (actual[key][1], expected_diff[key])
        for key in actual_keys & expected_keys
        if actual[key][1] != expected_diff[key]
    }
    if unwired or drift or mismatched:
        lines = [
            "resolved config.yaml does not match its expected diff against the reference recipe:"
        ]
        if unwired:
            lines.append(
                f"  UNWIRED (expected to change, but didn't reach the resolved config): "
                f"{sorted(unwired)}"
            )
        if drift:
            lines.append(
                f"  DRIFT (changed, but no CLI flag/decision explains it): "
                f"{ {k: actual[k] for k in sorted(drift)} }"
            )
        if mismatched:
            lines.append(
                f"  MISMATCHED (changed, but not to the predicted value; (actual, expected)): "
                f"{ {k: mismatched[k] for k in sorted(mismatched)} }"
            )
        raise ValueError("\n".join(lines))
    return actual

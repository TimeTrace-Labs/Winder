"""CKPT-01/02/03: full training-state checkpoints for `winder.jepa.train.fit`.

A checkpoint bundle is one directory containing three files:

    <checkpoint_dir>/
        state.pt     `model.state_dict()`, `optimizer.state_dict()`, the integer global step
                     (steps *completed*, so `start_step=` this value resumes correctly), and a
                     `{name: torch.Generator.get_state()}` dict for every named generator stream
                     the caller hands to `save_checkpoint` (CKPT-01) -- e.g. `"mask"`,
                     `"sigreg"`, `"data_order"`. Loaded with `weights_only=False`: this is a
                     first-party artifact this project only ever writes and reads itself, never
                     an untrusted download, and optimizer state (AdamW's per-parameter step/
                     exp_avg/exp_avg_sq) plus this dict-of-generator-states is not guaranteed to
                     clear torch's `weights_only` unpickling allowlist across torch versions.
        config.yaml  the fully-resolved `JepaConfig` (`"jepa:"`) and `TrainConfig` (`"train:"`)
                     that built this run, as OmegaConf YAML text supplied by the caller
                     (`resolved_config_yaml`) -- written verbatim, never re-serialized here, so
                     "byte-identical to the merge used at train start" (CKPT-02) holds as long as
                     the caller snapshots that string *before* `fit()` runs, not after.
        meta.json    data provenance the caller assembles: manifest/lead_stats SHA-256, fold
                     list, git SHA (CKPT-03). This module has no opinion on what a "manifest" or
                     "lead_stats" is -- it only writes whatever `dict` it is given.

`SeededDropout`'s own per-instance generator is deliberately NOT part of the `generators` dict
callers pass to `save_checkpoint`: it already rides inside `model.state_dict()` via its own
`get_extra_state`/`set_extra_state` hook (see that module's docstring) -- naming it again here
would double-save the same bits under two keys, and `load_checkpoint` restoring it a second time
via a generic `"dropout"` entry would silently clobber whatever `load_state_dict` already set.

`load_checkpoint` never calls `model.eval()`/`model.train()`. `SeededDropout` only advances its
own generator in training mode; a loader that forced eval mode would silently make a *resumed
training* step replay without dropout while its state still says otherwise, diverging from the
uninterrupted reference (CKPT-04). Callers evaluating a frozen checkpoint (s3) must call
`model.eval()` themselves after `load_checkpoint` returns.

The transport arm's operator (`winder.operators.harmonic.HarmonicTransport`) is a SEPARATE
`nn.Module`, deliberately not a submodule of `model` (`winder.jepa.model`'s own docstring: the
operator/JEPA decoupling is preserved by construction). `save_checkpoint`/`load_checkpoint`'s
optional `operator=` argument saves/restores its `state_dict()` under a SIBLING top-level key
(`"operator_state_dict"`), never inside `model_state_dict` -- a control-arm checkpoint (no
operator) stays loadable into a plain `JepaModel` exactly as before this feature existed, and a
transport-arm checkpoint's `model_state_dict` alone still loads into that same plain model (e.g.
for `winder.eval.probe`'s existing frozen-encoder eval surfaces, which have no use for the
operator at all). `resolved_config_yaml`'s optional `arm_config=` similarly adds a SIBLING
`"arm:"` YAML section (reusing `winder.config.ArmConfig`'s existing operator-name-plus-params
schema rather than inventing a second one) -- `arm_config_from_yaml` returns `None` on a
checkpoint with no such section, unlike `jepa_config_from_yaml`/`train_config_from_yaml`, which
raise: the jepa/train sections are load-bearing for every checkpoint this module has ever
written, but "arm" is genuinely optional and every pre-existing checkpoint lacks it.

Caveat on `"data_order"`-style generator streams (real `DataLoader` shuffling, not this project's
own pure-generator synthetic/test batch streams): restoring a saved generator's state reproduces
the *next full epoch's* shuffle order, not "batch K of the epoch already in progress at save
time" -- `DataLoader(shuffle=True)` consumes its generator once per `__iter__` call to build one
whole-epoch permutation, not once per batch. Saving this state is still exactly what CKPT-01's
mission asks for and is exactly restorable at epoch granularity; it does not, by itself, make a
mid-epoch resume of a real `DataLoader` bitwise-identical to never having stopped. The
exact-resume test (CKPT-04) proves the stronger, batch-granular claim on a pure generator-driven
stream (this project's own synthetic/test data path), where no such caveat applies.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from winder.config import ArmConfig
from winder.jepa.model import JepaConfig
from winder.jepa.train import TrainConfig

__all__ = [
    "STATE_FILENAME",
    "CONFIG_FILENAME",
    "META_FILENAME",
    "LoadedCheckpoint",
    "resolved_config_yaml",
    "jepa_config_from_yaml",
    "train_config_from_yaml",
    "arm_config_from_yaml",
    "save_checkpoint",
    "load_checkpoint",
]

STATE_FILENAME = "state.pt"
CONFIG_FILENAME = "config.yaml"
META_FILENAME = "meta.json"


def resolved_config_yaml(
    jepa_config: "JepaConfig | DictConfig",
    train_config: TrainConfig,
    arm_config: "ArmConfig | DictConfig | None" = None,
) -> str:
    """OmegaConf YAML text of `{"jepa": <resolved JepaConfig>, "train": <resolved TrainConfig>}`,
    plus an optional sibling `"arm": <resolved ArmConfig>` section when `arm_config` is given --
    CKPT-02's config snapshot. Call this BEFORE `fit()` runs and hold the returned string for
    `save_checkpoint`'s `config_yaml=` argument: that is what makes "byte-identical to the merge
    used at train start" true by construction, rather than an assumption that nothing between
    train start and checkpoint time mutated either config object."""
    jepa_node = (
        jepa_config if isinstance(jepa_config, DictConfig) else OmegaConf.structured(jepa_config)
    )
    train_node = OmegaConf.structured(train_config)
    merged_dict: dict[str, Any] = {"jepa": jepa_node, "train": train_node}
    if arm_config is not None:
        merged_dict["arm"] = (
            arm_config if isinstance(arm_config, DictConfig) else OmegaConf.structured(arm_config)
        )
    merged = OmegaConf.create(merged_dict)
    return str(OmegaConf.to_yaml(merged))


def jepa_config_from_yaml(config_yaml: str) -> DictConfig:
    """Extracts the `"jepa:"` section of a `resolved_config_yaml`/`save_checkpoint` config.yaml
    and merges it onto `JepaConfig`'s own schema -- the sanctioned way to reconstruct a
    checkpoint's exact architecture (mirrors `winder.jepa.model.load_jepa_config`'s own merge
    recipe), rather than guessing at a config a caller happens to have lying around."""
    full = OmegaConf.create(config_yaml)
    return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(JepaConfig), full.jepa))


def train_config_from_yaml(config_yaml: str) -> DictConfig:
    """Extracts the `"train:"` section of a `resolved_config_yaml`/`save_checkpoint` config.yaml
    and merges it onto `TrainConfig`'s own schema -- e.g. so a reader can recover the exact
    `seed_pretrain` a checkpoint's pretraining run used."""
    full = OmegaConf.create(config_yaml)
    return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(TrainConfig), full.train))


def arm_config_from_yaml(config_yaml: str) -> DictConfig | None:
    """Extracts the OPTIONAL `"arm:"` section of a `resolved_config_yaml`/`save_checkpoint`
    config.yaml and merges it onto `winder.config.ArmConfig`'s own schema -- `None` if this
    checkpoint has no such section (every checkpoint saved before the transport arm existed,
    or any control-arm run that never passed `arm_config=` to `resolved_config_yaml`). Unlike
    `jepa_config_from_yaml`/`train_config_from_yaml`, which raise on a missing section (both are
    load-bearing for every checkpoint this module has ever written), this section is genuinely
    optional -- see this module's own docstring."""
    full = OmegaConf.create(config_yaml)
    if "arm" not in full:
        return None
    return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(ArmConfig), full.arm))


def save_checkpoint(
    checkpoint_dir: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    generators: Mapping[str, torch.Generator],
    config_yaml: str,
    meta: Mapping[str, Any],
    operator: nn.Module | None = None,
) -> str:
    """Writes `state.pt`/`config.yaml`/`meta.json` under `checkpoint_dir` (created if missing).
    `step` is the number of steps *completed* (not the last step's index) -- a resumed `fit()`
    call reads this back as its own `start_step`. `operator`, if given, is saved under a SIBLING
    `"operator_state_dict"` key -- never inside `model_state_dict` (module docstring). Returns
    `checkpoint_dir`."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    state: dict[str, Any] = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "generator_states": {name: gen.get_state() for name, gen in generators.items()},
    }
    if operator is not None:
        state["operator_state_dict"] = operator.state_dict()
    torch.save(state, os.path.join(checkpoint_dir, STATE_FILENAME))

    with open(os.path.join(checkpoint_dir, CONFIG_FILENAME), "w", encoding="utf-8") as fh:
        fh.write(config_yaml)

    with open(os.path.join(checkpoint_dir, META_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(dict(meta), fh, indent=2, default=float)

    return checkpoint_dir


@dataclass
class LoadedCheckpoint:
    step: int
    generator_states: dict[str, torch.Tensor]
    config_yaml: str
    meta: dict[str, Any]


def load_checkpoint(
    checkpoint_dir: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    operator: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Loads `state.pt` into `model` (and `optimizer`/`operator`, if given) in place via
    `load_state_dict` (`strict=True`, the default -- an architecture mismatch raises immediately
    rather than silently loading a partial/wrong-shaped state), and returns the global step,
    every saved generator's raw state tensor (apply via `torch.Generator.set_state`), and the two
    provenance files' raw text/dict. Does not call `model.eval()`/`model.train()` -- see this
    module's docstring.

    Passing `operator=` against a checkpoint saved WITHOUT one raises immediately (named-field,
    actionable) rather than silently leaving `operator` at its own construction-time init --
    that would be a checkpoint claiming to restore a trained operator while actually not."""
    state_path = os.path.join(checkpoint_dir, STATE_FILENAME)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"{state_path} not found -- not a winder.jepa.checkpoint bundle")
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if operator is not None:
        if "operator_state_dict" not in state:
            raise ValueError(
                f"{checkpoint_dir} has no saved operator_state_dict, but an operator was passed "
                f"to load into -- this checkpoint was saved without a transport arm "
                f"(save_checkpoint's own operator= argument was omitted or None)."
            )
        operator.load_state_dict(state["operator_state_dict"])

    config_path = os.path.join(checkpoint_dir, CONFIG_FILENAME)
    with open(config_path, encoding="utf-8") as fh:
        config_yaml = fh.read()

    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    return LoadedCheckpoint(
        step=int(state["step"]),
        generator_states=dict(state["generator_states"]),
        config_yaml=config_yaml,
        meta=meta,
    )

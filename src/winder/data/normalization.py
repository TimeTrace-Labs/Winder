"""Amplitude normalisation strategies: raw (no-op) vs. per-beat RMS.

Relocated from ttl-phase's `src/model/psi0.py` (`beat_rms_per_token`), where this data-prep
function lived in the model package despite being applied entirely within the data
pipeline (`scripts/s3_train.py::build_pool`) -- a data-layer concern historically
mislocated upstream. `build_pool` called it with `centres=np.arange(T)`, i.e. per-*sample*,
not per-token, so relocating it here changes nothing about what it computes.

Ported the shared-scalar variant specifically -- the one `build_pool` actually applies: one
RMS scalar per beat, shared across all 12 leads, so inter-lead ratios and QRS morphology
survive but absolute mV scale does not. `phase.py`'s deferred `beat_matrix` function has a
SECOND, incompatible "perbeat" definition (per-lead RMS) that is NOT ported here and must
never be conflated with this one under the same name -- they have different clinical
consequences (per-lead normalisation additionally destroys inter-lead ratios).

`NormConfig.mode` is a REQUIRED field (OmegaConf `MISSING`, no default) -- not because
`raw` or `perbeat` is wrong, but because of a measured finding: the shared-scalar `perbeat`
convention drops an LVH-criterion (Sokolow-Lyon) discriminative power from AUC 0.796 to
0.577 by destroying absolute voltage, on a downstream classification task winder does not
train yet. The fix is "never let this default silently," not "pick a winner now."

Follows winder's established tag+registry pattern (see `operators/registry.py`): OmegaConf
structured configs can't express `Union[RawNormConfig, PerBeatNormConfig]` directly (unions
of containers are unsupported), so `NormConfig` carries a string tag resolved through
`NORM_REGISTRY`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from omegaconf import MISSING, DictConfig, OmegaConf

__all__ = [
    "RawNormConfig",
    "PerBeatNormConfig",
    "CorpusStatsNormConfig",
    "NormConfig",
    "NORM_REGISTRY",
    "beat_rms",
    "apply_raw",
    "apply_perbeat",
    "apply_corpus_stats",
    "normalize",
    "resolve_norm_config",
]


def beat_rms(sig: np.ndarray, rpeaks: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """RMS of the beat containing each sample (or token) centre, across all leads --
    ONE scalar per beat, shared across leads (not per-lead; see module docstring).

    A centre before the first or after the last R-peak gets the RMS of the nearest beat;
    those positions carry `bin_id = BIN_EXCLUDE` and are dropped downstream anyway, so the
    choice is inconsequential but is made explicit rather than left to a NaN.

    `sig` and `rpeaks` must be in the same time-index units (both at 100 Hz, both at
    500 Hz, ...) -- this function is rate-agnostic.

    `sig` must be time-major 2-D, shape (T, n_leads). Unlike `phase._as_time_major`, this
    function does NOT auto-transpose a lead-major array or accept a 1-D single-lead
    signal -- a caller passing either used to get a silently wrong result (found by
    audit: a (T,) signal broadcasts against an (T,1) divisor into a (T,T) array with no
    error) rather than a shape error. Reshape/transpose before calling if needed.
    """
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (T, n_leads) time-major, got shape {sig.shape}")
    if len(rpeaks) < 2:
        return np.full(len(centres), np.sqrt((sig**2).mean()), dtype=np.float64)
    edges = np.asarray(rpeaks, dtype=np.float64)
    rms = np.empty(len(edges) - 1, dtype=np.float64)
    for i in range(len(edges) - 1):
        a, b = int(np.floor(edges[i])), int(np.ceil(edges[i + 1]))
        a, b = max(a, 0), min(b, sig.shape[0])
        seg = sig[a:b]
        rms[i] = np.sqrt((seg**2).mean()) if seg.size else np.nan
    if np.isnan(rms).any():
        good = np.nanmean(rms) if np.isfinite(np.nanmean(rms)) else 1.0
        rms = np.where(np.isnan(rms), good, rms)
    which = np.clip(np.searchsorted(edges, centres, side="right") - 1, 0, len(rms) - 1)
    return rms[which]


@dataclass
class RawNormConfig:
    """No-op: the signal passes through unchanged."""


@dataclass
class PerBeatNormConfig:
    scale_floor: float = 1e-9


def apply_raw(sig: np.ndarray, rpeaks: np.ndarray, config: RawNormConfig) -> np.ndarray:
    return sig


def apply_perbeat(sig: np.ndarray, rpeaks: np.ndarray, config: PerBeatNormConfig) -> np.ndarray:
    """Divide every sample by its beat's RMS (all leads sharing one scalar divisor).

    Matches `build_pool`'s exact floor semantics: below `scale_floor`, divide by 1.0 (a
    no-op for that sample), not by the floor value itself.
    """
    centres = np.arange(sig.shape[0], dtype=np.float64)
    scale = beat_rms(sig, rpeaks, centres)
    divisor = np.where(scale > config.scale_floor, scale, 1.0)
    return sig / divisor[:, None]


@dataclass
class CorpusStatsNormConfig:
    """Fitted per-lead training-set mean/std -- see `winder.data.norm_stats.LeadStats`, the
    pydantic cross-run artifact this config's numbers actually come from.

    `mean_mv`/`std_mv` are `MISSING`, for exactly `NormConfig.mode`'s reason above: an identity
    default (e.g. mean=0, std=1) would silently no-op a forgotten stats load instead of failing
    loudly. `LeadStats.to_norm_config()` is the one sanctioned bridge from the fitted artifact to
    this config -- nobody should hand-copy `mean_mv`/`std_mv` elsewhere.
    """

    mean_mv: list[float] = MISSING
    std_mv: list[float] = MISSING
    scale_floor: float = 1e-6


def apply_corpus_stats(
    sig: np.ndarray, rpeaks: np.ndarray, config: CorpusStatsNormConfig
) -> np.ndarray:
    """Per-lead z-score using CORPUS-level (not per-record) training-set statistics.

    `rpeaks` is accepted and ignored: this normalizer has no per-beat structure to consult.
    Honouring `NORM_REGISTRY`'s existing `(sig, rpeaks, config)` signature rather than
    special-casing it away is what keeps this a registry entry instead of a second, parallel
    code path.

    Per-record standardisation would destroy relative amplitude *between* records the same way
    `perbeat`'s per-beat RMS destroys absolute voltage within one (Sokolow-Lyon AUC 0.796 ->
    0.577, module docstring above): a single per-lead divisor shared across the whole corpus
    preserves both the absolute-scale relationship between records and the inter-lead ratios
    within one record, which a per-record statistic cannot.

    Floor semantics match `apply_perbeat` exactly: below `scale_floor`, divide by 1.0 (a no-op
    for that lead), not by the floor value itself.
    """
    mean = np.asarray(config.mean_mv, dtype=np.float64)
    std = np.asarray(config.std_mv, dtype=np.float64)
    if mean.shape[0] != sig.shape[1] or std.shape[0] != sig.shape[1]:
        raise ValueError(
            f"CorpusStatsNormConfig has {mean.shape[0]} mean_mv / {std.shape[0]} std_mv "
            f"entries but sig has {sig.shape[1]} leads; set mean_mv/std_mv to match, e.g. via "
            f"LeadStats.to_norm_config()"
        )
    divisor = np.where(std > config.scale_floor, std, 1.0)
    return (sig - mean) / divisor


#: name -> (config schema, apply function). See operators/registry.py for why this is a
#: tag+registry rather than a Union in the config schema.
NORM_REGISTRY: dict[str, tuple[type, Callable[[np.ndarray, np.ndarray, Any], np.ndarray]]] = {
    "raw": (RawNormConfig, apply_raw),
    "perbeat": (PerBeatNormConfig, apply_perbeat),
    "corpus_stats": (CorpusStatsNormConfig, apply_corpus_stats),
}


@dataclass
class NormConfig:
    mode: str = MISSING  # "raw" | "perbeat" -- required, resolved via NORM_REGISTRY
    params: dict[str, Any] = field(default_factory=dict)


def resolve_norm_config(norm_cfg: "NormConfig | DictConfig") -> DictConfig:
    """Merge `norm_cfg.params` onto its tagged schema from NORM_REGISTRY."""
    schema_cls, _ = NORM_REGISTRY[norm_cfg.mode]
    return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(schema_cls), norm_cfg.params))


def normalize(mode: str, sig: np.ndarray, rpeaks: np.ndarray, config: object) -> np.ndarray:
    _, apply_fn = NORM_REGISTRY[mode]
    return apply_fn(sig, rpeaks, config)

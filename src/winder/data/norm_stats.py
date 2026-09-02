"""Fitted per-lead normalization statistics: `LeadStats`, the pydantic cross-run artifact
`normalization.CorpusStatsNormConfig`'s numbers actually come from.

`LeadStats` is pydantic, not a plain dataclass -- this crosses winder's established pydantic
boundary (see `config.py`'s docstring, and `data/manifest.py::RecordRow`'s identical reasoning):
fit once by one run, read back by every later val/probe run, possibly hand-inspected or
version-drifted. `LeadStats.to_norm_config()` is the *only* sanctioned bridge from this artifact
into the dataclass config `apply_corpus_stats` actually consumes -- nobody should hand-copy
`mean_mv`/`std_mv` elsewhere.

`fit_lead_stats` accumulates plain (not Welford) running sums in float64: for this corpus's scale
(~18k records x 1000 samples x 12 leads, physical mV -- not raw ADU -- so no huge dynamic range),
a naive `E[X^2] - E[X]^2` in float64 is numerically ample; Welford's extra bookkeeping buys
stability this scale doesn't need.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from winder.data.folds import FoldConfig
from winder.data.normalization import CorpusStatsNormConfig
from winder.data.ptbxl import LEAD_ORDER

__all__ = ["LeadStats", "fit_lead_stats"]


class LeadStats(BaseModel):
    """One corpus's fitted per-lead mean/std, at a fixed sampling rate, over a fixed fold set.

    Units: millivolts (matches `wfdb_io.read_record`'s physical-unit output), fitted on the
    *decimated* signal at `fs` -- the rate the model actually consumes, not PTB-XL's native rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    leads: tuple[str, ...]
    mean_mv: tuple[float, ...]
    std_mv: tuple[float, ...]
    fs: int
    folds: tuple[int, ...]
    n_records: int
    n_samples: int
    winder_git_sha: str | None = None
    created_utc: str

    @field_validator("mean_mv")
    @classmethod
    def _mean_finite(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        bad = [x for x in v if not np.isfinite(x)]
        if bad:
            raise ValueError(f"mean_mv must be finite for every lead, got non-finite {bad}")
        return v

    @field_validator("std_mv")
    @classmethod
    def _std_finite_and_positive(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        bad = [x for x in v if not (np.isfinite(x) and x > 0)]
        if bad:
            raise ValueError(f"std_mv must be finite and > 0 for every lead, got {bad}")
        return v

    @field_validator("folds")
    @classmethod
    def _folds_subset_of_train_folds(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        train_folds = set(FoldConfig().train_folds)
        leaked = set(v) - train_folds
        if leaked:
            raise ValueError(
                f"folds must be a subset of the training folds {sorted(train_folds)}; got "
                f"{sorted(leaked)}, which would leak the validation fold (9) or the sealed test "
                f"fold (10) into the fitted normalization statistics"
            )
        return v

    @model_validator(mode="after")
    def _lengths_and_lead_order_match(self) -> "LeadStats":
        if not (len(self.leads) == len(self.mean_mv) == len(self.std_mv)):
            raise ValueError(
                f"leads/mean_mv/std_mv must have matching lengths, got "
                f"{len(self.leads)}/{len(self.mean_mv)}/{len(self.std_mv)}"
            )
        if self.leads != LEAD_ORDER:
            raise ValueError(f"leads must equal ptbxl.LEAD_ORDER exactly, got {self.leads}")
        return self

    def to_norm_config(self) -> CorpusStatsNormConfig:
        """The one sanctioned bridge from this fitted artifact to the dataclass config
        `apply_corpus_stats` consumes."""
        return CorpusStatsNormConfig(mean_mv=list(self.mean_mv), std_mv=list(self.std_mv))

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def from_json(cls, path: str) -> "LeadStats":
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())


def fit_lead_stats(
    records: Iterable[np.ndarray],
    *,
    leads: Sequence[str] = LEAD_ORDER,
    fs: int,
    folds: Sequence[int],
    winder_git_sha: str | None = None,
) -> LeadStats:
    """Streaming float64 per-lead mean/std over `records`, each `(n_samples, len(leads))`.

    Raises if `records` is empty -- a fit with zero records is a caller bug (an over-restrictive
    filter, an empty split), not a legitimate all-zero/all-one result to silently produce.
    """
    n_leads = len(leads)
    total = np.zeros(n_leads, dtype=np.float64)
    total_sq = np.zeros(n_leads, dtype=np.float64)
    n_samples_seen = 0
    n_records = 0
    for sig in records:
        if sig.ndim != 2 or sig.shape[1] != n_leads:
            raise ValueError(f"expected each record to be (n_samples, {n_leads}), got {sig.shape}")
        x = sig.astype(np.float64)
        total += x.sum(axis=0)
        total_sq += (x * x).sum(axis=0)
        n_samples_seen += x.shape[0]
        n_records += 1
    if n_records == 0:
        raise ValueError("fit_lead_stats received no records -- refusing to fit on an empty set")

    mean = total / n_samples_seen
    var = np.maximum(total_sq / n_samples_seen - mean * mean, 0.0)  # guards float roundoff only
    std = np.sqrt(var)

    return LeadStats(
        leads=tuple(leads),
        mean_mv=tuple(float(m) for m in mean),
        std_mv=tuple(float(s) for s in std),
        fs=fs,
        folds=tuple(sorted(set(folds))),
        n_records=n_records,
        n_samples=n_samples_seen,
        winder_git_sha=winder_git_sha,
        created_utc=datetime.now(UTC).isoformat(),
    )

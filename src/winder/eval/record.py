"""`EvalRecord`: the pydantic cross-run artifact for one probe evaluation.

Pydantic, not a plain dataclass -- crosses winder's established pydantic boundary (see
`config.py`'s docstring, `data/manifest.py::RecordRow`'s identical reasoning): written by one
evaluation run, read back later, possibly compared across many runs or hand-inspected.

`split: Literal["val"]` is deliberately narrow. Fold 10 is sealed
(`winder.data.folds.folds(unseal=False)` by default); widening this field to also accept
`"test"` is a real, reviewable code change for the day a pre-registered protocol actually reads
it -- not something a caller can opt into by passing a string. `l2_selected_on: Literal["none"]`
records, for the same audit reason, that this MVP does not select `l2` on the validation fold --
so a later run that *does* select it cannot be silently compared against this one as if the two
used the same protocol.

`participation_ratio_*` and `effective_rank_*` are `winder.jepa.diagnostics.stable_rank`/
`effective_rank` under this artifact's own field names (the design spec calls the trace-only
quantity "participation ratio"; `diagnostics.py` calls the identical formula "stable rank" --
same computation, two names from two literatures, not a second implementation).

`*_probe_features` (added post-hoc, this session's adversarial MVP-staging review): the SAME two
diagnostics, but measured on the tensor the probe (and the anomaly score) actually read --
`model.predictor_hidden_states`, pooled -- rather than `*_tokens`/`*_pooled` above, which are
always the raw encoder's own output (`model.embed()`) regardless of what a downstream probe
reads. `winder.eval.collapse_gate.rank_gate`'s pre-registered E1-01 threshold was written against
the raw-encoder fields and stamps `INVALID_REPRESENTATION` on checkpoints whose encoder collapses
-- but a checkpoint can fail that gate while the tensor its own probe/anomaly score consumes is
perfectly healthy (the probe repointing moved the probe downstream of the projector's nonlinearity,
which the primer's own Table 13 shows can manufacture rank the encoder itself does not have).
Before this field existed, that "scope mismatch" defense was asserted, not measurable -- these
two fields make it falsifiable per checkpoint.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from winder.data.ptbxl import SUPERCLASSES

__all__ = ["EvalRecord"]


class EvalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    winder_git_sha: str | None = None
    created_utc: str

    split: Literal["val"]

    seed_pretrain: int
    seed_probe: int
    train_folds: tuple[int, ...]
    eval_fold: int

    n_train_records: int
    n_train_patients: int
    n_eval_records: int
    n_eval_patients: int

    lambda_sig: float
    n_pretrain_steps: int
    final_pred_loss: float
    final_sigreg_loss: float

    latent_width: int
    proj_width: int
    n_tokens: int

    macro_auroc: float
    macro_auroc_lo: float
    macro_auroc_hi: float
    per_superclass_auroc: dict[str, float]

    effective_rank_tokens: float
    participation_ratio_tokens: float
    effective_rank_pooled: float
    participation_ratio_pooled: float
    effective_rank_probe_features: float
    participation_ratio_probe_features: float

    encoder_rf_samples: int
    min_span_novel_fraction: float

    l2_selected_on: Literal["none"]
    config_yaml: str

    @field_validator("per_superclass_auroc")
    @classmethod
    def _keys_match_superclasses(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != set(SUPERCLASSES):
            raise ValueError(
                f"per_superclass_auroc keys must equal {SUPERCLASSES}, got {sorted(v)}"
            )
        return v

    @field_validator("macro_auroc", "macro_auroc_lo", "macro_auroc_hi")
    @classmethod
    def _auroc_in_bounds_or_nan(cls, v: float) -> float:
        if not (math.isnan(v) or 0.0 <= v <= 1.0):
            raise ValueError(f"AUROC values must be in [0, 1] or NaN, got {v}")
        return v

    @model_validator(mode="after")
    def _ci_ordered(self) -> "EvalRecord":
        vals = (self.macro_auroc_lo, self.macro_auroc, self.macro_auroc_hi)
        if not any(math.isnan(x) for x in vals) and not (vals[0] <= vals[1] <= vals[2]):
            raise ValueError(
                f"expected macro_auroc_lo <= macro_auroc <= macro_auroc_hi, got {vals}"
            )
        return self

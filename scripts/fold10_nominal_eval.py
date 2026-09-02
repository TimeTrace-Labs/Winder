"""Gate 1 of `notes/fold10_preregistration.md`'s five-gate ceremony: the one, single,
pre-agreed-filename script that may ever score fold 10, structurally constrained to call ONLY
already-validated `winder.*` functions -- no new statistical or eval logic written inline.

**What this file's own existence authorizes: nothing, yet.** Writing this script is gate 1.
Running it against fold 9 (already-published, already-spent data) and sanity-checking the result
is gate 2. Neither gate opens fold 10. `artifacts/fold10_authorization.json` still does not exist
on this repo as this file is committed; `winder.data.fold10_authorization.authorized_unseal`
therefore still refuses every call, including the one this file itself makes at `--target-fold
10` (see `resolve_target_fold_frames` below). Gates 3-5 (blind second-agent review, hash-pinned
sign-off, the CTO's own acceptance statement) come after this file is committed and reviewed, not
before.

**One call site, hash-gated, and it is the only place in this whole file that can ever touch fold
10.** `resolve_target_fold_frames(labeled, target_fold)` imports and calls `winder.data.
fold10_authorization.authorized_unseal` in exactly one branch, taken iff `target_fold == 10`
literally. There is no override flag, no alternate code path, and no way to reach that import by
passing any other value.

**A deliberate departure from a plausible-sounding design that would have broken a standing repo
invariant.** The obvious way to read "fold 9, the already-spent dry-run fold" would be to call
`folds()` with its own sealed-fold-release keyword set to `True`, directly in this script, since
fold 9 is not the sealed fold and the seal-invariant check does not protect it. That would,
however, require this file's own source to carry the literal spelling of that keyword-set-to-true
call site -- and `tests/test_folds.py::test_no_call_site_unseals` scans `src/` AND `scripts/` for
exactly that spelling (in any of its whitespace/quoting variants), exempting only two modules
under `src/winder/data/`, neither of which is this file. Writing it here would fail that test
outright, and disguising the keyword to dodge the scan would be gaming a safety check the project
explicitly built to catch self-motivated shortcuts under time pressure -- precisely the failure
mode this whole ceremony exists to prevent. The actual, already-established precedent for reading
a non-sealed fold in this codebase is `winder.eval.acceptance.build_split_frames`, which reads its
own "already-spent" fold (9) via `folds(labeled, LEGACY_FOLD_CONFIG)["val"]` -- the `"val"` key is
exposed unconditionally, with that sealed-fold-release keyword appearing nowhere in the call at
all, because `LEGACY_FOLD_CONFIG` simply points `val_fold` at the fold of interest.
`resolve_target_fold_frames` follows that exact, already-tested precedent for every `target_fold
!= 10`: it builds a `FoldConfig` with `val_fold=target_fold` and reads `folds(labeled, cfg)
["val"]` -- that keyword's own literal spelling, set true, appears as a call site nowhere in this
file, for either branch. (`scripts/detection_battery.py`'s own module docstring independently
documents the same discipline for its own file.)

**The train_folds exclusion is required, not stylistic.** `FoldConfig`'s seal invariant
(`folds._check_seal_invariant`) raises unless `test_fold` is absent from `train_folds` and unequal
to `val_fold`; `calibration_subset`/`train_minus_calibration` enforce the same check. A `cfg`
built with the class default `train_folds=(1,...,9)` and `test_fold=9` (or any fold in that
range) would put the eval fold inside the training pool and raise immediately -- this is not an
edge case, it is guaranteed to fire for every `target_fold` between 1 and 9. This file therefore
always excludes `target_fold` from `train_folds`: `tuple(f for f in range(1, 10) if f !=
target_fold)`. At `target_fold == 10` this is a no-op (10 is never in `range(1, 10)`), so the
train/cal split is built from EXACTLY `train_folds=(1,...,9)` -- what the four real checkpoints
were actually pretrained on. At `target_fold == 9` it reduces `train_folds` to `(1,...,8)`,
identical to `LEGACY_FOLD_CONFIG`'s own train set; the resulting probe cohort's `n_train`/`n_cal`
counts are therefore expected to reproduce `winder.eval.acceptance.build_split_frames`'s own
counts exactly (`build_p9_cohort`'s recorded `14521`/`2563`, per `artifacts/reports/
p9_eval_suite.json`). This IS a real, legitimate departure from "what the checkpoints were
trained on" for the dry-run branch, exactly as the pre-registration's own gate 2 text anticipates
("this cohort-building protocol legitimately differs from Phase P9's LEGACY-protocol diagnostic")
-- there is no way to dry-run against fold 9 as an isolated eval split while also training on it.

**The detection cohort's own missing independent seal, closed in `detection.py` itself, not
worked around here.** An earlier draft of this file constructed a second, separate `FoldConfig`
(`val_fold=10`) for the detection battery, reasoning that building it "strictly after" the probe
path's own `authorized_unseal` call already succeeded made it safe -- because
`winder.eval.detection.build_detection_cohort` read its eval split via `folds(labeled,
fold_config)["val"]` directly, and the `"val"` key carried no seal at all, at any `val_fold`
value. An independent review caught this: "safe by ordering" is exactly the failure mode this
whole ceremony exists to resist, and it is a GENERAL gap, not one specific to this file --
`winder.data.folds._check_seal_invariant` was hardened, the same session, to reject
`val_fold==10` unconditionally, closing the gap at its actual source rather than routing around
it here. `build_detection_cohort` was given a `frame=` parameter (an already-resolved frame, as
an alternative to `fold_config=`), and this file now passes `resolve_target_fold_frames`'s own
`eval_frame` straight through -- the probe and detection batteries score the identical record
set, resolved by the one authorized call, never re-derived.

**`run_detection_battery`'s own contract does not support the pre-registered per-step battery.**
Its own module docstring: it resolves `checkpoint_names` (roster ARM names) to
`"{roster_dir}/{name}/checkpoint"` ONLY -- the bare final snapshot, no `checkpoint_step<N>`
support. The pre-registration's steps are 5,000 / 20,000 / 30,000 for all four arms; 30,000
happens to BE the final snapshot (`discover_seed_checkpoints`'s own convention), but 5,000 and
20,000 are not reachable through that function's public contract at all. This file therefore
calls the two lower-level functions `run_detection_battery` itself is built from --
`build_detection_cohort` (once, shared across every arm/step, mirroring both `run_detection_
battery`'s own body and this file's own probe-cohort convention) and `run_checkpoint_detection_
battery` (once per resolved checkpoint directory, reusing the SAME `load_model_and_operator` call
already paid for the probe/gain/G1/robustness battery at that checkpoint) -- both already exported
in `winder.eval.detection.__all__`. This is a substitution of which already-validated function is
called, not new statistical logic: the composition is exactly `run_detection_battery`'s own loop,
re-pointed at explicit per-step checkpoint directories instead of per-arm roster names.

**`detection_gap_ci` is not wired.** `winder.eval.detection.cells_for` already exists on disk as
of this session (the companion background port has landed it), so it is *possible* to wire
`winder.eval.gates.detection_gap_ci` through it. Doing so requires choosing a "trained"
`(theta, detector)` cell and an anomaly type to score it against -- a substantive analysis
decision the pre-registration document does not itself pin down, and inventing one here would be
exactly the kind of judgment call gate 1's own "no new statistical logic written inline" restraint
is meant to keep out of a thin, fast-reviewable file. Deferred as a documented, known future
addition (this is the explicit escape hatch the commissioning brief names), not built.

**No comparison table.** The pre-registration's own "Full battery" bullet list names five
things: patient-clustered probe AUROC (both task variants), G1, the robustness suite, the
detection battery, and the transport gain/geometry report. It does not name `winder.eval.
comparison.arm_comparison_table` or `winder.eval.tasks.select_step`. `scripts/eval_suite.py`
already established the discipline of computing per-arm quantities and leaving cross-arm framing
for a human reader rather than baking in a derived delta or a crowned step; this file follows
that same discipline for the same reason -- a signal-vs-control delta CI is exactly the kind of
aggregation the "no aggregation/crowning" instruction below forbids introducing.

**Six previously-unfrozen parameters, now pinned by the authorization record itself (gate-3's B4
finding, closed 2026-08-18).** The pre-registration's own "Statistical plan, frozen now" section
already named and enforced `n_boot=1000`, `seed=0`, and `LinearProbeConfig()` defaults at
`--target-fold 10`. `n_strata`, `gain_limit`, `n_replicates`, `geometry_limit`, `causal_window`,
and the detection battery's own `n_records` remained CLI-configurable with no record of which
values backed "the" event -- an independent review correctly flagged this as a real gap, not a
cosmetic one: a single valid authorization could otherwise back multiple differently-parameterized
runs. These six now default to this project's existing, independently-verified conventions
(`scripts/eval_suite.py`'s own defaults for the first four; `scripts/detection_battery.py`'s own
default for the last two -- both checked directly against those scripts' own argparse defaults,
not merely copied from this file's prior draft), AND `main()` additionally requires, at
`--target-fold 10` only, that every resolved value matches the authorization record's own
`frozen_parameters` block exactly (`winder.data.fold10_authorization.load_frozen_parameters`) --
mismatched or missing pinning raises before any expensive work starts. Dry runs at any
`target_fold != 10` remain free to vary these (gate 2's own reduced-bootstrap-width tooling is
legitimate; it proves code correctness, not the frozen scale).

**`operator_report` is additive bookkeeping, not a new statistic.** `full_battery_for_checkpoint`
computes `operator_report` alongside the pre-registration's named `gain_report`/`geometry_report`
-- it characterises the harmonic operator's own spectrum (shared across every arm/step at a given
config), not a per-arm/per-step scored quantity. Named here explicitly after gate-3 flagged its
absence from the pre-registration document's own enumeration.

**The structural invariant closing gate-3's round-3 finding: past `main()`'s own argument
resolution, no raw directory crosses a function boundary on the `target_fold == 10` path -- only
resolved, individually pinned file paths.** An earlier version of `build_fold10_style_cohort`
took `artifacts_dir` and re-derived `theta_tokens.npz`/`manifest.parquet`'s paths internally; a
real, hash-verified `theta_tokens_path` existed in `main()`, but this function silently recomputed
its own instead, making the corresponding half of `_check_target_fold_10_frozen_inputs`'s check
decorative -- and `manifest.parquet` had no entry in `frozen_inputs` at all, so a caller could
repoint `--artifacts-dir` and feed a wholly different, unreviewed file into every scored quantity
that reads `rr_median_ms` (the entire robustness suite) while every OTHER check still passed.
`build_fold10_style_cohort` now takes `theta_tokens_path`/`manifest_path` as explicit, resolved
parameters (never `artifacts_dir`); `_resolve_frozen_inputs`'s own docstring records the complete
input surface this invariant is checked against, enumerated by tracing a real run
(`strace -f -e trace=openat`), not guessed. A future change that reintroduces a raw
`artifacts_dir`-shaped parameter anywhere on this path reopens exactly this bypass class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from winder.data.folds import FoldConfig, calibration_subset, folds, train_minus_calibration
from winder.data.integrity import git_sha
from winder.data.norm_stats import LeadStats
from winder.data.ptbxl import MULTIHOT_COLS, load_metadata, load_scp_statements
from winder.eval.comparison import EvalCohort
from winder.eval.detection import (
    DEFAULT_CAUSAL_WINDOW,
    DetectionCohort,
    build_detection_cohort,
    run_checkpoint_detection_battery,
)
from winder.eval.gates import g1_shuffled_theta_gain_null
from winder.eval.pooling import masked_mean_pool
from winder.eval.probe import LinearProbeConfig
from winder.eval.readout import (
    assert_lead_stats_matches_checkpoint,
    discover_seed_checkpoints,
    encode_z,
    load_model_and_operator,
    preflight_check_checkpoints,
    read_waveforms,
    theta_for_frame,
)
from winder.eval.robustness import robustness_suite
from winder.eval.tasks import (
    CLASSES,
    MIN_POSITIVES,
    ci_row,
    fit_and_score,
    subclass_code_map,
    subclass_multihot,
    superclass_multihot,
    surviving_columns,
)
from winder.jepa import checkpoint
from winder.jepa.dataset import EcgWindowDataset
from winder.paths import default_data_root
from winder.transport.dataset import load_theta_tokens
from winder.transport.report import gain_report, geometry_report, operator_report

MILESTONE_ID = "fold10-gate1and2-nominal-eval-script"

#: The pre-registered arms and steps (notes/fold10_preregistration.md, "What gets scored", the
#: CTO's 2026-08-18 executive decision). Never hardcoded a second time anywhere else in this file.
DEFAULT_ARMS: tuple[str, ...] = ("signal_seed0", "signal_seed1", "control_seed0", "control_seed1")
DEFAULT_STEPS: tuple[int, ...] = (5000, 20000, 30000)

#: Shared with scripts/print_fold10_authorization_template.py, so both scripts resolve --data-root
#: to the same default without a second, independently-maintained copy of this path.
DEFAULT_DATA_ROOT = default_data_root()

#: The pre-registration's own frozen statistical-plan values, enforced literally at
#: --target-fold 10 (see main()'s guard below); left CLI-configurable for target_fold != 10,
#: where a dry run legitimately wants to explore cheaper settings.
_FROZEN_N_BOOT = 1000
_FROZEN_SEED = 0

#: The six gate-3 B4 parameters' own frozen values -- named here as the single source of truth
#: for both this file's own argparse defaults AND scripts/print_fold10_authorization_template.py's
#: frozen_parameters block, so the two can never drift apart (the exact bug class gate-3 round 3
#: closed for build_fold10_style_cohort's own paths). DEFAULT_CAUSAL_WINDOW (imported above) is
#: the sixth; not redefined here.
_FROZEN_N_STRATA = 16
_FROZEN_GAIN_LIMIT = 250
_FROZEN_N_REPLICATES = 2000
_FROZEN_GEOMETRY_LIMIT = 1200
_FROZEN_DETECTION_N_RECORDS = 400

_SUPERCLASS_COLUMNS = list(range(len(CLASSES)))


# ==================================================================== target-fold branch point


@dataclass(frozen=True)
class TargetFoldFrames:
    """The output of the one shared branch point every fold-10-adjacent cohort in this file goes
    through: metadata-only (no waveform decode) train/cal/eval frames. `eval_frame` is reused
    directly for BOTH the probe cohort and the detection cohort -- they were always meant to score
    the exact same set of records, and `winder.eval.detection.build_detection_cohort`'s `frame=`
    parameter (added this session) accepts an already-resolved frame directly, so there is no
    second derivation and no second `FoldConfig` pointed anywhere near the sealed fold."""

    train_frame: pd.DataFrame
    cal_frame: pd.DataFrame
    eval_frame: pd.DataFrame


def resolve_target_fold_frames(labeled: pd.DataFrame, target_fold: int) -> TargetFoldFrames:
    """The one, single branch point this file's whole safety argument rests on.

    `target_fold == 10` is the ONLY value for which the `winder.data.fold10_authorization`
    import below is ever reached; every other value takes the `folds(labeled, cfg)["val"]`
    branch, matching `winder.eval.acceptance.build_split_frames`'s own established precedent for
    reading an already-spent fold (module docstring). See this module's own docstring for why the
    `train_folds` exclusion is load-bearing, not stylistic.

    **Revised this session, after an independent review caught a second bypass class.** The
    first draft additionally built a SEPARATE `FoldConfig(val_fold=10, test_fold=0)` for the
    detection cohort, reasoning that constructing it "strictly after" the real authorization call
    already succeeded made it safe. `winder.data.folds._check_seal_invariant` was hardened, the
    same session, to reject `val_fold==10` unconditionally, regardless of what `test_fold` says or
    when the config is built -- "safe by ordering" is exactly the property that check now
    forecloses, on purpose. `eval_frame` (below) is now reused directly for the detection cohort
    instead: authorization happens exactly once, at the one gated call site, and the resulting
    frame -- not a config that could re-derive fold membership -- flows down from there.
    """
    train_folds = tuple(f for f in range(1, 10) if f != target_fold)
    if target_fold == 10:
        probe_cfg = FoldConfig(train_folds=train_folds, val_fold=0, test_fold=10)
        # The one call site in this entire file that can ever reach fold 10. Gated by a
        # hash-pinned sign-off record this repo does not carry today (winder.data.
        # fold10_authorization's own module docstring); raises AuthorizationError otherwise.
        from winder.data.fold10_authorization import authorized_unseal

        eval_frame = authorized_unseal(labeled, probe_cfg)["test"]
    else:
        probe_cfg = FoldConfig(train_folds=train_folds, val_fold=target_fold, test_fold=10)
        eval_frame = folds(labeled, probe_cfg)["val"]
    train_frame = train_minus_calibration(labeled, probe_cfg)
    cal_frame = calibration_subset(labeled, probe_cfg)
    return TargetFoldFrames(train_frame, cal_frame, eval_frame)


# ============================================================================ cohort construction


@dataclass(frozen=True)
class Fold10Cohort:
    """Everything one target_fold's probe battery needs, decoded once and shared across every
    (arm, step) cell -- the same "pay once, reuse" discipline `scripts/eval_suite.py`'s own
    `build_p9_cohort` established."""

    eval_cohort: EvalCohort  #: superclass-5 labels in `.labels`, per `EvalCohort`'s own contract
    subclass_labels: dict[str, np.ndarray]  #: `(N, len(subclass_classes))`, row-aligned to splits
    subclass_classes: tuple[str, ...]
    #: The SAME metadata-only eval frame `eval_cohort` was built from (not re-derived) -- passed
    #: to `winder.eval.detection.build_detection_cohort`'s `frame=` parameter, so the detection
    #: battery scores the identical record set the probe does, with zero second call anywhere
    #: near fold selection. See `resolve_target_fold_frames`'s own docstring for why this replaced
    #: an earlier `FoldConfig`-based design.
    detection_eval_frame: pd.DataFrame
    bookkeeping: dict[str, Any]


def build_fold10_style_cohort(
    data_root: str,
    lead_stats_path: str,
    theta_tokens_path: str,
    manifest_path: str,
    *,
    target_fold: int,
) -> Fold10Cohort:
    """The train/cal/eval probe cohort for `target_fold`, waveforms decoded against
    `lead_stats_path` -- winder-nominal's own `lead_stats_f1to9.json` by convention, matching what
    the four real checkpoints were actually trained with (`scripts/eval_suite.py`'s own
    "lead-stats trap" docstring), passed explicitly rather than hardcoded so a caller cannot
    silently drift from that convention.

    Takes `theta_tokens_path`/`manifest_path` as already-resolved file paths, never a directory to
    derive them from -- gate-3's round-3 finding: an earlier version took `artifacts_dir` and
    re-derived these two paths internally, which meant `_check_target_fold_10_frozen_inputs`'s own
    hash check on `theta_tokens_path` was decorative for this call site (a real, resolved,
    hash-verified path existed in `main()`, but this function silently recomputed its own instead)
    and `manifest.parquet` was never in the `frozen_inputs` schema at all. See this module's own
    docstring for the invariant this establishes.
    """
    metadata = load_metadata(data_root)
    labeled = metadata.loc[metadata[list(MULTIHOT_COLS)].sum(axis=1) > 0]
    resolved = resolve_target_fold_frames(labeled, target_fold)
    frames = {
        "train": resolved.train_frame,
        "cal": resolved.cal_frame,
        "eval": resolved.eval_frame,
    }

    lead_stats = LeadStats.from_json(lead_stats_path)
    theta_by_id, theta_meta = load_theta_tokens(theta_tokens_path)
    n_tokens = cast(int, theta_meta["n_tokens"])
    patch_width = cast(int, theta_meta["patch_width"])

    waveforms = {
        k: read_waveforms(EcgWindowDataset(f, data_root, lead_stats=lead_stats))
        for k, f in frames.items()
    }
    thetas = {k: theta_for_frame(f, theta_by_id, n_tokens) for k, f in frames.items()}
    superclass_labels = {k: superclass_multihot(f) for k, f in frames.items()}
    patient_ids = {k: f["patient_id"].to_numpy() for k, f in frames.items()}

    code_map = subclass_code_map(load_scp_statements(data_root))
    subclass_classes = tuple(sorted(set(code_map.values())))
    if len(subclass_classes) != 23:
        raise ValueError(
            f"expected exactly 23 diagnostic subclasses from subclass_code_map, got "
            f"{len(subclass_classes)}: {subclass_classes}"
        )
    subclass_labels = {
        k: subclass_multihot(f, code_map, subclass_classes) for k, f in frames.items()
    }

    manifest = pd.read_parquet(manifest_path)
    rr_lookup = dict(zip(manifest["ecg_id"], manifest["rr_median_ms"], strict=True))
    rr_median_ms = {
        k: np.array([rr_lookup.get(int(e), np.nan) for e in f["ecg_id"]], dtype=np.float64)
        for k, f in frames.items()
    }

    eval_cohort = EvalCohort(
        waveforms=waveforms,
        thetas=thetas,
        labels=superclass_labels,
        patient_ids=patient_ids,
        rr_median_ms=rr_median_ms,
        patch_width=patch_width,
        gain_limit=250,
    )
    bookkeeping = {
        "target_fold": target_fold,
        "n_train": len(frames["train"]),
        "n_cal": len(frames["cal"]),
        "n_eval": len(frames["eval"]),
        "lead_stats_path": lead_stats_path,
        "subclass_classes": list(subclass_classes),
    }
    return Fold10Cohort(
        eval_cohort=eval_cohort,
        subclass_labels=subclass_labels,
        subclass_classes=subclass_classes,
        detection_eval_frame=resolved.eval_frame,
        bookkeeping=bookkeeping,
    )


# ======================================================================= full battery, one cell


def full_battery_for_checkpoint(
    name: str,
    ckpt_dir: str,
    cohort: Fold10Cohort,
    detection_cohort: DetectionCohort | None,
    *,
    device: torch.device,
    seed: int,
    n_boot: int,
    n_strata: int,
    gain_limit: int,
    n_replicates: int,
    geometry_limit: int,
    causal_window: int,
) -> dict[str, Any]:
    """One checkpoint's full pre-registered battery: operator spectrum + geometry report +
    transport gain + G1 shuffled-theta null + superclass-5 `z/mean` probe AUROC (patient-clustered
    CI) + subclass-23 `z/mean` probe AUROC (both the survivor-filtered and the full 23-class
    macro-average, patient-clustered CI, from the SAME fitted scores) + the full robustness suite
    + the detection/localisation battery -- one model load, one `encode_z` pass per split, one
    `HarmonicTransport` shared across every piece, exactly mirroring `scripts/eval_suite.py`'s own
    `full_battery_for_checkpoint` for the pieces it already covers.
    """
    ec = cohort.eval_cohort
    model, operator = load_model_and_operator(ckpt_dir, seed=seed, device=device)
    if operator is None:
        raise ValueError(f"{name} ({ckpt_dir}): no transport operator -- full battery undefined")
    try:
        z_by_split = {s: encode_z(model, wf, device) for s, wf in ec.waveforms.items()}
        op_report = operator_report(name, operator)
        op_cpu = operator.to("cpu")

        g_lim = min(geometry_limit, z_by_split["eval"].shape[0])
        geometry = geometry_report(
            z_by_split["eval"][:g_lim].double(), ec.thetas["eval"][:g_lim].double(), op_cpu
        )

        gl = min(gain_limit, z_by_split["eval"].shape[0])
        z_gain = z_by_split["eval"][:gl].float()
        th_gain = ec.thetas["eval"][:gl].float()
        pid_gain = ec.patient_ids["eval"][:gl]
        gain = gain_report(z_gain, th_gain, op_cpu, pid_gain, n_strata=n_strata, seed=seed)
        g1 = g1_shuffled_theta_gain_null(
            z_gain,
            th_gain,
            op_cpu,
            pid_gain,
            n_strata=n_strata,
            n_replicates=n_replicates,
            seed=seed,
        )

        feats = {
            s: masked_mean_pool(z_by_split[s], ec.thetas[s]).numpy()
            for s in ("train", "cal", "eval")
        }
        probe_cfg = LinearProbeConfig(seed_probe=seed)

        scores_super, _ = fit_and_score(
            feats["train"],
            ec.labels["train"],
            feats["cal"],
            ec.labels["cal"],
            feats["eval"],
            ec.labels["eval"],
            CLASSES,
            probe_cfg,
        )
        probe_superclass5 = ci_row(
            scores_super,
            ec.labels["eval"],
            ec.patient_ids["eval"],
            _SUPERCLASS_COLUMNS,
            n_boot,
            seed,
        )

        sub = cohort.subclass_labels
        scores_sub, _ = fit_and_score(
            feats["train"],
            sub["train"],
            feats["cal"],
            sub["cal"],
            feats["eval"],
            sub["eval"],
            cohort.subclass_classes,
            probe_cfg,
        )
        surviving = surviving_columns(sub["eval"], MIN_POSITIVES)
        excluded = [i for i in range(len(cohort.subclass_classes)) if i not in surviving]
        probe_subclass23_filtered = {
            **ci_row(scores_sub, sub["eval"], ec.patient_ids["eval"], surviving, n_boot, seed),
            "min_positives": MIN_POSITIVES,
            "surviving_classes": [cohort.subclass_classes[i] for i in surviving],
            "excluded_classes": [cohort.subclass_classes[i] for i in excluded],
        }
        all_columns = list(range(len(cohort.subclass_classes)))
        probe_subclass23_full = {
            **ci_row(scores_sub, sub["eval"], ec.patient_ids["eval"], all_columns, n_boot, seed),
            "classes": list(cohort.subclass_classes),
        }

        robustness = robustness_suite(
            model,
            op_cpu,
            z_by_split,
            ec.thetas,
            ec.waveforms,
            ec.labels,
            ec.patient_ids["eval"],
            ec.rr_median_ms,
            ec.patch_width,
            probe_cfg,
            device,
            seed=seed,
        )

        detection: dict[str, Any] | None = None
        if detection_cohort is not None:
            detection, _dump = run_checkpoint_detection_battery(
                model,
                op_cpu,
                detection_cohort,
                device,
                ckpt_name=name,
                causal_window=causal_window,
                dump_per_record=False,
            )
    finally:
        del model, operator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "operator": op_report,
        "geometry": geometry,
        "gain": gain,
        "g1": g1,
        "probe_superclass5_z_mean": probe_superclass5,
        "probe_subclass23_z_mean_filtered": probe_subclass23_filtered,
        "probe_subclass23_z_mean_full": probe_subclass23_full,
        "robustness": robustness,
        "detection": detection,
    }


# ================================================================================== atomic write


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write `payload` to `path` via a sibling `.tmp` + `os.replace` -- atomic on the same
    filesystem, matching `scripts/eval_suite.py`'s own convention."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    os.replace(tmp_path, path)


def _envelope(
    target_fold: int,
    status: str,
    metrics: dict[str, Any],
    decisions: list[str],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """`split_status`/`headline` are derived from `target_fold` alone, automatically -- never a
    separately-settable flag a caller could set inconsistently with the fold actually scored."""
    split_status = (
        "true_holdout" if target_fold == 10 else f"dry_run_fold{target_fold}_differential_check"
    )
    headline = target_fold == 10
    return {
        "status": status,
        "milestone_id": MILESTONE_ID,
        "split_status": split_status,
        "headline": headline,
        "metrics": metrics,
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": params,
            "seed": seed,
        },
        "decisions": [
            "Gate 1 (this file, thin glue over already-validated winder.* functions only) and "
            "gate 2 (a real, executed differential run against an already-spent fold) are the "
            "only gates this script's own execution can ever satisfy. Gates 3 (blind second-agent "
            "review), 4 (hash-pinned authorization record), and 5 (the CTO's own acceptance "
            "statement) happen outside this script, after it is committed and reviewed.",
            "The one call site that can reach fold 10 (winder.data.fold10_authorization."
            "authorized_unseal) is taken iff target_fold == 10 literally; every other value "
            "reads its eval fold via folds(labeled, cfg)['val'], matching winder.eval.acceptance."
            "build_split_frames's own established precedent -- this file's own source never "
            "carries the sealed-fold-release keyword's literal spelling (see module docstring).",
            "No comparison table, no pairwise delta, no crowned step: every (arm, step) cell is "
            "reported independently, exactly as notes/fold10_preregistration.md's 'no single "
            "step is crowned' instruction requires.",
            "winder.eval.gates.detection_gap_ci is not wired: it requires choosing a trained "
            "(theta, detector) cell and an anomaly type, a substantive analysis decision the "
            "pre-registration document does not itself pin down. Documented future addition, "
            "not a missing requirement (module docstring).",
            "n_strata, gain_limit, n_replicates, geometry_limit, causal_window, and "
            "detection_n_records actually used this run -- see provenance.parameters for the "
            "exact values. At target_fold == 10, main() requires these to match the authorization "
            "record's own frozen_parameters block exactly (winder.data.fold10_authorization."
            "load_frozen_parameters); at any other target_fold this is a dry run and these remain "
            "CLI-configurable, defaulted to scripts/eval_suite.py's and scripts/"
            "detection_battery.py's own established conventions.",
            *decisions,
        ],
        "questions": [],
    }


def frozen_parameters_dict() -> dict[str, int]:
    """The six gate-3 B4 parameters' own frozen values, as a single named source of truth used
    both for this file's own argparse defaults and by `scripts/print_fold10_authorization_
    template.py` to build the authorization record's `frozen_parameters` block -- so gate-1's
    defaults and gate-4's printed template can never silently drift apart.
    """
    return {
        "n_strata": _FROZEN_N_STRATA,
        "gain_limit": _FROZEN_GAIN_LIMIT,
        "n_replicates": _FROZEN_N_REPLICATES,
        "geometry_limit": _FROZEN_GEOMETRY_LIMIT,
        "causal_window": DEFAULT_CAUSAL_WINDOW,
        "detection_n_records": _FROZEN_DETECTION_N_RECORDS,
    }


def resolve_default_paths(
    artifacts_dir: str,
    *,
    roster_dir: str | None = None,
    lead_stats_path: str | None = None,
    rpeaks_npz_path: str | None = None,
    theta_tokens_path: str | None = None,
    manifest_path: str | None = None,
) -> dict[str, str]:
    """The one place `<artifacts_dir>`-relative defaults are derived, shared by `main()` and
    `scripts/print_fold10_authorization_template.py` -- a second, independently-maintained copy
    of this logic is exactly the drift risk gate-3's round-3 finding closed for
    `build_fold10_style_cohort`'s own paths.
    """
    return {
        "roster_dir": roster_dir or os.path.join(artifacts_dir, "roster"),
        "lead_stats_path": lead_stats_path or os.path.join(artifacts_dir, "lead_stats_f1to9.json"),
        "rpeaks_npz_path": rpeaks_npz_path
        or os.path.join(artifacts_dir, "reference", "phase", "rpeaks.npz"),
        "theta_tokens_path": theta_tokens_path
        or os.path.join(artifacts_dir, "phase", "theta_tokens.npz"),
        "manifest_path": manifest_path or os.path.join(artifacts_dir, "manifest.parquet"),
    }


def resolve_checkpoints(
    roster_dir: str, arm_names: Sequence[str], steps: Sequence[int]
) -> dict[str, str]:
    """`{"{arm}/step{n}": ckpt_dir}` for every requested (arm, step) pair, discovered fresh from
    `roster_dir` -- the one place this resolution happens, shared by `main()` and
    `scripts/print_fold10_authorization_template.py`.
    """
    arm_dirs = {name: os.path.join(roster_dir, name) for name in arm_names}
    for name, d in arm_dirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(f"roster arm dir not found: {d} (arm={name!r})")
    per_arm_steps = {name: discover_seed_checkpoints(d) for name, d in arm_dirs.items()}
    checkpoints: dict[str, str] = {}
    for name in arm_names:
        for step in steps:
            if step not in per_arm_steps[name]:
                raise ValueError(
                    f"requested step {step} not found for arm {name!r} among "
                    f"{sorted(per_arm_steps[name])}"
                )
            checkpoints[f"{name}/step{step}"] = per_arm_steps[name][step]
    return checkpoints


def _count_battery_errors(metrics: dict[str, Any]) -> int:
    """Pure, unit-testable core of gate-3's status-field fix. Counts every per-cell/per-stage
    failure already tolerated by this script's own `except Exception` guards (preflight, the
    detection cohort, and each (arm, step) battery cell) -- these guards exist so one bad piece
    does not sink the whole run, but a run with real failures inside it must not read identically
    to a run with none. Returns 0 iff nothing failed anywhere.
    """
    n = len(metrics.get("preflight", {}).get("failed", {}))
    if "error" in metrics.get("detection_cohort", {}):
        n += 1
    for per_step in metrics.get("full_battery", {}).values():
        for cell in per_step.values():
            if "error" in cell:
                n += 1
    return n


def _check_target_fold_10_frozen_parameters(
    resolved: dict[str, int], frozen: dict[str, int]
) -> None:
    """Pure, unit-testable core of gate-3's B4 fix. `target_fold == 10` must use exactly the six
    statistical/scale parameters pinned in the authorization record's own `frozen_parameters`
    block -- these are not covered by the arms/steps/n_boot/seed check above, and an independent
    review found that a single valid authorization could otherwise back multiple
    differently-parameterized runs with no record of which was "the" event. Raises `SystemExit`
    (never proceeds on a mismatch) before any expensive work starts.
    """
    if resolved != frozen:
        raise SystemExit(
            f"target_fold=10 requires exactly the authorization record's own frozen_parameters: "
            f"frozen={frozen}, resolved (from this run's CLI args)={resolved}."
        )


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _resolve_frozen_inputs(
    *,
    data_root: str,
    roster_dir: str,
    lead_stats_path: str,
    rpeaks_npz_path: str,
    theta_tokens_path: str,
    manifest_path: str,
    checkpoints: dict[str, str],
) -> dict[str, Any]:
    """This run's own actual content hashes for every discrete file input gate-3's A-2 finding
    flagged as unpinned -- previously, an authorization could be satisfied while `--roster-dir`/
    `--data-root` pointed at different underlying checkpoints or data, since only the script's own
    hash, arms/steps, and the six statistical parameters were checked. `data_root`/`roster_dir`
    are recorded as exact path strings, not hashed -- the waveform corpus under `data_root` is
    deliberately path-trusted (hashing every WFDB record is impractical); its content-level
    coverage is the metadata hash below, the pre-existing lead-stats trap
    (`assert_lead_stats_matches_checkpoint`), and the acceptance gate's own split-shape checks --
    not a corpus hash.

    **The enumerated input surface (gate-3 round 3, empirically traced via `strace -f -e
    trace=openat` over a real dry run, not guessed):** every non-waveform, non-interpreter file
    the fold-10-style eval path opens is one of: `ptbxl_database.csv`, `scp_statements.csv`
    (`metadata_sha256`), `lead_stats_path`, `rpeaks_npz_path`, `theta_tokens_path`,
    `manifest_path` (each hashed below), and per-checkpoint `state.pt`/`config.yaml`/`meta.json`
    (`checkpoint_sha256`, nested per (arm, step) key -- `config.yaml`/`meta.json` were traced but
    NOT hashed prior to this fix; the trace showed no other checkpoint-dir file is read). Nothing
    else was observed; `manifest.parquet` was traced being opened but was entirely absent from
    this schema before this fix -- the exact gap round 3's live-constructed bypass exploited via
    `--artifacts-dir`.

    **One more traced read, named explicitly rather than silently covered (gate-3 round 4):**
    `discover_seed_checkpoints` (called by `main()` before this function, to build the
    `checkpoints` dict this function's own `checkpoint_sha256` block hashes) unconditionally
    opens the FINAL `checkpoint/`'s own `config.yaml` for every arm, to resolve its step number --
    regardless of whether step 30000 was actually requested. At `target_fold == 10` this is
    always covered anyway: `main()`'s arms/steps hard-equality gate forces every real event to
    request exactly `DEFAULT_STEPS = (5000, 20000, 30000)`, so `checkpoints["{arm}/step30000"]`
    always resolves to that same `checkpoint/` dir, and its `config.yaml`/`state.pt`/`meta.json`
    are hashed under that key by the ordinary mechanism below -- no separate entry needed. This
    is a real, named cross-file dependency, not an independent guarantee of this function alone:
    it is the arms/steps gate (not `_resolve_frozen_inputs`) that keeps the final checkpoint dir
    inside the pinned set. Hashing every arm's final `checkpoint/config.yaml` unconditionally,
    even for dry runs that never request step 30000, would be enforcement theater -- nothing
    scored ever depends on it at any `target_fold != 10`.
    """
    return {
        "data_root": data_root,
        "roster_dir": roster_dir,
        "metadata_sha256": {
            "ptbxl_database.csv": _sha256_file(os.path.join(data_root, "ptbxl_database.csv")),
            "scp_statements.csv": _sha256_file(os.path.join(data_root, "scp_statements.csv")),
        },
        "lead_stats_sha256": _sha256_file(lead_stats_path),
        "rpeaks_npz_sha256": _sha256_file(rpeaks_npz_path),
        "theta_tokens_npz_sha256": _sha256_file(theta_tokens_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "checkpoint_sha256": {
            key: {
                "state.pt": _sha256_file(os.path.join(ckpt_dir, checkpoint.STATE_FILENAME)),
                "config.yaml": _sha256_file(os.path.join(ckpt_dir, checkpoint.CONFIG_FILENAME)),
                "meta.json": _sha256_file(os.path.join(ckpt_dir, checkpoint.META_FILENAME)),
            }
            for key, ckpt_dir in sorted(checkpoints.items())
        },
    }


def _check_target_fold_10_frozen_inputs(resolved: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Pure, unit-testable core of gate-3's A-2 fix. `target_fold == 10` must use exactly the
    data/checkpoint inputs pinned in the authorization record's own `frozen_inputs` block -- see
    `_resolve_frozen_inputs`. Raises `SystemExit` (never proceeds on a mismatch) before any
    checkpoint is deserialized (`preflight_check_checkpoints`'s `torch.load`) or scored.

    Not literally "before any file is opened at all" (gate-3 round 4's precision fix to this
    docstring): `main()` calls this immediately after `checkpoints` is built from `discover_seed_
    checkpoints`'s own scan, which has already opened each checkpoint dir's `config.yaml` to
    resolve step numbers -- that read is the irreducible pre-verification I/O this whole path
    needs before it can even know which directory to hash under which key. A mismatch here still
    aborts before `preflight_check_checkpoints`, the lead-stats loop, and `build_fold10_style_
    cohort`/the one real unseal call -- nothing heavier than that one discovery-time read ever
    touches a tampered checkpoint's bytes.
    """
    if resolved != frozen:
        raise SystemExit(
            f"target_fold=10 requires exactly the authorization record's own frozen_inputs: "
            f"frozen={frozen}, resolved (from this run's actual files)={resolved}."
        )


# ======================================================================================== main


def main(argv: list[str] | None = None) -> int:
    """Parse args, resolve the target-fold cohort (probe + detection), preflight every requested
    checkpoint, run the full battery per (arm, step) cell -- reported separately, never
    aggregated -- and write the report JSON atomically after every stage."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target-fold",
        type=int,
        required=True,
        help="10 for the real, pre-registered event (requires artifacts/fold10_authorization.json "
        "to already exist and name this exact script by content hash); any other value (e.g. 9) "
        "for a differential dry run against already-spent data.",
    )
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument(
        "--lead-stats-path", default=None, help="default <artifacts-dir>/lead_stats_f1to9.json"
    )
    ap.add_argument(
        "--rpeaks-npz-path",
        default=None,
        help="default <artifacts-dir>/reference/phase/rpeaks.npz -- the only copy on disk; "
        "R-peak timings are protocol/fold-independent, so this is shared by every target_fold",
    )
    ap.add_argument(
        "--theta-tokens-path", default=None, help="default <artifacts-dir>/phase/theta_tokens.npz"
    )
    ap.add_argument(
        "--manifest-path", default=None, help="default <artifacts-dir>/manifest.parquet"
    )
    ap.add_argument("--out", default="artifacts/reports/fold10_nominal_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=_FROZEN_SEED)
    ap.add_argument("--n-boot", type=int, default=_FROZEN_N_BOOT)
    ap.add_argument("--n-strata", type=int, default=_FROZEN_N_STRATA)
    ap.add_argument("--gain-limit", type=int, default=_FROZEN_GAIN_LIMIT)
    ap.add_argument("--n-replicates", type=int, default=_FROZEN_N_REPLICATES)
    ap.add_argument("--geometry-limit", type=int, default=_FROZEN_GEOMETRY_LIMIT)
    ap.add_argument("--causal-window", type=int, default=DEFAULT_CAUSAL_WINDOW)
    ap.add_argument(
        "--detection-n-records",
        type=int,
        default=_FROZEN_DETECTION_N_RECORDS,
        help=(
            "at --target-fold 10, must match the authorization record's own "
            "frozen_parameters.detection_n_records exactly"
        ),
    )
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--steps", default=",".join(str(s) for s in DEFAULT_STEPS))
    args = ap.parse_args(argv)

    t0 = time.time()
    device = torch.device(args.device)
    resolved_paths = resolve_default_paths(
        args.artifacts_dir,
        roster_dir=args.roster_dir,
        lead_stats_path=args.lead_stats_path,
        rpeaks_npz_path=args.rpeaks_npz_path,
        theta_tokens_path=args.theta_tokens_path,
        manifest_path=args.manifest_path,
    )
    roster_dir = resolved_paths["roster_dir"]
    lead_stats_path = resolved_paths["lead_stats_path"]
    rpeaks_npz_path = resolved_paths["rpeaks_npz_path"]
    theta_tokens_path = resolved_paths["theta_tokens_path"]
    manifest_path = resolved_paths["manifest_path"]
    arm_names = [a for a in args.arms.split(",") if a]
    steps = sorted({int(s) for s in args.steps.split(",") if s})

    # The real event's own scope is frozen by notes/fold10_preregistration.md, not by whatever
    # --arms/--steps a caller happens to type; a scoped-down subset is legitimate ONLY for a
    # target_fold != 10 differential/dry-run gate (this run's own gate 2 uses exactly that).
    if args.target_fold == 10:
        if sorted(arm_names) != sorted(DEFAULT_ARMS) or steps != list(DEFAULT_STEPS):
            raise SystemExit(
                "target_fold=10 is the real, pre-registered event: --arms/--steps must exactly "
                f"match the frozen set (arms={sorted(DEFAULT_ARMS)}, steps={list(DEFAULT_STEPS)}); "
                f"got arms={sorted(arm_names)}, steps={steps}. A scoped-down subset is only "
                "permitted for a target_fold != 10 differential/dry-run gate."
            )
        if args.n_boot != _FROZEN_N_BOOT or args.seed != _FROZEN_SEED:
            raise SystemExit(
                f"target_fold=10 requires the pre-registration's own frozen statistical plan: "
                f"n_boot={_FROZEN_N_BOOT}, seed={_FROZEN_SEED}; got n_boot={args.n_boot}, "
                f"seed={args.seed}."
            )
        from winder.data.fold10_authorization import load_frozen_parameters

        _check_target_fold_10_frozen_parameters(
            {
                "n_strata": args.n_strata,
                "gain_limit": args.gain_limit,
                "n_replicates": args.n_replicates,
                "geometry_limit": args.geometry_limit,
                "causal_window": args.causal_window,
                "detection_n_records": args.detection_n_records,
            },
            load_frozen_parameters(),
        )

    params = {
        "target_fold": args.target_fold,
        "data_root": args.data_root,
        "artifacts_dir": args.artifacts_dir,
        "roster_dir": roster_dir,
        "lead_stats_path": lead_stats_path,
        "rpeaks_npz_path": rpeaks_npz_path,
        "theta_tokens_path": theta_tokens_path,
        "manifest_path": manifest_path,
        "device": str(device),
        "n_boot": args.n_boot,
        "n_strata": args.n_strata,
        "gain_limit": args.gain_limit,
        "n_replicates": args.n_replicates,
        "geometry_limit": args.geometry_limit,
        "causal_window": args.causal_window,
        "detection_n_records": args.detection_n_records,
        "arms": arm_names,
        "steps": steps,
    }

    # --------------------------------------------------------- Task 1: discovery + preflight
    checkpoints = resolve_checkpoints(roster_dir, arm_names, steps)

    # Gate-3 A-2/round-4 fix: at target_fold==10, every discrete file input (checkpoints,
    # lead-stats, rpeaks, theta-tokens, manifest, metadata) must match the authorization record's
    # own frozen_inputs block exactly -- checked HERE, immediately after `checkpoints` is built
    # and before any other file is opened for real work (preflight's torch.load, the lead-stats
    # loop, or build_fold10_style_cohort/the one real unseal call below). Gate-3 round 4 found
    # this check used to run after preflight and the lead-stats loop had already read checkpoint
    # bytes -- never a leakage path (a mismatch still aborts before cohort-building/unseal either
    # way), but this ordering is the tighter, correct one: discovery's own `config.yaml` reads
    # (to resolve step numbers) are the only irreducible pre-verification I/O on this path.
    if args.target_fold == 10:
        from winder.data.fold10_authorization import load_frozen_inputs

        _check_target_fold_10_frozen_inputs(
            _resolve_frozen_inputs(
                data_root=args.data_root,
                roster_dir=roster_dir,
                lead_stats_path=lead_stats_path,
                rpeaks_npz_path=rpeaks_npz_path,
                theta_tokens_path=theta_tokens_path,
                manifest_path=manifest_path,
                checkpoints=checkpoints,
            ),
            load_frozen_inputs(),
        )

    preflight_failed = preflight_check_checkpoints(checkpoints, seed=args.seed, device=device)
    metrics: dict[str, Any] = {
        "preflight": {
            "n_checkpoints": len(checkpoints),
            "n_ok": len(checkpoints) - len(preflight_failed),
            "failed": preflight_failed,
        }
    }
    _write_json_atomic(
        args.out, _envelope(args.target_fold, "RUNNING", metrics, [], params, args.seed)
    )

    # ------------------------------------------ the lead-stats trap: a real, executed assertion
    # (dry-run defense: at target_fold==10 a mismatch is already caught above via
    # frozen_inputs.lead_stats_sha256; this loop remains the one that fires for any other
    # target_fold, where frozen_inputs is never checked at all.)
    hash_failures: dict[str, str] = {}
    for key, ckpt_dir in checkpoints.items():
        try:
            assert_lead_stats_matches_checkpoint(ckpt_dir, lead_stats_path)
        except AssertionError as e:
            hash_failures[key] = str(e)
    metrics["lead_stats_hash_check"] = {
        "lead_stats_path": lead_stats_path,
        "n_checked": len(checkpoints),
        "n_mismatched": len(hash_failures),
        "mismatched": hash_failures,
    }
    if hash_failures:
        report = _envelope(
            args.target_fold,
            "FAIL",
            metrics,
            [
                f"STOPPED before any encoding: {len(hash_failures)} checkpoint(s) declare a "
                "lead_stats_sha256 that does not match the actual sha256 of the lead-stats file "
                "this run was about to use."
            ],
            params,
            args.seed,
        )
        _write_json_atomic(args.out, report)
        print(
            f"[fold10_nominal_eval] status=FAIL (lead-stats mismatch) wrote {args.out}",
            flush=True,
        )
        return 1

    # --------------------------------------------------------------------- build cohorts once
    # Gate-3 round-3 invariant: past this point, every path fed into cohort-building is an
    # already-resolved, individually-pinned file path -- never a raw directory a callee could
    # derive its own paths from (see build_fold10_style_cohort's own docstring for why).
    cohort = build_fold10_style_cohort(
        args.data_root,
        lead_stats_path,
        theta_tokens_path,
        manifest_path,
        target_fold=args.target_fold,
    )
    metrics["cohort"] = cohort.bookkeeping
    _write_json_atomic(
        args.out, _envelope(args.target_fold, "RUNNING", metrics, [], params, args.seed)
    )

    detection_cohort: DetectionCohort | None
    try:
        detection_cohort = build_detection_cohort(
            args.data_root,
            frame=cohort.detection_eval_frame,
            n_records=args.detection_n_records,
            rpeaks_npz_path=rpeaks_npz_path,
            lead_stats_path=lead_stats_path,
            theta_tokens_path=theta_tokens_path,
        )
        metrics["detection_cohort"] = {
            "n_records": len(detection_cohort.frame),
            "theta_coverage_offline": detection_cohort.theta_coverage_offline,
            "theta_coverage_causal": detection_cohort.theta_coverage_causal,
        }
    except Exception as e:  # noqa: BLE001 -- a bad detection cohort must not sink the probe battery
        detection_cohort = None
        metrics["detection_cohort"] = {"error": f"{type(e).__name__}: {e}"}
    _write_json_atomic(
        args.out, _envelope(args.target_fold, "RUNNING", metrics, [], params, args.seed)
    )

    # -------------------------------------------------- Task 2: full battery, per (arm, step)
    battery: dict[str, dict[str, Any]] = {name: {} for name in arm_names}
    for name in arm_names:
        for step in steps:
            key = f"{name}/step{step}"
            try:
                battery[name][str(step)] = {
                    "step": step,
                    **full_battery_for_checkpoint(
                        key,
                        checkpoints[key],
                        cohort,
                        detection_cohort,
                        device=device,
                        seed=args.seed,
                        n_boot=args.n_boot,
                        n_strata=args.n_strata,
                        gain_limit=args.gain_limit,
                        n_replicates=args.n_replicates,
                        geometry_limit=args.geometry_limit,
                        causal_window=args.causal_window,
                    ),
                }
            except Exception as e:  # noqa: BLE001 -- one bad cell must not sink the whole battery
                battery[name][str(step)] = {"step": step, "error": f"{type(e).__name__}: {e}"}
            metrics["full_battery"] = battery
            _write_json_atomic(
                args.out, _envelope(args.target_fold, "RUNNING", metrics, [], params, args.seed)
            )

    n_errors = _count_battery_errors(metrics)
    metrics["n_errors"] = n_errors
    status = "PASS" if n_errors == 0 else "PARTIAL"
    report = _envelope(args.target_fold, status, metrics, [], params, args.seed)
    report["metrics"]["elapsed_sec"] = time.time() - t0
    _write_json_atomic(args.out, report)
    print(
        f"[fold10_nominal_eval] status={status} ({n_errors} error(s)) wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fold discipline: train/val/sealed-test split, patient-disjoint calibration subset.

Ported near-verbatim from ttl-phase's `src/data/ptbxl.py` (pinned at
cfe2e60a5592e30a32ef1f1863ee4fb449e80714), split into its own module -- the train/val/
sealed-test pattern isn't PTB-XL-specific, unlike the metadata/labelling logic in
`ptbxl.py`.

Bug fix #1 vs. ttl-phase: ttl-phase's `TRAIN_FOLDS`/`VAL_FOLD`/`TEST_FOLD` were hardcoded
module constants, never passed in by a caller, so "fold 10 ends up in train_folds" was
structurally impossible there. Making fold membership OmegaConf-configurable (`FoldConfig`,
below) removes that structural guarantee -- a config file COULD set
`train_folds=(1,...,10)` by a typo. `folds()` therefore adds an explicit, unconditional
check (`test_fold` must not be in `train_folds` or equal `val_fold`) that fires regardless
of `unseal`, restoring the guarantee the original's hardcoded constants provided for free.
`unseal` (renamed from `allow_test`) controls only whether the *output* carries a "test"
key -- it was never meant to, and must never be able to, license training on fold 10.

winder-nominal deviation from the reference repo's default (see PR notes for this port):
`FoldConfig`'s default here is `train_folds=(1,...,9)`, `val_fold=0`. PTB-XL's real
`strat_fold` values are 1..10, so `val_fold=0` is a deliberate non-existent-fold sentinel --
`folds()` with this default returns a `"val"` split that is always empty on real data. This
default is required specifically because it moves fold 9, previously the validation fold,
into the training pool: if `val_fold` had stayed at 9, the seal/patient-disjoint invariant
check below would fail the moment fold 9 also appeared in `train_folds` (a patient cannot
legitimately be in both). `LEGACY_FOLD_CONFIG`, below, preserves the reference repo's
original default exactly, for an acceptance-test harness that needs to reproduce its exact
protocol later; nothing in this port consumes it yet.
"""

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FoldConfig",
    "LEGACY_FOLD_CONFIG",
    "folds",
    "calibration_subset",
    "train_minus_calibration",
]


@dataclass
class FoldConfig:
    train_folds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    val_fold: int = 0  # sentinel: PTB-XL's real strat_fold values are 1..10, so this is empty
    test_fold: int = 10  # SEALED until a pre-registered protocol opens it, once
    calibration_frac: float = 0.15
    calibration_seed: int = 0


#: The reference repo's (winder-theory-exp) original default, preserved verbatim under this
#: name for a later acceptance-test harness that must reproduce its exact protocol.
LEGACY_FOLD_CONFIG = FoldConfig(
    train_folds=(1, 2, 3, 4, 5, 6, 7, 8),
    val_fold=9,
    test_fold=10,
    calibration_frac=0.15,
    calibration_seed=0,
)


#: PTB-XL's actual sealed fold. Hardcoded, deliberately NOT derived from `cfg.test_fold` --
#: `test_fold` is a config FIELD a caller controls, and the whole point of the check below is
#: that the seal must hold regardless of what that field is set to. (Bug found this session,
#: 2026-08-18: a caller could set `val_fold=10` while relabelling `test_fold` to something else
#: entirely, e.g. 0. `_check_seal_invariant`'s original form only ever asked "is `cfg.test_fold`
#: reachable", so that config sailed through -- and `val` is exposed UNCONDITIONALLY in `folds()`'s
#: return value, unlike `test`, so real fold-10 rows came back with no warning, no `unseal`, no
#: gate at all. This is a distinct bypass from the two others hardened the same night, and the
#: more dangerous of the three: the other two required NOT calling `folds()`, or calling it with a
#: dict-unpacking trick; this one calls it completely normally, through its own public contract.)
_SEALED_FOLD = 10


def _check_seal_invariant(cfg: FoldConfig) -> None:
    """Unconditional: test_fold must never be reachable via train_folds or val_fold, AND the
    literal sealed fold (`_SEALED_FOLD`, always 10) must never be reachable via train_folds or
    val_fold no matter what `test_fold` is configured to. The second check is not redundant with
    the first whenever `cfg.test_fold != _SEALED_FOLD` -- see `_SEALED_FOLD`'s own comment."""
    if cfg.test_fold in cfg.train_folds or cfg.test_fold == cfg.val_fold:
        raise ValueError(
            f"test_fold={cfg.test_fold} must not appear in train_folds={cfg.train_folds} "
            f"or equal val_fold={cfg.val_fold} -- this check is unconditional, unlike "
            f"`unseal`, which only controls whether the *output* carries a 'test' key."
        )
    if _SEALED_FOLD in cfg.train_folds:
        raise ValueError(
            f"fold {_SEALED_FOLD} (the sealed fold) must not appear in train_folds="
            f"{cfg.train_folds}, regardless of what cfg.test_fold is set to -- training on it is "
            f"never legitimate under any config."
        )
    if cfg.val_fold == _SEALED_FOLD:
        raise ValueError(
            f"val_fold must not equal {_SEALED_FOLD} (the sealed fold), regardless of what "
            f"cfg.test_fold is set to -- `folds()`'s 'val' key is exposed unconditionally "
            f"(never gated by `unseal`), so this would release real sealed-fold rows with no "
            f"warning at all. There is exactly one gated path to fold {_SEALED_FOLD}: "
            f"`cfg.test_fold == {_SEALED_FOLD}` plus `unseal=True` (or, for real use, "
            f"`winder.data.fold10_authorization.authorized_unseal`)."
        )


def folds(
    df: pd.DataFrame, cfg: FoldConfig | None = None, *, unseal: bool = False
) -> dict[str, pd.DataFrame]:
    """Split on PTB-XL's `strat_fold` per `cfg`: train, val, and (only if unsealed) test.

    Returns `{"train": ..., "val": ...}` and, ONLY when `unseal=True`, additionally
    `{"test": ...}`. Fold 10 is sealed by default, so the default return does not
    contain the key at all: `folds(df)["test"]` raises `KeyError` by design rather than
    handing back held-out data. Requesting it prints a banner to stderr and raises a
    `UserWarning`, so an accidental use is visible in every log.

    Also asserts the splits are patient-disjoint (PTB-XL's stratification guarantees
    it; we check rather than trust, since a leak would invalidate every number).
    """
    cfg = cfg or FoldConfig()
    _check_seal_invariant(cfg)
    if "strat_fold" not in df.columns or "patient_id" not in df.columns:
        raise ValueError("df must carry 'strat_fold' and 'patient_id'")
    f = df["strat_fold"].to_numpy()

    # Compute all three splits unconditionally -- the disjointness check below must see
    # the sealed test fold even when it isn't exposed in the return value. (Bug found by
    # audit: checking only `out`'s present keys made any excluded fold, not just the
    # sealed one, invisible to this check -- a patient split across a train fold and the
    # sealed fold passed silently under the default unseal=False.)
    train = df.loc[np.isin(f, cfg.train_folds)]
    val = df.loc[f == cfg.val_fold]
    test = df.loc[f == cfg.test_fold]

    # Backstop on the DATA itself, not just the config, deliberately redundant with
    # `_check_seal_invariant`'s own `val_fold`/`train_folds` checks above. Three distinct bypass
    # classes were found and closed in one night; the argument for checking actual rows, not just
    # the spelling of the config that produced them, is that a future refactor of `train`/`val`'s
    # own construction (above) could reintroduce a leak that a config-only check cannot see --
    # this fires on the split CONTENT regardless of how it got built.
    if (train["strat_fold"] == _SEALED_FOLD).any() or (val["strat_fold"] == _SEALED_FOLD).any():
        raise ValueError(
            f"a 'train' or 'val' split contains a row from the sealed fold ({_SEALED_FOLD}) -- "
            f"this must never happen regardless of cfg; this is the output-level backstop, "
            f"`_check_seal_invariant` should have already raised before this point."
        )

    seen: dict[int, str] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        for p in part["patient_id"].unique():
            prev = seen.setdefault(int(p), name)
            if prev != name:
                raise ValueError(
                    f"patient {p} appears in both {prev} and {name}: fold split is not "
                    f"patient-disjoint"
                )

    out = {"train": train, "val": val}
    if unseal:
        msg = (
            f"SEALED FOLD {cfg.test_fold} RELEASED: folds(unseal=True) was called. "
            f"This is legitimate only under a pre-registered protocol and must be recorded."
        )
        print("\n" + "!" * 78 + f"\n!! {msg}\n" + "!" * 78 + "\n", file=sys.stderr, flush=True)
        warnings.warn(msg, UserWarning, stacklevel=2)
        out["test"] = test
    return out


def calibration_subset(df: pd.DataFrame, cfg: FoldConfig | None = None) -> pd.DataFrame:
    """Patient-disjoint ~`cfg.calibration_frac` subset of `cfg.train_folds`, threshold
    setting ONLY.

    These records set permutation-null thresholds and must never enter a primary
    estimate. Folds outside `train_folds` are untouched by construction.

    Rule (deterministic given `cfg.calibration_seed`): take the unique `patient_id`s in
    `train_folds` in ascending order, permute them with
    `np.random.default_rng(seed)`, then walk the permutation accumulating whole
    patients until the cumulative RECORD count first reaches
    `calibration_frac * n_records`. Patients are the sampling unit, so the subset is
    exactly patient-disjoint from its complement (`train_minus_calibration`) and the
    realised fraction overshoots the target by at most one patient's worth of records.

    Returns the subset in ascending `ecg_id` order.
    """
    cfg = cfg or FoldConfig()
    _check_seal_invariant(cfg)
    if not 0.0 < cfg.calibration_frac < 1.0:
        raise ValueError(f"calibration_frac must be in (0, 1), got {cfg.calibration_frac}")
    pool = df.loc[np.isin(df["strat_fold"].to_numpy(), cfg.train_folds)]
    if len(pool) == 0:
        raise ValueError(f"no records in train_folds={cfg.train_folds}")
    counts = pool.groupby("patient_id", sort=True).size()
    pids = counts.index.to_numpy()
    sizes = counts.to_numpy()
    perm = np.random.default_rng(cfg.calibration_seed).permutation(len(pids))
    target = cfg.calibration_frac * len(pool)
    take = perm[: int(np.searchsorted(np.cumsum(sizes[perm]), target) + 1)]
    chosen = set(pids[take].tolist())
    sub = pool.loc[pool["patient_id"].isin(chosen)]
    return sub.sort_values("ecg_id", kind="stable")


def train_minus_calibration(df: pd.DataFrame, cfg: FoldConfig | None = None) -> pd.DataFrame:
    """Exact complement of `calibration_subset(df, cfg)` inside `cfg.train_folds`.

    Same `cfg` gives the same partition, so the primary estimate can exclude the
    calibration records without a second sampling decision. Patient-disjoint from the
    calibration subset by construction.
    """
    cfg = cfg or FoldConfig()
    _check_seal_invariant(cfg)  # calibration_subset() re-checks too; explicit here for clarity
    pool = df.loc[np.isin(df["strat_fold"].to_numpy(), cfg.train_folds)]
    cal = calibration_subset(df, cfg)
    keep = ~pool["patient_id"].isin(set(cal["patient_id"].unique().tolist()))
    return pool.loc[keep].sort_values("ecg_id", kind="stable")

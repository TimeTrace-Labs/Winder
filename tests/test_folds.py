"""folds.py tests.

Ported behaviour tests plus the seal-invariant test that IS bug fix #1: ttl-phase's
TRAIN_FOLDS/VAL_FOLD/TEST_FOLD were hardcoded, so "fold 10 in train_folds" was structurally
impossible there. Making fold membership config-driven removes that guarantee unless
`folds()` checks for it explicitly and unconditionally -- which is what these tests pin.
"""

import pathlib
import re
import warnings

import pandas as pd
import pytest

from winder.data.folds import FoldConfig, calibration_subset, folds, train_minus_calibration


def _toy_metadata(n_patients: int = 40, records_per_patient: int = 1) -> pd.DataFrame:
    """10 folds, patients assigned round-robin, one or more records per patient."""
    rows = []
    ecg_id = 1
    for pid in range(n_patients):
        fold = (pid % 10) + 1
        for _ in range(records_per_patient):
            rows.append({"ecg_id": ecg_id, "patient_id": pid, "strat_fold": fold})
            ecg_id += 1
    return pd.DataFrame(rows)


def test_folds_default_excludes_test_key() -> None:
    df = _toy_metadata()
    out = folds(df)
    assert set(out.keys()) == {"train", "val"}
    with pytest.raises(KeyError):
        _ = out["test"]


def test_folds_unseal_adds_test_key_and_warns() -> None:
    df = _toy_metadata()
    with pytest.warns(UserWarning, match="SEALED FOLD"):
        out = folds(df, unseal=True)
    assert "test" in out
    assert set(out["test"]["strat_fold"].unique()) == {10}


def test_folds_are_patient_disjoint() -> None:
    df = _toy_metadata()
    with pytest.warns(UserWarning, match="SEALED FOLD"):
        out = folds(df, unseal=True)
    all_pids = [set(part["patient_id"].unique()) for part in out.values()]
    for i in range(len(all_pids)):
        for j in range(i + 1, len(all_pids)):
            assert all_pids[i].isdisjoint(all_pids[j])


def test_folds_raises_on_patient_spanning_two_folds() -> None:
    # winder-nominal deviation: fold 0 (this repo's val_fold sentinel) is used here instead
    # of the reference repo's original fold-9 literal, because FoldConfig()'s default
    # train_folds is (1..9) here -- fold 9 is now a train fold, not val, so a patient with
    # records in folds 1 and 9 would land in the SAME split (train) and no longer trip this
    # check. Fold 0 reproduces the intended train-vs-val collision.
    df = _toy_metadata()
    # Corrupt: patient 0 has two records, one claimed for fold 1 and one for fold 0 (val).
    df.loc[len(df)] = {"ecg_id": 9999, "patient_id": 0, "strat_fold": 0}
    with pytest.raises(ValueError, match="not patient-disjoint"):
        folds(df)


# ------------------------------------------------------------------- seal invariant (bug #1)
def test_seal_invariant_rejects_test_fold_in_train_folds() -> None:
    df = _toy_metadata()
    cfg = FoldConfig(train_folds=(1, 2, 3, 10), test_fold=10)
    with pytest.raises(ValueError, match="must not appear in train_folds"):
        folds(df, cfg)


def test_seal_invariant_rejects_test_fold_equal_to_val_fold() -> None:
    df = _toy_metadata()
    cfg = FoldConfig(val_fold=10, test_fold=10)
    with pytest.raises(ValueError, match="must not appear in train_folds"):
        folds(df, cfg)


def test_seal_invariant_fires_even_when_unsealed() -> None:
    """The critical case: `unseal=True` must NOT bypass the seal-invariant check --
    unsealing controls whether the output carries a 'test' key, nothing else."""
    df = _toy_metadata()
    cfg = FoldConfig(train_folds=(1, 2, 3, 10), test_fold=10)
    with pytest.raises(ValueError, match="must not appear in train_folds"):
        folds(df, cfg, unseal=True)


# ------------------------------------------------------- seal invariant, bypass class 3 (found
# 2026-08-18: relabelling which FIELD holds the sealed fold walks it out through the
# unconditionally-exposed "val" key, since the original check only ever asked whether
# `cfg.test_fold` was reachable -- not whether the literal sealed fold was, under any name).
def test_seal_invariant_rejects_val_fold_equal_sealed_fold_even_if_test_fold_relabelled() -> None:
    """The actual bypass found this session: `val_fold=10` with `test_fold` moved somewhere else
    entirely (here, 0, `FoldConfig`'s own empty-sentinel convention) used to sail straight through
    `_check_seal_invariant`, because that check only ever compared against `cfg.test_fold`'s
    current value, never against the literal constant 10. `val` is exposed unconditionally in
    `folds()`'s return value (never gated by `unseal`), so this returned real fold-10 rows with
    no warning at all -- confirmed empirically on this exact config before the fix."""
    df = _toy_metadata()
    cfg = FoldConfig(train_folds=(1, 2, 3), val_fold=10, test_fold=0)
    with pytest.raises(ValueError, match="val_fold must not equal 10"):
        folds(df, cfg)


def test_seal_invariant_rejects_the_sealed_fold_in_train_folds_even_with_test_fold_relabelled() -> (
    None
):
    """Same bypass shape, the `train_folds` side: fold 10 could previously be smuggled into
    `train_folds` as long as `cfg.test_fold` was set to anything else."""
    df = _toy_metadata()
    # test_fold must differ from both val_fold and every entry of train_folds, so the FIRST
    # (pre-existing) check does not fire before this new one gets a chance to.
    cfg = FoldConfig(train_folds=(1, 2, 3, 10), val_fold=0, test_fold=5)
    with pytest.raises(ValueError, match="fold 10 \\(the sealed fold\\) must not appear"):
        folds(df, cfg)


def test_output_level_backstop_fires_on_a_hand_built_violating_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately redundant with the config-level checks above: proves `folds()` also refuses
    a config that computes real fold-10 rows into `train`/`val`, even if some future refactor of
    the config-level check were to miss a case -- this is a check on the DATA, not the config.

    Reaches the real module object via `sys.modules`, not `import winder.data.folds`/a dotted
    monkeypatch target string: `winder/data/__init__.py` re-exports the `folds` FUNCTION under
    the same name as its own defining submodule, so `winder.data.folds` as an attribute path
    resolves to the function, not the module, once `winder.data` has been imported.
    """
    import sys

    df = _toy_metadata()
    folds_module = sys.modules["winder.data.folds"]
    monkeypatch.setattr(folds_module, "_check_seal_invariant", lambda cfg: None)
    cfg = FoldConfig(train_folds=(1, 2, 3), val_fold=10, test_fold=0)
    with pytest.raises(ValueError, match="contains a row from the sealed fold"):
        folds(df, cfg)


def test_seal_invariant_passes_for_the_default_config() -> None:
    df = _toy_metadata()
    folds(df, FoldConfig())  # must not raise


def test_seal_invariant_fires_for_calibration_subset_too() -> None:
    """Audit-found bug: calibration_subset/train_minus_calibration built their pool
    directly from cfg.train_folds without ever calling the seal check, so a config with
    the sealed fold folded into train_folds would silently leak sealed records into the
    "threshold setting ONLY" calibration set and its complement."""
    df = _toy_metadata()
    cfg = FoldConfig(train_folds=(1, 2, 3, 10), test_fold=10)
    with pytest.raises(ValueError, match="must not appear in train_folds"):
        calibration_subset(df, cfg)
    with pytest.raises(ValueError, match="must not appear in train_folds"):
        train_minus_calibration(df, cfg)


def test_disjointness_check_sees_the_sealed_fold_even_when_unseal_is_false() -> None:
    """Audit-found bug: the disjointness loop used to iterate only the splits present in
    the return value, so with the default unseal=False the sealed test fold was invisible
    to it -- a patient split across a train fold and the sealed fold passed silently."""
    df = _toy_metadata()
    # patient 0 already has a fold-1 record (round-robin assignment); add a second
    # record for the same patient in the sealed fold 10.
    df.loc[len(df)] = {"ecg_id": 9999, "patient_id": 0, "strat_fold": 10}
    with pytest.raises(ValueError, match="not patient-disjoint"):
        folds(df)  # unseal=False (the default) must still catch this


# --------------------------------------------------------------------------- calibration
def test_calibration_subset_is_patient_disjoint_from_complement() -> None:
    df = _toy_metadata(n_patients=200, records_per_patient=3)
    cfg = FoldConfig(calibration_frac=0.2, calibration_seed=0)
    cal = calibration_subset(df, cfg)
    rest = train_minus_calibration(df, cfg)
    assert set(cal.patient_id).isdisjoint(set(rest.patient_id))
    assert len(cal) + len(rest) == sum(df.strat_fold.isin(cfg.train_folds))


def test_calibration_subset_only_touches_train_folds() -> None:
    df = _toy_metadata(n_patients=200, records_per_patient=2)
    cfg = FoldConfig()
    cal = calibration_subset(df, cfg)
    assert set(cal.strat_fold.unique()) <= set(cfg.train_folds)


def test_calibration_subset_is_deterministic_given_seed() -> None:
    df = _toy_metadata(n_patients=200, records_per_patient=2)
    cfg = FoldConfig(calibration_seed=42)
    a = calibration_subset(df, cfg)
    b = calibration_subset(df, cfg)
    assert list(a.ecg_id) == list(b.ecg_id)


def test_calibration_subset_realises_approximately_the_requested_fraction() -> None:
    df = _toy_metadata(n_patients=500, records_per_patient=3)
    cfg = FoldConfig(calibration_frac=0.15, calibration_seed=0)
    cal = calibration_subset(df, cfg)
    train_n = int((df.strat_fold.isin(cfg.train_folds)).sum())
    assert abs(len(cal) / train_n - 0.15) < 0.05


def test_calibration_frac_out_of_range_raises() -> None:
    df = _toy_metadata()
    with pytest.raises(ValueError):
        calibration_subset(df, FoldConfig(calibration_frac=1.5))


def test_no_unexpected_warnings_on_the_sealed_path() -> None:
    """Guard against a regression that makes the default (sealed) path noisy."""
    df = _toy_metadata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folds(df)  # must not warn
        calibration_subset(df)
        train_minus_calibration(df)


# ------------------------------------------------------- winder-nominal fold-discipline
def test_default_fold_config_is_folds_1_through_9() -> None:
    """winder-nominal's default differs from the reference repo's: train_folds=(1..9),
    val_fold=0 (a sentinel -- PTB-XL's real strat_fold values are 1..10, so val_fold=0
    never matches a real record). Empirically verified (not assumed): folds() does not
    error on a val_fold with zero matching rows -- it returns a "val" key holding an
    empty-but-present DataFrame, the same shape contract as a non-empty split.
    """
    cfg = FoldConfig()
    assert cfg.train_folds == (1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert cfg.val_fold == 0
    assert cfg.test_fold == 10

    df = _toy_metadata()  # real strat_fold values 1..10, round-robin; fold 0 never appears
    out = folds(df, cfg)
    assert set(out.keys()) == {"train", "val"}
    assert len(out["val"]) == 0  # fold 0 selects no real records
    assert len(out["train"]) > 0


#: Modules deliberately excluded from `test_no_call_site_unseals`'s scan, and why. Exactly two,
#: both audited, neither added lightly:
#: - `folds.py` DEFINES `unseal` -- the literal string appears only in its own docstring and in
#:   the stderr banner it prints when `unseal=True` is legitimately passed by a caller elsewhere.
#: - `fold10_authorization.py` is the one, single, hash-gated GATE in front of `folds()` --
#:   `winder.data.fold10_authorization.authorized_unseal`'s own module docstring explains why a
#:   second module, not zero, is the right count: the real day-of eval script calls
#:   `authorized_unseal(df, cfg)` and never needs to write the literal pattern itself at all.
_UNSEAL_SCAN_EXEMPT_MODULES = ("folds.py", "fold10_authorization.py")

#: Broadened past a single literal substring (advisor ruling, 2026-08-18 fold-10 review-ceremony
#: panel: "the current seal is a literal string-scan... defeatable by `**{'unseal': True}` or by
#: filtering `strat_fold == 10` directly"). Catches `unseal=True`, `unseal = True`,
#: `unseal:True`/`"unseal": True`/`'unseal': True` (the dict-literal/`**{...}` unpacking form),
#: regardless of internal whitespace.
_UNSEAL_TRUE_PATTERN = re.compile(r"""unseal["']?\s*[:=]\s*True""")


def test_no_call_site_unseals() -> None:
    """Standing invariant for this repo's own development: fold 10 must never be opened.

    Scans src/ and scripts/ for any spelling of "unseal" being set to `True` -- not just the
    single literal substring `unseal=True`, which a `**{"unseal": True}` unpacking call or a
    stray space (`unseal = True`) would previously have slipped past. `scripts/` is empty at
    this point in the build for the real day-of driver; the scan still runs so it starts
    enforcing the moment a script exists. See `_UNSEAL_SCAN_EXEMPT_MODULES` for the exactly-two
    modules excluded, and why.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    exempt = {repo_root / "src" / "winder" / "data" / name for name in _UNSEAL_SCAN_EXEMPT_MODULES}
    offenders: list[str] = []
    for base in ("src", "scripts"):
        base_dir = repo_root / base
        if not base_dir.is_dir():
            continue
        for path in base_dir.rglob("*.py"):
            if path in exempt:
                continue
            if _UNSEAL_TRUE_PATTERN.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == [], f"found an 'unseal' set to True outside the exempt modules: {offenders}"


#: Heuristic, not a proof: catches `strat_fold` compared against the literal `10` in either
#: token order, within a short window, so a record could plausibly be read as "is this fold 10"
#: without ever calling `folds()` at all -- the bypass the same advisor ruling flagged. A
#: determined rewrite (e.g. aliasing `10` to a named constant first) can still evade a text scan;
#: this raises the bar against an accidental or casual bypass, it is not a security boundary.
_STRAT_FOLD_TEN_PATTERN = re.compile(
    r"strat_fold[^\n]{0,30}==[^\n]{0,10}\b10\b|"
    r"\b10\b[^\n]{0,10}==[^\n]{0,30}strat_fold"
)


def test_no_call_site_bypasses_folds_via_direct_strat_fold_comparison() -> None:
    """`folds()` is not the only way to touch `strat_fold` -- it is just a DataFrame column, so
    any module could filter `df.strat_fold == 10` directly and reach the sealed fold without
    ever calling `folds()`, its seal-invariant check, its disjointness check, or its warning
    banner. This scans for that specific bypass shape. `folds.py` itself is exempt: it is the
    module whose own `_check_seal_invariant`/`folds()` implementation legitimately compares
    `test_fold` (not the literal `10`) against fold membership.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    exempt = repo_root / "src" / "winder" / "data" / "folds.py"
    offenders: list[str] = []
    for base in ("src", "scripts"):
        base_dir = repo_root / base
        if not base_dir.is_dir():
            continue
        for path in base_dir.rglob("*.py"):
            if path == exempt:
                continue
            if _STRAT_FOLD_TEN_PATTERN.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == [], (
        f"found a direct strat_fold-vs-10 comparison outside folds.py -- this bypasses the "
        f"seal entirely, route it through folds()/authorized_unseal() instead: {offenders}"
    )

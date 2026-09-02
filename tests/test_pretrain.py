"""Tests for scripts/pretrain.py (Phase P7): the lean training entrypoint.

Split, as elsewhere in this project, into fast always-run unit tests on pure logic (CLI
validation, the integrity-set filter, the config-diff guard's own expected-diff computation -- no
I/O, no model, no GPU) and a skip-gated real-data integration test that exercises the ACTUAL
wiring end-to-end (real PTB-XL, the copied-in reference artifacts, a real conv_trunk model) at a
handful of steps -- fast enough to run in the standard `pytest` gate, but real compute, not a
mock: this is deliberately NOT a substitute for the separate 200-step x 4-arm smoke
(`scripts/p7_smoke.py`), which stays a manually-invoked script for the same reason
`scripts/accept.py`'s own numeric reproduction isn't bundled into `pytest` either.

`tests/conftest.py` puts `scripts/` on `sys.path`, so `import pretrain` here means
`scripts/pretrain.py`, not the `winder` package.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import pretrain
import pytest
import torch

from winder.config import resolve_operator_config
from winder.determinism import generator
from winder.jepa import checkpoint as ckpt_mod
from winder.jepa.model import build_jepa
from winder.jepa.train import StepMetrics
from winder.operators.harmonic import HarmonicTransport
from winder.operators.registry import OPERATOR_REGISTRY
from winder.paths import default_data_root

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFERENCE_ROOT = os.path.join(_REPO_ROOT, "artifacts", "reference")
_PTBXL_ROOT = default_data_root()
_HAS_PTBXL_ROOT = os.path.isfile(os.path.join(_PTBXL_ROOT, "ptbxl_database.csv"))
_LEGACY_LEAD_STATS = os.path.join(_REFERENCE_ROOT, "lead_stats_f1to8_legacy.json")
_MANIFEST_PATH = os.path.join(_REFERENCE_ROOT, "manifest.parquet")
_THETA_TOKENS_PATH = os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz")
_FIN_SEED0_CONFIG = os.path.join(_REFERENCE_ROOT, "FIN_seed0", "checkpoint_step5000", "config.yaml")
_HAS_SMOKE_INPUTS = (
    _HAS_PTBXL_ROOT
    and os.path.isfile(_LEGACY_LEAD_STATS)
    and os.path.isfile(_MANIFEST_PATH)
    and os.path.isfile(_THETA_TOKENS_PATH)
    and os.path.isfile(_FIN_SEED0_CONFIG)
)

_RECIPE_ARGS = [
    "--lambda-sig",
    "0.15",
    "--k0",
    "4",
    "--n-j",
    "1,2,3,4,5,6,7,8,9,10",
    "--k-j",
    "24,24,20,16,12,10,8,6,4,2",
    "--encoder-name",
    "conv_trunk",
    "--predictor-json",
    '{"n_layers":4}',
    "--augment",
    "gauss,powerline,wander,ampmod,leaddrop,leadgain",
    "--augment-prob",
    "0.5",
    "--train-folds",
    "1,2,3,4,5,6,7,8,9",
]


def _smoke_argv(*, seed: int, lambda_trans: float, steps: int, artifacts_dir: str) -> list[str]:
    """The recipe's own fixed flags, at reduced `steps` and the legacy (folds 1-8) lead-stats
    used only for this smoke -- never `artifacts/lead_stats_f1to9.json`, which is Phase P8's own
    refit, not yet built."""
    return [
        *_RECIPE_ARGS,
        "--batch-size",
        "8",
        "--steps",
        str(steps),
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--artifacts-dir",
        artifacts_dir,
        "--data-root",
        _PTBXL_ROOT,
        "--lead-stats-path",
        _LEGACY_LEAD_STATS,
        "--manifest-path",
        _MANIFEST_PATH,
        "--theta-tokens-path",
        _THETA_TOKENS_PATH,
        "--transport-arm",
        "cyclic",
        "--lambda-trans",
        str(lambda_trans),
    ]


# =========================================================== _phase_tagged_batches's num_workers


def test_phase_tagged_batches_forwards_num_workers_to_the_dataloader(monkeypatch: Any) -> None:
    """--num-workers's only job is to reach DataLoader's own num_workers= kwarg -- checked here
    without a real dataset/model (steps=0 means the generator's own while-loop body never runs,
    so the stub DataLoader below is never actually iterated), since the default-preserving and
    real-training-continuity claims are already covered by the resume tests below, which run
    this at num_workers=0 against real data."""
    captured: dict[str, object] = {}

    class _StubDataLoader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        def __iter__(self) -> object:
            raise AssertionError("steps=0 must never iterate the loader")

    monkeypatch.setattr(pretrain, "DataLoader", _StubDataLoader)
    gen = generator(0, "test")
    dataset: list[int] = []
    list(pretrain._phase_tagged_batches(dataset, 8, 0, gen, torch.device("cpu"), num_workers=3))  # type: ignore[arg-type]
    assert captured["num_workers"] == 3


# ============================================================================ pure CLI validation


def test_augment_prob_out_of_range_exits() -> None:
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")
    argv += ["--augment-prob", "1.5"]
    with pytest.raises(SystemExit):
        pretrain.main(argv)


def test_augment_prob_without_augment_is_rejected_as_silently_inert() -> None:
    argv = [a for a in _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")]
    # Replace --augment's value with "" and bump --augment-prob off its 0.5 default.
    idx = argv.index("--augment")
    argv[idx + 1] = ""
    argv[argv.index("--augment-prob") + 1] = "0.9"
    with pytest.raises(SystemExit):
        pretrain.main(argv)


def test_malformed_n_j_exits_with_a_named_flag_error() -> None:
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")
    argv[argv.index("--n-j") + 1] = "1,two,3"
    with pytest.raises(SystemExit):
        pretrain.main(argv)


def test_malformed_predictor_json_exits() -> None:
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")
    argv[argv.index("--predictor-json") + 1] = "not json"
    with pytest.raises(SystemExit):
        pretrain.main(argv)


def test_negative_checkpoint_at_step_exits() -> None:
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")
    argv += ["--checkpoint-at", "-5"]
    with pytest.raises(SystemExit):
        pretrain.main(argv)


def test_unknown_transport_arm_choice_exits() -> None:
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")
    argv[argv.index("--transport-arm") + 1] = "none"
    with pytest.raises(SystemExit):
        pretrain.main(argv)


def test_unknown_encoder_name_choice_exits() -> None:
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir="unused")
    argv[argv.index("--encoder-name") + 1] = "not_a_real_encoder"
    with pytest.raises(SystemExit):
        pretrain.main(argv)


# ================================================================= _restrict_to_integrity_set etc.


def test_restrict_to_integrity_set_drops_only_the_three_excluded_reason_codes() -> None:
    metadata = pd.DataFrame({"ecg_id": [1, 2, 3, 4]})
    manifest_df = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "reason_code": ["", "READ_ERROR", "HIGH_RR_CV", "WRONG_SHAPE"],
            "quality_flags": ["", "", "", ""],
        }
    )
    kept = pretrain._restrict_to_integrity_set(metadata, manifest_df)
    # HIGH_RR_CV is a phase-clock QC code, not in _INTEGRITY_EXCLUDE_CODES -- record 3 survives.
    assert sorted(kept["ecg_id"].tolist()) == [1, 3]


def test_restrict_to_integrity_set_raises_on_a_metadata_id_missing_from_the_manifest() -> None:
    metadata = pd.DataFrame({"ecg_id": [1, 2]})
    manifest_df = pd.DataFrame({"ecg_id": [1], "reason_code": [""], "quality_flags": [""]})
    with pytest.raises(ValueError, match="no row in the manifest"):
        pretrain._restrict_to_integrity_set(metadata, manifest_df)


def test_flat_signal_count_counts_only_requested_ids() -> None:
    manifest_df = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "reason_code": ["FLAT_SIGNAL", "FLAT_SIGNAL", ""],
            "quality_flags": ["", "", ""],
        }
    )
    assert pretrain._flat_signal_count(manifest_df, pd.Series([1, 3])) == 1
    assert pretrain._flat_signal_count(manifest_df, pd.Series([1, 2, 3])) == 2


# ======================================================================= _expected_config_diff


def _base_args(**overrides: object) -> argparse.Namespace:
    """A real `argparse.Namespace`, naming only the fields `_expected_config_diff` actually
    reads -- `_expected_config_diff` is typed against `argparse.Namespace` exactly, so this stays
    a real instance rather than a duck-typed stand-in."""
    defaults: dict[str, object] = dict(
        steps=30000,
        lambda_sig=0.15,
        lambda_trans=1.0,
        seed=0,
        augment="gauss,powerline,wander,ampmod,leaddrop,leadgain",
        augment_prob=0.5,
        encoder_name="conv_trunk",
        transport_arm="cyclic",
        k0=4,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


_REFERENCE_FLAT = {
    "train.n_steps": 30000,
    "train.lambda_sig": 0.15,
    "train.lambda_trans": 1.0,
    "train.seed_pretrain": 0,
    "train.augment": "gauss,powerline,wander,ampmod,leaddrop,leadgain",
    "train.augment_prob": 0.5,
    "jepa.encoder_name": "conv_trunk",
    "arm.name": "cyclic_seed0",
    "arm.seed": 0,
    "arm.operator_name": "cyclic",
    "arm.operator.k0": 4,
    "arm.operator.n_j": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "arm.operator.k_j": [24, 24, 20, 16, 12, 10, 8, 6, 4, 2],
    "jepa.predictor.n_layers": 4,
}
_N_J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_K_J = [24, 24, 20, 16, 12, 10, 8, 6, 4, 2]


def test_expected_config_diff_is_empty_for_the_reference_recipe_itself() -> None:
    diff = pretrain._expected_config_diff(
        _base_args(), "cyclic_seed0", _N_J, _K_J, {"n_layers": 4}, _REFERENCE_FLAT
    )
    assert diff == {}


def test_expected_config_diff_predicts_only_lambda_trans_for_the_control_arm() -> None:
    diff = pretrain._expected_config_diff(
        _base_args(lambda_trans=0.0), "cyclic_seed0", _N_J, _K_J, {"n_layers": 4}, _REFERENCE_FLAT
    )
    assert diff == {"train.lambda_trans": 0.0}


def test_expected_config_diff_predicts_seed_fields_but_not_jepa_seed_pretrain() -> None:
    diff = pretrain._expected_config_diff(
        _base_args(seed=1),
        "cyclic_seed1",
        _N_J,
        _K_J,
        {"n_layers": 4},
        _REFERENCE_FLAT,
    )
    assert diff == {
        "train.seed_pretrain": 1,
        "arm.seed": 1,
        "arm.name": "cyclic_seed1",
    }
    assert "jepa.seed_pretrain" not in diff


def test_expected_config_diff_predicts_n_steps_for_a_reduced_step_smoke_run() -> None:
    diff = pretrain._expected_config_diff(
        _base_args(steps=200), "cyclic_seed0", _N_J, _K_J, {"n_layers": 4}, _REFERENCE_FLAT
    )
    assert diff == {"train.n_steps": 200}


# ================================ Task 4: the 4 real P8 launches' expected config-diff, empirically
#
# Unlike the hardcoded-_REFERENCE_FLAT tests above (which check `_expected_config_diff`'s own
# logic in isolation), these drive `winder.ablations.resolve_arm` -- the ACTUAL argv each of the
# four real GPU launches will use -- through `_expected_config_diff`, against `flatten_yaml` of
# the REAL `artifacts/reference/FIN_seed0/checkpoint_step5000/config.yaml` file on disk. This is
# "confirm empirically, by calling the function" (P8 prep brief, Task 4), not a re-derivation by
# hand: if `resolve_arm`'s flags or the on-disk reference config ever drift from what this test
# expects, this fails here rather than being discovered mid-launch.


def _typed_args_from_resolved_flags(flags: dict[str, str]) -> argparse.Namespace:
    """Coerces `winder.ablations.parse_flag_pairs(resolve_arm(...))`'s string-valued flag dict
    into the typed `argparse.Namespace` `_expected_config_diff` reads -- argparse itself does this
    coercion for a real CLI invocation (`type=int`/`type=float` on each `add_argument`), so this
    mirrors that, not reinvents it. Uncoerced strings would produce spurious diffs, e.g.
    `steps="30000"` (str) never equalling `reference_flat["train.n_steps"] == 30000` (int)."""
    return argparse.Namespace(
        steps=int(flags["--steps"]),
        lambda_sig=float(flags["--lambda-sig"]),
        lambda_trans=float(flags["--lambda-trans"]),
        seed=int(flags["--seed"]),
        augment=flags["--augment"],
        augment_prob=float(flags["--augment-prob"]),
        encoder_name=flags["--encoder-name"],
        transport_arm=flags["--transport-arm"],
        k0=int(flags["--k0"]),
    )


#: combo name -> (ABLATION_ARMS registry key, seed) -- Phase P8's own four planned launches.
_P8_LAUNCH_COMBOS: dict[str, tuple[str, int]] = {
    "signal_seed0": ("signal", 0),
    "control_seed0": ("control", 0),
    "signal_seed1": ("signal", 1),
    "control_seed1": ("control", 1),
}

#: The brief's own pre-registered expected diffs, one per combo (module docstring above).
_P8_LAUNCH_EXPECTED_DIFFS: dict[str, dict[str, object]] = {
    "signal_seed0": {},
    "control_seed0": {"train.lambda_trans": 0.0},
    "signal_seed1": {"train.seed_pretrain": 1, "arm.seed": 1, "arm.name": "cyclic_seed1"},
    "control_seed1": {
        "train.lambda_trans": 0.0,
        "train.seed_pretrain": 1,
        "arm.seed": 1,
        "arm.name": "cyclic_seed1",
    },
}


@pytest.mark.skipif(
    not os.path.isfile(_FIN_SEED0_CONFIG), reason="real reference FIN_seed0 config.yaml absent"
)
@pytest.mark.parametrize("combo_name", sorted(_P8_LAUNCH_COMBOS))
def test_p8_launch_expected_config_diff_matches_the_real_reference_config(combo_name: str) -> None:
    from winder.ablations import parse_flag_pairs, resolve_arm
    from winder.config import flatten_yaml

    arm_key, seed = _P8_LAUNCH_COMBOS[combo_name]
    resolved = parse_flag_pairs(resolve_arm(arm_key, seed=seed, artifacts_dir="unused"))
    args = _typed_args_from_resolved_flags(resolved)
    n_j = pretrain._parse_int_csv(resolved["--n-j"], "--n-j")
    k_j = pretrain._parse_int_csv(resolved["--k-j"], "--k-j")
    predictor_overrides = pretrain._parse_json_dict(
        resolved["--predictor-json"], "--predictor-json"
    )
    arm_name = f"{args.transport_arm}_seed{args.seed}"

    with open(_FIN_SEED0_CONFIG, encoding="utf-8") as fh:
        reference_flat = flatten_yaml(fh.read())

    diff = pretrain._expected_config_diff(
        args, arm_name, n_j, k_j, predictor_overrides, reference_flat
    )
    assert diff == _P8_LAUNCH_EXPECTED_DIFFS[combo_name]


# ============================================================ full-pipeline guard-has-teeth check


@pytest.mark.skipif(not _HAS_SMOKE_INPUTS, reason="real PTB-XL data or reference artifacts absent")
def test_config_diff_guard_fires_loud_before_touching_any_data(tmp_path: object) -> None:
    """A deliberately WRONG --reference-config-path (one this run's own resolved config can never
    match) must raise before `main()` even reaches `load_metadata` -- proving the guard actually
    executes and has teeth, not merely exists. Uses a bogus --data-root too, so if the guard were
    ever silently skipped, this test would instead fail on a FileNotFoundError from load_metadata
    (a different, distinguishing failure), not silently pass."""
    bad_reference = os.path.join(str(tmp_path), "bad_reference_config.yaml")
    with open(_FIN_SEED0_CONFIG, encoding="utf-8") as fh:
        text = fh.read()
    with open(bad_reference, "w", encoding="utf-8") as fh:
        # Mutate a field no CLI flag here explains -- e.g. n_tokens -- so the guard's own DRIFT
        # category must fire.
        fh.write(text.replace("n_tokens: 125", "n_tokens: 999"))

    argv = _smoke_argv(
        seed=0, lambda_trans=1.0, steps=1, artifacts_dir=os.path.join(str(tmp_path), "artifacts")
    )
    argv += ["--reference-config-path", bad_reference, "--data-root", "/definitely/not/a/real/path"]
    with pytest.raises(ValueError, match="DRIFT"):
        pretrain.main(argv)


# ============================================================ skip-gated real-data integration


@pytest.mark.skipif(not _HAS_SMOKE_INPUTS, reason="real PTB-XL data or reference artifacts absent")
def test_main_runs_two_real_steps_and_writes_a_strictly_reloadable_checkpoint(
    tmp_path: object,
) -> None:
    """The real end-to-end wiring, at a handful of steps: real PTB-XL, the copied-in legacy
    lead-stats/manifest/theta artifacts, a real conv_trunk model -- fast enough for the standard
    gate, genuine compute (per this project's own "a baby-run on compute, not a mock" standard),
    never a substitute for the full 200-step x 4-arm smoke."""
    artifacts_dir = os.path.join(str(tmp_path), "signal_seed0")
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=2, artifacts_dir=artifacts_dir)
    argv += ["--checkpoint-at", "1"]

    exit_code = pretrain.main(argv)
    assert exit_code == 0

    # StepMetrics field names match exactly -- s2_history.jsonl rows are asdict(StepMetrics), not
    # a renamed/reshaped subset.
    history_path = os.path.join(artifacts_dir, "s2_history.jsonl")
    with open(history_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 2
    expected_fields = {f.name for f in StepMetrics.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    assert set(rows[0]) == expected_fields

    # No NaN in fields that must be finite for EVERY arm; the signal-only transport fields must
    # ALSO be finite here (lambda_trans=1.0); the frame/record-level fields stay NaN by design
    # (sigreg_frame="raw", lambda_sig_record=0.0 -- neither is exposed as a CLI flag).
    always_finite = {
        "lr",
        "pred_loss",
        "persistence_loss",
        "sigreg_loss",
        "total_loss",
        "grad_norm",
        "cutoff_mean",
    }
    signal_only_finite = {"trans_loss", "trans_floor", "trans_gain", "trans_directional"}
    always_nan = {"trans_radial", "theta_valid_frac", "sigreg_n_records", "sigreg_record_loss"}
    for row in rows:
        for field in always_finite | signal_only_finite:
            assert row[field] is not None and row[field] == row[field], f"{field} was NaN: {row}"
        for field in always_nan:
            assert row[field] != row[field], f"{field} should be NaN, got {row[field]}"

    # Both checkpoints (mid-run at step 1, final at step 2) strict-load, 80 model keys + an
    # operator_state_dict, matching Phase P6's own validated contract for the reference
    # checkpoints -- including this arm's own operator, since it is ALWAYS constructed.
    for step_dir in ("checkpoint_step1", "checkpoint"):
        state = torch.load(os.path.join(artifacts_dir, step_dir, "state.pt"), weights_only=False)
        assert len(state["model_state_dict"]) == 80
        assert "operator_state_dict" in state

        with open(os.path.join(artifacts_dir, step_dir, "config.yaml"), encoding="utf-8") as fh:
            config_yaml = fh.read()
        jepa_cfg = ckpt_mod.jepa_config_from_yaml(config_yaml)
        arm_cfg = ckpt_mod.arm_config_from_yaml(config_yaml)
        assert arm_cfg is not None
        model = build_jepa(jepa_cfg, generator=generator(0, "handshake"))
        _, operator_ctor = OPERATOR_REGISTRY[arm_cfg.operator_name]
        operator = operator_ctor(resolve_operator_config(arm_cfg))
        assert isinstance(operator, HarmonicTransport)  # true of every registered operator
        loaded = ckpt_mod.load_checkpoint(
            os.path.join(artifacts_dir, step_dir), model=model, operator=operator
        )
        assert loaded.step > 0

    # The config-diff guard's own verified diff is recorded in the summary for provenance.
    with open(os.path.join(artifacts_dir, "s2_summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["config_diff_vs_reference"] == {"train.n_steps": [30000, 2]}


@pytest.mark.skipif(not _HAS_SMOKE_INPUTS, reason="real PTB-XL data or reference artifacts absent")
def test_control_arm_checkpoint_has_the_same_structure_as_the_signal_arm(
    tmp_path: object,
) -> None:
    """The control's operator is ALWAYS constructed, just zero-weighted -- its checkpoint must
    show the SAME 80-key model_state_dict and an operator_state_dict, never a structurally
    different, operator-less bundle."""
    artifacts_dir = os.path.join(str(tmp_path), "control_seed0")
    argv = _smoke_argv(seed=0, lambda_trans=0.0, steps=1, artifacts_dir=artifacts_dir)
    assert pretrain.main(argv) == 0

    state = torch.load(os.path.join(artifacts_dir, "checkpoint", "state.pt"), weights_only=False)
    assert len(state["model_state_dict"]) == 80
    assert "operator_state_dict" in state

    with open(os.path.join(artifacts_dir, "s2_history.jsonl"), encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    # lambda_trans=0.0 structurally skips the transport block -- every trans_*/closure_residual
    # field is the NaN "not applicable" sentinel here, unlike the signal arm above.
    skipped_fields = (
        "trans_loss",
        "trans_floor",
        "trans_gain",
        "trans_directional",
        "closure_residual",
    )
    for field in skipped_fields:
        assert row[field] != row[field], (
            f"{field} should be NaN for the control arm, got {row[field]}"
        )


@pytest.mark.skipif(not _HAS_SMOKE_INPUTS, reason="real PTB-XL data or reference artifacts absent")
def test_checkpoint_meta_discriminates_train_folds_from_lead_stats_folds(
    tmp_path: object,
) -> None:
    """P8 prep Task 3: `meta.json` must independently reflect the ACTUAL `--train-folds` pool
    this run trained over and the ACTUAL fold set the `--lead-stats-path` artifact was itself
    fitted on -- two provenance surfaces this run deliberately mismatches (folds 1-9 for
    training, the LEGACY folds-1-8 lead-stats file for normalization) specifically so a bug that
    conflated the two (e.g. `meta["lead_stats_folds"]` silently mirroring `--train-folds` instead
    of reading `lead_stats.folds`) would be caught here, not discovered only once a real fold-9
    vs. fold-10-blinded eval needed to trust this distinction (build plan's P9 labeling / the
    eventual sealed-fold protocol)."""
    from winder.data.integrity import sha256_file

    artifacts_dir = os.path.join(str(tmp_path), "signal_seed0")
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=1, artifacts_dir=artifacts_dir)
    assert pretrain.main(argv) == 0

    with open(os.path.join(artifacts_dir, "checkpoint", "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    # train_folds mirrors --train-folds (1-9), NOT the legacy lead-stats file's own fold set.
    assert meta["train_folds"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # lead_stats_folds mirrors the LEGACY lead-stats artifact's own folds (1-8) -- a genuinely
    # different set from train_folds in this run, by construction.
    assert meta["lead_stats_folds"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert meta["lead_stats_path"] == os.path.abspath(_LEGACY_LEAD_STATS)
    assert meta["lead_stats_sha256"] == sha256_file(_LEGACY_LEAD_STATS)
    assert meta["manifest_path"] == os.path.abspath(_MANIFEST_PATH)
    assert meta["manifest_sha256"] == sha256_file(_MANIFEST_PATH)


# ==================================================================================== --resume-from


@pytest.mark.skipif(not _HAS_SMOKE_INPUTS, reason="real PTB-XL data or reference artifacts absent")
def test_resume_from_continues_the_same_schedule_not_a_new_one(tmp_path: object) -> None:
    """The actual guarantee `--resume-from` exists for: a checkpoint written mid-way through an
    8-step schedule, resumed with the SAME `--steps 8`, must continue that schedule -- not
    restart a fresh one sized to the remaining step count. `lr_schedule` is a pure function of
    `(step, cfg.n_steps, cfg.warmup_steps)` with no randomness, so the LR at step 7 is checked
    exactly against an uninterrupted 8-step run's own LR at step 7. `warmup_steps=5` (this
    recipe's config), so step 7 is past warmup, in the cosine-decay branch whose denominator is
    exactly the bug this test guards: feeding `cfg.n_steps` the REMAINING count (5) instead of
    the schedule TOTAL (8) changes that denominator and desyncs the LR silently, not loudly.

    Losses are checked for finiteness only, not equality against the uninterrupted run: resuming
    `data_order` is epoch-granular (`checkpoint.py`'s own documented caveat) -- a freshly
    constructed DataLoader on resume draws a new shuffle permutation rather than continuing
    mid-epoch, so the resumed run's step-7 batch is a genuinely different sample of the pool than
    the uninterrupted run's, and their losses are expected to differ.
    """
    uninterrupted_dir = os.path.join(str(tmp_path), "uninterrupted")
    argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=8, artifacts_dir=uninterrupted_dir)
    argv += ["--checkpoint-at", "8"]
    assert pretrain.main(argv) == 0
    with open(os.path.join(uninterrupted_dir, "s2_history.jsonl"), encoding="utf-8") as fh:
        uninterrupted_rows = [json.loads(line) for line in fh]
    assert len(uninterrupted_rows) == 8
    assert uninterrupted_rows[-1]["step"] == 7

    # Simulates a kill-after-checkpoint. `--checkpoint-at 3` does not stop the run early -- it is
    # a MID-run trigger fired while training continues to `--steps`' full length -- so this call
    # runs all 8 steps to completion for convenience; only `checkpoint_step3/` (written after
    # step index 2, i.e. 3 steps completed) is ever read below. `checkpoint.save_checkpoint` is
    # synchronous (module docstring's own kill-safety claim), so that bundle is byte-identical to
    # what a real process actually killed right after writing it would have left on disk. The
    # history file this same run also wrote is deleted below: a genuinely killed process's own
    # `s2_history.jsonl` would hold only steps 0-2, not the full run this convenience shortcut
    # happened to also produce.
    resumed_dir = os.path.join(str(tmp_path), "resumed")
    first_argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=8, artifacts_dir=resumed_dir)
    first_argv += ["--checkpoint-at", "3"]
    assert pretrain.main(first_argv) == 0
    history_path = os.path.join(resumed_dir, "s2_history.jsonl")
    with open(history_path, encoding="utf-8") as fh:
        first_call_rows = [json.loads(line) for line in fh]
    with open(history_path, "w", encoding="utf-8") as fh:
        for row in first_call_rows[:3]:
            fh.write(json.dumps(row) + "\n")

    second_argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=8, artifacts_dir=resumed_dir)
    second_argv += [
        "--checkpoint-at",
        "8",
        "--resume-from",
        os.path.join(resumed_dir, "checkpoint_step3"),
    ]
    assert pretrain.main(second_argv) == 0
    with open(os.path.join(resumed_dir, "s2_history.jsonl"), encoding="utf-8") as fh:
        resumed_rows = [json.loads(line) for line in fh]
    # main() only truncates s2_history.jsonl `if not args.resume_from` -- the resumed call
    # APPENDS to the (here, hand-truncated-to-simulate-a-kill) history file, giving one
    # continuous log across the resume boundary, not a fresh 5-row file starting at step 3.
    assert [r["step"] for r in resumed_rows] == [0, 1, 2, 3, 4, 5, 6, 7]

    resumed_final = resumed_rows[-1]
    uninterrupted_final = uninterrupted_rows[-1]
    assert resumed_final["step"] == uninterrupted_final["step"] == 7
    assert resumed_final["lr"] == pytest.approx(uninterrupted_final["lr"])
    for field in ("pred_loss", "sigreg_loss", "total_loss", "trans_loss", "trans_gain", "lr"):
        assert resumed_final[field] == resumed_final[field]  # not NaN
        assert abs(resumed_final[field]) != float("inf")


@pytest.mark.skipif(not _HAS_SMOKE_INPUTS, reason="real PTB-XL data or reference artifacts absent")
def test_resume_from_rejects_a_checkpoint_saved_under_different_flags(tmp_path: object) -> None:
    """Resuming with a DIFFERENT recipe than the one that wrote the checkpoint (here,
    lambda_trans changed 1.0 -> 0.0) must raise rather than silently continuing a schedule under
    a config the checkpoint was never trained under."""
    artifacts_dir = os.path.join(str(tmp_path), "signal_seed0")
    first_argv = _smoke_argv(seed=0, lambda_trans=1.0, steps=2, artifacts_dir=artifacts_dir)
    first_argv += ["--checkpoint-at", "1"]
    assert pretrain.main(first_argv) == 0

    second_argv = _smoke_argv(seed=0, lambda_trans=0.0, steps=2, artifacts_dir=artifacts_dir)
    second_argv += [
        "--checkpoint-at",
        "2",
        "--resume-from",
        os.path.join(artifacts_dir, "checkpoint_step1"),
    ]
    with pytest.raises(SystemExit, match="does not match this invocation"):
        pretrain.main(second_argv)

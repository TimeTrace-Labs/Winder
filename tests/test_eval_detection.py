"""Tests for winder.eval.detection: the time-localised detection/localisation battery.

Split, as elsewhere in this project, into fast always-run tests on pure logic and small synthetic
fixtures (no PTB-XL, no GPU) and skip-gated integration tests that touch real PTB-XL metadata and
the copied-in reference checkpoints/phase artifacts.

**The load-bearing test.** `test_ectopic_beat_cells_reproduce_the_reference_panels_published_
numbers` runs the real port against the reference repo's own two copied-in 30,000-step
checkpoints under `LEGACY_FOLD_CONFIG`, `n_records=400`, `seed=0` -- the exact protocol
`artifacts/campaign_closeout/detection_panel30k/localisation.json` (winder-theory-exp, verified
this session as the corrected, post-revert version) was produced under -- and asserts the five
published `mean_auroc` numbers reproduce to within 1e-4. Scoped to `perturbations={"ectopic_beat":
...}` only: every one of the five target cells is an `ectopic_beat` cell, and the full 6-
perturbation x ~5-amplitude sweep this test does not need would cost roughly 6x the GPU time for
zero additional coverage of the numbers this test actually gates on -- matching this project's own
precedent (`tests/test_eval_suite.py`'s real end-to-end test: "drastically reduced scope... never
a substitute for the full run").

**The full answer-key parity test.** `test_full_battery_reproduces_every_published_cell_in_the_
reference_answer_key` extends the same idea to the WHOLE combinatorial answer key -- all six
perturbation families, both checkpoints, every admissible `(theta, detector)` cell --
against `artifacts/campaign_closeout/detection_panel30k/localisation.json`'s full 330-cell-per-
checkpoint panel. This is the PRIMARY correctness gate for this module, not a supplement to the
property tests below: it is real compute reproducing real published numbers, expected to take
tens of minutes, and is skip-gated on exactly the same flags as the ectopic-beat test above.

**Five theory-derived property tests** (`test_sham_null_...` through `test_causal_scores_are_
exactly_unaffected_...`) check structural/theoretical properties the battery must have regardless
of whether the reference repo's numbers were ever computed -- fast, CPU-only, synthetic, always
on, no skip marker. Each is built by hand at the z/theta level, never through the real injection
pipeline and never against real PTB-XL data.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

from winder.config import ArmConfig
from winder.data.folds import LEGACY_FOLD_CONFIG, FoldConfig
from winder.data.norm_stats import LeadStats
from winder.data.perturb import PERTURBATIONS
from winder.data.phase import phase_from_rpeaks
from winder.data.ptbxl import LEAD_ORDER, MULTIHOT_COLS
from winder.determinism import generator, init_parameters
from winder.eval.detection import (
    DetectionCohort,
    build_detection_cohort,
    cells_for,
    detection_cell_key,
    patch_ms_from_patch_width,
    rpeaks_at_output_rate,
    run_checkpoint_detection_battery,
    run_detection_battery,
    sample_theta_grids,
    score_all,
    score_one_perturbation,
    token_theta_from_samples,
)
from winder.eval.gates import detection_gap_ci
from winder.eval.readout import load_model_and_operator
from winder.jepa import checkpoint
from winder.jepa.model import JepaConfig, JepaModel, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.harmonic import HarmonicTransport
from winder.paths import default_data_root
from winder.transport.localisation import (
    causal_phase_from_rpeaks,
    deviation_scores,
    identity_residual_scores,
    radial_scores,
    transport_residual_scores,
    within_record_auroc,
)

TWO_PI = 2.0 * math.pi

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PTBXL_ROOT = default_data_root()
_HAS_PTBXL_ROOT = os.path.isfile(os.path.join(_PTBXL_ROOT, "ptbxl_database.csv"))
_REFERENCE_ROOT = os.path.join(_REPO_ROOT, "artifacts", "reference")
_HAS_REFERENCE_PHASE = os.path.isfile(
    os.path.join(_REFERENCE_ROOT, "phase", "rpeaks.npz")
) and os.path.isfile(os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz"))
_HAS_REFERENCE_CKPTS = os.path.isdir(
    os.path.join(_REFERENCE_ROOT, "FIN_seed0", "checkpoint")
) and os.path.isdir(os.path.join(_REFERENCE_ROOT, "FIN_LAM0_seed0", "checkpoint"))


# =============================================================== rpeaks_at_output_rate (exact)


def test_rpeaks_at_output_rate_divides_every_position_by_the_decimation_factor() -> None:
    native = np.array([0.0, 250.0, 500.0, 999.5])
    out = rpeaks_at_output_rate(native, 5.0)
    np.testing.assert_allclose(out, native / 5.0)


def test_rpeaks_at_output_rate_leaves_theta_unchanged_composed_with_phase_from_rpeaks() -> None:
    """The exactness claim, not just the arithmetic: rescaling every R-peak AND the query grid by
    the same decimation factor must leave theta at a shared instant unchanged (module docstring)."""
    native = np.array([100.0, 600.0, 1100.0, 1600.0])
    decimated = rpeaks_at_output_rate(native, 5.0)
    theta_native = phase_from_rpeaks(native, 2000)[:, 0]
    theta_decimated = phase_from_rpeaks(decimated, 400)[:, 0]
    # sample n of the decimated grid corresponds to native sample 5n exactly (decimate_to's own
    # contract, module docstring) -- theta at that shared instant must match.
    native_positions = np.arange(400) * 5
    np.testing.assert_allclose(theta_decimated, theta_native[native_positions], atol=1e-9)


# =============================================================================== patch_ms helper


def test_patch_ms_from_patch_width_matches_the_reference_scripts_hardcoded_constant() -> None:
    """patch_width=8 at FS=100 Hz -> 80.0 ms, the reference script's own hardcoded `PATCH_MS`."""
    assert patch_ms_from_patch_width(8) == pytest.approx(80.0)


def test_patch_ms_from_patch_width_scales_with_fs() -> None:
    assert patch_ms_from_patch_width(8, fs=200.0) == pytest.approx(40.0)


# ============================================================================ detection_cell_key


def test_detection_cell_key_matches_the_gates_module_docstring_format() -> None:
    key = detection_cell_key("ectopic_beat", 1.0, "offline", "transport_offline")
    assert key == "ectopic_beat|1.0|offline|transport_offline"


def test_detection_cell_key_amplitude_zero_formats_as_point_zero() -> None:
    """`0.0`, not `0` -- must match the sham cell's own key exactly (Python's default float
    formatting inside an f-string), since a caller matches against the literal string."""
    key = detection_cell_key("st_shift", 0.0, "causal", "deviation")
    assert key == "st_shift|0.0|causal|deviation"


# ===================================================================== token_theta_from_samples


def test_token_theta_from_samples_reads_each_tokens_centre_sample() -> None:
    n_samples, patch_width, n_tokens = 16, 4, 4
    sample_theta = np.arange(n_samples, dtype=np.float64)[None, :].repeat(2, axis=0)
    out = token_theta_from_samples(sample_theta, n_tokens, patch_width)
    # token j covers samples [4j, 4j+3]; centre = 4j + 1.5, rounds to 4j+2 under int() truncation
    # of (j+0.5)*patch_width -- token 0 -> sample 2, token 1 -> sample 6, etc.
    expected = torch.tensor([2.0, 6.0, 10.0, 14.0])
    torch.testing.assert_close(out[0], expected)
    torch.testing.assert_close(out[1], expected)


def test_token_theta_from_samples_clips_the_last_token_to_the_final_sample() -> None:
    """A token whose nominal centre would fall past the record's own last sample is clamped, not
    indexed out of bounds."""
    sample_theta = np.arange(10, dtype=np.float64)[None, :]
    out = token_theta_from_samples(sample_theta, n_tokens=4, patch_width=4)  # covers samples 0-15
    assert out[0, -1] == 9.0  # clamped to the last real sample, index 9


# =========================================================================== sample_theta_grids


def _write_synthetic_rpeaks_npz(path: str, rpeaks_by_ecg_id: dict[int, np.ndarray]) -> None:
    ecg_ids = np.array(sorted(rpeaks_by_ecg_id), dtype=np.int64)
    flat = np.concatenate([rpeaks_by_ecg_id[int(e)] for e in ecg_ids])
    offsets = np.concatenate(
        [[0], np.cumsum([len(rpeaks_by_ecg_id[int(e)]) for e in ecg_ids])]
    ).astype(np.int64)
    np.savez(path, ecg_ids=ecg_ids, offsets=offsets, rpeaks=flat, fs=500.0)


def test_sample_theta_grids_matches_phase_from_rpeaks_directly(tmp_path: Path) -> None:
    npz_path = str(tmp_path / "rpeaks.npz")
    native = np.array([100.0, 600.0, 1100.0, 1600.0, 2100.0])  # RR=500 native samples
    _write_synthetic_rpeaks_npz(npz_path, {7: native, 9: native * 1.5})
    ecg_ids = np.array([7, 9, 999])  # 999 absent -> all-NaN row
    n_samples = 400
    offline, causal = sample_theta_grids(npz_path, ecg_ids, n_samples, decimation_factor=5.0)
    assert offline.shape == (3, n_samples)
    assert causal.shape == (3, n_samples)
    assert np.isnan(offline[2]).all()  # absent ecg_id -> untouched NaN row
    assert np.isnan(causal[2]).all()

    at_rate = rpeaks_at_output_rate(native, 5.0)
    expected_offline = phase_from_rpeaks(at_rate, n_samples)[:, 0]
    expected_causal = causal_phase_from_rpeaks(at_rate, n_samples)
    np.testing.assert_allclose(offline[0], expected_offline, equal_nan=True)
    np.testing.assert_allclose(causal[0], expected_causal, equal_nan=True)


# ============================================================================= score_all / cells


_K0, _N_J, _K_J = 2, [1, 2, 3], [1, 2, 1]  # matches tests/test_eval_gates.py's toy operator


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


def test_score_all_returns_the_seven_declared_detectors() -> None:
    op = _toy_operator()
    gen = torch.Generator().manual_seed(0)
    n, t = 3, 12
    theta = torch.rand(n, t, generator=gen) * TWO_PI
    z = torch.randn(n, t, op.dimension, generator=gen)
    scores = score_all(z, theta, op, window=4)
    assert set(scores) == {
        "transport_offline",
        "transport_causal",
        "identity_offline",
        "identity_causal",
        "deviation",
        "radial_offline",
        "radial_causal",
    }
    for s in scores.values():
        assert s.shape == (n, t)


def test_score_one_perturbation_skips_offline_detectors_under_causal_theta() -> None:
    op = _toy_operator()
    gen = torch.Generator().manual_seed(1)
    n, t = 4, 16
    theta = torch.rand(n, t, generator=gen) * TWO_PI
    z = torch.randn(n, t, op.dimension, generator=gen)
    mask = torch.zeros(n, t, dtype=torch.bool)
    mask[:, 5] = True

    cells = score_one_perturbation(
        z, {"offline": theta, "causal": theta}, mask, op, causal_window=4, patch_ms=80.0
    )
    causal_keys = {k for k in cells if k.startswith("causal|")}
    offline_keys = {k for k in cells if k.startswith("offline|")}
    assert not any(k.endswith("offline") for k in causal_keys)  # the skip rule under test
    assert offline_keys == {
        "offline|transport_offline",
        "offline|transport_causal",
        "offline|identity_offline",
        "offline|identity_causal",
        "offline|deviation",
        "offline|radial_offline",
        "offline|radial_causal",
    }


def test_score_one_perturbation_cell_carries_auroc_localisation_and_latency_fields() -> None:
    op = _toy_operator()
    gen = torch.Generator().manual_seed(2)
    n, t = 5, 20
    theta = torch.rand(n, t, generator=gen) * TWO_PI
    z = torch.randn(n, t, op.dimension, generator=gen)
    mask = torch.zeros(n, t, dtype=torch.bool)
    mask[:, 10] = True

    cells = score_one_perturbation(
        z, {"offline": theta}, mask, op, causal_window=None, patch_ms=80.0
    )
    cell = cells["offline|transport_offline"]
    assert set(cell) >= {
        "theta",
        "detector",
        "mean_auroc",
        "median_auroc",
        "n_records",
        "localisation",
        "latency",
        "_auroc",
    }
    assert isinstance(cell["mean_auroc"], float)
    assert "median_ms" in cell["localisation"]
    assert "median_ms" in cell["latency"]
    assert len(cell["_auroc"]["per_record"]) == len(cell["_auroc"]["record_index"])


# ================================================================ run_checkpoint_detection_battery


_N_SAMPLES = 1000
_N_TOKENS = 250
_PATCH_WIDTH = 4
_OP_K0, _OP_N_J, _OP_K_J = 2, [1, 2], [2, 2]
_DIM = _OP_K0 + 2 * sum(_OP_K_J)


def _tiny_jepa_config() -> JepaConfig:
    return JepaConfig(
        n_leads=12,
        n_samples=_N_SAMPLES,
        n_tokens=_N_TOKENS,
        encoder_name="residual_cnn",
        encoder={},
        projector_name="mlp",
        projector={"input_width": 256, "hidden_width": 16, "output_width": _DIM},
        predictor_name="transformer",
        predictor={"width": _DIM, "n_heads": 2, "feedforward_width": 16},
        mask_sampler_name="causal_block",
        mask_sampler={},
        prediction_loss_name="mse",
        prediction_loss={},
        regularizer_name="sigreg",
        regularizer={"n_directions": 4, "chunk": 4},
    )


def _build_model_and_operator(
    tmp_path: Path, *, seed: int = 0
) -> tuple[JepaModel, HarmonicTransport]:
    jepa_cfg = _tiny_jepa_config()
    model = build_jepa(jepa_cfg, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    operator = CyclicOperator(CyclicOperatorConfig(k0=_OP_K0, n_j=_OP_N_J, k_j=_OP_K_J))
    arm_cfg = ArmConfig(
        name="tiny_cyclic",
        seed=seed,
        operator_name="cyclic",
        operator={"k0": _OP_K0, "n_j": _OP_N_J, "k_j": _OP_K_J},
    )
    config_yaml = checkpoint.resolved_config_yaml(
        jepa_cfg, TrainConfig(n_steps=10, seed_pretrain=seed), arm_config=arm_cfg
    )
    ckpt_dir = str(tmp_path / "checkpoint")
    checkpoint.save_checkpoint(
        ckpt_dir,
        model=model,
        optimizer=optimizer,
        step=10,
        generators={},
        config_yaml=config_yaml,
        meta={},
        operator=operator,
    )
    loaded_model, loaded_operator = load_model_and_operator(
        ckpt_dir, seed=seed, device=torch.device("cpu")
    )
    assert loaded_operator is not None
    return loaded_model, loaded_operator


def _tiny_synthetic_cohort(n_records: int = 5) -> DetectionCohort:
    """A synthetic but ECG-SHAPED cohort: 10 evenly-spaced beats (RR=100 samples=1000 ms) over a
    1000-sample record, same theta for every record -- enough beat structure for
    `winder.data.perturb`'s arc/wrap logic (`ST_ARC`/`T_ARC`/`ectopic_beat`'s own beat-boundary
    search) to have somewhere real to act, without needing real PTB-XL data."""
    rpeaks = np.arange(50, 1000, 100, dtype=np.float64)  # 10 peaks, 9 beats, RR=100
    offline_row = phase_from_rpeaks(rpeaks, _N_SAMPLES)[:, 0]
    causal_row = causal_phase_from_rpeaks(rpeaks, _N_SAMPLES)
    offline_grid = np.tile(offline_row, (n_records, 1))
    causal_grid = np.tile(causal_row, (n_records, 1))

    gen = torch.Generator().manual_seed(0)
    clean = torch.randn(n_records, 12, _N_SAMPLES, generator=gen)
    frame = pd.DataFrame({"ecg_id": np.arange(n_records)})
    return DetectionCohort(
        frame=frame,
        ecg_ids=frame["ecg_id"].to_numpy(),
        clean=clean,
        offline_samples=torch.from_numpy(offline_grid).float(),
        theta_tokens={
            "offline": token_theta_from_samples(offline_grid, _N_TOKENS, _PATCH_WIDTH),
            "causal": token_theta_from_samples(causal_grid, _N_TOKENS, _PATCH_WIDTH),
        },
        lead_std=torch.ones(12),
        n_tokens=_N_TOKENS,
        patch_width=_PATCH_WIDTH,
        theta_coverage_offline=float(np.isfinite(offline_grid).mean()),
        theta_coverage_causal=float(np.isfinite(causal_grid).mean()),
    )


def test_run_checkpoint_detection_battery_covers_every_perturbation_and_admissible_cell(
    tmp_path: Path,
) -> None:
    model, operator = _build_model_and_operator(tmp_path)
    cohort = _tiny_synthetic_cohort()
    per_ckpt, dump = run_checkpoint_detection_battery(
        model,
        operator,
        cohort,
        torch.device("cpu"),
        ckpt_name="tiny",
        causal_window=4,
        dump_per_record=True,
    )
    assert per_ckpt  # non-empty
    perturbations_seen = {v["perturbation"] for v in per_ckpt.values()}
    assert perturbations_seen == set(PERTURBATIONS)
    # every cell's amplitude-0.0 sham exists for every (perturbation, theta, detector) that
    # exists at all -- the same-pipeline null the module docstring promises.
    for pert_name, (_fn, _sweep, _family) in PERTURBATIONS.items():
        sham_keys = [k for k in per_ckpt if k.startswith(f"{pert_name}|0.0|")]
        assert sham_keys, pert_name
    # no offline-suffixed detector survives under causal theta
    assert not any(
        v["theta"] == "causal" and v["detector"].endswith("offline") for v in per_ckpt.values()
    )
    for cell in per_ckpt.values():
        assert set(cell) == {
            "perturbation",
            "family",
            "amplitude",
            "theta",
            "detector",
            "mean_auroc",
            "median_auroc",
            "n_records",
            "localisation",
            "latency",
        }

    # dump_per_record: exactly one "|auroc" and one "|record_index" key per per_ckpt cell, both
    # prefixed by ckpt_name, in EXACTLY winder.eval.gates's expected key format.
    assert len(dump) == 2 * len(per_ckpt)
    for key in per_ckpt:
        assert f"tiny|{key}|auroc" in dump
        assert f"tiny|{key}|record_index" in dump
        assert len(dump[f"tiny|{key}|auroc"]) == len(dump[f"tiny|{key}|record_index"])


def test_run_checkpoint_detection_battery_can_be_scoped_to_one_perturbation(
    tmp_path: Path,
) -> None:
    """The scoping this module's own load-bearing real-checkpoint test relies on: passing a
    `perturbations` subset must touch nothing outside it."""
    model, operator = _build_model_and_operator(tmp_path)
    cohort = _tiny_synthetic_cohort()
    per_ckpt, _dump = run_checkpoint_detection_battery(
        model,
        operator,
        cohort,
        torch.device("cpu"),
        ckpt_name="tiny",
        causal_window=4,
        perturbations={"ectopic_beat": PERTURBATIONS["ectopic_beat"]},
    )
    assert per_ckpt
    assert {v["perturbation"] for v in per_ckpt.values()} == {"ectopic_beat"}


# ======================================== theory-derived property tests (synthetic, always-on)
#
# Fast, CPU-only, no PTB-XL, no checkpoints, no GPU -- each builds z/theta by hand and checks a
# structural/theoretical property of the battery's own maths, independent of whether the
# reference repo's numbers were ever computed. See module docstring.


def test_sham_null_auroc_is_exactly_0point5_by_construction() -> None:
    """Deliverable 2(a): the "amplitude 0.0" case, built directly at the z/theta level (no
    `winder.data.perturb` call, no real waveform).

    Every token in every record carries the exact SAME unit-normalised latent direction `u`, so
    `deviation_scores`'s own comparison ("each token vs. the record's own mean direction",
    module docstring) is `1 - <u, u>` at EVERY token -- an exact tie, not merely a statistically
    small difference. `within_record_auroc`'s rank formula on a perfect tie of `p` positive and
    `q` negative tokens works out to exactly `(N-p)/(2q) = q/(2q) = 0.5` for ANY split of `p`/`q`
    (worked from `winder.transport.localisation._auroc`'s own mid-rank formula: tied ranks all
    equal `(N+1)/2`, so `sum(ranks[:p]) - p(p+1)/2 = p(N+1-p-1)/2 = p*q/2`, divided by `p*q` gives
    `1/2` exactly) -- so 0.5 here is a hard equality up to float rounding, not a fudge factor.
    NOT run on real PTB-XL data: the real sham AUROC is 0.454 (ST-arc region is intrinsically
    higher-variance, measured this session), which is neither exact nor the property under test.
    """
    n, t, k = 6, 24, 8
    gen = torch.Generator().manual_seed(0)
    u = torch.randn(n, 1, k, generator=gen)
    u = u / u.norm(dim=-1, keepdim=True)
    z = u.expand(n, t, k).clone()  # the SAME direction at every token of every record
    theta = torch.linspace(0.0, TWO_PI, t).unsqueeze(0).expand(n, t).clone()  # arbitrary, finite

    scores = deviation_scores(z, theta)
    token_mask = torch.zeros(n, t, dtype=torch.bool)
    token_mask[:, 5:11] = True  # an arbitrary, non-trivial "positive" window

    auroc = within_record_auroc(scores, token_mask)
    assert auroc["mean_auroc"] == pytest.approx(0.5, abs=1e-6), auroc


def test_weak_monotonicity_of_mean_auroc_with_perturbation_amplitude() -> None:
    """Deliverable 2(b): a growing additive offset injected into a SMALL token window of an
    otherwise uniform synthetic latent, at amplitudes `(0, 0.5, 1.0, 2.0, 4.0)`.

    Baseline (as in test (a)): every token carries the SAME unit direction `u`, so at
    `amplitude=0` `identity_residual_scores` is an exact 0 everywhere -- an exact null starting
    point, not merely a low one. The window (3 of 24 tokens, kept deliberately SMALL relative to
    the record) is then pushed toward a fixed, unrelated direction by a growing offset:
    `identity_residual_scores` compares a token PAIRWISE against every OTHER token (never
    itself, `winder.transport.localisation`'s own `_pair_reference_mask`), so a growing window
    residual against the still-mostly-`u` rest of the record is not self-referential the way a
    record-mean-based statistic would be -- there is no aggregate for the perturbation to "pull
    toward itself" and no risk of the signal inverting as amplitude grows (an earlier version of
    this test used `deviation_scores`, whose mean IS record-wide and self-referential, and it
    inverted: past a certain amplitude the perturbed window dominated its own mean enough that
    its deviation FELL as amplitude grew -- a real, reportable property of that statistic, not a
    test bug, and the reason this test deliberately does not use it).

    Asserts WEAK monotonicity only -- a positive Spearman rank correlation between amplitude and
    `mean_auroc` across the sweep -- not strict step-by-step ordering: with `n=8` records per
    amplitude, sampling noise can still flip two ADJACENT amplitudes' AUROCs even though the
    overall trend is unambiguous.

    Observed sweep (this construction, this seed): `mean_auroc = [0.5, 1.0, 1.0, 1.0, 1.0]` for
    `amplitude = [0.0, 0.5, 1.0, 2.0, 4.0]` -- `rho = 0.707`. The signal SATURATES by
    `amplitude=0.5` rather than climbing gradually (an unnormalised `randn(k)` direction already
    has norm ~sqrt(k), so even the smallest non-zero step dominates the window's own unit-norm
    baseline) -- non-decreasing, which satisfies "weak monotonicity" even more directly than a
    gradual climb would, but it means this sweep is not shaped like a realistic sensitivity
    curve. That is a property of this particular synthetic construction, not a claim about the
    real battery's own amplitude sweeps (which DO climb gradually, e.g. the load-bearing test's
    own `ectopic_beat` numbers above).
    """
    n, t, k = 8, 24, 8
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(n, 1, k, generator=gen)
    u = u / u.norm(dim=-1, keepdim=True)
    base = u.expand(n, t, k).clone()  # exact tie baseline (test (a)'s construction)
    theta = torch.linspace(0.0, TWO_PI, t).unsqueeze(0).expand(n, t).clone()
    window = slice(10, 13)
    direction = torch.randn(k, generator=gen)

    amplitudes = (0.0, 0.5, 1.0, 2.0, 4.0)
    mean_aurocs = []
    for amp in amplitudes:
        z = base.clone()
        z[:, window, :] = z[:, window, :] + amp * direction
        mask = torch.zeros(n, t, dtype=torch.bool)
        mask[:, window] = True
        mean_aurocs.append(
            within_record_auroc(identity_residual_scores(z, theta), mask)["mean_auroc"]
        )

    rho, _p = spearmanr(amplitudes, mean_aurocs)
    assert rho > 0.0, (amplitudes, mean_aurocs)
    assert mean_aurocs[-1] > mean_aurocs[0] + 0.05, mean_aurocs  # a clear, non-marginal margin


def test_transport_residual_beats_identity_residual_on_a_phase_coherent_defect() -> None:
    """Deliverable 2(c) -- THE mechanism claim of the whole battery, on synthetic latents built
    to be exactly equivariant except at one lesioned window.

    `z_t = R(theta_t) z0` for a fixed canonical `z0`: an EXACTLY equivariant record (every token
    is the transport of every other, by direct construction, not by training). At a window of
    tokens, an extra rotation by a FIXED angle unrelated to theta is applied -- a phase-coherent
    defect a phase-aware detector should catch. `transport_residual_scores` corrects for theta
    before comparing, so its baseline residual is ~0 and the defect spikes cleanly. `identity_
    residual_scores` does not correct for theta at all, so even the CLEAN tokens already differ
    from each other by their own theta-driven rotation -- it cannot tell "expected phase change"
    from "defect", and its AUROC should sit far closer to the null. The margin below (>0.2) was
    read off this exact construction before being pinned; if a future change to the detectors or
    this construction shrinks it, that is the real finding to report, not a threshold to loosen.
    """
    op = _toy_operator()
    k = op.dimension
    n, t = 12, 30
    gen = torch.Generator().manual_seed(2)
    theta = torch.rand(n, t, generator=gen) * TWO_PI

    z0 = torch.randn(n, 1, k, generator=gen)
    z0 = z0 / z0.norm(dim=-1, keepdim=True)
    z = op.transport(z0.expand(n, t, k), theta)  # exactly equivariant: z_t = R(theta_t) z0

    window = slice(12, 16)
    w = window.stop - window.start
    defect_angle = math.pi  # a phase-coherent rotation defect, unrelated to theta
    z_defect = z.clone()
    z_defect[:, window, :] = op.transport(z[:, window, :], torch.full((n, w), defect_angle))
    mask = torch.zeros(n, t, dtype=torch.bool)
    mask[:, window] = True

    transport_auroc = within_record_auroc(transport_residual_scores(z_defect, theta, op), mask)[
        "mean_auroc"
    ]
    identity_auroc = within_record_auroc(identity_residual_scores(z_defect, theta), mask)[
        "mean_auroc"
    ]
    margin = transport_auroc - identity_auroc
    assert margin > 0.2, (transport_auroc, identity_auroc, margin)


def test_radial_scores_alone_detect_a_pure_norm_rescale_the_others_are_exactly_blind() -> None:
    """Deliverable 2(d) -- the sharpest of the five: EXACT, not fuzzy, because "radial
    blindness" is a structural guarantee of `winder.transport.localisation`'s own construction,
    not a statistical tendency.

    `transport_residual_scores`/`identity_residual_scores`/`deviation_scores` all normalise `z`
    to unit length before doing anything else (`_normalise`, `winder.transport.localisation`'s
    own source, read before writing this test) -- a lesion that rescales a token's latent by a
    POSITIVE scalar, with no change of direction, is invisible to them BY CONSTRUCTION: the unit
    vector is unchanged. `radial_scores` is the one score built to see exactly that channel (its
    own module docstring: "radial blindness (theory notes sec 8)").

    Construction, so the blindness is exact rather than merely small: every token in a record
    carries the SAME unit direction `u` (as in test (a)) under a CONSTANT theta (every pairwise
    phase delta is exactly 0, so every (query, reference) pair is computed on literally identical
    inputs -- an exact tie regardless of what the operator does with a zero delta). A window of
    tokens is then rescaled by `amp=4.0`, a power of two: `_normalise`'s `+1e-8` epsilon is well
    below HALF a float32 ULP at both 1.0 (ULP ~1.19e-7) and 4.0 (ULP ~4.77e-7) -- the threshold
    that actually matters for round-to-nearest -- so it rounds away identically at both scales,
    and 4.0/0.25 are EXACT (no mantissa rounding) -- so the unit-normalised vector
    at a rescaled token is BIT-IDENTICAL to the unrescaled one. (An amplitude like 3.0 would NOT
    give this exact cancellation: the epsilon's rounding differs subtly between norm 1 and norm
    3, which is enough to break AUROC's rank statistic away from 0.5 -- exactly the trap this
    construction is designed to avoid.) `radial_scores` reads the raw (non-normalised) norm, so
    the rescaled window's log-norm differs from the record's own median by exactly `log(4)`,
    while every other token differs by exactly 0 -- perfect rank separation, AUROC = 1.0 exactly.
    """
    op = _toy_operator()
    k = op.dimension
    n, t = 6, 16
    gen = torch.Generator().manual_seed(3)
    u = torch.randn(n, 1, k, generator=gen)
    u = u / u.norm(dim=-1, keepdim=True)
    z = u.expand(n, t, k).clone()  # unit norm, SAME direction at every token (see test (a))
    theta = torch.zeros(n, t)  # CONSTANT theta: every pairwise delta is exactly 0 (see docstring)

    window = slice(6, 9)
    amp = 4.0  # power of two -- see docstring for why this is load-bearing, not cosmetic
    z_rescaled = z.clone()
    z_rescaled[:, window, :] = z_rescaled[:, window, :] * amp
    mask = torch.zeros(n, t, dtype=torch.bool)
    mask[:, window] = True

    blind = {
        "transport": transport_residual_scores(z_rescaled, theta, op),
        "identity": identity_residual_scores(z_rescaled, theta),
        "deviation": deviation_scores(z_rescaled, theta),
    }
    for name, scores in blind.items():
        auroc = within_record_auroc(scores, mask)["mean_auroc"]
        assert auroc == pytest.approx(0.5, abs=1e-6), (name, auroc)

    radial_auroc = within_record_auroc(radial_scores(z_rescaled, theta), mask)["mean_auroc"]
    assert radial_auroc == pytest.approx(1.0, abs=1e-9), radial_auroc


def test_causal_scores_are_exactly_unaffected_by_anything_after_the_query_token() -> None:
    """Deliverable 2(e), exact: a causal score at token `t` must not depend on anything at a
    token index `>= t`.

    Two versions of the same `(z, theta)` differ ONLY at and after a cutoff `t0`, replaced with
    unrelated, arbitrary FINITE values (not NaN: `_residual`'s masking is multiplicative --
    `defect * pair` -- and `NaN * 0 == NaN`, so an actually-invalid future token would poison a
    masked-out pairing rather than cleanly zero it; a finite-but-arbitrary replacement is both
    safe and the STRONGER claim, since it proves causality even when the "future" looks like
    perfectly ordinary, validly-phased data). `_pair_reference_mask`'s own causal branch admits
    only `reference_index < query_index`, so for any query `q < t0` every admissible reference
    satisfies `r < q < t0` -- strictly before the cutoff in BOTH versions, hence identical -- and
    the (query, reference) pairs with `r >= t0` are zeroed by the pair mask in the dense
    computation before being summed, contributing nothing regardless of what the replaced values
    are. Asserted at a 1e-10 tolerance with `equal_nan=True` (query 0 has no valid causal
    reference at all -- `counts=0` -- so BOTH versions legitimately emit NaN there; that is the
    correctly-undefined case, not a mismatch). A failure here would be a real finding about a
    lookahead leak, not a test-tolerance problem, and is reported as such rather than loosened.
    """
    op = _toy_operator()
    k = op.dimension
    n, t, t0 = 4, 20, 10
    gen = torch.Generator().manual_seed(4)
    z = torch.randn(n, t, k, generator=gen)
    theta = torch.rand(n, t, generator=gen) * TWO_PI

    future_gen = torch.Generator().manual_seed(99)
    z_altered = z.clone()
    theta_altered = theta.clone()
    z_altered[:, t0:, :] = torch.randn(n, t - t0, k, generator=future_gen) * 100.0
    theta_altered[:, t0:] = torch.rand(n, t - t0, generator=future_gen) * TWO_PI

    detectors = (
        lambda zz, th: transport_residual_scores(zz, th, op, causal=True, window=None),
        lambda zz, th: identity_residual_scores(zz, th, causal=True, window=None),
    )
    for detector in detectors:
        s1 = detector(z, theta)
        s2 = detector(z_altered, theta_altered)
        torch.testing.assert_close(s1[:, :t0], s2[:, :t0], atol=1e-10, rtol=0.0, equal_nan=True)


# ==================================================================== build_detection_cohort


def test_build_detection_cohort_raises_loudly_on_the_nominal_empty_val_sentinel() -> None:
    if not _HAS_PTBXL_ROOT:
        pytest.skip(f"PTB-XL data root not found at {_PTBXL_ROOT}")
    with pytest.raises(ValueError, match="empty"):
        build_detection_cohort(
            _PTBXL_ROOT,
            fold_config=FoldConfig(),
            n_records=10,
            rpeaks_npz_path="/does/not/matter",
            lead_stats_path="/does/not/matter",
            theta_tokens_path="/does/not/matter",
        )


def test_build_detection_cohort_raises_when_both_fold_config_and_frame_are_given() -> None:
    """The mutual-exclusivity guard is the FIRST thing the function checks -- before `data_root`
    is ever touched -- so this needs no real PTB-XL root. `frame=` is the exact parameter the
    fold-10 event's own `resolve_target_fold_frames` uses (gate-3 round 4 flagged this guard, on
    the parameter the real event actually exercises, as having zero isolated test coverage)."""
    with pytest.raises(ValueError, match="exactly one of fold_config or frame"):
        build_detection_cohort(
            "/does/not/matter",
            fold_config=FoldConfig(),
            frame=pd.DataFrame({"ecg_id": [1], "patient_id": [1], "strat_fold": [1]}),
            n_records=10,
            rpeaks_npz_path="/does/not/matter",
            lead_stats_path="/does/not/matter",
            theta_tokens_path="/does/not/matter",
        )


def test_build_detection_cohort_raises_when_neither_fold_config_nor_frame_is_given() -> None:
    """Companion negative case for the same guard."""
    with pytest.raises(ValueError, match="exactly one of fold_config or frame"):
        build_detection_cohort(
            "/does/not/matter",
            n_records=10,
            rpeaks_npz_path="/does/not/matter",
            lead_stats_path="/does/not/matter",
            theta_tokens_path="/does/not/matter",
        )


def _toy_metadata_for_cohort(n_patients: int, records_per_patient: int) -> pd.DataFrame:
    """A synthetic in-memory stand-in for `winder.data.ptbxl.load_metadata`'s own output: the
    same round-robin fold assignment `tests/test_folds.py::_toy_metadata` uses for the identical
    purpose (kept local rather than imported -- this module's own commit does not touch
    `test_folds.py`), extended with the extra columns `build_detection_cohort`'s own pipeline
    needs before anything ever reaches `folds()`: `EcgWindowDataset`'s `_REQUIRED_COLUMNS`
    (`filename_lr`/`filename_hr`, never actually read here -- see the test's own `read_waveforms`
    monkeypatch) and `winder.data.ptbxl.MULTIHOT_COLS` (so every row survives `build_detection_
    cohort`'s own "labeled" filter, `metadata[MULTIHOT_COLS].sum(axis=1) > 0`).
    """
    rows = []
    ecg_id = 1
    for pid in range(n_patients):
        fold = (pid % 10) + 1
        for _ in range(records_per_patient):
            row = {
                "ecg_id": ecg_id,
                "patient_id": pid,
                "strat_fold": fold,
                "filename_lr": f"records100/{ecg_id:05d}_lr",
                "filename_hr": f"records500/{ecg_id:05d}_hr",
            }
            row.update({col: (1 if col == MULTIHOT_COLS[0] else 0) for col in MULTIHOT_COLS})
            rows.append(row)
            ecg_id += 1
    return pd.DataFrame(rows)


def _toy_lead_stats(path: str) -> None:
    """A minimal, VALID `LeadStats` artifact at `fs=100` (`EcgWindowDataset`'s own requirement),
    written to `path` -- content is otherwise arbitrary, since the test's `read_waveforms`
    monkeypatch means no real waveform is ever normalised against it."""
    stats = LeadStats(
        leads=LEAD_ORDER,
        mean_mv=tuple(0.0 for _ in LEAD_ORDER),
        std_mv=tuple(1.0 for _ in LEAD_ORDER),
        fs=100,
        folds=(1, 2, 3),  # subset of the DEFAULT FoldConfig().train_folds, per the model's own
        # validator -- independent of whichever fold_config this test's cohort call itself uses
        n_records=1,
        n_samples=1,
        created_utc="2026-01-01T00:00:00Z",
    )
    stats.to_json(path)


def _toy_theta_tokens_npz(
    path: str, *, n_tokens: int, patch_width: int, decimation_factor: float
) -> None:
    """A minimal, structurally valid `theta_tokens.npz` -- `build_detection_cohort` only reads
    this archive's METADATA (`n_tokens`/`patch_width`/`decimation_factor`), never its per-ecg_id
    theta rows (`load_theta_tokens`'s return value is unpacked as `_theta_by_id, theta_meta`, the
    first name explicitly discarded), so the `ecg_ids`/`theta` payload's actual content does not
    need to correspond to this test's own toy records at all."""
    np.savez(
        path,
        ecg_ids=np.array([0], dtype=np.int64),
        theta=np.zeros((1, n_tokens), dtype=np.float32),
        patch_width=np.int64(patch_width),
        n_tokens=np.int64(n_tokens),
        decimation_factor=np.float64(decimation_factor),
        timestamp=np.str_("toy"),
    )


def test_build_detection_cohort_works_under_an_arbitrary_non_legacy_fold_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliverable 3: `build_detection_cohort` must work under ANY well-formed `FoldConfig`, not
    only the two named configurations already exercised elsewhere in this file (the nominal
    empty-`val_fold=0` sentinel above, and `LEGACY_FOLD_CONFIG`'s `val_fold=9` in the skip-gated
    real-checkpoint test below). This proves the exact code path a future pre-registered
    fold-10 protocol will eventually exercise has already run successfully, on safe synthetic
    data, at a DIFFERENT val_fold -- never `test_fold=10`, never `unseal=True` (neither appears
    anywhere in this test).

    `load_metadata` and `read_waveforms` are monkeypatched -- the two real-I/O boundaries
    `build_detection_cohort` cannot be driven through without either a real PTB-XL root or a full
    synthetic WFDB signal archive, neither of which this test needs: everything this deliverable
    is actually about (fold selection via `folds()`, `n_records` truncation, phase-grid sampling
    via `sample_theta_grids`, `LeadStats`/`theta_tokens` archive loading, final `DetectionCohort`
    assembly) is REAL code, exercised end to end.
    """
    n_patients, records_per_patient = 40, 1
    metadata = _toy_metadata_for_cohort(n_patients, records_per_patient)
    monkeypatch.setattr("winder.eval.detection.load_metadata", lambda data_root: metadata)

    n_samples, patch_width = 200, 8
    n_tokens = n_samples // patch_width

    def _stub_read_waveforms(dataset: Any, batch_size: int = 128) -> torch.Tensor:
        return torch.zeros(len(dataset), 12, n_samples)

    monkeypatch.setattr("winder.eval.detection.read_waveforms", _stub_read_waveforms)

    val_fold = 4
    fold_config = FoldConfig(train_folds=(1, 2, 3, 5, 6, 7, 8, 9), val_fold=val_fold, test_fold=10)
    expected_val_ecg_ids = metadata.loc[metadata["strat_fold"] == val_fold, "ecg_id"].tolist()
    assert len(expected_val_ecg_ids) == n_patients // 10  # sanity: round-robin gives 4 patients

    lead_stats_path = str(tmp_path / "lead_stats.json")
    _toy_lead_stats(lead_stats_path)
    decimation_factor = 5.0
    theta_tokens_path = str(tmp_path / "theta_tokens.npz")
    _toy_theta_tokens_npz(
        theta_tokens_path,
        n_tokens=n_tokens,
        patch_width=patch_width,
        decimation_factor=decimation_factor,
    )
    rpeaks_npz_path = str(tmp_path / "rpeaks.npz")
    native_range = int(decimation_factor * n_samples)
    rpeaks_native = np.arange(50, native_range, 100, dtype=np.float64)  # >=3 peaks -> real theta
    _write_synthetic_rpeaks_npz(
        rpeaks_npz_path, {ecg_id: rpeaks_native for ecg_id in expected_val_ecg_ids}
    )

    n_records = 2  # strictly fewer than the 4 available in val_fold -- also proves truncation
    cohort = build_detection_cohort(
        "/does/not/matter",  # never touched: load_metadata and read_waveforms are both patched
        fold_config=fold_config,
        n_records=n_records,
        rpeaks_npz_path=rpeaks_npz_path,
        lead_stats_path=lead_stats_path,
        theta_tokens_path=theta_tokens_path,
    )

    assert len(cohort.frame) == n_records
    assert set(cohort.frame["strat_fold"].unique().tolist()) == {val_fold}
    assert cohort.clean.shape == (n_records, 12, n_samples)
    assert cohort.n_tokens == n_tokens
    assert cohort.patch_width == patch_width
    assert cohort.theta_tokens["offline"].shape == (n_records, n_tokens)
    assert cohort.theta_tokens["causal"].shape == (n_records, n_tokens)
    # real rpeaks fed through the real sample_theta_grids path -- coverage must be genuinely
    # non-zero, not a degenerate all-NaN grid the monkeypatches happened to paper over.
    assert cohort.theta_coverage_offline > 0.0
    assert cohort.theta_coverage_causal > 0.0


# ================================================================================== cells_for


def test_cells_for_decodes_a_flat_per_record_dump_into_the_gates_expected_shape() -> None:
    """Deliverable 4: `cells_for` decodes exactly the keys `run_checkpoint_detection_battery(...,
    dump_per_record=True)` emits (`detection_cell_key`'s own format, prefixed by `{ckpt}|` and
    suffixed by `|auroc`/`|record_index`) into `{(theta, detector): {amplitude: (auroc,
    record_index)}}` -- `winder.eval.gates.detection_gap_ci`'s `untrained_cells` shape.

    A plain `dict` stands in for an npz object -- no file on disk. Extra keys for a DIFFERENT
    anomaly and a DIFFERENT checkpoint are included deliberately, to prove both filters actually
    filter, not just that matching keys pass through.
    """
    dump = {
        "ckptA|ectopic_beat|0.0|offline|transport_offline|auroc": np.array([0.5, 0.6]),
        "ckptA|ectopic_beat|0.0|offline|transport_offline|record_index": np.array([0, 1]),
        "ckptA|ectopic_beat|1.0|offline|transport_offline|auroc": np.array([0.9, 0.8]),
        "ckptA|ectopic_beat|1.0|offline|transport_offline|record_index": np.array([0, 1]),
        "ckptA|ectopic_beat|1.0|causal|transport_causal|auroc": np.array([0.7]),
        "ckptA|ectopic_beat|1.0|causal|transport_causal|record_index": np.array([0]),
        # a DIFFERENT anomaly, same checkpoint -- must be excluded from the "ectopic_beat" decode
        "ckptA|st_shift|0.0|offline|transport_offline|auroc": np.array([0.4]),
        "ckptA|st_shift|0.0|offline|transport_offline|record_index": np.array([0]),
        # a DIFFERENT checkpoint, same anomaly -- must be excluded too
        "ckptB|ectopic_beat|0.0|offline|transport_offline|auroc": np.array([0.55]),
        "ckptB|ectopic_beat|0.0|offline|transport_offline|record_index": np.array([0]),
    }
    out = cells_for(dump, "ckptA", "ectopic_beat")
    assert set(out) == {("offline", "transport_offline"), ("causal", "transport_causal")}
    assert set(out[("offline", "transport_offline")]) == {"0.0", "1.0"}
    assert set(out[("causal", "transport_causal")]) == {"1.0"}

    auroc, idx = out[("offline", "transport_offline")]["1.0"]
    np.testing.assert_array_equal(auroc, [0.9, 0.8])
    np.testing.assert_array_equal(idx, [0, 1])


def test_cells_for_output_can_be_fed_directly_into_detection_gap_ci() -> None:
    """Deliverable 4's real plumbing proof, not just "does not raise": `cells_for`'s decoded
    output, with zero adapter code, IS `detection_gap_ci`'s `trained_severity`/`untrained_cells`
    argument shape -- the two previously-disconnected pieces this module's own docstring names
    ("`winder.eval.gates.detection_gap_ci` was ported in Phase P5 but left permanently
    unusable...") are now connected. Mirrors `tests/test_eval_gates.py::test_detection_gap_ci_
    computes_the_baseline_corrected_gap`'s own numbers exactly, but starting from flat dump keys
    (as a real per-record dump would produce) rather than hand-built nested dicts.
    """
    trained_dump = {
        "trained|ectopic_beat|0.0|offline|transport_offline|auroc": np.array([0.5, 0.5, 0.5, 0.5]),
        "trained|ectopic_beat|0.0|offline|transport_offline|record_index": np.array([0, 1, 2, 3]),
        "trained|ectopic_beat|1.0|offline|transport_offline|auroc": np.array([0.9, 0.8, 0.9, 0.8]),
        "trained|ectopic_beat|1.0|offline|transport_offline|record_index": np.array([0, 1, 2, 3]),
    }
    untrained_dump = {
        "untrained|ectopic_beat|0.0|offline|matched_filter|auroc": np.array([0.5, 0.5, 0.5, 0.5]),
        "untrained|ectopic_beat|0.0|offline|matched_filter|record_index": np.array([0, 1, 2, 3]),
        "untrained|ectopic_beat|1.0|offline|matched_filter|auroc": np.array([0.6, 0.6, 0.6, 0.6]),
        "untrained|ectopic_beat|1.0|offline|matched_filter|record_index": np.array([0, 1, 2, 3]),
    }
    trained_severity = cells_for(trained_dump, "trained", "ectopic_beat")[
        ("offline", "transport_offline")
    ]
    untrained_cells = cells_for(untrained_dump, "untrained", "ectopic_beat")

    patient_ids = np.array([10, 10, 20, 20])
    out = detection_gap_ci(trained_severity, untrained_cells, patient_ids, n_replicates=200, seed=0)

    assert out is not None
    assert out["untrained_best_detector"] == "matched_filter/offline"
    assert out["gap"]["mean"] == pytest.approx(0.35 - 0.1, abs=1e-9)
    assert out["n_records_paired"] == 4
    assert out["ci_excludes_zero"] is True


# ========================================================== skip-gated: real reference artifacts


_SKIP_REASON = (
    "PTB-XL data root, artifacts/reference phase archives, or reference checkpoints absent"
)


@pytest.mark.skipif(
    not (_HAS_PTBXL_ROOT and _HAS_REFERENCE_PHASE and _HAS_REFERENCE_CKPTS), reason=_SKIP_REASON
)
def test_ectopic_beat_cells_reproduce_the_reference_panels_published_numbers() -> None:
    """THE LOAD-BEARING TEST (module docstring). Reproduces, to 1e-4 absolute, the three
    `FINs0_30k` cells and two `FINLAM0_30k` cells published in the reference repo's
    `artifacts/campaign_closeout/detection_panel30k/localisation.json` (winder-theory-exp,
    post-revert, verified this session).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = run_detection_battery(
        data_root=_PTBXL_ROOT,
        roster_dir=_REFERENCE_ROOT,
        checkpoint_names=["FIN_seed0", "FIN_LAM0_seed0"],
        rpeaks_npz_path=os.path.join(_REFERENCE_ROOT, "phase", "rpeaks.npz"),
        lead_stats_path=os.path.join(_REFERENCE_ROOT, "lead_stats_f1to8_legacy.json"),
        theta_tokens_path=os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz"),
        fold_config=LEGACY_FOLD_CONFIG,
        n_records=400,
        causal_window=40,
        seed=0,
        device=device,
        perturbations={"ectopic_beat": PERTURBATIONS["ectopic_beat"]},
    )
    assert report["skipped"] == {}
    assert report["config"]["n_records"] == 400

    # Loose sanity bound against the published theta_coverage_offline=0.89627 /
    # theta_coverage_causal=0.812675 -- NOT the strict 1e-4 reproduction gate the five AUROC
    # targets below get (per the design brief: "sanity-check, report, not gate on an exact
    # number"). This only catches a GROSS upstream error (wrong decimation factor, wrong theta
    # grid, wrong rpeaks archive) -- a loud, early signal before the AUROC numbers are even read.
    assert report["config"]["theta_coverage_offline"] == pytest.approx(0.89627, abs=0.01)
    assert report["config"]["theta_coverage_causal"] == pytest.approx(0.812675, abs=0.01)

    targets = {
        ("FIN_seed0", "ectopic_beat|0.0|offline|transport_offline"): 0.4543768877271185,
        ("FIN_seed0", "ectopic_beat|1.0|offline|transport_offline"): 0.8307946236460486,
        ("FIN_seed0", "ectopic_beat|1.0|causal|transport_causal"): 0.7955706107623158,
        ("FIN_LAM0_seed0", "ectopic_beat|1.0|offline|transport_offline"): 0.6614836220141929,
        ("FIN_LAM0_seed0", "ectopic_beat|1.0|causal|transport_causal"): 0.5787963876875425,
    }
    for (ckpt_name, key), expected in targets.items():
        measured = report[ckpt_name][key]["mean_auroc"]
        assert measured == pytest.approx(expected, abs=1e-4), (ckpt_name, key, measured, expected)


#: The full combinatorial answer key, produced by an earlier run of this exact protocol -- lives
#: in the SIBLING reference repo (winder-theory-exp), not this one; winder-nominal's own copied-in
#: `artifacts/reference/` never carried this file, only the checkpoints/phase archives it was
#: computed from.
_ANSWER_KEY_PATH = "/home/blaised/winder-theory-exp/artifacts/campaign_closeout/detection_panel30k/localisation.json"  # noqa: E501

#: `run_detection_battery`'s own report keys (matching `_REFERENCE_ROOT`'s directory names) ->
#: the answer key's own checkpoint names for the SAME two 30,000-step runs.
_CKPT_TO_ANSWER_KEY_NAME = {"FIN_seed0": "FINs0_30k", "FIN_LAM0_seed0": "FINLAM0_30k"}

#: One localisation-error/detection-latency "quantum". Every raw value either metric emits is
#: `crossed_index * patch_ms` for an integer index, and `patch_ms=80.0` for this roster (the same
#: `patch_ms` the load-bearing test above already reports) -- so every RAW value is an exact
#: multiple of 80 ms, and the MEDIAN of an even-sized sample (about half of `n_records=400`,
#: after within-record masking) is a multiple of 40 ms: one HALF quantum. This tolerance is fixed
#: at that one half-quantum, decided before this test was ever run against real numbers -- not
#: fitted after seeing the observed deltas.
_MS_QUANTUM_TOLERANCE = 40.0


@pytest.mark.skipif(
    not (_HAS_PTBXL_ROOT and _HAS_REFERENCE_PHASE and _HAS_REFERENCE_CKPTS), reason=_SKIP_REASON
)
@pytest.mark.skipif(
    not os.path.isfile(_ANSWER_KEY_PATH),
    reason=f"answer key not found at {_ANSWER_KEY_PATH} -- lives in the sibling reference repo "
    "(winder-theory-exp), not this one; a machine with PTB-XL and the reference checkpoints but "
    "not that sibling repo checked out cannot run this test",
)
def test_full_battery_reproduces_every_published_cell_in_the_reference_answer_key() -> None:
    """THE PRIMARY correctness gate for this module (module docstring) -- not a supplement to the
    five theory-derived property tests, the real main check. Runs the full, UNRESTRICTED
    `run_detection_battery` (`perturbations` left at its own default, `winder.data.perturb.
    PERTURBATIONS`, all six families) against BOTH reference checkpoints, at the exact protocol
    (`LEGACY_FOLD_CONFIG`, `n_records=400`, `causal_window=40`, `seed=0`) the reference repo's
    full answer key (`_ANSWER_KEY_PATH`, winder-theory-exp, verified this session as the
    corrected, post-revert version) was produced under -- 330 cells per checkpoint, 660 total.

    Deliberately NOT subsampled below `n_records=400`: `within_record_auroc` averages over
    whichever records survive ITS OWN per-record masking, so a smaller `n_records` reproduces a
    DIFFERENT (self-consistent for that smaller sample, but not comparable) number, not merely a
    noisier estimate of the published one -- subsampling would silently break the exact
    reproduction this test exists to check. Expected wall-clock: tens of minutes (six
    perturbation families x their own amplitude sweeps x two checkpoints, roughly 6x the
    existing five-cell load-bearing test's own ~8 minutes) -- real compute validating real
    published numbers, not a unit test, and not something to shrink for wall-clock convenience.

    Compares, for every one of the 660 cells present in BOTH winder-nominal's own report and the
    answer key (key-SET equality asserted per checkpoint first, so a shrunk intersection can
    never silently pass):
      - `mean_auroc` to 1e-4 absolute -- the primary gate, unchanged from the existing 5-cell
        load-bearing test above; never loosened.
      - `localisation.median_ms` / `latency.median_ms` to one half-quantum, `_MS_QUANTUM_
        TOLERANCE` (see its own comment for why 40 ms is the right number, not an arbitrary
        looser bound). Both-NaN (e.g. an all-miss `latency.median_ms`, a real and correct value
        for a cell where no record ever crosses its own threshold) counts as a match, not a
        failure; only exactly one side being NaN, or a finite delta past the quantum, is
        recorded as a mismatch.
    ALL mismatches, across BOTH metrics and BOTH checkpoints, are collected before any assertion
    fires -- at this wall-clock cost, failing on the first mismatch and forcing a full re-run to
    see the second would be a real waste, not a safety margin.
    """
    with open(_ANSWER_KEY_PATH, encoding="utf-8") as fh:
        answer_key = json.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = run_detection_battery(
        data_root=_PTBXL_ROOT,
        roster_dir=_REFERENCE_ROOT,
        checkpoint_names=list(_CKPT_TO_ANSWER_KEY_NAME),
        rpeaks_npz_path=os.path.join(_REFERENCE_ROOT, "phase", "rpeaks.npz"),
        lead_stats_path=os.path.join(_REFERENCE_ROOT, "lead_stats_f1to8_legacy.json"),
        theta_tokens_path=os.path.join(_REFERENCE_ROOT, "phase", "theta_tokens.npz"),
        fold_config=LEGACY_FOLD_CONFIG,
        n_records=400,
        causal_window=40,
        seed=0,
        device=device,
        # perturbations intentionally omitted: this is the ONE test in this module that must run
        # the module's own full default, all six perturbation families.
    )
    assert report["skipped"] == {}
    assert report["config"]["n_records"] == 400

    mean_auroc_mismatches: list[tuple[str, str, float, float]] = []
    ms_mismatches: list[tuple[str, str, str, float, float]] = []
    n_cells_compared = 0

    for ckpt_name, answer_ckpt_name in _CKPT_TO_ANSWER_KEY_NAME.items():
        ours = report[ckpt_name]
        theirs = answer_key[answer_ckpt_name]
        assert set(ours) == set(theirs), (
            f"{ckpt_name}: cell key SETS differ (not just a subset check) -- ours has "
            f"{len(ours)} cells, the answer key has {len(theirs)}, symmetric difference: "
            f"{set(ours) ^ set(theirs)}"
        )
        for key, their_cell in theirs.items():
            our_cell = ours[key]
            n_cells_compared += 1

            our_mean, their_mean = our_cell["mean_auroc"], their_cell["mean_auroc"]
            if abs(our_mean - their_mean) > 1e-4:
                mean_auroc_mismatches.append((ckpt_name, key, our_mean, their_mean))

            for field in ("localisation", "latency"):
                our_ms = our_cell[field]["median_ms"]
                their_ms = their_cell[field]["median_ms"]
                our_nan, their_nan = math.isnan(our_ms), math.isnan(their_ms)
                if our_nan and their_nan:
                    continue  # both-NaN is a match (e.g. an all-miss latency median)
                if our_nan or their_nan or abs(our_ms - their_ms) > _MS_QUANTUM_TOLERANCE:
                    ms_mismatches.append((ckpt_name, key, field, our_ms, their_ms))

    expected_total = sum(len(answer_key[name]) for name in _CKPT_TO_ANSWER_KEY_NAME.values())
    assert n_cells_compared == expected_total == 660, (
        f"compared {n_cells_compared} cells, answer key holds {expected_total} -- expected "
        f"330 cells/checkpoint x 2 checkpoints = 660; nothing should have been silently dropped"
    )
    assert not mean_auroc_mismatches, (
        f"{len(mean_auroc_mismatches)}/{n_cells_compared} cells exceeded the 1e-4 mean_auroc "
        f"gate (ckpt, key, ours, theirs): {mean_auroc_mismatches}"
    )
    assert not ms_mismatches, (
        f"{len(ms_mismatches)}/{n_cells_compared} cells exceeded the "
        f"{_MS_QUANTUM_TOLERANCE} ms median_ms tolerance (ckpt, key, field, ours, theirs): "
        f"{ms_mismatches}"
    )

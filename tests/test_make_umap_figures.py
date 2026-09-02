"""Tests for `scripts/make_umap_figures.py`, the UMAP figure suite.

`tests/conftest.py` puts `scripts/` on `sys.path`, so `import make_umap_figures` here means
`scripts/make_umap_figures.py`, not the `winder` package.

Three layers:
  - Pure-function tests with NO optional dependency at all: the ported beat-landmark numerics on
    a synthetic ECG whose PQRST positions are known by construction, the measured-only stage
    labelling rule, the dominance collapse of multi-label superclasses, and the paired-sampling
    contract (same records, same tokens, stratified, seeded, reproducible).
  - Figure-layout tests on synthetic embeddings, which need matplotlib but not umap: the
    title/caption and axis-label contracts, the tickless arrowed axes, shared limits across
    panels, and byte-identical re-rendering.
  - One end-to-end test of the joint fit itself, behind `pytest.importorskip("umap")`, so the
    suite stays green on a machine without the dependency (`umap-learn` is deliberately NOT in
    `pyproject.toml`; the script is run through `uv run --with`).
"""

from __future__ import annotations

import math
import os

import make_umap_figures as muf
import numpy as np
import pytest
import torch

TWO_PI = 2.0 * math.pi

#: These renders rasterize 6 x N-point scatters, so they are legitimately bigger than the pure
#: vector figures 1-10 (which the paper set holds under 500 kB). Still a budget, not a blank
#: cheque: a figure over this has almost certainly lost its `rasterized=True`.
_PDF_BUDGET_BYTES = 1_200 * 1024


# ----------------------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------------------


def _synthetic_beat_waveforms(
    n_records: int = 6, n_samples: int = 1000, patch_width: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """A 12-lead cohort whose lead-II trace is a known PQRST shape on a known phase clock.

    Theta is exact and linear over four beats per record, so the sample-to-phase-bin map is
    unambiguous; lead II carries a sharp R spike at phase 0, a broad T hump at phase ~pi/2 and a
    smaller, broader P hump at phase ~3pi/2. The landmark finder must recover exactly those.
    """
    n_tokens = n_samples // patch_width
    theta = torch.empty(n_records, n_tokens, dtype=torch.float32)
    for r in range(n_records):
        theta[r] = torch.remainder(torch.linspace(0.0, 4.0 * TWO_PI, n_tokens) + 0.11 * r, TWO_PI)
    sample_theta = theta.repeat_interleave(patch_width, dim=1)[:, :n_samples]

    def _hump(centre: float, width: float, height: float) -> torch.Tensor:
        delta = torch.remainder(sample_theta - centre + math.pi, TWO_PI) - math.pi
        return height * torch.exp(-0.5 * (delta / width) ** 2)

    lead_ii = (
        _hump(0.0, 0.06, 6.0) + _hump(math.pi / 2, 0.30, 1.4) + _hump(3 * math.pi / 2, 0.30, 1.0)
    )
    waveforms = torch.zeros(n_records, 12, n_samples)
    waveforms[:, 1] = lead_ii  # LEAD_ORDER index 1 is lead II
    return waveforms, theta


# ----------------------------------------------------------------------------------------
# The measured beat
# ----------------------------------------------------------------------------------------


def test_ensemble_beat_recovers_landmarks_planted_at_known_phases() -> None:
    """QRS at phase 0, T at pi/2, P at 3pi/2 by construction -- the finder must land on those."""
    waveforms, theta = _synthetic_beat_waveforms()
    beat = muf.ensemble_beat(waveforms, theta, patch_width=8, n_fine=128)
    measured = beat["measured"]
    centres = np.asarray(beat["fine_centers"])
    assert centres[measured["qrs_fine_bin"]] == pytest.approx(0.0, abs=0.2)
    assert centres[measured["t_wave_fine_bin"]] == pytest.approx(math.pi / 2, abs=0.2)
    assert centres[measured["p_wave_fine_bin"]] == pytest.approx(3 * math.pi / 2, abs=0.2)
    assert measured["qrs_phase_bin"] == 0
    assert measured["t_wave_phase_bin"] == 1  # pi/2 is the centre of bin 1 of 8
    assert measured["p_wave_phase_bin"] == 5


def test_ensemble_beat_rejects_a_time_major_waveform_tensor() -> None:
    """`EcgWindowDataset` emits (N, 12, T); indexing it (N, T, 12) silently empties most bins."""
    waveforms, theta = _synthetic_beat_waveforms()
    with pytest.raises(ValueError, match="lead-major"):
        muf.ensemble_beat(waveforms.transpose(1, 2), theta, patch_width=8)


def test_stage_bin_labels_names_only_measured_landmarks() -> None:
    beat = {"measured": {"qrs_phase_bin": 0, "t_wave_phase_bin": 2, "p_wave_phase_bin": None}}
    assert muf.stage_bin_labels(beat) == {0: "QRS", 2: "T Wave"}


def test_stage_bin_labels_joins_two_landmarks_that_share_a_bin() -> None:
    """A collision must be visible in the label, not resolved by whichever key is read last."""
    beat = {"measured": {"qrs_phase_bin": 3, "t_wave_phase_bin": 3}}
    assert muf.stage_bin_labels(beat) == {3: "QRS / T Wave"}


def test_stage_phase_centres_are_bin_centres_not_bin_edges() -> None:
    centres = muf.stage_phase_centres({0: "QRS", 4: "Diastole"}, n_bins=8)
    assert centres["QRS"] == pytest.approx(TWO_PI * 0.5 / 8)
    assert centres["Diastole"] == pytest.approx(TWO_PI * 4.5 / 8)


# ----------------------------------------------------------------------------------------
# Superclass collapse and paired sampling
# ----------------------------------------------------------------------------------------


def test_dominant_superclass_lets_pathology_outrank_norm() -> None:
    """A record co-asserting NORM and a pathology is not a normal ECG (ptbxl rule R5)."""
    classes = ("NORM", "MI", "STTC", "CD", "HYP")
    labels = np.array(
        [
            [1, 0, 0, 0, 0],  # NORM only
            [1, 0, 0, 1, 0],  # NORM + CD  -> CD
            [0, 1, 1, 0, 0],  # MI + STTC  -> MI (MI is first in DOMINANCE_ORDER)
            [0, 0, 0, 0, 0],  # nothing    -> -1
        ],
        dtype=np.float32,
    )
    assert list(muf.dominant_superclass(labels, classes)) == [0, 3, 1, -1]


def test_longest_finite_run_finds_the_longer_of_two_runs() -> None:
    mask = np.array([True, True, False, True, True, True, False, True])
    assert muf.longest_finite_run(mask) == (3, 3)


def test_longest_finite_run_handles_an_all_true_mask() -> None:
    assert muf.longest_finite_run(np.ones(5, dtype=bool)) == (0, 5)


def test_longest_finite_run_returns_zero_length_when_nothing_is_finite() -> None:
    assert muf.longest_finite_run(np.zeros(5, dtype=bool)) == (0, 0)


def _sampling_fixture(n: int = 40, n_tokens_total: int = 30) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(7)
    theta = torch.rand(n, n_tokens_total) * TWO_PI
    # Punch a NaN gap into a third of the records, so the contiguity rule has work to do.
    for i in range(0, n, 3):
        theta[i, 4:7] = float("nan")
    labels = np.zeros((n, 5), dtype=np.float32)
    labels[np.arange(n), rng.integers(0, 5, size=n)] = 1.0
    return theta, labels


def test_sample_token_pairs_returns_only_finite_theta_positions() -> None:
    theta, labels = _sampling_fixture()
    records, tokens = muf.sample_token_pairs(
        theta,
        labels,
        ("NORM", "MI", "STTC", "CD", "HYP"),
        n_records=10,
        n_tokens=12,
        rng=np.random.default_rng(0),
    )
    assert records.shape == tokens.shape == (10, 12)
    assert torch.isfinite(theta[records.ravel(), tokens.ravel()]).all()


def test_sample_token_pairs_draws_a_contiguous_run_per_record() -> None:
    """Figure 12 draws these tokens as a path in time order; gaps would make that a lie."""
    theta, labels = _sampling_fixture()
    _records, tokens = muf.sample_token_pairs(
        theta,
        labels,
        ("NORM", "MI", "STTC", "CD", "HYP"),
        n_records=8,
        n_tokens=10,
        rng=np.random.default_rng(1),
    )
    assert np.all(np.diff(tokens, axis=1) == 1)


def test_sample_token_pairs_is_reproducible_under_the_same_seed() -> None:
    theta, labels = _sampling_fixture()
    classes = ("NORM", "MI", "STTC", "CD", "HYP")
    first = muf.sample_token_pairs(
        theta, labels, classes, n_records=9, n_tokens=11, rng=np.random.default_rng(3)
    )
    second = muf.sample_token_pairs(
        theta, labels, classes, n_records=9, n_tokens=11, rng=np.random.default_rng(3)
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_sample_token_pairs_is_stratified_over_the_eligible_pool() -> None:
    """Every class that has eligible records must be represented, not just the common ones."""
    theta = torch.rand(50, 30) * TWO_PI
    labels = np.zeros((50, 5), dtype=np.float32)
    labels[:40, 0] = 1.0  # 40 NORM
    labels[40:, 4] = 1.0  # 10 HYP
    classes = ("NORM", "MI", "STTC", "CD", "HYP")
    records, _tokens = muf.sample_token_pairs(
        theta, labels, classes, n_records=20, n_tokens=10, rng=np.random.default_rng(0)
    )
    dominant = muf.dominant_superclass(labels, classes)[records[:, 0]]
    assert int((dominant == 0).sum()) == 16
    assert int((dominant == 4).sum()) == 4


def test_sample_token_pairs_raises_when_too_few_records_are_eligible() -> None:
    theta = torch.full((5, 30), float("nan"))
    labels = np.zeros((5, 5), dtype=np.float32)
    labels[:, 0] = 1.0
    with pytest.raises(ValueError, match="consecutive"):
        muf.sample_token_pairs(
            theta,
            labels,
            ("NORM", "MI", "STTC", "CD", "HYP"),
            n_records=3,
            n_tokens=10,
            rng=np.random.default_rng(0),
        )


def test_dominance_order_is_a_permutation_of_the_five_superclasses() -> None:
    """`DOMINANCE_ORDER` is kept for `sample_token_pairs`'s own stratification even though the
    display names it used to feed (fig13's `SUPERCLASS_DISPLAY`) were retired with fig13."""
    assert set(muf.DOMINANCE_ORDER) == {"NORM", "MI", "STTC", "CD", "HYP"}


# ----------------------------------------------------------------------------------------
# Figure layout (matplotlib only, no umap)
# ----------------------------------------------------------------------------------------


def test_the_umap_figures_report_step_5000_only() -> None:
    """CTO policy of 2026-08-20: the joint fit spans the two step-5,000 cells and nothing else."""
    assert muf.UMAP_STEPS == (muf.ANCHOR_STEP,) == (5000,)
    assert len(muf.ARM_CLASSES) * len(muf.UMAP_STEPS) == 2


def test_figure_stems_cover_every_deliverable_and_reject_a_typo() -> None:
    """`--only` is validated against this tuple, so a mistyped stem raises rather than writing a
    `status=PASS` record with zero figures in it."""
    assert set(muf.FIGURE_STEMS) == {"umap_embedding_cache", "fig14_phase_ring_staged"}
    with pytest.raises(SystemExit, match="unknown figure stem"):
        muf.main(["--only", "fig11_umap_phase_seed2"])


# ----------------------------------------------------------------------------------------
# The joint fit itself (needs umap-learn, which is deliberately not a project dependency)
# ----------------------------------------------------------------------------------------


def test_fit_joint_umap_splits_one_fit_back_into_its_cells_in_order() -> None:
    pytest.importorskip("umap")
    rng = np.random.default_rng(0)
    cells = {"a": rng.normal(size=(60, 8)), "b": rng.normal(size=(40, 8)) + 4.0}
    embeddings, version = muf.fit_joint_umap(
        cells, n_neighbors=5, min_dist=0.1, metric="cosine", random_state=0
    )
    assert list(embeddings) == ["a", "b"]
    assert embeddings["a"].shape == (60, 2)
    assert embeddings["b"].shape == (40, 2)
    assert version.startswith("0.5.")


def test_knn_phase_coherence_separates_a_phase_ordered_layout_from_its_own_null() -> None:
    """A ring laid out in phase order must score near 1; the shuffled-phi null must score low."""
    phi = np.linspace(0.0, TWO_PI, 600, endpoint=False)
    ring = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    value, null = muf.knn_phase_coherence(ring, phi, k=10, seed=0)
    assert value > 0.98
    assert null < 0.35
    assert value > null


def test_knn_phase_coherence_is_invariant_to_rotation_reflection_and_scale() -> None:
    """A UMAP layout is free to rotate, mirror and rescale; the diagnostic must not notice."""
    rng = np.random.default_rng(0)
    phi = rng.uniform(0.0, TWO_PI, 400)
    points = np.stack([np.cos(phi), np.sin(phi)], axis=1) + 0.05 * rng.normal(size=(400, 2))
    angle = 0.7
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mirrored = (points @ rotate.T) * np.array([3.0, -3.0])
    first, _n1 = muf.knn_phase_coherence(points, phi, k=10, seed=0)
    second, _n2 = muf.knn_phase_coherence(mirrored, phi, k=10, seed=0)
    assert first == pytest.approx(second, abs=1e-9)


def test_knn_phase_coherence_collapses_to_the_null_on_a_phase_blind_layout() -> None:
    """Random positions carry no phase structure, so the statistic and its null must agree."""
    rng = np.random.default_rng(1)
    phi = rng.uniform(0.0, TWO_PI, 500)
    points = rng.normal(size=(500, 2))
    value, null = muf.knn_phase_coherence(points, phi, k=10, seed=0)
    assert abs(value - null) < 0.12


def test_fit_joint_umap_is_deterministic_under_a_fixed_random_state() -> None:
    pytest.importorskip("umap")
    rng = np.random.default_rng(1)
    cells = {"a": rng.normal(size=(80, 8))}
    first, _v1 = muf.fit_joint_umap(
        cells, n_neighbors=5, min_dist=0.1, metric="cosine", random_state=0
    )
    second, _v2 = muf.fit_joint_umap(
        cells, n_neighbors=5, min_dist=0.1, metric="cosine", random_state=0
    )
    np.testing.assert_allclose(first["a"], second["a"])


# ----------------------------------------------------------------------------------------
# Against the real rendered output
# ----------------------------------------------------------------------------------------

_FIGURE_DIR = os.path.expanduser("~/winder-paper/figures")
#: fig11/fig12/fig13's own static PDFs stay on disk (retired from this script, not deleted from
#: the manuscript), but this script no longer generates them, so only fig14 belongs in a test of
#: what THIS script's own run actually produces.
_UMAP_STEMS = ("fig14_phase_ring_staged",)


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_FIGURE_DIR, "fig14_phase_ring_staged.pdf")),
    reason="UMAP figures not rendered on this machine",
)
def test_every_rendered_umap_figure_exists_within_the_size_budget() -> None:
    for stem in _UMAP_STEMS:
        for extension in ("pdf", "png"):
            path = os.path.join(_FIGURE_DIR, f"{stem}.{extension}")
            assert os.path.isfile(path), f"{path} missing"
        pdf = os.path.join(_FIGURE_DIR, f"{stem}.pdf")
        assert os.path.getsize(pdf) < _PDF_BUDGET_BYTES, f"{stem} exceeds the PDF size budget"

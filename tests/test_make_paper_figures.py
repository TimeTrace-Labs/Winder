"""Tests for `scripts/make_paper_figures.py`.

`tests/conftest.py` puts `scripts/` on `sys.path`, so `import make_paper_figures` here means
`scripts/make_paper_figures.py`, not the `winder` package.

The manuscript's sealed-fold-report-driven figures (2-13, 15-16) and their pure-function tests
were retired along with that code once the manuscript shipped (see the module's own docstring).
What remains here covers what the script still actively builds: `apply_paper_style`/
`_assert_axis_labels` (shared style contracts), `assert_operators_share_state` (relocated from
the retired `make_figures.py`, its own tests ported with it), and a reproducibility check on
`phase_ring_grid_figure` -- the fig01/fig14 engine -- replacing the one that used to run against
the now-deleted `figure_headline_auroc`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import make_paper_figures as mpf
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from winder.config import ArmConfig
from winder.determinism import generator, init_parameters
from winder.eval.readout import operator_from_checkpoint
from winder.jepa import checkpoint
from winder.jepa.model import JepaConfig, build_jepa
from winder.jepa.train import TrainConfig
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig

# ----------------------------------------------------------------------------------------
# The style contracts
# ----------------------------------------------------------------------------------------


def test_apply_paper_style_sets_one_typography_scale_and_is_idempotent() -> None:
    """Every builder calls this; a per-figure font size drifting back in is the failure it stops."""

    # Indexed through a plain dict: matplotlib's stubs type `RcParams.__getitem__` against a
    # Literal union of every known key, which a `str` loop variable cannot satisfy.
    def _snapshot() -> dict[str, Any]:
        current = cast(dict[str, Any], dict(plt.rcParams))
        return {key: current[key] for key in mpf._TYPOGRAPHY}

    mpf.apply_paper_style()
    first = _snapshot()
    mpf.apply_paper_style()
    assert first == _snapshot()
    assert first["axes.labelsize"] == mpf._TYPOGRAPHY["axes.labelsize"]
    assert first["legend.fontsize"] < first["xtick.labelsize"] < first["axes.labelsize"]


def test_assert_axis_labels_rejects_a_wholly_unlabelled_axes() -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    with pytest.raises(AssertionError, match="ylabel"):
        mpf._assert_axis_labels(fig)
    plt.close(fig)


def test_assert_axis_labels_accepts_a_label_reached_through_the_shared_group() -> None:
    """An inner panel of a shared-axis grid legitimately carries no label of its own."""
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].set_ylabel("y")
    for ax in axes:
        ax.set_xlabel("x")
    mpf._assert_axis_labels(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# assert_operators_share_state -- relocated from the retired make_figures.py, tests ported
# ----------------------------------------------------------------------------------------

_N_SAMPLES = 1000
_N_TOKENS = 250
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


def _write_tiny_checkpoint(ckpt_dir: Path, *, k_j: list[int] | None = None, seed: int = 0) -> str:
    k_j = k_j if k_j is not None else _OP_K_J
    jepa_cfg = _tiny_jepa_config()
    model = build_jepa(jepa_cfg, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    operator = CyclicOperator(CyclicOperatorConfig(k0=_OP_K0, n_j=_OP_N_J, k_j=k_j))
    arm_cfg = ArmConfig(
        name="tiny_cyclic",
        seed=seed,
        operator_name="cyclic",
        operator={"k0": _OP_K0, "n_j": _OP_N_J, "k_j": k_j},
    )
    config_yaml = checkpoint.resolved_config_yaml(
        jepa_cfg, TrainConfig(n_steps=10, seed_pretrain=seed), arm_config=arm_cfg
    )
    checkpoint.save_checkpoint(
        str(ckpt_dir),
        model=model,
        optimizer=optimizer,
        step=10,
        generators={},
        config_yaml=config_yaml,
        meta={},
        operator=operator,
    )
    return str(ckpt_dir)


def test_assert_operators_share_state_returns_the_operator_when_all_agree(tmp_path: Path) -> None:
    ckpt_a = _write_tiny_checkpoint(tmp_path / "a" / "checkpoint", seed=0)
    ckpt_b = _write_tiny_checkpoint(tmp_path / "b" / "checkpoint", seed=1)  # seed irrelevant here
    op_a = operator_from_checkpoint(ckpt_a)
    op_b = operator_from_checkpoint(ckpt_b)
    assert op_a is not None and op_b is not None
    shared = mpf.assert_operators_share_state({"a": op_a, "b": op_b})
    assert shared is op_a  # first by insertion order


def test_assert_operators_share_state_raises_on_a_structural_mismatch(tmp_path: Path) -> None:
    ckpt_a = _write_tiny_checkpoint(tmp_path / "a" / "checkpoint", k_j=[2, 2])
    ckpt_b = _write_tiny_checkpoint(tmp_path / "b" / "checkpoint", k_j=[3, 1])
    op_a = operator_from_checkpoint(ckpt_a)
    op_b = operator_from_checkpoint(ckpt_b)
    assert op_a is not None and op_b is not None
    with pytest.raises(ValueError, match="differs"):
        mpf.assert_operators_share_state({"a": op_a, "b": op_b})


def test_assert_operators_share_state_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mpf.assert_operators_share_state({})


# ----------------------------------------------------------------------------------------
# phase_ring_grid_figure -- the fig01/fig14 engine
# ----------------------------------------------------------------------------------------


def _synthetic_loops(names: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    n_bins = 24
    return {name: (rng.normal(size=n_bins), rng.normal(size=n_bins)) for name in names}


def test_phase_ring_grid_figure_renders_one_column_per_arm() -> None:
    for names in (("signal_seed0",), ("signal_seed0", "control_seed0")):
        fig = mpf.phase_ring_grid_figure(_synthetic_loops(names), harmonic=1)
        # Two rows (raw, RMS-normalised) x len(names) columns, plus one colorbar axes.
        assert len(fig.axes) == 2 * len(names) + 1
        plt.close(fig)


def test_phase_ring_grid_figure_rejects_an_unregistered_arm_name() -> None:
    """`_ARM_LABEL` is a fixed nominal-arm vocabulary; an ablation arm name is out of scope for
    this figure at this stage (the ablation supplement is table-only, per the porting plan)."""
    with pytest.raises(KeyError):
        mpf.phase_ring_grid_figure(_synthetic_loops(("no_augmentation_seed0",)), harmonic=1)


def test_phase_ring_grid_figure_rendering_is_reproducible(tmp_path: Path) -> None:
    """Same input, same code, byte-identical raster output -- matplotlib's PNG writer embeds no
    timestamp, so this is a real determinism check rather than a tautology (the PDF writer does
    embed one, which is why the comparison is on the PNG)."""
    loops = _synthetic_loops(("signal_seed0", "control_seed0"))
    first = mpf._save(mpf.phase_ring_grid_figure(loops, harmonic=1), str(tmp_path / "a"), "fig")
    second = mpf._save(mpf.phase_ring_grid_figure(loops, harmonic=1), str(tmp_path / "b"), "fig")
    with open(first["png"], "rb") as fh_a, open(second["png"], "rb") as fh_b:
        assert fh_a.read() == fh_b.read()


@pytest.mark.parametrize("rows", ["top", "bottom"])
def test_phase_ring_grid_figure_single_row_has_one_row_of_axes_per_arm(rows: str) -> None:
    """`rows="top"`/`"bottom"` renders exactly one row instead of two -- one axes per arm plus
    one colorbar axes, not the full two-row grid's `2 * len(names) + 1`."""
    names = ("signal_seed0", "control_seed0")
    fig = mpf.phase_ring_grid_figure(_synthetic_loops(names), harmonic=1, rows=rows)  # type: ignore[arg-type]
    assert len(fig.axes) == len(names) + 1
    plt.close(fig)


def test_phase_ring_grid_figure_bottom_row_alone_still_gets_stage_labels() -> None:
    """`rows="bottom"` must not silently drop the stage-label annotation -- it is the ONLY row
    left, and stage labels are `phase_ring_grid_figure`'s whole reason to exist as fig14 rather
    than reusing fig01's own call. Checked via the annotation call succeeding without error
    against real (not out-of-range) stage bins, rather than asserting on annotation internals."""
    fig = mpf.phase_ring_grid_figure(
        _synthetic_loops(("signal_seed0",)),
        harmonic=1,
        stage_labels={0: "QRS", 4: "T Wave"},
        n_stage_bins=8,
        rows="bottom",
    )
    assert len(fig.axes) == 2  # one arm column + colorbar
    plt.close(fig)


def test_phase_ring_grid_figure_row_default_matches_explicit_both() -> None:
    """The new `rows` parameter must be fully backward compatible: every pre-existing call site
    omits it, so its default has to be indistinguishable from `rows="both"`, not just similar."""
    loops = _synthetic_loops(("signal_seed0", "control_seed0"))
    default_fig = mpf.phase_ring_grid_figure(loops, harmonic=1)
    explicit_fig = mpf.phase_ring_grid_figure(loops, harmonic=1, rows="both")
    assert len(default_fig.axes) == len(explicit_fig.axes)
    plt.close(default_fig)
    plt.close(explicit_fig)

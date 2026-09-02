import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.plotting.latents import phase_ring_comparison_figure, phase_ring_figure
from winder.plotting.style import assert_no_baked_in_title_or_caption

# Same toy spectrum as tests/test_transport_loop_projection.py -- K = 2 + 2*6 = 14. The
# z_parity fixture the design brief prefers (tests/fixtures/z_parity_fin_seed0_step5000.npz) does
# not exist yet: it is generated in Phase P6 (acceptance gate), which runs after this one. This
# synthetic fixture is the brief's own documented fallback.
_K0, _N_J, _K_J = 2, [1, 2, 3], [3, 2, 1]


def _toy_operator() -> CyclicOperator:
    return CyclicOperator(CyclicOperatorConfig(k0=_K0, n_j=_N_J, k_j=_K_J))


def _winding_z_theta(
    operator: CyclicOperator, *, n_records: int = 6, n_frames: int = 48, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(z, theta)` with a real, non-degenerate winding pattern: `theta` sweeps `[0, 2*pi)` per
    record and `z` is the operator's own exact transport of a random per-record start point --
    the same construction as `test_transport_loop_projection.py::
    test_consumes_phase_resolved_trajectory_output_directly`, reused rather than re-derived. This
    is deliberately NOT i.i.d. noise: harmonic-`j`'s block genuinely rotates `n_j` times with
    theta, so `phase_ring_figure` has a real loop to render, not empty/degenerate axes.
    """
    gen = torch.Generator().manual_seed(seed)
    theta = (
        (torch.arange(n_frames, dtype=torch.float64) * (2 * math.pi / n_frames))
        .expand(n_records, n_frames)
        .contiguous()
    )
    z0 = torch.randn(n_records, 1, operator.dimension, dtype=torch.float64, generator=gen)
    z = operator.transport(z0.expand(n_records, n_frames, -1), theta)
    return z, theta


def test_phase_ring_figure_renders_without_raising() -> None:
    operator = _toy_operator()
    z, theta = _winding_z_theta(operator)
    fig = phase_ring_figure(z, theta, operator, harmonic=1, n_bins=8)
    assert len(fig.axes) == 1
    # The rendered loop must have real, non-collapsed content: at least two distinct (Re, Im)
    # points, not every bin landing on the same point (the phase-blind null).
    (line,) = fig.axes[0].lines
    xs = np.asarray(line.get_xdata()).tolist()
    ys = np.asarray(line.get_ydata()).tolist()
    assert len(set(zip(xs, ys, strict=True))) > 1
    plt.close(fig)


def test_phase_ring_comparison_figure_renders_one_panel_per_arm() -> None:
    operator = _toy_operator()
    arms = {
        "signal": _winding_z_theta(operator, seed=1),
        "control": _winding_z_theta(operator, seed=2),
    }
    fig = phase_ring_comparison_figure(arms, operator, harmonic=2)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_phase_ring_figure_has_no_baked_in_title_or_caption() -> None:
    """The convention is "labelled axes, no title" -- both halves need their own assertion, or
    a figure with no text at all would pass a naive "no title" check too."""
    operator = _toy_operator()
    z, theta = _winding_z_theta(operator)
    arms = {"signal": (z, theta), "control": _winding_z_theta(operator, seed=3)}

    single = phase_ring_figure(z, theta, operator)
    comparison = phase_ring_comparison_figure(arms, operator)
    try:
        for fig in (single, comparison):
            assert_no_baked_in_title_or_caption(fig)
            for ax in fig.axes:
                assert ax.get_xlabel() != ""
                assert ax.get_ylabel() != ""
    finally:
        plt.close(single)
        plt.close(comparison)


def test_harmonic_is_1_indexed_and_out_of_range_raises_with_the_callers_own_number() -> None:
    """`harmonic=1` must mean the fundamental (`n_j=1`, `harmonic_index=0`), and an out-of-range
    `harmonic` must be reported in the caller's own 1-indexed vocabulary, not the internally
    shifted `harmonic_index`."""
    operator = _toy_operator()
    z, theta = _winding_z_theta(operator)

    fig = phase_ring_figure(z, theta, operator, harmonic=1)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_text() == "n_j=1"
    plt.close(fig)

    with pytest.raises(ValueError, match=r"harmonic must be 1-indexed in \[1, 3\]"):
        phase_ring_figure(z, theta, operator, harmonic=0)
    with pytest.raises(ValueError, match=r"harmonic must be 1-indexed in \[1, 3\]"):
        phase_ring_figure(z, theta, operator, harmonic=len(_N_J) + 1)


def test_phase_ring_comparison_figure_rejects_an_empty_arms_dict() -> None:
    operator = _toy_operator()
    with pytest.raises(ValueError, match="non-empty"):
        phase_ring_comparison_figure({}, operator)

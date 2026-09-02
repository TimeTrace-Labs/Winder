"""The phase-ring figure: a gauge-invariant picture of whether a checkpoint's latent actually
stages the cardiac cycle in order, built directly on `winder.transport.geometry`'s two
closed-form functions (`phase_resolved_trajectory`, `harmonic_loop_projection`) -- no UMAP, no
re-derivation of either function's maths.

Visual idea ported from the reference repo's `scripts/p2_panel_figures.py::b4_phase_staging_loop`
(aspect-equal axes, bins connected in cyclic order and closed back to bin 0, a colour progression
across bins): that function reads precomputed `geometry.json`/`beat.json`; this module computes
directly from `z`/`theta`/`operator` tensors, since winder-nominal has no campaign-script/JSON
layer.

**The `harmonic` argument is 1-INDEXED, `harmonic_index` is 0-INDEXED.** Both functions here take
`harmonic: int = 1` meaning "the fundamental" (n_j=1), matching the design brief's own
pseudocode. `harmonic_loop_projection` itself takes `harmonic_index`, 0-indexed into the
operator's `n_j` list (`harmonic_index=0` is n_j=1, the FIRST harmonic). Every call site below
does `harmonic_index = harmonic - 1` explicitly (`_validate_harmonic`), and validates `harmonic`
itself (1-indexed, in the caller's own vocabulary) before that translation -- so an out-of-range
value raises with the number the caller actually passed, not an internally-shifted one.

**Style ownership.** Both public functions call `apply_style()` themselves rather than requiring
the caller to remember to. Rationale: `winder.plotting.style.apply_style` mutates global
matplotlib rcParams with no visible effect if skipped (a forgotten call silently renders in
matplotlib's default style, not an error) -- exactly the kind of reproducibility footgun a
figure-producing script should not be able to hit by omission. The cost is a global side effect
on every call; `apply_style` is documented as idempotent so repeated calls (e.g. once per panel
in `phase_ring_comparison_figure`) are harmless.

**Titles and captions.** Neither function ever calls `ax.set_title` or `fig.suptitle`. Arm
identity and the harmonic's winding number go in a legend entry per panel instead
(`winder.plotting.style.assert_no_baked_in_title_or_caption` checks this).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from winder.operators.harmonic import HarmonicTransport
from winder.plotting.style import apply_style
from winder.transport.geometry import harmonic_loop_projection, phase_resolved_trajectory

__all__ = ["phase_ring_figure", "phase_ring_comparison_figure"]

#: Bin count for `phase_ring_comparison_figure`, which has no `n_bins` parameter of its own (the
#: design brief's signature fixes it) -- matches `phase_ring_figure`'s own default and
#: `winder.transport.report.N_PHASE_BINS`, the reference repo's clinical-staging bin count
#: (~105 ms at PTB-XL's cohort-median ~842.6 ms RR interval).
_DEFAULT_N_BINS = 8

_XLABEL = "Re ⟨u, v_b⟩"
_YLABEL = "Im ⟨u, v_b⟩"


def _validate_harmonic(harmonic: int, operator: HarmonicTransport) -> int:
    """1-indexed `harmonic` -> 0-indexed `harmonic_index`, raising on the caller's own number."""
    n_harmonics = len(operator.n_j)
    if not 1 <= harmonic <= n_harmonics:
        raise ValueError(
            f"harmonic must be 1-indexed in [1, {n_harmonics}] (harmonic=1 is the fundamental "
            f"n_j=1), got {harmonic}"
        )
    return harmonic - 1


def _ring_xy(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    *,
    harmonic_index: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """`(z, theta)` -> the loop's `(Re, Im)` points and the harmonic's own winding number `n_j`.

    Exactly the composition `winder.transport.report.geometry_report` already uses (binned means
    from `phase_resolved_trajectory`, round-tripped through a float64 tensor, fed to
    `harmonic_loop_projection`) -- reused here rather than re-derived. A reference bin (bin 0)
    with zero or non-finite harmonic energy raises `ValueError` and is NOT caught: unlike
    `geometry_report`, which loops every harmonic and records a per-harmonic error string, a
    single figure has no sensible degenerate rendering for "this harmonic has no signal in bin
    0" and should fail loudly rather than plot nothing.
    """
    traj = phase_resolved_trajectory(z, theta, operator, n_bins=n_bins)
    binned = torch.tensor(traj["binned_means"], dtype=torch.float64)
    proj = harmonic_loop_projection(binned, operator, harmonic_index=harmonic_index)
    return np.asarray(proj["real"]), np.asarray(proj["imag"]), int(proj["n_j"])


def _draw_ring(ax: Axes, re: np.ndarray, im: np.ndarray, *, label: str) -> None:
    """Draw one arm/checkpoint's loop on `ax`: bins in cyclic order, closed back to bin 0, a
    colour progression across bins (visual idea from `b4_phase_staging_loop`, module docstring).
    Axis labels are set here (non-empty, per the title/caption contract); no title is ever set.
    """
    n_bins = len(re)
    order = [*range(n_bins), 0]
    ax.plot(re[order], im[order], "-", color="0.6", linewidth=1.0, zorder=1)
    cmap = matplotlib.colormaps["twilight"]
    ax.scatter(
        re,
        im,
        c=[cmap(b / n_bins) for b in range(n_bins)],
        s=40,
        zorder=2,
        edgecolor="0.15",
        linewidth=0.3,
        label=label,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.legend()


def phase_ring_figure(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    *,
    harmonic: int = 1,
    n_bins: int = 8,
) -> Figure:
    """The gauge-invariant phase-ring loop for one checkpoint's `(z, theta)`, one harmonic.

    `harmonic=1` (the default) is the fundamental, n_j=1 -- see module docstring for the
    1-indexed/0-indexed mapping. The harmonic's own winding number `n_j` is reported as the
    plotted loop's legend label, never a title.
    """
    apply_style()
    harmonic_index = _validate_harmonic(harmonic, operator)
    re, im, n_j = _ring_xy(z, theta, operator, harmonic_index=harmonic_index, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    _draw_ring(ax, re, im, label=f"n_j={n_j}")
    return fig


def phase_ring_comparison_figure(
    arms: dict[str, tuple[torch.Tensor, torch.Tensor]],
    operator: HarmonicTransport,
    *,
    harmonic: int = 1,
) -> Figure:
    """Small-multiples phase-ring figure: one panel per arm in `arms`, each built exactly as
    `phase_ring_figure`'s single-arm case, at the shared `_DEFAULT_N_BINS` bin count. Per-panel
    arm names are the panel's legend label, never a subplot title.
    """
    if not arms:
        raise ValueError("arms must be non-empty -- nothing to compare")
    apply_style()
    harmonic_index = _validate_harmonic(harmonic, operator)
    n_arms = len(arms)
    fig, axes_arr = plt.subplots(1, n_arms, figsize=(3.2 * n_arms, 3.2), squeeze=False)
    axes: list[Axes] = list(axes_arr[0])
    for ax, (name, (z, theta)) in zip(axes, arms.items(), strict=True):
        re, im, _n_j = _ring_xy(
            z, theta, operator, harmonic_index=harmonic_index, n_bins=_DEFAULT_N_BINS
        )
        _draw_ring(ax, re, im, label=name)
    return fig

"""Shared aurora presentation toolkit: cyclic colormap, panel framing, and phase colorbar.

Extracted from the former `scripts/render_umap_aurora.py` (the cosmetic joint-UMAP re-render),
since `scripts/render_latent_projections.py` (fig17, actively maintained) depends on these
symbols too. The re-render script itself was retired -- it only restyled an already-cached
embedding and shipped no figure the manuscript uses.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

TWO_PI = 2.0 * np.pi

#: Verbatim from the reference repo's `fig_c2b_umap_gallery.AURORA`. Cyclic by construction.
AURORA = LinearSegmentedColormap.from_list(
    "aurora",
    ["#2b1e66", "#1f7a8c", "#7fb069", "#e8c547", "#d1495b", "#7d3c98", "#2b1e66"],
)

#: Measured landmarks (reference repo `_bin_labels` + campaign_finale beat.json): phase-bin left
#: edges of 8 bins. Capitalised per CTO convention. Only measured landmarks get names.
STAGE_LABELS = {0: "QRS", 2: "T Wave", 4: "Diastole", 6: "P Wave"}
N_PHASE_BINS = 8

#: CTO 2026-08-20: no step text in the panel tags (the whole document is step-5,000 only,
#: and the step lives in the caption/README, not the image), and the signal arm is named
#: by the method -- "Winder" -- in these presentation figures.
PANEL_NAMES = {"signal@5000": "Winder", "control@5000": "Control"}


def frame(ax: Axes, kind: str) -> None:
    """`box` = all four spines, no ticks. `arrows` = left+bottom arrowed spines, no ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    if kind == "box":
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        return
    for spine in ax.spines.values():
        spine.set_visible(False)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    akw = dict(
        arrowstyle="-|>,head_width=0.16,head_length=0.38",
        color="0.35",
        lw=0.9,
        shrinkA=0,
        shrinkB=0,
    )
    ax.annotate("", xy=(x1, y0), xytext=(x0, y0), arrowprops=akw, annotation_clip=False)
    ax.annotate("", xy=(x0, y1), xytext=(x0, y0), arrowprops=akw, annotation_clip=False)


def _draw_mixed_size_pieces(ax: Axes, pieces: list[tuple[str, float]], *, y: float) -> None:
    """Draw `pieces` (text, fontsize) left-to-right, baseline-aligned, centred as one
    block at axes-fraction (0.5, y). Widths are measured with the real renderer so
    pieces at different sizes concatenate with no gap and no overlap."""
    fig = ax.figure
    fig.canvas.draw()
    # Agg-specific: guaranteed present at runtime (`matplotlib.use("Agg")`, module top),
    # not on the generic `FigureCanvasBase` mypy sees the attribute through.
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    scratch = [ax.text(0, 0, s, fontsize=fs, alpha=0) for s, fs in pieces]
    fig.canvas.draw()
    widths = [t.get_window_extent(renderer).width for t in scratch]
    for t in scratch:
        t.remove()

    anchor_disp = ax.transAxes.transform((0.5, y))
    x_disp = anchor_disp[0] - sum(widths) / 2
    inv = ax.transData.inverted()
    for (s, fs), w in zip(pieces, widths, strict=True):
        x_data, y_data = inv.transform((x_disp, anchor_disp[1]))
        ax.text(x_data, y_data, s, fontsize=fs, va="baseline", ha="left", clip_on=False)
        x_disp += w


def panel_tag(ax: Axes, text: str, *, fontsize: float = 8.0, y: float = 1.09) -> None:
    """Panel title, placed ABOVE the axes (not overlapping the data) with real headroom.

    Any `text` beginning with "Winder" renders that word as pseudo small-caps --
    matplotlib does not synthesise small-caps glyphs for arbitrary fonts under Agg
    (`FontProperties(variant="small-caps")` is silently ignored for this repo's
    STIXGeneral font -- confirmed empirically), so this hand-rolls the effect: "W" at
    `fontsize`, "INDER" immediately after at 0.72x, both baseline-aligned. Any text
    after "Winder" (e.g. ", seed 0") follows in plain style at the same size and
    baseline. Text not starting with "Winder" (e.g. "Control...") renders as one
    plain piece, so every panel tag in the family shares one baseline position.
    """
    if not text.startswith("Winder"):
        ax.text(
            0.5,
            y,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="baseline",
            ha="center",
            clip_on=False,
        )
        return
    remainder = text[len("Winder") :]
    pieces = [("W", fontsize), ("INDER", fontsize * 0.72)]
    if remainder:
        pieces.append((remainder, fontsize))
    _draw_mixed_size_pieces(ax, pieces, y=y)


def add_phase_colorbar(fig: Figure, axes: list[Axes]) -> None:
    """Vertical phi colourbar with numeric ticks LEFT of the bar and the capitalised measured
    stage names on the free right side -- no collisions. Shared by the projection scripts."""
    sm = ScalarMappable(norm=Normalize(0.0, TWO_PI), cmap=AURORA)
    # pad 0.06, not 0.03: the rotated axis label sits LEFT of the bar, and at 0.03 it
    # collided with the right-hand panel whenever that panel's cloud filled its axes.
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.06)
    cb.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWO_PI])
    cb.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=7)
    cb.ax.yaxis.set_ticks_position("left")
    # CTO 2026-08-20: the axis label sits to the RIGHT of the bar, beyond the stage
    # names (labelpad clears them); numeric ticks stay on the left.
    cb.ax.yaxis.set_label_position("right")
    cb.set_label(r"Cardiac phase $\varphi$ (rad)", fontsize=8, labelpad=42)
    for b, name in STAGE_LABELS.items():
        cb.ax.text(
            1.35,
            TWO_PI * b / N_PHASE_BINS,
            name,
            fontsize=6.5,
            va="center",
            ha="left",
            transform=cb.ax.get_yaxis_transform(),
        )

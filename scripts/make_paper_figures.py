#!/usr/bin/env python3
"""fig01_phase_ring, and the shared `phase_ring_grid_figure` engine `make_umap_figures.py`
(fig14) builds on.

The manuscript (arXiv:2608.21147) shipped with the full Results figure set (figs 2-13, 15-16)
built from this script and its siblings; those builders were retired here once the manuscript
was final -- the deleted code is recoverable from git history if a camera-ready revision ever
needs it. Going forward this repo only actively maintains fig01/fig14 (phase-ring geometry) and
fig17 (own-space UMAP, in `render_latent_projections.py`).

**On the phase-ring figure and why it is not `phase_ring_comparison_figure` verbatim.** The
projection maths is `winder.transport.geometry`'s, unchanged: `phase_resolved_trajectory` then
`harmonic_loop_projection`, the identical composition `winder.plotting.latents._ring_xy` performs.
What this script does not reuse is that module's LAYOUT, and for a specific reason. The existing
rendered example, `artifacts/figures/p9_phase_ring_comparison.pdf`, autoscales every panel
independently while holding `set_aspect("equal")`. The control arm's loop is ~7x smaller in RMS
radius than the signal arm's (measured: 1.02 vs 7.10 at step 5000, harmonic n_j=1, 2000 records),
so autoscaling inflates it to full panel width and it reads as a LARGER, more
dramatic deformation than the signal arm's clean ring -- the exact opposite of the truth. A
correct comparison needs shared limits (so radius is comparable) and, separately, a
scale-normalised row (so shape is comparable at matched size). Neither is expressible through
`phase_ring_comparison_figure`'s signature, which fixes a 1xN row and per-panel autoscaling.

**Style.** `apply_paper_style()` for every figure -- `winder.plotting.style.apply_style()` plus
one shared typography scale (`_TYPOGRAPHY`), so labels, ticks and legends are the same size on
every page; no `ax.set_title`, no
`fig.suptitle`, ever -- `assert_no_baked_in_title_or_caption` is called on every figure inside
`_save`, and `_assert_axis_labels` checks that every axes reaches a non-empty label through its
own shared-axis group. Arm identity in the phase-ring grid is encoded by column position and the
`_ARM_LABEL` legend text, not by colour -- the cyclic phase colormap (`_PHASE_CMAP`) is the only
colour channel in play, and it encodes phase, not arm.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from winder.data.integrity import git_sha
from winder.operators.harmonic import HarmonicTransport
from winder.paths import default_data_root
from winder.plotting.style import apply_style, assert_no_baked_in_title_or_caption
from winder.transport.geometry import harmonic_loop_projection, phase_resolved_trajectory

MILESTONE_ID = "paper-results-figure-set"

#: The four pre-registered arms, in the order every legend uses.
ARMS: tuple[str, ...] = ("signal_seed0", "signal_seed1", "control_seed0", "control_seed1")
_MUTED = "#898781"
_INK = "#0b0b0b"
_ARM_LABEL = {
    "signal_seed0": "signal, seed 0",
    "signal_seed1": "signal, seed 1",
    "control_seed0": "control, seed 0",
    "control_seed1": "control, seed 1",
}

_TYPOGRAPHY: dict[str, Any] = {
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
}
_PHASE_CMAP = "twilight_shifted"


def apply_paper_style() -> None:
    """`winder.plotting.style.apply_style()` plus this set's one shared typography scale.

    Every figure builder calls this instead of `apply_style()` directly, for the same reason
    `winder.plotting.latents` gives for calling `apply_style()` itself: a forgotten call has no
    visible failure mode, it just silently renders one figure at a different size from its
    neighbours. Idempotent, because both halves are.
    """
    apply_style()
    # Same stub-strictness mismatch `winder.plotting.style.apply_style` documents: matplotlib's
    # own stubs type `RcParams.update` against a Literal union of every known rc key.
    plt.rcParams.update(_TYPOGRAPHY)  # type: ignore[arg-type]


def _assert_axis_labels(fig: Figure) -> None:
    """Every axes must reach a non-empty x and y label through its own shared-axis group.

    A real check, not decoration: in a shared-axis grid the inner panels legitimately carry empty
    labels, but every panel must still be readable, i.e. some panel it shares an axis with must
    be labelled. A lone unlabelled axes fails.
    """
    for ax in fig.axes:
        # matplotlib tags colorbar axes with the literal label `<colorbar>`. They carry exactly
        # one meaningful label -- the scale's, set via `Colorbar.set_label` -- and the cross axis
        # is empty by construction, so requiring both would be requiring a nonsense label. The
        # colorbar's own label is checked separately, at the call site that creates it.
        if ax.get_label() == "<colorbar>":
            continue
        for axis_name, getter in (("x", Axes.get_xlabel), ("y", Axes.get_ylabel)):
            group = (
                ax.get_shared_x_axes() if axis_name == "x" else ax.get_shared_y_axes()
            ).get_siblings(ax)
            if not any(getter(sibling) != "" for sibling in group):
                raise AssertionError(
                    f"axes {ax.get_subplotspec()} has no non-empty {axis_name}label anywhere in "
                    "its shared-axis group"
                )


def _save(
    fig: Figure, outdir: str, stem: str, *, dpi: int = 220, pdf_dpi: int | None = None
) -> dict[str, Any]:
    """Enforce the title/caption and axis-label contracts, then write `<stem>.pdf` and `.png`.

    `pdf_dpi` is the resolution of any RASTERIZED artist in the PDF (`Artist.set_rasterized`);
    vector text and axes are unaffected. Every figure here is pure vector and leaves it at
    matplotlib's default; `make_umap_figures.py` rasterizes its 72,000-point scatters, which
    would otherwise be a multi-megabyte vector object per panel, and passes a value.
    """
    assert_no_baked_in_title_or_caption(fig)
    _assert_axis_labels(fig)
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    png_path = os.path.join(outdir, f"{stem}.png")
    if pdf_dpi is None:
        fig.savefig(pdf_path)
    else:
        fig.savefig(pdf_path, dpi=pdf_dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return {
        "stem": stem,
        "pdf": pdf_path,
        "png": png_path,
        "pdf_bytes": os.path.getsize(pdf_path),
        "png_bytes": os.path.getsize(png_path),
    }


def ring_projection(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: Any,
    *,
    harmonic_index: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """`(Re, Im, n_j)` of one checkpoint's phase-binned harmonic loop.

    Exactly `winder.plotting.latents._ring_xy`'s composition -- binned means from
    `phase_resolved_trajectory`, round-tripped through float64, fed to
    `harmonic_loop_projection` -- with no re-derivation of either function's maths.
    """
    trajectory = phase_resolved_trajectory(z, theta, operator, n_bins=n_bins)
    binned = torch.tensor(trajectory["binned_means"], dtype=torch.float64)
    projection = harmonic_loop_projection(binned, operator, harmonic_index=harmonic_index)
    return (
        np.asarray(projection["real"]),
        np.asarray(projection["imag"]),
        int(projection["n_j"]),
    )


def _draw_loop(ax: Axes, re: np.ndarray, im: np.ndarray, *, label: str) -> None:
    """One loop: bins in cyclic order, closed back to bin 0, cyclic colour across bins.

    Markers carry a hairline ink ring. Any cyclic ramp has one phase whose colour approaches the
    surface (see `_PHASE_CMAP`); at 24 bins the ring costs nothing and guarantees that bin is
    still a visible marker rather than a gap in the loop.
    """
    n_bins = len(re)
    order = [*range(n_bins), 0]
    ax.plot(re[order], im[order], "-", color=_MUTED, linewidth=0.8, zorder=1)
    cmap = matplotlib.colormaps[_PHASE_CMAP]
    ax.scatter(
        re,
        im,
        c=[cmap(b / n_bins) for b in range(n_bins)],
        s=14,
        zorder=2,
        edgecolor=_INK,
        linewidth=0.25,
        label=label,
    )
    ax.set_aspect("equal")


def _identity_legend(ax: Axes) -> None:
    """A legend that carries a panel's identity as TEXT only.

    The marker would be the first bin's cyclic-ramp colour, which encodes phase, not arm --
    drawing it beside an arm name invites the reader to think the colour means the arm.
    `markerscale=0` and zero handle length suppress it, leaving the label, which is the whole
    content. (Panel identity lives in a legend rather than a title by the repo's own
    title/caption convention.)
    """
    ax.legend(
        loc="upper center",
        markerscale=0,
        handlelength=0,
        handletextpad=0,
        borderpad=0.15,
    )


def _annotate_stage_labels(
    ax: Axes,
    re: np.ndarray,
    im: np.ndarray,
    stage_labels: dict[int, str],
    *,
    n_stage_bins: int,
    limit: float,
) -> None:
    """Write each MEASURED clinical stage name radially outward from its own bin on the loop.

    `stage_labels` is keyed by index into `n_stage_bins` coarse bins; the loop is drawn at a
    finer `len(re)` bin count, so each coarse bin is mapped to the fine bin at its own CENTRE
    (`round((k + 0.5) * len(re) / n_stage_bins)`, wrapped). Only bins present in `stage_labels`
    are written -- a bin whose landmark was not located on this cohort's own ensemble-averaged
    beat gets no name, never a textbook one.

    "Radially outward" is taken from the loop's own centroid. On a collapsed loop (the control
    arm's) two bins can lie in almost the same direction from that centroid, so a final pass
    pushes any pair of labels that would overlap apart along y; the leader line still ends on the
    label's own bin, so the position being asserted never moves, only the text does.
    """
    n_bins = len(re)
    centre_r, centre_i = float(np.mean(re)), float(np.mean(im))
    anchors: list[tuple[str, tuple[float, float]]] = []
    targets: list[list[float]] = []
    for stage_bin, label in sorted(stage_labels.items()):
        fine = int(round((stage_bin + 0.5) * n_bins / n_stage_bins)) % n_bins
        dx, dy = float(re[fine]) - centre_r, float(im[fine]) - centre_i
        norm = float(np.hypot(dx, dy)) or 1.0
        anchors.append((label, (float(re[fine]), float(im[fine]))))
        targets.append([centre_r + dx / norm * limit * 0.90, centre_i + dy / norm * limit * 0.90])

    # Thresholds in DATA units, derived from the panel: a 5.5 pt label on a panel spanning
    # 2 * `limit` is roughly 0.14 * limit tall and 0.65 * limit wide at these names' lengths.
    min_dy, near_x = 0.16 * limit, 0.60 * limit
    order = sorted(range(len(targets)), key=lambda i: targets[i][1])
    for previous, current in zip(order, order[1:], strict=False):
        if (
            abs(targets[current][0] - targets[previous][0]) < near_x
            and targets[current][1] - targets[previous][1] < min_dy
        ):
            targets[current][1] = targets[previous][1] + min_dy

    for (label, anchor), target in zip(anchors, targets, strict=True):
        text = ax.annotate(
            label,
            xy=anchor,
            xytext=(target[0], target[1]),
            fontsize=5.5,
            color=_INK,
            ha="center",
            va="center",
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "color": _MUTED,
                "linewidth": 0.4,
                "shrinkA": 1.0,
                "shrinkB": 2.0,
            },
        )
        # The control arm's collapsed loop reaches almost to the panel edge along one direction,
        # so its outermost label lands on top of its own markers. A surface-coloured stroke round
        # the glyphs keeps the name readable without moving it off its own bin's radius.
        text.set_path_effects([path_effects.withStroke(linewidth=1.8, foreground="white")])


def phase_ring_grid_figure(
    loops: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    harmonic: int,
    stage_labels: dict[int, str] | None = None,
    n_stage_bins: int = 8,
    rows: Literal["both", "top", "bottom"] = "both",
) -> Figure:
    """Two rows of phase-ring loops: raw (shared limits) above, RMS-normalised below.

    Row 1 answers "how big is the loop", on one shared scale for all arms -- the comparison the
    existing `p9_phase_ring_comparison.pdf` destroys by autoscaling each panel. Row 2 answers
    "what shape is the loop", by dividing each loop by its own RMS radius so shape is legible at
    matched size; the measured RMS radius goes in that panel's legend entry, never a title.

    `stage_labels` (default: none, which is Figure 1) adds the MEASURED clinical stage names to
    the normalised row -- Figure 14. It is `{coarse_bin: name}` over `n_stage_bins` bins; see
    `_annotate_stage_labels`.

    `rows` (default `"both"`, every existing caller's behaviour, unchanged) restricts the figure
    to just the raw row (`"top"`) or just the RMS-normalised row (`"bottom"`) when a single row
    is the deliverable -- e.g. a presentation figure that only wants the shape comparison, not
    the size one. Both limits are still computed from every loop regardless of `rows`, so a
    single-row figure's axis scale is identical to what that same row would show inside the full
    two-row grid -- never re-autoscaled to the shown subset alone.
    """
    apply_paper_style()
    names = list(loops)
    n_rows = 1 if rows != "both" else 2
    # `sharey="row"` (not `"all"`) because the two rows are on deliberately different scales: the
    # top row is in the projection's own units, the bottom row in units of each loop's own RMS
    # radius. Within a row the scale IS shared -- that is the entire point of the top row.
    fig, axes_arr = plt.subplots(
        n_rows,
        len(names),
        figsize=(1.85 * len(names), 4.3 if n_rows == 2 else 2.5),
        squeeze=False,
        sharex="row",
        sharey="row",
        layout="constrained",
    )
    limit = 1.08 * max(float(np.max(np.abs(np.concatenate([re, im])))) for re, im in loops.values())
    # The normalised row's own limit, measured rather than hardcoded. The previous fixed +/-2.0
    # was smaller than the control arm's normalised extent, so its outermost bins were drawn
    # sitting on the right spine -- a clipped marker reads as a data point at the axis maximum,
    # which is exactly the wrong impression on the panel whose whole message is "this loop is
    # crumpled". Same 1.08 / 1.32 headroom convention as the raw row above.
    norm_limit = 1.08 * max(
        float(np.max(np.abs(np.concatenate([re, im])))) / float(np.sqrt(np.mean(re**2 + im**2)))
        for re, im in loops.values()
    )
    for col, name in enumerate(names):
        re, im = loops[name]
        radius = float(np.sqrt(np.mean(re**2 + im**2)))
        row_cursor = 0

        if rows in ("both", "top"):
            top = axes_arr[row_cursor][col]
            _draw_loop(top, re, im, label=_ARM_LABEL[name])
            top.set_xlim(-limit, limit)
            top.set_ylim(-limit, 1.32 * limit)
            _identity_legend(top)
            top.set_xlabel(r"$C_b$")
            if col > 0:
                top.tick_params(labelleft=False)
            row_cursor += 1

        if rows in ("both", "bottom"):
            bottom = axes_arr[row_cursor][col]
            _draw_loop(bottom, re / radius, im / radius, label=f"RMS radius {radius:.2f}")
            bottom.set_xlim(-norm_limit, norm_limit)
            bottom.set_ylim(-norm_limit, 1.32 * norm_limit)
            if stage_labels:
                _annotate_stage_labels(
                    bottom,
                    re / radius,
                    im / radius,
                    stage_labels,
                    n_stage_bins=n_stage_bins,
                    limit=norm_limit,
                )
            _identity_legend(bottom)
            bottom.set_xlabel(r"$C_b$ / RMS radius")
            if col > 0:
                bottom.tick_params(labelleft=False)

    if rows in ("both", "top"):
        axes_arr[0][0].set_ylabel(r"$S_b$" + f"  (harmonic $n_j$={harmonic})")
    if rows == "bottom":
        axes_arr[0][0].set_ylabel(r"$S_b$" + " / RMS radius")
    elif rows == "both":
        axes_arr[1][0].set_ylabel(r"$S_b$" + " / RMS radius")

    # The bin colour is carrying real information -- whether the loop is traversed in phase
    # order -- so it needs a scale, not just a nice look.
    n_bins = len(next(iter(loops.values()))[0])
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0.0, 2 * np.pi), cmap=matplotlib.colormaps[_PHASE_CMAP]
    )
    colorbar = fig.colorbar(
        mappable,
        ax=axes_arr.ravel().tolist(),
        orientation="horizontal",
        fraction=0.045,
        pad=0.02,
        aspect=60,
    )
    colorbar.set_label(rf"Cardiac phase $\varphi$ at bin centre (rad), {n_bins} bins")
    colorbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    colorbar.ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    if colorbar.ax.get_xlabel() == "":
        raise AssertionError("phase-ring colorbar lost its scale label")
    return fig


# --------------------------------------------------------------------------------------
# The step-5,000 cut (CTO policy, 2026-08-20): a second, single-step render of four of the
# step-spanning figures, for the bottom-line document.
# --------------------------------------------------------------------------------------

_PHASE_RING_STEM = "fig01_phase_ring"


def assert_operators_share_state(
    operators: dict[str, HarmonicTransport],
) -> HarmonicTransport:
    """Assert every arm's own operator carries an IDENTICAL state_dict, then return one of them
    (arbitrarily the first, by insertion order) for use across every panel.

    Relocated from the retired `make_figures.py` -- `encode_phase_ring_loops` below is its only
    remaining consumer. Raises `ValueError` naming the two arms and the exact tensor key that
    differs, rather than silently picking one arm's operator and using it for all.
    """
    if not operators:
        raise ValueError("operators must be non-empty")
    names = list(operators)
    reference_name = names[0]
    reference = operators[reference_name]
    ref_state = reference.state_dict()
    for other_name in names[1:]:
        other_state = operators[other_name].state_dict()
        if set(ref_state) != set(other_state):
            raise ValueError(
                f"operator state_dict keys differ between {reference_name!r} and "
                f"{other_name!r}: {sorted(ref_state)} vs {sorted(other_state)}"
            )
        for key, ref_val in ref_state.items():
            if not torch.equal(ref_val, other_state[key]):
                raise ValueError(
                    f"operator state {key!r} differs between {reference_name!r} and "
                    f"{other_name!r} -- cannot share one operator across the comparison "
                    "figure's panels"
                )
    return reference


def encode_phase_ring_loops(
    args: argparse.Namespace,
    *,
    cohort: Any | None = None,
    arms: Sequence[str] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """`({arm: (Re, Im)}, n_j)` from the folds-1--9 eval split under each arm's own checkpoint.

    Split out from `_build_phase_ring` because this half is the expensive, non-deterministic-cost
    half (a full PTB-XL decode plus four encoder passes, minutes) while the layout half is
    instant -- keeping them separate means the projection can be computed once and re-plotted.

    `cohort` (an already-built `winder.eval.comparison.EvalCohort`) skips the PTB-XL decode.
    `make_umap_figures.py` builds the cohort once for its own six-cell encoding and passes it in
    here for Figure 14; without the hook that script would decode the identical 2,146-record eval
    split a second time in the same process.

    `arms` selects which roster arms to encode -- default `ARMS` (all four nominal signal/control
    seeds, today's shipped fig01/fig14 layout). Passing a subset (one arm for a single-panel
    loop, two for a joint pair) is what lets `phase_ring_grid_figure`'s existing N-column layout
    render single- or two-panel variants with no change to its own code.

    Imports the roster/cohort plumbing lazily so the nine JSON-only figures never pay for a torch
    checkpoint load or a PTB-XL decode.
    """
    import eval_suite

    from winder.eval.readout import (
        discover_seed_checkpoints,
        encode_z,
        load_model_and_operator,
        operator_from_checkpoint,
    )

    device = torch.device(args.device)
    roster_dir = args.roster_dir or os.path.join(args.artifacts_dir, "roster")
    lead_stats_path = args.lead_stats_path or os.path.join(
        args.artifacts_dir, "lead_stats_f1to9.json"
    )
    checkpoint_dirs: dict[str, str] = {}
    for arm in arms if arms is not None else ARMS:
        steps = discover_seed_checkpoints(os.path.join(roster_dir, arm))
        if args.phase_ring_step not in steps:
            raise SystemExit(
                f"[figures] {arm}: step {args.phase_ring_step} not found (have {sorted(steps)})"
            )
        checkpoint_dirs[arm] = steps[args.phase_ring_step]

    operators = {arm: operator_from_checkpoint(d) for arm, d in checkpoint_dirs.items()}
    missing = [arm for arm, op in operators.items() if op is None]
    if missing:
        raise SystemExit(f"[figures] arm(s) with no declared transport operator: {missing}")
    shared_operator = assert_operators_share_state(
        {arm: op for arm, op in operators.items() if op is not None}
    )
    harmonic_index = args.harmonic - 1

    if cohort is None:
        cohort, _bookkeeping = eval_suite.build_p9_cohort(
            args.data_root, args.artifacts_dir, lead_stats_path, train_limit=1
        )
    n = min(args.n_records, cohort.waveforms["eval"].shape[0])
    waveforms = cohort.waveforms["eval"][:n]
    theta = cohort.thetas["eval"][:n]

    loops: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_j = args.harmonic
    for arm, ckpt_dir in checkpoint_dirs.items():
        model, _operator = load_model_and_operator(ckpt_dir, seed=args.seed, device=device)
        z = encode_z(model, waveforms, device)
        re, im, n_j = ring_projection(
            z, theta, shared_operator, harmonic_index=harmonic_index, n_bins=args.phase_ring_bins
        )
        loops[arm] = (re, im)
        del model, z
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[figures] phase ring: {arm} encoded {n} records", flush=True)
    return loops, n_j


def _build_phase_ring(args: argparse.Namespace) -> Figure:
    """Encode, project, and lay out the folds-1--9 phase-ring grid (train_contaminated).

    `args.phase_ring_arms` (default: unset -> all four nominal arms, today's shipped
    fig01 layout) lets the CLI ask for a single-panel or two-panel subset instead.
    """
    arms = args.phase_ring_arms.split(",") if args.phase_ring_arms else None
    loops, n_j = encode_phase_ring_loops(args, arms=arms)
    return phase_ring_grid_figure(loops, harmonic=n_j)


def _phase_ring_stem(args: argparse.Namespace) -> str:
    """`fig01_phase_ring`, or `fig01_phase_ring_<arm1>_<arm2>...` for an explicit subset --
    so a single-/two-panel render never clobbers the shipped four-panel fig01."""
    if not args.phase_ring_arms:
        return _PHASE_RING_STEM
    return "_".join([_PHASE_RING_STEM, *args.phase_ring_arms.split(",")])


def main(argv: list[str] | None = None) -> int:
    """Render fig01 (the phase-ring grid) -- the only figure this script still builds; every
    sealed-fold-report-driven figure (2-13, 15-16) and the legacy `make_figures.py` comparison
    were retired once the manuscript shipped, per the ablation-porting cleanup. Returns 0 iff
    the figure rendered and passed the no-title/no-caption and axis-label contracts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.expanduser("~/winder-paper/figures"))
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated figure stems to render (default: fig01, unless --skip-phase-ring)",
    )
    ap.add_argument(
        "--skip-phase-ring",
        action="store_true",
        help="skip fig01, the only figure that loads checkpoints and PTB-XL waveforms",
    )
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument(
        "--lead-stats-path", default=None, help="default <artifacts-dir>/lead_stats_f1to9.json"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-records", type=int, default=2000, help="fig01 eval records per arm")
    ap.add_argument("--phase-ring-step", type=int, default=5000)
    ap.add_argument("--phase-ring-bins", type=int, default=24)
    ap.add_argument(
        "--phase-ring-arms",
        default=None,
        help="comma-separated roster arms for fig01 (default: all four nominal arms). "
        "One arm renders a single-panel loop, two render a joint pair -- output stem gains "
        "the arm names as a suffix so it never overwrites the default four-panel fig01.",
    )
    ap.add_argument("--harmonic", type=int, default=1, help="1-indexed; 1 = the fundamental")
    ap.add_argument("--report-out", default="artifacts/reports/paper_figures.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    requested = set(args.only.split(",")) if args.only else None
    rendered: list[dict[str, Any]] = []

    ring_stem = _phase_ring_stem(args)
    want_ring = (requested is None and not args.skip_phase_ring) or (
        requested is not None and ring_stem in requested
    )
    if want_ring:
        rendered.append(_save(_build_phase_ring(args), args.outdir, ring_stem))
        print(f"[figures] {ring_stem} -> {rendered[-1]['pdf_bytes'] / 1024:.0f} kB pdf")

    payload = {
        "status": "PASS",
        "milestone_id": MILESTONE_ID,
        "split_status": "train_contaminated",
        "headline": False,
        "metrics": {
            "figures": rendered,
            "n_figures": len(rendered),
            "elapsed_sec": time.time() - t0,
        },
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": vars(args),
            "seed": args.seed,
        },
        "decisions": [
            "Figure 1 encodes folds-1--9 checkpoints on the fold-9 eval split, which is TRAINING "
            "data for those checkpoints -- split_status train_contaminated, recorded per-figure "
            "in figures/README.md. Figures 2-13/15-16 (sealed-fold-report-driven and UMAP-PCA) "
            "and the legacy make_figures.py comparison were retired from this script once the "
            "manuscript shipped; their static output remains on disk in winder-paper/figures/.",
        ],
        "questions": [],
    }
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    tmp = args.report_out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, args.report_out)
    print(
        f"[figures] status=PASS {len(rendered)} figures -> {args.outdir} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

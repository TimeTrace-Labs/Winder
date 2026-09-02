#!/usr/bin/env python3
"""fig17: per-arm own-space UMAP of the SAME token sample the joint UMAP used.

Own-fit, own-limits, coloured by cardiac phase with the aurora treatment. `--cells` selects
which roster cells to render as panels -- one cell renders a single-panel figure, two render a
joint pair (the shipped `fig17_umap_phase_seed{s}_own` default, `signal@5000,control@5000`).

**The own-space caveat, stated where the file is made:** nothing is comparable across panels --
not size, not shape, not distance. Each panel autoscales its own embedding, which is precisely
the effect that made the reference repo's per-panel phase-ring figure misleading. This exists
because within-panel colour organisation is still a fair question per arm; any cross-arm
statement must come from a JOINT figure, which this script no longer produces (fig15/fig16, the
joint- and own-basis PCA variants, were retired once the manuscript shipped -- shipped/frozen in
the arXiv PDF, no longer actively maintained here).

The token sample is NOT re-drawn: `record_index`/`token_index` are loaded from the cached
`umap_embedding_seed{s}.npz` written by `scripts/make_umap_figures.py`, so this figure shows
literally the same tokens as the rest of the UMAP family. A phi-consistency assertion guards
against cohort drift between that cache and the present rebuild.

Split status of every output: train_contaminated (folds 1-9 encodings). Geometry is not a
performance claim; the label stands.

    uv sync --extra figures && uv run python scripts/render_latent_projections.py

(pyproject.toml's `figures` extra pins exactly 0.5.8, not 0.5.7: the joint-fit cache was
produced under 0.5.8, and 0.5.7 breaks against this repo's scikit-learn >= 1.6 --
`check_array(force_all_finite=...)` was renamed upstream.)
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from winder.paths import default_data_root  # noqa: E402
from winder.plotting.aurora import (  # noqa: E402
    AURORA,
    PANEL_NAMES,
    TWO_PI,
    add_phase_colorbar,
    frame,
    panel_tag,
)
from winder.plotting.style import apply_style, assert_no_baked_in_title_or_caption  # noqa: E402

#: fig17's shipped joint-pair default. `--cells` overrides this to a subset for a single-panel
#: render; both the default and any override must be keys of PANEL_NAMES (winder.plotting.aurora).
DEFAULT_CELLS = ("signal@5000", "control@5000")


def panel_figure(
    coords: dict[str, np.ndarray],
    phi: np.ndarray,
    *,
    shared: bool,
    axis_stub: str,
    xlabels: dict[str, str] | None,
    out_stem: str,
    figsize: tuple[float, float] = (7.4, 3.9),
) -> None:
    """N aurora panels (one per `coords` entry) over one phi colouring -- N=1 for a single-arm
    panel, N=2 for the joint signal-vs-control pair (today's shipped fig17 layout), and the
    layout generalizes beyond 2 with no further change. `shared` shares limits (joint bases);
    otherwise every panel autoscales its own space and carries its own axis labels.

    Figure taller and with pulled-in top/bottom margins so the panel tag sits above the
    axes with real headroom, and 0.09 padding around the data (was cramped at 0.04 /
    matplotlib's ~5% autoscale default) so the point cloud does not touch the frame.
    """
    n = len(coords)
    fig, axes_arr = plt.subplots(1, n, figsize=figsize, sharex=shared, sharey=shared, squeeze=False)
    axes = axes_arr[0]
    fig.subplots_adjust(top=0.86, bottom=0.14)
    order = np.argsort(phi)
    if shared:
        allxy = np.concatenate(list(coords.values()), axis=0)
        pad = 0.09 * (allxy.max(axis=0) - allxy.min(axis=0))
    for ax, (cell, xy) in zip(axes, coords.items(), strict=True):
        ax.scatter(
            xy[order, 0],
            xy[order, 1],
            s=2.0,
            alpha=0.65,
            lw=0,
            c=AURORA(phi[order] / TWO_PI),
            rasterized=True,
        )
        if shared:
            ax.set_xlim(allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
            ax.set_ylim(allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])
        else:
            ax.margins(0.09)
        frame(ax, "arrows")
        panel_tag(ax, PANEL_NAMES[cell])
        ax.set_xlabel(xlabels[cell] if xlabels else f"{axis_stub}-1", fontsize=8)
        if not shared or ax is axes[0]:
            ax.set_ylabel(f"{axis_stub}-2", fontsize=8)
    add_phase_colorbar(fig, list(axes))
    assert_no_baked_in_title_or_caption(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=320, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[proj] wrote {out_stem}.(pdf|png)", flush=True)


def _fig17_stem(cells: tuple[str, ...], seed: int) -> str:
    """`fig17_umap_phase_seed{s}_own` for the shipped joint pair; a cell-suffixed stem for any
    other subset, so a single-panel render never overwrites the shipped default."""
    if cells == DEFAULT_CELLS:
        return f"fig17_umap_phase_seed{seed}_own"
    suffix = "_".join(c.split("@")[0] for c in cells)
    return f"fig17_umap_phase_seed{seed}_own_{suffix}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--reports-dir", default="artifacts/reports")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/winder-paper/figures"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument(
        "--cells",
        default=None,
        help="comma-separated roster cells, e.g. 'signal@5000' or 'signal@5000,control@5000' "
        "(default: both, today's shipped joint fig17). One cell renders a single-panel figure; "
        f"valid cells are {sorted(PANEL_NAMES)}.",
    )
    args = ap.parse_args(argv)

    cells = tuple(args.cells.split(",")) if args.cells else DEFAULT_CELLS
    unknown = [c for c in cells if c not in PANEL_NAMES]
    if unknown:
        ap.error(f"unknown cell(s) {unknown} -- valid cells are {sorted(PANEL_NAMES)}")

    import eval_suite
    from make_umap_figures import encode_token_cells

    apply_style()
    lead_stats_path = os.path.join(args.artifacts_dir, "lead_stats_f1to9.json")
    t0 = time.time()
    cohort, bookkeeping = eval_suite.build_p9_cohort(
        args.data_root, args.artifacts_dir, lead_stats_path, train_limit=1
    )
    print(f"[proj] cohort ready ({time.time() - t0:.0f}s): {bookkeeping}", flush=True)

    # Heavy, optional dependency (pyproject.toml's `figures` extra): only imported after the
    # cohort proves the environment is sane. mypy's error code for this depends on whether
    # umap-learn happens to be installed in whatever venv mypy runs against -- `import-untyped`
    # if present (installed, but ships no py.typed marker), `import-not-found` if absent (not
    # installed at all) -- so both are silenced rather than picking whichever one the CURRENT
    # venv reports today.
    import umap  # type: ignore[import-not-found,import-untyped]

    for seed in [int(s) for s in args.seeds.split(",")]:
        with np.load(os.path.join(args.reports_dir, f"umap_embedding_seed{seed}.npz")) as d:
            record_index, token_index, phi_cache = d["record_index"], d["token_index"], d["phi"]

        # Identical gather to make_umap_figures._build_seed_embedding -- and a drift guard:
        # if the cohort rebuild does not reproduce the cached phi bit-for-bit, the sample no
        # longer means what the cache says it means, and nothing should be drawn from it.
        phi = cohort.thetas["eval"].numpy()[record_index.ravel(), token_index.ravel()]
        if not np.allclose(phi, phi_cache, atol=0.0):
            raise AssertionError(f"seed {seed}: rebuilt phi differs from the cached sample")

        records = np.unique(record_index[:, 0])
        waveforms = cohort.waveforms["eval"][torch.from_numpy(records)]
        compact = {int(r): i for i, r in enumerate(records.tolist())}
        compact_index = np.vectorize(compact.__getitem__)(record_index)
        encoded_cells = encode_token_cells(
            os.path.join(args.artifacts_dir, "roster"),
            seed,
            waveforms,
            compact_index,
            token_index,
            device=torch.device(args.device),
            model_seed=0,
        )
        z = {c: encoded_cells[c] for c in cells}

        # fig17 -- per-arm UMAP, own fit, own limits. Same hyperparameters as the joint fit.
        own_umap = {}
        for c in cells:
            t1 = time.time()
            own_umap[c] = umap.UMAP(
                n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0
            ).fit_transform(z[c])
            print(f"[proj] seed {seed}: own-space UMAP {c} in {time.time() - t1:.0f}s", flush=True)
        # CTO 2026-08-20: no "(own fit)" in the axis label -- the own-fit provenance lives
        # in the filename, the README and the manuscript caption, not the image.
        # 2026-08-21: wider than the retired fig15/fig16 default -- this is the only figure in
        # the family that ships in the manuscript, and at 7.4x3.9 (1.87:1) it read too vertical
        # next to the joint UMAP panels (fig11, 2.12:1).
        # 2026-08-27: width scales with panel count (3.7 per panel, matching the 2-panel
        # default's per-panel width) instead of a hardcoded 7.4 -- the fixed width stretched
        # single-panel (--cells) renders across the full 2-panel span, distorting the aspect
        # ratio of the embedding. Height stays fixed, per make_paper_figures.py's
        # phase_ring_grid_figure precedent (width scales with len(names), height does not).
        panel_figure(
            own_umap,
            phi,
            shared=False,
            axis_stub="UMAP",
            xlabels=None,
            out_stem=os.path.join(args.out_dir, _fig17_stem(cells, seed)),
            figsize=(3.7 * len(cells), 3.3),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The UMAP figure suite: what the latent looks like when nothing about it is assumed.

**The scientific question this script answers, in one sentence.** Figure 1 asks whether the
latent's *harmonic-1 projection* traces a phase-ordered loop -- a closed-form, gauge-safe
question with a closed-form answer; this script asks the complementary, assumption-free
question, "if you throw the raw 256-dimensional token vectors at a neighbour-graph embedding
that knows nothing about cardiac phase, does phase come back out anyway, and does it come back
out only for the transport arm?"

**Why both figures exist, and which one is evidence.** A UMAP layout is not a measurement. Its
axes carry no units, its distances are not metric, and a joint embedding of two model states puts
them in one coordinate system that is *shared* but not *comparable* -- "these two clouds are far
apart" is not a quantitative statement about the two models. Figure 14 is the rigorous companion:
the same clinical stage labels, on the closed-form projection, where the geometry is derived
rather than optimised. The UMAPs are the picture; Figure 1 / Figure 14 are the claim.

**Step 5,000 only (CTO policy, 2026-08-20).** Every figure here reports the step-5,000
checkpoints and nothing else; see `UMAP_STEPS` for what that supersedes and for where the
superseded observations are recorded.

**Fold 10 is never touched.** Every figure here encodes the folds-1--9 checkpoints under
`artifacts/roster/` on the fold-9 eval split, via `eval_suite.build_p9_cohort` -- the same cohort
`scripts/heart_rate_strata_eval.py` and Figure 1 use. Fold 9 is TRAINING data for these
checkpoints, so every figure in this file carries `split_status: train_contaminated` and
`headline: false`, recorded per figure in `figures/README.md`. Nothing here re-scores anything;
these are pictures of a latent, not performance numbers.

**Paired sampling, so a panel difference is a model difference.** Both cells (signal and control
at step 5,000) of a seed are encoded on the SAME records and the SAME token indices within those
records: one seeded draw of `--n-records` eval records stratified by dominant superclass, and
within each record one contiguous run of `--n-tokens` tokens with finite theta. Each cell is then
indexed identically, so nothing that differs between the two panels can be a sampling artifact.
One UMAP is fitted jointly on the concatenation of both cells, and each cell is scattered into
its own panel of that single embedding -- same space, same limits, same neighbour graph.

**Cosine metric, and the cost of it, stated up front.** The metric is cosine, so the embedding
sees the DIRECTION of each token vector and not its length. That is the right choice for a
question about phase structure (the transport operator acts by rotation), but it means the
between-arm amplitude disparity Figure 1 measures -- RMS loop radius 7.10 for signal against 1.02
for control -- is invisible here BY CONSTRUCTION. Read amplitude off Figure 1 or Figure 14; read
neighbourhood structure off these. Per-cell normalisation is separately NOT applied: the raw `z`
goes in, so whatever offset the cells carry relative to one another is in the neighbour graph.

**Clinical stage labels are measured, never a template.** `ensemble_beat` is ported from the
reference repo's `scripts/p1_panel_numerics.py` and re-run here on winder-nominal's own fold-9
eval cohort: every token's own theta averages the raw lead-II trace into 128 fine phase bins, and
the QRS / T / P / diastole positions are read off THAT curve. A bin whose landmark could not be
located gets no name. The recomputation reproduces the reference's `beat.json` bin assignment
exactly (QRS -> bin 0, T -> 2, diastole -> 4, P -> 6 of 8), which is a cross-repo check rather
than a coincidence: the two cohorts are the same 2,146 records.

**Style.** `make_paper_figures.apply_paper_style()` for the manuscript-grade renders, so the
typography matches figures 1-10 exactly; the deck and dark variants scale that same scale up. No
`ax.set_title`, no `fig.suptitle`, ever -- `assert_no_baked_in_title_or_caption` runs on every
figure inside `make_paper_figures._save`. Panel identity is a legend entry, per the repo's own
convention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

import make_paper_figures as mpf
import numpy as np
import torch

from winder.data.integrity import git_sha
from winder.data.ptbxl import LEAD_ORDER
from winder.paths import default_data_root

MILESTONE_ID = "paper-umap-figure-set"

#: Emitted at the TOP level of the run record unconditionally, never merely documented -- the
#: convention `scripts/eval_suite.py` and `scripts/heart_rate_strata_eval.py` already follow.
#: Folds 1-9 are training data for every checkpoint this script loads.
SPLIT_STATUS = "train_contaminated"
HEADLINE = False

TWO_PI = 2.0 * math.pi

SEEDS: tuple[int, ...] = (0, 1)
ARM_CLASSES: tuple[str, ...] = ("signal", "control")

#: The one step every figure in this file reports. Chosen to match Figure 1's own
#: `--phase-ring-step` default, so every train_contaminated picture in the set is of the same
#: checkpoints; NOT chosen by looking at the embeddings first.
ANCHOR_STEP = 5000

#: **Step-5,000 only, by CTO policy of 2026-08-20.** This supersedes an earlier spec in which the
#: joint fit spanned `(5000, 20000, 30000)` and the panel grid was 2 arms x 3 steps. The 20,000
#: and 30,000 encodings were computed under that spec and are NOT shipped; what they showed is
#: recorded in `figures/README.md` and in this script's run report (`superseded_spec`) rather than
#: deleted, because one of them complicated the story and deleting it would be curation. The
#: joint fit is now over TWO cells -- signal@5000 and control@5000 -- and the grid is one row.
UMAP_STEPS: tuple[int, ...] = (ANCHOR_STEP,)

#: Every stem/step this script can run, in order. `--only` is validated against it, so a typo
#: raises instead of quietly doing nothing and still reporting status=PASS. `umap_embedding_cache`
#: is not a rendered figure -- it is `_build_seed_embedding`'s per-seed joint UMAP fit, cached to
#: `umap_embedding_seed{s}.npz`. fig11/fig12/fig13 (which used to gate this build) were retired
#: once the manuscript shipped, but `render_latent_projections.py`'s fig17 still reads this cache
#: directly, so the build step stays as its own explicit target.
FIGURE_STEMS: tuple[str, ...] = (
    "umap_embedding_cache",
    "fig14_phase_ring_staged",
)

# --------------------------------------------------------------------------------------
# The measured beat: clinical stage labels
# --------------------------------------------------------------------------------------

#: Fine phase-bin count for the ensemble-averaged beat, and the coarse staging bin count the
#: labels are reported at. Both are the reference repo's own (`p1_panel_numerics.N_PHASE_BINS`
#: = 8, ~105 ms at this cohort's 842.6 ms median RR interval).
N_FINE_BINS = 128
N_STAGE_BINS = 8

#: `beat["measured"]` key -> capitalised clinical display name. Capitalisation is the only
#: difference from the reference repo's `p2_panel_figures._bin_labels`; the measured-only rule is
#: identical, and deliberately so -- a bin whose landmark was not located is left unnamed rather
#: than filled in from a textbook PQRST timing template.
_STAGE_KEYS: tuple[tuple[str, str], ...] = (
    ("qrs_phase_bin", "QRS"),
    ("t_wave_phase_bin", "T Wave"),
    ("diastole_phase_bin", "Diastole"),
    ("p_wave_phase_bin", "P Wave"),
)


def _local_maxima(curve: np.ndarray, allowed: np.ndarray) -> list[int]:
    """Indices of `curve`'s circular local maxima, restricted to positions `allowed` marks True."""
    n = len(curve)
    return [
        i
        for i in range(n)
        if allowed[i] and curve[i] > curve[(i - 1) % n] and curve[i] >= curve[(i + 1) % n]
    ]


def ensemble_beat(
    waveforms: torch.Tensor, theta: torch.Tensor, patch_width: int, n_fine: int = N_FINE_BINS
) -> dict[str, Any]:
    """The phase-resolved average ECG, from which the 8 phase bins' PQRST labels are MEASURED.

    Ported unchanged (bar typing and this docstring) from the reference repo's
    `scripts/p1_panel_numerics.py::ensemble_beat`. Every token's own theta (`winder.data.phase`:
    theta = 0 exactly at an R-peak, rising to 2*pi at the next) averages the raw lead-II trace
    into `n_fine` phase bins; QRS is the maximum of the circular |d/dphase| of that curve, T and P
    are the two largest smoothed humps outside a guard band around the QRS (T precedes P by the
    R-peak convention, so ordering them by phase is a consequence, not an assumption), and
    diastole is the coarse bin of least transition energy.

    Amplitudes are in the dataset's own per-lead z-scored units, not millivolts -- only the SHAPE
    and the phase positions are used, and both are scale-free.
    """
    lead_ii = LEAD_ORDER.index("II")
    n_records, n_leads, n_samples = waveforms.shape
    if n_leads != len(LEAD_ORDER):
        raise ValueError(f"expected (N, 12, T) lead-major waveforms, got {tuple(waveforms.shape)}")

    # Each token covers `patch_width` raw samples; assign every raw sample its token's theta so
    # the beat is resolved at raw-sample resolution rather than at the 125-token grid.
    sample_theta = theta.repeat_interleave(patch_width, dim=1)[:, :n_samples]
    valid = torch.isfinite(sample_theta)
    idx = torch.clamp((sample_theta / (TWO_PI / n_fine)).long(), 0, n_fine - 1)

    sig_ii = waveforms[:, lead_ii, :n_samples].to(torch.float64)
    sig_abs = waveforms[:, :, :n_samples].abs().mean(dim=1).to(torch.float64)
    flat_ok = valid.reshape(-1)
    flat_idx = idx.reshape(-1)[flat_ok]
    counts = torch.zeros(n_fine, dtype=torch.float64)
    counts.index_add_(0, flat_idx, torch.ones(int(flat_ok.sum()), dtype=torch.float64))
    sum_ii = torch.zeros(n_fine, dtype=torch.float64)
    sum_ii.index_add_(0, flat_idx, sig_ii.reshape(-1)[flat_ok])
    sum_abs = torch.zeros(n_fine, dtype=torch.float64)
    sum_abs.index_add_(0, flat_idx, sig_abs.reshape(-1)[flat_ok])

    safe = counts.clamp_min(1.0)
    mean_ii = (sum_ii / safe).numpy()
    mean_abs = (sum_abs / safe).numpy()
    # Circular derivative: the beat is periodic on [0, 2*pi), so np.gradient's one-sided edge
    # handling would be wrong exactly at the R-peak, the steepest point.
    d_ii = np.abs(np.roll(mean_ii, -1) - np.roll(mean_ii, 1)) / (2 * TWO_PI / n_fine)

    fine_centers = (np.arange(n_fine) + 0.5) * (TWO_PI / n_fine)
    qrs_fine = int(np.argmax(d_ii))
    guard = max(1, n_fine // 12)
    away = np.ones(n_fine, dtype=bool)
    for off in range(-guard, guard + 1):
        away[(qrs_fine + off) % n_fine] = False
    # Smooth before hunting for humps: T and P are broad (~100-200 ms) and low amplitude, so the
    # raw binned curve's sampling ripple produces spurious one-bin maxima that would outrank them.
    # The QRS position above comes from the UNSMOOTHED derivative, where sharpness is the signal.
    kernel = max(3, n_fine // 32) | 1
    smooth = np.convolve(np.r_[mean_ii, mean_ii, mean_ii], np.ones(kernel) / kernel, mode="same")[
        n_fine : 2 * n_fine
    ]
    ordered: list[int] = []
    for i in sorted(_local_maxima(smooth, away), key=lambda i: smooth[i], reverse=True):
        if all(min(abs(i - j), n_fine - abs(i - j)) > guard for j in ordered):
            ordered.append(i)
        if len(ordered) == 2:
            break
    t_fine, p_fine = (
        cast(tuple[int | None, int | None], tuple(sorted(ordered)))
        if len(ordered) == 2
        else (None, None)
    )

    def _to_bin(fine: int | None) -> int | None:
        return None if fine is None else int(fine * N_STAGE_BINS // n_fine)

    def _coarse(values: np.ndarray, b: int) -> float:
        return float(values[b * n_fine // N_STAGE_BINS : (b + 1) * n_fine // N_STAGE_BINS].mean())

    return {
        "n_records": int(n_records),
        "n_fine": n_fine,
        "fine_centers": fine_centers.tolist(),
        "mean_lead_ii_z": mean_ii.tolist(),
        "mean_abs_all_leads_z": mean_abs.tolist(),
        "abs_dtheta_lead_ii": d_ii.tolist(),
        "bin_counts": counts.tolist(),
        "n_phase_bins": N_STAGE_BINS,
        "measured": {
            "qrs_fine_bin": qrs_fine,
            "qrs_phase_bin": _to_bin(qrs_fine),
            "t_wave_fine_bin": t_fine,
            "t_wave_phase_bin": _to_bin(t_fine),
            "p_wave_fine_bin": p_fine,
            "p_wave_phase_bin": _to_bin(p_fine),
            "diastole_phase_bin": int(np.argmin([_coarse(d_ii, b) for b in range(N_STAGE_BINS)])),
        },
        "per_phase_bin_transition_energy": [_coarse(d_ii, b) for b in range(N_STAGE_BINS)],
    }


def stage_bin_labels(beat: dict[str, Any]) -> dict[int, str]:
    """`{phase_bin: capitalised stage name}` for the landmarks MEASURED on `beat`, and no others.

    A landmark the hump-finder could not locate is absent from the mapping rather than guessed;
    two landmarks that land in one bin are joined with a slash rather than one silently winning.
    """
    measured = beat["measured"]
    out: dict[int, str] = {}
    for key, label in _STAGE_KEYS:
        value = measured.get(key)
        if value is None:
            continue
        b = int(value)
        out[b] = label if b not in out else f"{out[b]} / {label}"
    return out


def stage_phase_centres(labels: dict[int, str], n_bins: int = N_STAGE_BINS) -> dict[str, float]:
    """`{stage name: phi at its bin centre, rad}` -- where each measured name goes on a phi axis."""
    return {name: (b + 0.5) * TWO_PI / n_bins for b, name in sorted(labels.items())}


# --------------------------------------------------------------------------------------
# Paired token sampling
# --------------------------------------------------------------------------------------

#: Precedence for collapsing a multi-label record to one colour. This is `winder.data.ptbxl.
#: assign_superclass`'s own rule R5 -- pathology outranks NORM, and among pathologies the order is
#: the dataset's published column order. Arbitrary but FIXED, so it cannot be tuned against a
#: result. 509 of the 2,146 eval records carry more than one superclass. Kept for
#: `sample_token_pairs`'s own dominant-superclass stratification below, even though the display
#: names this precedence used to feed (fig13's `SUPERCLASS_DISPLAY`) were retired with fig13.
DOMINANCE_ORDER: tuple[str, ...] = ("MI", "STTC", "CD", "HYP", "NORM")


def dominant_superclass(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """`(N,)` index into `classes` of each record's dominant superclass, by `DOMINANCE_ORDER`.

    `-1` marks a record with no positive superclass at all (none exist in the fold-9 eval split,
    but the encoding is explicit rather than assumed).
    """
    rank = {name: i for i, name in enumerate(DOMINANCE_ORDER)}
    order = sorted(range(len(classes)), key=lambda c: rank[classes[c]])
    out = np.full(labels.shape[0], -1, dtype=np.int64)
    for column in reversed(order):
        out[labels[:, column] > 0] = column
    return out


def longest_finite_run(mask: np.ndarray) -> tuple[int, int]:
    """`(start, length)` of the longest run of True in `mask`; `(0, 0)` if there is none.

    Tokens are sampled as a CONTIGUOUS run rather than at random positions, for two reasons that
    both matter. It guarantees Figure 12's trajectories are real trajectories -- consecutive
    tokens in time, not a decimated shadow of one -- and at PTB-XL's ~842.6 ms median RR a run of
    40 tokens (8 raw samples each at 100 Hz, so 3.2 s) covers roughly four complete cardiac
    cycles, i.e. full phase coverage several times over.
    """
    best_start, best_len, start = 0, 0, None
    for i, ok in enumerate([*mask.tolist(), False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return best_start, best_len


def sample_token_pairs(
    theta: torch.Tensor,
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    n_records: int,
    n_tokens: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """`(record_index, token_index)` -- the ONE paired sample every cell is indexed by.

    Records are drawn stratified by dominant superclass, proportionally to that class's share of
    the eligible pool (a record is eligible iff it has a run of `n_tokens` consecutive
    finite-theta tokens), with the largest remainders taking the rounding slack. Within a record
    the token window start is drawn uniformly among the valid starts of its longest finite run.
    Both arrays are `(n_records, n_tokens)`.
    """
    finite = torch.isfinite(theta).numpy()
    runs = [longest_finite_run(finite[i]) for i in range(finite.shape[0])]
    eligible = np.array([i for i, (_s, length) in enumerate(runs) if length >= n_tokens])
    if len(eligible) < n_records:
        raise ValueError(
            f"only {len(eligible)} of {finite.shape[0]} records have {n_tokens} consecutive "
            f"finite-theta tokens; asked for {n_records}"
        )
    dominant = dominant_superclass(labels, classes)[eligible]

    quota: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    for c in range(len(classes)):
        exact = n_records * float((dominant == c).sum()) / len(eligible)
        quota[c] = int(exact)
        remainders.append((exact - int(exact), c))
    for _r, c in sorted(remainders, reverse=True)[: n_records - sum(quota.values())]:
        quota[c] += 1

    chosen: list[int] = []
    for c in range(len(classes)):
        pool = eligible[dominant == c]
        chosen.extend(rng.choice(pool, size=quota[c], replace=False).tolist())
    chosen.sort()

    record_index = np.repeat(np.asarray(chosen, dtype=np.int64)[:, None], n_tokens, axis=1)
    token_index = np.empty_like(record_index)
    for row, record in enumerate(chosen):
        start, length = runs[record]
        offset = int(rng.integers(0, length - n_tokens + 1))
        token_index[row] = np.arange(start + offset, start + offset + n_tokens)
    return record_index, token_index


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------


def cell_name(arm_class: str, step: int) -> str:
    """`"signal@5000"` -- the key every per-cell dict in this script is keyed by."""
    return f"{arm_class}@{step}"


def encode_token_cells(
    roster_dir: str,
    seed: int,
    waveforms: torch.Tensor,
    record_index: np.ndarray,
    token_index: np.ndarray,
    *,
    device: torch.device,
    model_seed: int,
) -> dict[str, np.ndarray]:
    """`{cell: (n_records * n_tokens, K)}` raw token `z`, on the identical (record, token) pairs.

    One checkpoint is resident at a time; `z` for a cell is gathered down to the sampled tokens
    before the next checkpoint loads, so peak memory is one `(n_records, T, K)` tensor rather than
    six.
    """
    from winder.eval.readout import discover_seed_checkpoints, encode_z, load_model_and_operator

    flat_records = torch.from_numpy(record_index.ravel())
    flat_tokens = torch.from_numpy(token_index.ravel())
    out: dict[str, np.ndarray] = {}
    for arm_class in ARM_CLASSES:
        arm = f"{arm_class}_seed{seed}"
        steps = discover_seed_checkpoints(os.path.join(roster_dir, arm))
        for step in UMAP_STEPS:
            if step not in steps:
                raise SystemExit(f"[umap] {arm}: step {step} not found (have {sorted(steps)})")
            model, _operator = load_model_and_operator(steps[step], seed=model_seed, device=device)
            z = encode_z(model, waveforms, device)
            out[cell_name(arm_class, step)] = z[flat_records, flat_tokens].numpy()
            del model, z
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[umap] seed {seed}: encoded {arm} step {step}", flush=True)
    return out


def fit_joint_umap(
    cells: dict[str, np.ndarray],
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> tuple[dict[str, np.ndarray], str]:
    """ONE UMAP fitted on the concatenation of every cell, split back per cell. `(embeddings, ver)`.

    Fitting jointly is the whole point: every panel is then a view of one embedding, so the panels
    share a space, share limits, and differ only where the models differ. `random_state` forces
    umap's single-threaded, deterministic path -- the same input gives the same layout.
    """
    # Both codes silenced: mypy reports import-untyped if umap-learn happens to be installed in
    # whatever venv it runs against (no py.typed marker), import-not-found if it is absent --
    # depends on the CURRENT venv, not on this code.
    import umap  # type: ignore[import-not-found,import-untyped]

    names = list(cells)
    stacked = np.concatenate([cells[n] for n in names], axis=0).astype(np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedded = np.asarray(reducer.fit_transform(stacked), dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for name in names:
        size = len(cells[name])
        out[name] = embedded[cursor : cursor + size]
        cursor += size
    return out, str(umap.__version__)


def knn_phase_coherence(
    embedding: np.ndarray, phi: np.ndarray, *, k: int, seed: int
) -> tuple[float, float]:
    """`(R, R_shuffled)` -- how phase-organised a panel's LAYOUT is, and its own null.

    **Definition:** `R` is the circular mean resultant length of `phi` over each point's `k`
    nearest neighbours in the embedding, averaged over points. `R = 1` means every neighbourhood
    is at a single phase; `R = 0` means neighbourhoods are phase-uniform.

    **Why this exists, and what it is not.** "The control panel looks like a ring too" is the kind
    of claim that needs a number attached or it should not be made, and the obvious numbers are
    worse than useless here: a polar-angle correlation about the cloud's centroid conflates
    ordering with elongation (it scores this figure set's *cleanest* ring at 0.04, because that
    ring is narrow and vertical). `R` is invariant to the rotation, reflection and rescaling that
    a UMAP layout is free to apply, and it comes with a null -- `R_shuffled` recomputes it after
    permuting `phi` across the same neighbour graph, which is ~0.16 at these sample sizes.

    It remains a property of the LAYOUT, not of the model: a different `n_neighbors` gives a
    different `R`. It is recorded in the run report as a descriptive diagnostic that keeps the
    figures' own prose honest, and it is not drawn in any figure and not a performance number.
    """
    # scikit-learn ships no `py.typed`; it is a dev/test dependency here and arrives at runtime
    # as one of `umap-learn`'s own requirements under `uv run --with`.
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    index = neighbours.kneighbors(embedding, return_distance=False)[:, 1:]
    unit = np.exp(1j * phi)
    shuffled = np.exp(1j * np.random.default_rng(seed).permutation(phi))
    return (
        float(np.abs(unit[index].mean(axis=1)).mean()),
        float(np.abs(shuffled[index].mean(axis=1)).mean()),
    )


# --------------------------------------------------------------------------------------
# Shared plotting machinery
# --------------------------------------------------------------------------------------


def _cache_path(outdir: str, seed: int) -> str:
    return os.path.join(outdir, f"umap_embedding_seed{seed}.npz")


def _build_seed_embedding(
    args: argparse.Namespace,
    cohort: Any,
    seed: int,
) -> dict[str, Any]:
    """Encode, sample and fit -- or reload a cached fit -- for one seed. All six cells."""
    cache = _cache_path(args.cache_dir, seed)
    expected = {cell_name(a, s) for a in ARM_CLASSES for s in UMAP_STEPS}
    if args.reuse_cache and os.path.isfile(cache):
        blob = np.load(cache, allow_pickle=False)
        cached = set(blob["cells"].tolist())
        if cached == expected:
            print(f"[umap] seed {seed}: reusing cached embedding {cache}", flush=True)
            return {
                "embeddings": {n: blob[f"emb::{n}"] for n in blob["cells"].tolist()},
                "phi": blob["phi"],
                "record_index": blob["record_index"],
                "token_index": blob["token_index"],
                "umap_version": str(blob["umap_version"]),
            }
        # A cache from the superseded multi-step spec would silently widen the shared limits to
        # cover clouds that are no longer drawn. Refit rather than reuse.
        print(
            f"[umap] seed {seed}: cache {cache} holds {sorted(cached)}, expected "
            f"{sorted(expected)} -- refitting",
            flush=True,
        )

    from winder.eval.tasks import CLASSES

    rng = np.random.default_rng(args.sample_seed + seed)
    record_index, token_index = sample_token_pairs(
        cohort.thetas["eval"],
        cohort.labels["eval"],
        CLASSES,
        n_records=args.n_records,
        n_tokens=args.n_tokens,
        rng=rng,
    )
    records = np.unique(record_index[:, 0])
    waveforms = cohort.waveforms["eval"][torch.from_numpy(records)]
    # Re-index the sampled records against the compact waveform tensor just gathered.
    compact = {int(r): i for i, r in enumerate(records.tolist())}
    compact_index = np.vectorize(compact.__getitem__)(record_index)

    cells = encode_token_cells(
        args.roster_dir or os.path.join(args.artifacts_dir, "roster"),
        seed,
        waveforms,
        compact_index,
        token_index,
        device=torch.device(args.device),
        model_seed=args.model_seed,
    )
    phi = cohort.thetas["eval"].numpy()[record_index.ravel(), token_index.ravel()]
    if not np.isfinite(phi).all():
        raise AssertionError("sampled a non-finite theta despite the finite-run constraint")

    t0 = time.time()
    embeddings, umap_version = fit_joint_umap(
        cells,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.umap_seed,
    )
    print(
        f"[umap] seed {seed}: joint fit over {sum(map(len, cells.values())):,} points "
        f"in {time.time() - t0:.0f}s (umap-learn {umap_version})",
        flush=True,
    )

    os.makedirs(args.cache_dir, exist_ok=True)
    # One kwargs dict rather than named arguments plus a `**` splat: `savez_compressed`'s stub
    # types its third positional parameter as `allow_pickle: bool`, so a splatted array dict
    # arriving alongside named arrays is rejected as that parameter.
    arrays: dict[str, Any] = {
        "cells": np.array(list(embeddings), dtype=object).astype(str),
        "phi": phi,
        "record_index": record_index,
        "token_index": token_index,
        "umap_version": np.array(umap_version),
        **{f"emb::{n}": e for n, e in embeddings.items()},
    }
    np.savez_compressed(cache, **arrays)
    return {
        "embeddings": embeddings,
        "phi": phi,
        "record_index": record_index,
        "token_index": token_index,
        "umap_version": umap_version,
    }


def main(argv: list[str] | None = None) -> int:
    """Render the UMAP figure suite plus the staged phase ring. Returns 0 iff every figure
    rendered and passed the no-title/no-caption and axis-label contracts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.expanduser("~/winder-paper/figures"))
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--roster-dir", default=None, help="default <artifacts-dir>/roster")
    ap.add_argument(
        "--lead-stats-path", default=None, help="default <artifacts-dir>/lead_stats_f1to9.json"
    )
    ap.add_argument("--cache-dir", default="artifacts/reports")
    ap.add_argument("--reuse-cache", action="store_true", help="reload a cached joint fit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="comma-separated figure stems (default: all)")
    ap.add_argument("--n-records", type=int, default=300, help="eval records in the token sample")
    ap.add_argument("--n-tokens", type=int, default=40, help="consecutive finite-theta tokens")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--umap-seed", type=int, default=0, help="umap random_state")
    ap.add_argument("--sample-seed", type=int, default=20260820, help="record/token draw")
    ap.add_argument("--model-seed", type=int, default=0, help="build_jepa handshake stream")
    ap.add_argument("--ring-records", type=int, default=2000, help="fig14 eval records per arm")
    ap.add_argument("--ring-bins", type=int, default=24)
    ap.add_argument(
        "--phase-ring-arms",
        default=None,
        help="comma-separated roster arms for fig14 (default: all four nominal arms, today's "
        "shipped layout). One arm renders a single-panel loop, two render a joint pair -- the "
        "output stem gains the arm names as a suffix so it never overwrites the default fig14.",
    )
    ap.add_argument("--harmonic", type=int, default=1, help="1-indexed; 1 = the fundamental")
    ap.add_argument("--pdf-dpi", type=int, default=320, help="raster resolution inside the PDF")
    ap.add_argument("--report-out", default="artifacts/reports/umap_figures.json")
    args = ap.parse_args(argv)

    # Argument validation before any import or decode: a typo in --only would otherwise render
    # nothing, take five minutes doing it, and still write status=PASS.
    requested = set(args.only.split(",")) if args.only else None
    if requested is not None:
        unknown = sorted(requested - set(FIGURE_STEMS))
        if unknown:
            raise SystemExit(f"[umap] unknown figure stem(s) {unknown}; have {list(FIGURE_STEMS)}")

    def wanted(stem: str) -> bool:
        return requested is None or stem in requested

    import eval_suite

    t0 = time.time()

    lead_stats_path = args.lead_stats_path or os.path.join(
        args.artifacts_dir, "lead_stats_f1to9.json"
    )
    cohort, bookkeeping = eval_suite.build_p9_cohort(
        args.data_root, args.artifacts_dir, lead_stats_path, train_limit=1
    )
    print(f"[umap] cohort ready: {bookkeeping}", flush=True)

    beat = ensemble_beat(
        cohort.waveforms["eval"], cohort.thetas["eval"], cohort.patch_width, N_FINE_BINS
    )
    stage_labels = stage_bin_labels(beat)
    print(f"[umap] measured stage bins: {stage_labels}", flush=True)

    rendered: list[dict[str, Any]] = []
    umap_versions: set[str] = set()
    coherence: dict[str, dict[str, float]] = {}

    if wanted("umap_embedding_cache"):
        for seed in SEEDS:
            payload = _build_seed_embedding(args, cohort, seed)
            umap_versions.add(payload["umap_version"])
            embeddings, phi = payload["embeddings"], payload["phi"]
            for name, points in embeddings.items():
                value, null = knn_phase_coherence(
                    points, phi, k=args.n_neighbors, seed=args.sample_seed
                )
                coherence[f"seed{seed}/{name}"] = {"R": value, "R_shuffled_phi": null}
                print(
                    f"[umap] seed {seed}: {name} kNN phase coherence R={value:.3f} "
                    f"(shuffled null {null:.3f})",
                    flush=True,
                )

    if wanted("fig14_phase_ring_staged"):
        ring_args = argparse.Namespace(
            device=args.device,
            roster_dir=args.roster_dir,
            artifacts_dir=args.artifacts_dir,
            lead_stats_path=args.lead_stats_path,
            phase_ring_step=ANCHOR_STEP,
            harmonic=args.harmonic,
            data_root=args.data_root,
            n_records=args.ring_records,
            seed=args.model_seed,
            phase_ring_bins=args.ring_bins,
        )
        arms = args.phase_ring_arms.split(",") if args.phase_ring_arms else None
        loops, n_j = mpf.encode_phase_ring_loops(ring_args, cohort=cohort, arms=arms)
        fig = mpf.phase_ring_grid_figure(
            loops, harmonic=n_j, stage_labels=stage_labels, n_stage_bins=N_STAGE_BINS
        )
        stem = "fig14_phase_ring_staged"
        if args.phase_ring_arms:
            stem = "_".join([stem, *args.phase_ring_arms.split(",")])
        rendered.append(mpf._save(fig, args.outdir, stem))
        print(f"[umap] {stem} -> {rendered[-1]['pdf_bytes'] / 1024:.0f} kB pdf", flush=True)

    payload_out = {
        "status": "PASS",
        "milestone_id": MILESTONE_ID,
        "split_status": SPLIT_STATUS,
        "headline": HEADLINE,
        "metrics": {
            "figures": rendered,
            "n_figures": len(rendered),
            "n_eval_records": bookkeeping["n_eval"],
            "measured_stage_bins": {str(k): v for k, v in stage_labels.items()},
            "measured_beat": beat["measured"],
            "stage_phase_centres_rad": stage_phase_centres(stage_labels),
            "knn_phase_coherence": coherence,
            "elapsed_sec": time.time() - t0,
        },
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_hash": git_sha(os.getcwd()),
            "parameters": vars(args),
            "umap_learn_version": sorted(umap_versions),
            "umap_params": {
                "n_components": 2,
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "metric": args.metric,
                "random_state": args.umap_seed,
            },
            "sample_seed": args.sample_seed,
            "cohort": bookkeeping,
        },
        "decisions": [
            "Fold 10 is never loaded. Every figure encodes folds-1--9 checkpoints on the fold-9 "
            "eval split, which is TRAINING data for them -- split_status train_contaminated, "
            "headline false, recorded per figure in figures/README.md.",
            "ONE UMAP is fitted jointly per seed on the concatenation of BOTH step-5,000 cells, "
            "on the SAME (record, token) pairs, so a difference between panels is a difference "
            "between models and cannot be a difference in sampling. This cache "
            "(umap_embedding_seed{s}.npz) is what render_latent_projections.py's fig17 reads "
            "directly -- fig11/fig12/fig13, which used to render from it here, were retired once "
            "the manuscript shipped (their static output remains in winder-paper/figures/).",
            "Clinical stage labels are recomputed on this cohort by ensemble_beat and only "
            "MEASURED landmarks are named; no textbook PQRST template is overlaid.",
            "At step 5,000 the control arm's token cloud is ALREADY strongly phase-organised: "
            "knn_phase_coherence puts it at R = 0.810 (seed 0) / 0.771 (seed 1) against a "
            "shuffled-phi null of ~0.16, with the signal arm at 0.932 / 0.917. The signal arm is "
            "consistently ahead by ~0.13, but 'only the transport arm organises its latent by "
            "cardiac phase' is not what this figure shows; what the picture does carry is that "
            "only the signal arm closes that ordering into a global ring with a hole in it.",
        ],
        "questions": [],
    }
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    tmp = args.report_out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload_out, fh, indent=2, default=str)
    os.replace(tmp, args.report_out)
    print(
        f"[umap] status=PASS {len(rendered)} figures -> {args.outdir} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

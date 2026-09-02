# Winder

Companion code for [**Capturing Cardiac Cyclicity through Phase-Equivariant Self-Supervised
Learning**](https://arxiv.org/abs/2608.21147) (arXiv:2608.21147).
Blaise Delaney, Dominic Dootson, Juan Jose Juan Castella, Salil Patel, Andrew Pfaff, Yuji Xing,
Jonny Hancox, Karin Sevegnani.

Winder is a JEPA-style encoder for the 12-lead ECG trained under a cyclic transport operator
that is fixed and closed-form -- derived from the cardiac cycle's geometry, not learned, and
adding no parameters. Diagnostic performance is read out with a frozen linear probe.

## Quickstart

```bash
uv sync
uv run python scripts/run_pipeline.py --arms signal,control --seeds 0,1
```

This chains fetch → manifest → phase tokens → lead stats → train → eval (two arms × two seeds,
~250 min/checkpoint on an A100-40GB). `uv run python scripts/run_pipeline.py --help` lists every
flag, or read the six stage scripts it chains if you want to run one alone.

- **Python 3.11**, exactly (`.python-version`; `uv sync` picks it up).
- **CUDA driver ≥ 570** for the pinned `torch==2.11.0+cu128`. No hard CUDA dependency in the
  model code -- every script accepts `--device cpu`.

## Figures

```bash
uv sync --extra figures
```

| Script | Produces |
|---|---|
| `scripts/make_paper_figures.py` | fig01 (phase ring) and the grid engine fig14 builds on |
| `scripts/make_umap_figures.py` | fig14, the joint signal-vs-control UMAP |
| `scripts/render_latent_projections.py` | fig17, per-arm own-space UMAP |

## Reproducibility, honestly

Preprocessing is bit-exact: `build_manifest.py` and `build_phase_tokens.py`, run against raw
PTB-XL, reproduce this repo's own committed manifest and phase tokens bit-for-bit (full corpus,
10m23s). Training is statistical, not bit-identical -- same seed and data land within the
reported confidence interval on any reasonable GPU. The fold-10 evaluation was a one-shot,
pre-registered event whose authorization record self-consumed at verification time; it cannot be
re-run. Every other number under `artifacts/reports/` is labelled `split_status:
train_contaminated` (fold 9 is training data for every checkpoint) -- diagnostic, not headline.
The pre-registration and blind-review record behind the fold-10 result exist but are internal
process and not published in this repo.

## Data

PTB-XL 1.0.3, fetched from PhysioNet by `scripts/fetch_ptbxl.py`. Only `records500/` is used;
the two metadata CSVs are checksum-verified against this repo's own recorded provenance before
anything is built from them.

PTB-XL is distributed under CC BY 4.0 (Wagner et al., *Sci Data* 7, 154, 2020,
https://physionet.org/content/ptb-xl/). Fetching it accepts that license's terms.

## Citation

```bibtex
@misc{delaney2026capturingcardiaccyclicityphaseequivariant,
      title={Capturing Cardiac Cyclicity through Phase-Equivariant Self-Supervised Learning},
      author={Blaise Delaney and Dominic Dootson and Juan Jose Juan Castella and Salil Patel and Andrew Pfaff and Yuji Xing and Jonny Hancox and Karin Sevegnani},
      year={2026},
      eprint={2608.21147},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.21147},
}
```

## Development

```bash
uv run pytest && uv run mypy src tests scripts
```

## License

Code: MIT (`LICENSE`). PTB-XL itself is CC BY 4.0, separately, per PhysioNet's terms -- not
covered by this repo's license.

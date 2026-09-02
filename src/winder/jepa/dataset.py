"""`EcgWindowDataset`: reads PTB-XL's 500 Hz release and decimates to 100 Hz in-pipeline.

DATA-04's finding (`scripts/audit_records100_decimation.py`, `artifacts/data04_lowrate_audit.json`)
is why this reads `records500/` rather than the vendor's own pre-generated `records100/`: locally,
`records100/` is only 1.4% present (299/21,799 files) while `records500/` is 100% complete, and the
299 files that do exist differ *materially* from `decimate_to(records500, 500, 100)` (median
relative RMSE 0.112, 0.075 after correcting for each lead's own measured sub-sample timing lag --
a real filter-shape difference even with timing removed, not filter-order noise: resample_poly's
Kaiser FIR simply isn't the same filter PhysioNet's own tool used). Given records100 could cover
at most 1.4% of the corpus anyway,
mixing "vendor-filtered" and "self-decimated" provenance across one training corpus was judged
worse than a real but bounded per-filter difference on every record -- single provenance,
`decimate_to(records500)`, for the whole corpus, matching the sibling `ttl-phase` campaign's own
precedent (`~/anisotropy_scratch/ttl-phase/src/data/ptbxl.py`: "`records100/` is deliberately not
used"). The 299 real `records100/*_lr` files are therefore no longer read by this class at all --
they remain on disk purely as DATA-04's validation ground truth.

`filename_hr` points at `records500/<...>/<...>_hr`, the record *stem* (no extension) -- `.hea`
must be appended before handing it to `wfdb_io.read_record`. `filename_lr` is kept in
`_REQUIRED_COLUMNS` even though it is no longer this class's read path: CON-04/DATA-03 already
cite it as part of the shared eligibility contract, and DATA-04's own audit script depends on it
to locate each record's native-100Hz ground truth.

The read-and-decimate itself is `winder.data.ptbxl.read_and_decimate_500hz`, shared with
`scripts/s1_lead_stats.py` rather than duplicated here -- it asserts `(5000, 12)` (records500,
pre-decimation) and finite before decimating, then re-asserts the decimated output shape as a
defensive invariant. Canonical lead order is enforced by `wfdb_io.read_record`'s own
`expected_sig_name` parameter (raises on mismatch, never reorders silently) -- not reimplemented
here.

Throughput note: this reads and filters a 5000x12 signal per item instead of a pre-decimated
1000x12 one -- roughly 5x the I/O and one `resample_poly` call per `__getitem__`. Not addressed
here (DATA-01's real-corpus wiring is where that cost would actually be measured, and where a
cached decimated copy would be considered if it matters).

Labels come straight from `winder.data.ptbxl.assign_superclass`'s existing `MULTIHOT_COLS`
(threshold-free multi-hot over the five diagnostic superclasses, rules R1/R3) -- no new
label-parsing code. A record with no asserted diagnostic statement (`UNLABELED`, an all-zero
label vector) still gets an item here (usable for self-supervised pretraining); `has_label` in
the returned dict flags it so a caller building a *probe* dataset can exclude it, per the design
spec's rule that unlabeled records remain in the SSL set but not in probe train/val/test.

Which records this dataset is constructed over is a decision this module does not make: a caller
building the pretraining set should filter `metadata` by manifest *reason code*
(`READ_ERROR`/`WRONG_SHAPE`/`NAN` only), not by `status == "included"` -- phase-clock QC
exclusions (RR plausibility, phase yield, ...) have nothing to do with a JEPA that has no phase
clock, and filtering by `status` would silently drop records for a reason unrelated to this arm.

DATA-02 wires `CorpusStatsNorm` (`winder.data.normalization.apply_corpus_stats`) in here as a
required constructor argument, `lead_stats: winder.data.norm_stats.LeadStats` -- not a default,
not an optional flag: there is no code path through this class that returns an unnormalized
waveform. The parameter's type is `LeadStats` specifically, not the generic `NormConfig`/mode
string `normalization.py` otherwise supports, so `"perbeat"` (CM-08: forbidden on this causal
path -- it drops the LVH/Sokolow-Lyon AUC 0.796 -> 0.577 by destroying absolute voltage) is not
merely undocumented here, it is a type this class's signature cannot accept at all.
`lead_stats.to_norm_config()` is called once, in `__init__`, not per `__getitem__` call, and the
same resolved config is applied to every item -- normalizing with the exact stats the caller
passed in, never a silent refit. Applied to the decimated (T, n_leads) signal, before the
transpose to the (n_leads, T) tensor this class returns.
"""

import os
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from winder.data.norm_stats import LeadStats
from winder.data.normalization import apply_corpus_stats
from winder.data.ptbxl import LEAD_ORDER, MULTIHOT_COLS, read_and_decimate_500hz

__all__ = ["EcgWindowItem", "EcgWindowDataset"]

_REQUIRED_COLUMNS = {
    "ecg_id",
    "patient_id",
    "strat_fold",
    "filename_lr",
    "filename_hr",
    *MULTIHOT_COLS,
}


class EcgWindowItem(TypedDict):
    waveform: torch.Tensor  # (12, 1000) float32
    ecg_id: int
    patient_id: int
    strat_fold: int
    labels: torch.Tensor  # (5,) float32, multi-hot over ptbxl.SUPERCLASSES
    has_label: bool  # False iff labels is all-zero (ptbxl.UNLABELED) -- exclude from probe sets


class EcgWindowDataset(Dataset[EcgWindowItem]):
    """One item per row of `metadata`, read from `records500/` under `data_root` and decimated
    to 100 Hz via `winder.data.ptbxl.read_and_decimate_500hz` -- see this module's docstring
    for why (DATA-04).

    `metadata` must already carry the columns `winder.data.ptbxl.load_metadata` (plus
    `assign_superclass`) produces -- this class does no metadata loading or labelling of its
    own, only reads waveforms.

    `lead_stats` is required (see module docstring, DATA-02): the fitted `LeadStats` whose
    `to_norm_config()` bridges into `normalization.apply_corpus_stats`, applied to every item
    this dataset returns. Pass the same `LeadStats` a caller loaded from `s1_lead_stats.py`'s
    `lead_stats.json` -- this class never fits its own statistics.
    """

    def __init__(self, metadata: pd.DataFrame, data_root: str, lead_stats: LeadStats) -> None:
        missing = _REQUIRED_COLUMNS - set(metadata.columns)
        if missing:
            raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
        if not isinstance(lead_stats, LeadStats):
            raise TypeError(
                f"lead_stats must be a winder.data.norm_stats.LeadStats instance (the fitted "
                f"corpus normalization artifact), got {type(lead_stats).__name__}. "
                f"EcgWindowDataset applies CorpusStatsNorm unconditionally (DATA-02, CM-08) -- "
                f"there is no raw or perbeat mode on this path."
            )
        if lead_stats.fs != 100:
            raise ValueError(
                f"lead_stats.fs={lead_stats.fs} but this dataset decimates every record to "
                f"100 Hz (winder.data.ptbxl.read_and_decimate_500hz's default fs_out); a "
                f"LeadStats fitted at a different rate would silently normalize with the wrong "
                f"per-lead scale. Refit lead_stats at fs=100."
            )
        self.metadata = metadata.reset_index(drop=True)
        self.data_root = data_root
        self.lead_stats = lead_stats
        self._norm_config = lead_stats.to_norm_config()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> EcgWindowItem:
        row = self.metadata.iloc[index]
        hea_path = os.path.join(self.data_root, str(row["filename_hr"]) + ".hea")
        try:
            sig = read_and_decimate_500hz(hea_path, expected_sig_name=LEAD_ORDER)
            sig = apply_corpus_stats(sig, np.empty(0, dtype=np.float64), self._norm_config)
        except ValueError as exc:
            raise ValueError(f"ecg_id={row['ecg_id']}: {exc}") from exc
        waveform = torch.from_numpy(sig.T.copy()).float()  # (12, 1000)
        labels = torch.tensor([float(row[c]) for c in MULTIHOT_COLS], dtype=torch.float32)
        return {
            "waveform": waveform,
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(row["patient_id"]),
            "strat_fold": int(row["strat_fold"]),
            "labels": labels,
            "has_label": bool(float(labels.sum()) > 0.0),
        }

"""PTB-XL v1.0.3 metadata: raw CSV join, SCP-code parsing, diagnostic superclass labels.

Ported near-verbatim from ttl-phase's `src/data/ptbxl.py` (pinned at
cfe2e60a5592e30a32ef1f1863ee4fb449e80714 -- see tests/fixtures/MANIFEST.json), split from
that module's original scope: fold discipline moved to `folds.py` (it's not PTB-XL-specific
-- the train/val/sealed-test pattern is a general policy this repo might apply to a second
dataset one day), decimation already lives in `decimation.py` (PR2), and the bulk
signal-loading helpers (`iter_signals`/`load_signals`/`load_signals_500`/
`write_signals_memmap`) are not ported at all -- confirmed by grep that no ttl-phase
pipeline script called any of them; every script read signals directly via `wfdb_io`.

DATA-04 later added exactly one single-record helper, `read_and_decimate_500hz`, once two
different callers (`winder.jepa.dataset.EcgWindowDataset`, `scripts/s1_lead_stats.py`) both
needed "read one records500/ record, decimate it to 100 Hz" -- not one of the bulk
chunked/memmap loaders the paragraph above says were deliberately not ported.
"""

import ast
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from winder.data.decimation import decimate_to, out_len
from winder.data.wfdb_io import read_record

__all__ = [
    "SUPERCLASSES",
    "MULTIHOT_COLS",
    "WEIGHT_COLS",
    "UNLABELED",
    "LEAD_ORDER",
    "KEEP_COLS",
    "parse_scp_codes",
    "load_scp_statements",
    "assign_superclass",
    "load_metadata",
    "read_and_decimate_500hz",
]

#: PTB-XL diagnostic superclasses, in the dataset's published order.
SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
#: Multi-hot column names, one per superclass (int8, 0/1).
MULTIHOT_COLS: tuple[str, ...] = tuple(f"sc_{s}" for s in SUPERCLASSES)
#: Summed-likelihood column names, one per superclass (float32). Audit trail for the
#: dominance rule: `superclass` is the argmax of these, tie-broken deterministically.
WEIGHT_COLS: tuple[str, ...] = tuple(f"scw_{s}" for s in SUPERCLASSES)
#: Label used when no asserted diagnostic statement maps to a superclass.
UNLABELED = "UNLABELED"

#: PTB-XL lead order, as written in every records500 header. Verified, never assumed.
LEAD_ORDER: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

#: Metadata columns retained by `load_metadata`.
KEEP_COLS: tuple[str, ...] = (
    "ecg_id",
    "patient_id",
    "strat_fold",
    "age",
    "sex",
    "height",
    "weight",
    "device",
    "site",
    "recording_date",
    "filename_lr",
    "filename_hr",
    "scp_codes",
)


# ----------------------------------------------------------------------- metadata
def parse_scp_codes(s: Any) -> dict[str, float]:
    """Parse PTB-XL's `scp_codes` cell, a Python dict literal such as
    `"{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}"`, into `{code: likelihood}`.

    Uses `ast.literal_eval` (never `eval`). An empty/NaN cell gives `{}`. Keys are
    stripped; values are cast to float. Raises ValueError on anything that is not a
    dict literal of that shape -- a malformed cell is a data problem, not a record to
    silently zero out.
    """
    if isinstance(s, dict):
        return {str(k).strip(): float(v) for k, v in s.items()}
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    txt = str(s).strip()
    if not txt or txt.lower() in {"nan", "none", "{}"}:
        return {}
    obj = ast.literal_eval(txt)
    if not isinstance(obj, dict):
        raise ValueError(f"scp_codes cell is not a dict literal: {s!r}")
    return {str(k).strip(): float(v) for k, v in obj.items()}


def load_scp_statements(root: str | os.PathLike) -> pd.DataFrame:
    """Load `scp_statements.csv`, indexed by SCP code.

    Keeps the columns used downstream: `description`, `diagnostic`, `form`, `rhythm`,
    `diagnostic_class`, `diagnostic_subclass`. `diagnostic` is 1.0 for statements the
    dataset authors flag as diagnostic (as opposed to form/rhythm annotations) and NaN
    otherwise.
    """
    path = os.path.join(os.fspath(root), "scp_statements.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    scp = pd.read_csv(path, index_col=0)
    scp.index = scp.index.astype(str).str.strip()
    scp.index.name = "scp_code"
    if "diagnostic_class" not in scp.columns or "diagnostic" not in scp.columns:
        raise ValueError(f"{path}: missing 'diagnostic'/'diagnostic_class' columns")
    keep = [
        c
        for c in (
            "description",
            "diagnostic",
            "form",
            "rhythm",
            "diagnostic_class",
            "diagnostic_subclass",
        )
        if c in scp.columns
    ]
    return scp[keep]


def assign_superclass(
    df: pd.DataFrame,
    scp_statements: pd.DataFrame,
    *,
    zero_likelihood_as: float = 100.0,
    dominance_order: Sequence[str] = ("MI", "STTC", "CD", "HYP", "NORM"),
    unlabeled_label: str = UNLABELED,
) -> pd.DataFrame:
    """Attach the 5-way diagnostic superclass label to a PTB-XL metadata frame.

    Adds, on a copy of `df`:
      `sc_NORM, sc_MI, sc_STTC, sc_CD, sc_HYP`   int8 multi-hot vector
      `scw_NORM, ... , scw_HYP`                  float32 summed likelihood per class
      `superclass`                               single dominant label (str)
      `n_superclass`                             int8, number of asserted superclasses
      `n_scp_codes`                              int16, all statements on the record
      `n_diag_codes`                             int16, statements used for the label
      `scp_diag_codes`                           ';'-joined sorted codes used

    PRE-REGISTERED RESOLUTION RULES (fixed before any model was trained; changing one
    later means re-running everything downstream and saying so):

    R1. Eligible statements. A statement contributes iff it appears in
        `scp_statements` with `diagnostic == 1` AND a non-null `diagnostic_class`.
        Form and rhythm statements (SR, LVOLT, ABQRS, ...) are therefore ignored for
        the superclass label. They stay available in `scp_codes` for other analyses.

    R2. Likelihood 0.0 means ASSERTED-BUT-NOT-QUANTIFIED, not absent. PTB-XL only
        lists statements the cardiologist asserted; the value is a confidence in
        (0, 100] when quantified and exactly 0.0 when it was not. So a 0.0 statement
        sets the multi-hot bit, and for the dominance score it is credited
        `zero_likelihood_as` (default 100.0, i.e. treated as full confidence). This is
        the standard PTB-XL benchmark reading of the field; the alternative (dropping
        0.0) would discard ~a third of all assertions and is not used here.

    R3. Multi-hot is threshold-free. Bit `s` is 1 iff at least one eligible statement
        maps to superclass `s`. No likelihood cut-off is applied anywhere.

    R4. Dominant label = argmax over superclasses of the summed credited likelihood
        (R2). This is the only place a multi-label record is collapsed, and the
        multi-hot vector is always kept alongside so nothing is lost.

    R5. Ties in R4 are broken by the fixed precedence `dominance_order`, default
        (MI, STTC, CD, HYP, NORM): pathology outranks NORM, because a record that
        co-asserts NORM with a pathology is not a normal ECG and giving NORM the tie
        would produce a single label contradicting its own multi-hot vector. Among
        pathologies the order is the dataset's published column order. The rule is
        arbitrary but fixed, so it cannot be tuned against a result.

    R6. Records with no eligible statement get an all-zero multi-hot vector and
        `superclass == unlabeled_label`. They are NOT dropped here -- exclusion, if
        any, is a manifest decision with a logged reason code.
    """
    if "scp_codes" not in df.columns:
        raise ValueError("df must carry the raw 'scp_codes' column")
    missing = [s for s in SUPERCLASSES if s not in set(dominance_order)]
    if missing or len(set(dominance_order)) != len(SUPERCLASSES):
        raise ValueError(
            f"dominance_order must be a permutation of {SUPERCLASSES}, got {tuple(dominance_order)}"
        )

    diag = scp_statements
    eligible: dict[str, str] = {}
    for code, row in diag.iterrows():
        cls = row.get("diagnostic_class")
        flag = row.get("diagnostic")
        if pd.notna(cls) and pd.notna(flag) and float(flag) == 1.0:
            cls = str(cls).strip()
            if cls not in SUPERCLASSES:
                raise ValueError(
                    f"scp_statements: unexpected diagnostic_class {cls!r} for code {code!r}"
                )
            eligible[str(code)] = cls

    # precedence: lower rank wins a tie
    rank = {s: i for i, s in enumerate(dominance_order)}
    order = np.array([rank[s] for s in SUPERCLASSES])

    n = len(df)
    W = np.zeros((n, len(SUPERCLASSES)), np.float64)
    H = np.zeros((n, len(SUPERCLASSES)), np.int8)
    n_all = np.zeros(n, np.int16)
    n_diag = np.zeros(n, np.int16)
    used_codes: list[str] = []
    col = {s: j for j, s in enumerate(SUPERCLASSES)}

    for i, cell in enumerate(df["scp_codes"].to_numpy()):
        codes = parse_scp_codes(cell)
        n_all[i] = len(codes)
        used: list[str] = []
        for code, lik in codes.items():
            cls = eligible.get(code)
            if cls is None:
                continue
            j = col[cls]
            H[i, j] = 1
            W[i, j] += zero_likelihood_as if lik == 0.0 else lik
            used.append(code)
        n_diag[i] = len(used)
        used_codes.append(";".join(sorted(used)))

    out = df.copy()
    for j in range(len(SUPERCLASSES)):
        out[MULTIHOT_COLS[j]] = H[:, j]
        out[WEIGHT_COLS[j]] = W[:, j].astype(np.float32)
    # argmax with a deterministic tie-break: lexsort on (-weight, precedence rank)
    has_any = H.any(axis=1)
    key = np.stack([np.broadcast_to(order, W.shape).astype(np.float64), -W], axis=0)
    winner = np.lexsort(key, axis=1)[:, 0]  # sorts by -W first, then rank
    labels = np.array([SUPERCLASSES[k] for k in winner], dtype=object)
    labels[~has_any] = unlabeled_label
    out["superclass"] = pd.Categorical(labels, categories=[*SUPERCLASSES, unlabeled_label])
    out["n_superclass"] = H.sum(axis=1).astype(np.int8)
    out["n_scp_codes"] = n_all
    out["n_diag_codes"] = n_diag
    out["scp_diag_codes"] = used_codes
    return out


def load_metadata(
    root: str | os.PathLike,
    *,
    scp_statements: pd.DataFrame | None = None,
    extra_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Load `ptbxl_database.csv`, join `scp_statements.csv`, attach superclass labels.

    INDEX CONVENTION (fixed here, relied on downstream): the returned frame has a
    plain `RangeIndex` and carries `ecg_id` as a COLUMN, not the index -- pandas raises
    "both an index level and a column label" on any `sort_values`/`groupby`/`merge`
    touching a name that is both, and this frame gets grouped by `patient_id` and
    merged on `ecg_id`. Rows are in ascending `ecg_id` order and no row is dropped, so
    `df.iloc[i]` lines up with row `i` of a signal array loaded in the same order. For
    label lookups, use `df.set_index("ecg_id")` at the call site.

    Retained columns: ecg_id, patient_id (int64), strat_fold (int8), age, sex, height,
    weight, device, site, recording_date, filename_lr, filename_hr, scp_codes (kept as
    the RAW string -- a dict column is not parquet-safe; use `parse_scp_codes` on it),
    plus everything `assign_superclass` adds, plus any `extra_cols` requested.

    Validated on load: unique ecg_id, strat_fold in 1..10, non-null patient_id and
    filenames. A failure here is a corrupt download, not a modelling choice.
    """
    root = os.fspath(root)
    path = os.path.join(root, "ptbxl_database.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, low_memory=False)

    want = list(KEEP_COLS) + [c for c in extra_cols if c not in KEEP_COLS]
    absent = [c for c in want if c not in raw.columns]
    if absent:
        raise ValueError(f"{path}: missing expected columns {absent}")
    df = raw[want].copy()

    if df["ecg_id"].isna().any() or df["patient_id"].isna().any():
        raise ValueError(f"{path}: null ecg_id or patient_id")
    df["ecg_id"] = df["ecg_id"].astype(np.int64)
    df["patient_id"] = df["patient_id"].astype(np.int64)
    if df["ecg_id"].duplicated().any():
        dup = df.loc[df["ecg_id"].duplicated(), "ecg_id"].tolist()[:10]
        raise ValueError(f"{path}: duplicate ecg_id values, e.g. {dup}")
    if df["strat_fold"].isna().any():
        raise ValueError(f"{path}: null strat_fold")
    df["strat_fold"] = df["strat_fold"].astype(np.int8)
    bad = sorted(set(df["strat_fold"].unique()) - set(range(1, 11)))
    if bad:
        raise ValueError(f"{path}: strat_fold outside 1..10: {bad}")
    for c in ("filename_lr", "filename_hr"):
        if df[c].isna().any():
            raise ValueError(f"{path}: null {c}")
        df[c] = df[c].astype(str).str.strip()

    if scp_statements is None:
        scp_statements = load_scp_statements(root)
    df = assign_superclass(df, scp_statements)

    return df.sort_values("ecg_id", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------- single-record I/O
def read_and_decimate_500hz(
    hea_path: str,
    *,
    expected_sig_name: Sequence[str] | None = LEAD_ORDER,
    fs_out: float = 100.0,
) -> np.ndarray:
    """Read one `records500/` (500 Hz) record and decimate it to `fs_out` via `decimate_to`.

    DATA-04's single shared read path (see this module's docstring and
    notes/methodology/stage-03-data-pipeline.md's DATA-04 entry): `EcgWindowDataset` and
    `scripts/s1_lead_stats.py` both need "read records500, decimate to 100 Hz," so the logic
    lives here once rather than duplicated per caller.

    Asserts, in order: the header declares `fs == 500`; the raw signal is `(5000,
    len(expected_sig_name))` and finite; `decimate_to`'s own output shape matches
    `out_len(5000, 500, fs_out)`. Raises `ValueError` (not `wfdb_io.ReadError`) on any of these
    -- a rate-specific or shape-specific failure here is a bug for whichever contract governs
    this data path (DATA-04, CON-04), never a silent drop. Error messages reference `hea_path`
    (not an `ecg_id`, which this function has no metadata to look up) -- the caller's own
    record identifier, if it needs one in the message, should be attached by catching
    `ValueError` and re-raising, not duplicated inside this function's checks.
    """
    sig500, header = read_record(hea_path, expected_sig_name=expected_sig_name)
    n_leads = len(expected_sig_name) if expected_sig_name is not None else sig500.shape[1]
    if float(header.fs) != 500.0:
        raise ValueError(f"{hea_path}: header declares fs={header.fs}, expected 500")
    if sig500.shape != (5000, n_leads):
        raise ValueError(
            f"{hea_path}: expected (5000, {n_leads}) from records500/, got {sig500.shape}"
        )
    if not np.isfinite(sig500).all():
        raise ValueError(f"{hea_path}: non-finite samples in records500/ waveform")
    sig = decimate_to(sig500, header.fs, fs_out)
    want_n = out_len(5000, header.fs, fs_out)
    if sig.shape != (want_n, n_leads):
        raise ValueError(
            f"{hea_path}: decimate_to produced {sig.shape}, expected {(want_n, n_leads)}"
        )
    return sig

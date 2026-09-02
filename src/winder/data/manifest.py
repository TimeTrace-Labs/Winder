"""The per-record ledger.

Ported near-verbatim from ttl-phase's `src/data/manifest.py` (pinned at
cfe2e60a5592e30a32ef1f1863ee4fb449e80714). Ground rule this module makes executable rather
than aspirational: never drop a record silently.

  * every record a driver script touches gets exactly one row, whether it survives or not;
  * `status` is 'included' or 'excluded', and an excluded row MUST carry a `reason_code`
    drawn from the fixed enumeration `REASON_CODES`. An unlisted code is rejected -- a new
    failure mode cannot be swept into a free-text field, it forces a visible edit to this
    enumeration;
  * `assert_accounts_for(n_total)` fails unless included + excluded == n_total, so a
    record cannot be lost without an assertion firing;
  * `summary(by='superclass')` / `summary(by='reason_code')` produce the yield and
    exclusion tables from the parquet alone, no re-run needed.

`RecordRow` is pydantic, not a plain dataclass -- this is the data layer's one type that
actually crosses winder's established pydantic boundary (see `config.py`'s docstring):
written by one run, read back by a possibly-much-later one, possibly hand-inspected or
version-drifted. Pydantic buys real validators (status/reason_code consistency, a closed
`Literal` on status, exact-length `superclasses`) enforced on every construction path --
including `from_parquet`, not just `Manifest.add`.

Bug fixes vs. ttl-phase (see the port plan):
  * #4: `REASON_CODES` gains `RR_OUTLIERS`, split out from `IMPLAUSIBLE_RR`. The two flags
    (`PHASE_IMPLAUSIBLE_RR`: median RR out of range: `PHASE_RR_OUTLIERS`: too large a
    *fraction* of individual RRs out of range) were already distinct in `phase.py`; a
    driver script collapsed them into one manifest reason code. That collapse doesn't
    happen here -- there are now two codes to map onto.
  * #9: `Manifest.add()`'s old coercion of a non-tuple `quality_flags` iterated whatever
    was passed -- including a pre-joined string, character by character (`for c in "AB"`
    yields `"A"`, `"B"`, not `"AB"`). That code path does not exist here: `RecordRow`'s own
    validator rejects a bare `str` outright instead of silently iterating it.
  * #7 (vocabulary half): `REASON_CODES` also gains `FLAT_SIGNAL` and `LOW_CONFIDENCE`.
    `phase.py` can emit `PHASE_FLAT_SIGNAL`/`PHASE_LOW_CONFIDENCE`, and ttl-phase's driver
    script's flag-to-reason mapping covered neither, so a record carrying only one of these
    flags fell through to `status="included"` with nobody reading the flag. The driver
    script in this port (`scripts/s0_phase_manifest.py`) asserts total flag coverage against
    `phase.ALL_FLAGS` before processing a single record, so this can't recur silently.
"""

import os
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "REASON_CODES",
    "REASON_DESCRIPTIONS",
    "SUPERCLASSES",
    "NO_REASON",
    "RecordRow",
    "Manifest",
    "multihot",
]

# ------------------------------------------------------------------- the fixed vocabulary
#: Exhaustive, ordered exclusion vocabulary. Adding a code is a deliberate act: it changes
#: the paper's exclusion table. Removing one invalidates existing manifests.
REASON_CODES: tuple[str, ...] = (
    "READ_ERROR",  # .hea/.dat missing, unparseable header, or truncated record
    "WRONG_SHAPE",  # not (5000, 12) at 500 Hz after parsing
    "NAN",  # non-finite samples in the physical signal
    "NO_SUPERCLASS",  # scp_codes maps to no diagnostic superclass (no label to report by)
    "TOO_FEW_BEATS",  # n_beats < phase.min_beats; too few to define a phase field
    "IMPLAUSIBLE_RR",  # median RR outside [phase.rr_min_ms, phase.rr_max_ms]: detector failure
    "RR_OUTLIERS",  # too large a FRACTION of individual RRs out of bounds (bug #4)
    "HIGH_RR_CV",  # RR CV > phase.rr_cv_max: AF, ectopy, or detector failure
    "LOW_PHASE_YIELD",  # phase_yield < phase.min_phase_yield; too little valid theta
    "FLAT_SIGNAL",  # every lead's robust scale hit the dead-lead floor (bug #7)
    "LOW_CONFIDENCE",  # detector_confidence below a calibrated floor; inactive by default
    # (min_detector_confidence is None until calibrated) but the code
    # must exist so the flag has somewhere to go once it is (bug #7)
)

#: Human-readable gloss per code, so an exclusion table can be generated rather than
#: written from memory.
REASON_DESCRIPTIONS: dict[str, str] = {
    "READ_ERROR": "record could not be read or its WFDB header could not be parsed",
    "WRONG_SHAPE": "signal shape is not the expected (5000, 12) at 500 Hz",
    "NAN": "non-finite samples after conversion to mV",
    "NO_SUPERCLASS": "no diagnostic superclass in scp_statements",
    "TOO_FEW_BEATS": "fewer detected beats than the configured minimum",
    "IMPLAUSIBLE_RR": "median RR interval outside the physiological window",
    "RR_OUTLIERS": "too large a fraction of individual RR intervals outside the "
    "physiological window (median RR itself may be within range)",
    "HIGH_RR_CV": "RR coefficient of variation above the configured maximum",
    "LOW_PHASE_YIELD": "fraction of samples with valid phase below the configured minimum",
    "FLAT_SIGNAL": "every lead's robust scale was at or below the dead-lead floor",
    "LOW_CONFIDENCE": "detector confidence below a calibrated floor (beats mutually inconsistent)",
}

#: PTB-XL diagnostic superclasses, in the order used for every multi-hot column set.
SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")

#: reason_code of an included row. Empty string, not None: parquet round-trips it exactly.
NO_REASON = ""

_SC_COLS = tuple(f"sc_{s}" for s in SUPERCLASSES)

COLUMNS: tuple[str, ...] = (
    "ecg_id",
    "patient_id",
    "strat_fold",
    "superclass",
    *_SC_COLS,
    "age",
    "sex",
    "device",
    "site",
    "n_beats",
    "phase_yield",
    "rr_mean_ms",
    "rr_median_ms",
    "rr_sd_ms",
    "rr_cv",
    "jitter_ms",
    "quality_flags",
    "status",
    "reason_code",
    "reason_detail",
)

_INT_COLS = ("ecg_id", "patient_id", "strat_fold", "n_beats", *_SC_COLS)
_FLOAT_COLS = (
    "age",
    "sex",
    "phase_yield",
    "rr_mean_ms",
    "rr_median_ms",
    "rr_sd_ms",
    "rr_cv",
    "jitter_ms",
)
_STR_COLS = (
    "superclass",
    "device",
    "site",
    "quality_flags",
    "status",
    "reason_code",
    "reason_detail",
)


def _null_safe_str(v: Any) -> str:
    """pandas NaN/None -> "", this module's own documented convention for a missing
    string. Plain `str(v)` turns a null cell into the literal string "nan", which would
    then be accepted as legitimate data (a bogus flag tag, a fake device name) -- found by
    audit against `from_parquet`."""
    return "" if pd.isna(v) else str(v)


def multihot(labels: Iterable[str]) -> tuple[int, ...]:
    """Multi-hot over SUPERCLASSES, in SUPERCLASSES order.

    A record with {'MI', 'STTC'} gives (0, 1, 1, 0, 0). Unknown labels raise, because a
    typo'd superclass would otherwise silently produce an all-zero row that looks like
    NO_SUPERCLASS.
    """
    labs = list(labels)
    bad = sorted(set(labs) - set(SUPERCLASSES))
    if bad:
        raise ValueError(f"unknown superclass label(s) {bad}; expected {SUPERCLASSES}")
    s = set(labs)
    return tuple(int(sc in s) for sc in SUPERCLASSES)


# -------------------------------------------------------------------------------- the row
class RecordRow(BaseModel):
    """One record's ledger entry. Field names are the parquet column names.

    Every field has a default except `ecg_id` and `status`, so a read failure can still be
    logged with nothing but an id and a reason -- which is precisely the case where
    silently dropping is most tempting.

    Units: RR statistics and jitter in milliseconds; `phase_yield` a fraction in [0, 1];
    `sex` as PTB-XL codes it (0 male, 1 female) held as float so that missing is NaN.
    `jitter_ms` is the R-peak localisation jitter estimate (sub-sample refinement
    residual), the quantity compared against one bin width.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ecg_id: int
    status: Literal["included", "excluded"]
    patient_id: int = -1
    strat_fold: int = -1
    superclass: str = ""
    superclasses: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    age: float = float("nan")
    sex: float = float("nan")
    device: str = ""
    site: str = ""
    n_beats: int = 0
    phase_yield: float = float("nan")
    rr_mean_ms: float = float("nan")
    rr_median_ms: float = float("nan")
    rr_sd_ms: float = float("nan")
    rr_cv: float = float("nan")
    jitter_ms: float = float("nan")
    quality_flags: tuple[str, ...] = ()
    reason_code: str = NO_REASON
    reason_detail: str = ""

    @field_validator("superclasses", mode="before")
    @classmethod
    def _coerce_superclasses(cls, v: Any) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in v)
        except TypeError as exc:
            # Re-raise as ValueError: a bare TypeError from a non-iterable input (None, an
            # int, a NaN) would otherwise escape pydantic's validation entirely instead of
            # surfacing as ValidationError like every other bad input to this model does
            # (audit-found inconsistency in the pydantic-boundary error contract).
            raise ValueError(f"superclasses must be an iterable of 5 ints, got {v!r}") from exc

    @field_validator("quality_flags", mode="before")
    @classmethod
    def _reject_preformatted_flag_string(cls, v: Any) -> tuple[str, ...]:
        """Bug #9 fix. A pre-joined string (`";".join(flags)`) must never reach here: Python
        iterates a `str` character-by-character, which is exactly how ttl-phase's manifest
        corrupted `quality_flags` into single characters. Fail loudly instead.

        `bytes`/`bytearray` reproduce the identical failure mode (Python iterates them into
        per-byte ints) with no exception at all, silently -- found by audit -- so they're
        rejected here too, not just `str`.
        """
        if isinstance(v, str | bytes | bytearray):
            # ValueError (not TypeError): pydantic only wraps ValueError/AssertionError
            # raised inside a validator into its own ValidationError.
            raise ValueError(
                f"quality_flags must be an iterable of flag strings (e.g. a list or tuple), "
                f"not a pre-joined/atomic blob of type {type(v).__name__} ({v!r}). Pass the "
                f"flags list itself; `RecordRow.to_dict()` does the ';'.join for storage."
            )
        try:
            return tuple(str(f) for f in v)
        except TypeError as exc:
            # Same re-raise-as-ValueError reasoning as _coerce_superclasses above.
            raise ValueError(
                f"quality_flags must be an iterable of flag strings, got {v!r}"
            ) from exc

    @model_validator(mode="after")
    def _status_reason_consistency(self) -> "RecordRow":
        """An excluded row must carry a legal reason code; an included row must carry
        none -- an "included but excluded-because" row is a contradiction that would
        corrupt the yield table."""
        if self.status == "excluded":
            if self.reason_code not in REASON_CODES:
                raise ValueError(
                    f"unlisted exclusion reason {self.reason_code!r}. Legal codes: "
                    f"{list(REASON_CODES)}. If this is a genuinely new failure mode, add it "
                    f"to REASON_CODES (and to the paper's exclusion table) rather than "
                    f"logging it as free text."
                )
        elif self.reason_code != NO_REASON:
            raise ValueError(
                f"included record {self.ecg_id} was given reason_code {self.reason_code!r}; "
                f"an included row must have no reason code. Use quality_flags for advisory "
                f"tags."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Flatten to the parquet column layout: multi-hot -> sc_* columns, flag tuple ->
        ';'-joined string."""
        d = self.model_dump()
        sc = d.pop("superclasses")
        for col, v in zip(_SC_COLS, sc, strict=True):
            d[col] = int(v)
        d["quality_flags"] = ";".join(d["quality_flags"])
        return {c: d[c] for c in COLUMNS}


# --------------------------------------------------------------------------- the ledger
class Manifest:
    """Accumulate one row per record, then write it once to parquet.

    Usage (a driver script):

        man = Manifest()
        for meta in records:
            try:
                sig = load(meta)
            except Exception as exc:
                man.add_excluded(ecg_id=..., reason_code="READ_ERROR", reason_detail=str(exc))
                continue
            ...
            man.add_included(ecg_id=..., n_beats=..., phase_yield=..., ...)
        man.assert_accounts_for(len(records))
        man.to_parquet('artifacts/manifest.parquet')

    Every `add*` call constructs a `RecordRow` (whose own validators enforce status/reason
    consistency and the closed vocabulary) and rejects duplicate ecg_ids.
    """

    def __init__(self) -> None:
        """Empty ledger. No configuration: the row schema and the reason vocabulary are
        module-level constants precisely so that two runs cannot disagree about them."""
        self._rows: list[RecordRow] = []
        self._seen: set[int] = set()

    # ---- construction
    def add(
        self,
        ecg_id: int,
        status: Literal["included", "excluded"],
        reason_code: str | None = NO_REASON,
        reason_detail: str = "",
        **fields: Any,
    ) -> RecordRow:
        """Append one row. Returns the stored RecordRow.

        Validation (all of it is the honesty guard, none of it is decoration): status/
        reason-code consistency and the closed reason vocabulary are enforced by
        `RecordRow` itself; this method additionally rejects a duplicate `ecg_id`, since
        that is ledger-level state `RecordRow` cannot see on its own. `reason_code=None`
        is normalised to `NO_REASON` (ttl-phase's original explicitly accepted `None`
        here; the port's first pass dropped that normalisation, audit-found).
        """
        eid = int(ecg_id)
        if eid in self._seen:
            raise ValueError(f"duplicate ecg_id {eid}: the manifest is one row per record")
        code = NO_REASON if reason_code is None else str(reason_code)
        row = RecordRow(
            ecg_id=eid,
            status=status,
            reason_code=code,
            reason_detail=str(reason_detail),
            **fields,
        )
        self._rows.append(row)
        self._seen.add(eid)
        return row

    def add_included(self, ecg_id: int, **fields: Any) -> RecordRow:
        """Log a record that survives QC."""
        return self.add(ecg_id=ecg_id, status="included", **fields)

    def add_excluded(
        self, ecg_id: int, reason_code: str, reason_detail: str = "", **fields: Any
    ) -> RecordRow:
        """Log a record that does not survive QC. `reason_code` must be in REASON_CODES;
        `reason_detail` is free text for the specific value that tripped it (e.g.
        'rr_cv=0.51 > 0.35'), which is what makes the exclusion reproducible."""
        return self.add(
            ecg_id=ecg_id,
            status="excluded",
            reason_code=reason_code,
            reason_detail=reason_detail,
            **fields,
        )

    # ---- size / accounting
    def __len__(self) -> int:
        """Number of records logged (included + excluded)."""
        return len(self._rows)

    @property
    def n_included(self) -> int:
        """Count of rows with status 'included' -- the size of the analysis set."""
        return sum(r.status == "included" for r in self._rows)

    @property
    def n_excluded(self) -> int:
        """Count of rows with status 'excluded' -- every one of them carries a reason."""
        return sum(r.status == "excluded" for r in self._rows)

    @property
    def ecg_ids(self) -> np.ndarray:
        """(n,) int64 ecg_ids in insertion order."""
        return np.asarray([r.ecg_id for r in self._rows], dtype=np.int64)

    def included_ids(self) -> np.ndarray:
        """(n_included,) int64 ecg_ids with status 'included' -- the analysis set."""
        return np.asarray([r.ecg_id for r in self._rows if r.status == "included"], dtype=np.int64)

    def assert_accounts_for(self, n_total: int) -> None:
        """Raise unless included + excluded == n_total (and the rows are unique).

        Called at the end of a driver script with the number of rows read from the raw
        source, so a record skipped by a `continue` without being logged cannot reach an
        artifact unnoticed.
        """
        n_total = int(n_total)
        inc, exc = self.n_included, self.n_excluded
        if inc + exc != len(self._rows):
            raise AssertionError(
                f"manifest internally inconsistent: {inc} included + {exc} excluded "
                f"!= {len(self._rows)} rows (a status outside the enumeration got in)"
            )
        if len(self._seen) != len(self._rows):
            raise AssertionError(
                f"manifest has {len(self._rows)} rows but only {len(self._seen)} unique ecg_ids"
            )
        if inc + exc != n_total:
            missing = n_total - (inc + exc)
            raise AssertionError(
                f"manifest accounts for {inc + exc} records ({inc} included, {exc} "
                f"excluded) but {n_total} were offered: {missing:+d} unlogged. Every "
                f"record must be logged, including the ones that fail to load."
            )

    # ---- views
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame with the fixed COLUMNS order and stable dtypes (int64 / float64 /
        str). Empty manifests still return the full schema, so downstream code that
        selects columns does not need to special-case a zero-record run."""
        if self._rows:
            df = pd.DataFrame([r.to_dict() for r in self._rows], columns=list(COLUMNS))
        else:
            df = pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})
        for c in _INT_COLS:
            df[c] = df[c].astype("int64")
        for c in _FLOAT_COLS:
            df[c] = df[c].astype("float64")
        for c in _STR_COLS:
            df[c] = df[c].fillna("").astype(str)
        return df

    def summary(self, by: str | Sequence[str] = "superclass") -> pd.DataFrame:
        """Inclusion/exclusion counts broken down by one or more columns.

        `by='superclass'` gives the yield-by-superclass table; `by='reason_code'` gives
        the exclusion table, in which included rows appear under the '' code by
        construction; `by=('superclass','reason_code')` cross-tabulates the two, which is
        how "did QC preferentially drop HYP" is answered. `by='strat_fold'`, `'device'`,
        `'site'` also work.

        Columns: n_total, n_included, n_excluded, yield (= n_included / n_total). A TOTAL
        row is appended so the table cannot be read as summing to something else. Note
        that with `by='superclass'` the rows partition the corpus (one primary label per
        record); use `summary_by_superclass_multilabel()` for the multi-hot view, where a
        record with two superclasses is counted twice on purpose.
        """
        keys = [by] if isinstance(by, str) else list(by)
        df = self.to_dataframe()
        for k in keys:
            if k not in df.columns:
                raise ValueError(f"cannot summarise by {k!r}; columns are {list(COLUMNS)}")
        df = df.assign(
            _inc=(df["status"] == "included").astype(int),
            _exc=(df["status"] == "excluded").astype(int),
        )
        g = (
            df.groupby(keys, dropna=False)[["_inc", "_exc"]]
            .sum()
            .rename(columns={"_inc": "n_included", "_exc": "n_excluded"})
        )
        g.insert(0, "n_total", g["n_included"] + g["n_excluded"])
        g["yield"] = np.where(g["n_total"] > 0, g["n_included"] / g["n_total"], np.nan)
        n_t = int(g["n_total"].sum())
        idx: pd.Index | pd.MultiIndex
        if len(keys) == 1:
            idx = pd.Index(["TOTAL"], name=keys[0])
        else:
            idx = pd.MultiIndex.from_tuples([("TOTAL", *("",) * (len(keys) - 1))], names=keys)
        total = pd.DataFrame(
            {
                "n_total": [n_t],
                "n_included": [int(g["n_included"].sum())],
                "n_excluded": [int(g["n_excluded"].sum())],
                "yield": [float(g["n_included"].sum() / n_t) if n_t else np.nan],
            },
            index=idx,
        )
        return pd.concat([g.sort_index(), total])

    def summary_by_superclass_multilabel(self) -> pd.DataFrame:
        """Yield per superclass using the multi-hot columns, so a record carrying MI and
        STTC contributes to both rows. Totals therefore exceed the record count; that is
        the point, and it is why this is a separate method from `summary`."""
        df = self.to_dataframe()
        inc = df["status"] == "included"
        rows = []
        for sc, col in zip(SUPERCLASSES, _SC_COLS, strict=True):
            m = df[col] == 1
            n_t, n_i = int(m.sum()), int((m & inc).sum())
            rows.append(
                {
                    "superclass": sc,
                    "n_total": n_t,
                    "n_included": n_i,
                    "n_excluded": n_t - n_i,
                    "yield": (n_i / n_t) if n_t else np.nan,
                }
            )
        m = df[list(_SC_COLS)].sum(axis=1) == 0
        n_t, n_i = int(m.sum()), int((m & inc).sum())
        rows.append(
            {
                "superclass": "(none)",
                "n_total": n_t,
                "n_included": n_i,
                "n_excluded": n_t - n_i,
                "yield": (n_i / n_t) if n_t else np.nan,
            }
        )
        return pd.DataFrame(rows).set_index("superclass")

    def reason_table(self) -> pd.DataFrame:
        """Exclusion counts per reason code, with every legal code present (count 0 if it
        never fired) and its description. A code that never fires is information: it says
        the QC rule was inert on this corpus."""
        df = self.to_dataframe()
        exc = df.loc[df["status"] == "excluded", "reason_code"].value_counts()
        return pd.DataFrame(
            {
                "n_excluded": [int(exc.get(c, 0)) for c in REASON_CODES],
                "description": [REASON_DESCRIPTIONS[c] for c in REASON_CODES],
            },
            index=pd.Index(REASON_CODES, name="reason_code"),
        )

    # ---- persistence
    def to_parquet(self, path: str, overwrite: bool = True) -> str:
        """Write the ledger to parquet (pyarrow). Returns the path.

        The manifest is a derived artifact that a re-run legitimately regenerates, so
        overwriting is allowed by default; pass overwrite=False where clobbering would
        lose information.
        """
        if not overwrite and os.path.exists(path):
            raise FileExistsError(f"{path} exists and overwrite=False")
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False, engine="pyarrow")
        return path

    @classmethod
    def from_parquet(cls, path: str) -> "Manifest":
        """Rebuild a Manifest from parquet. Round-trips `to_parquet` exactly (same
        dataframe, same dtypes) and re-validates every row through `add`, so a
        hand-edited manifest with an unlisted reason code fails on load rather than at
        analysis time."""
        # engine="auto" (not "pyarrow"): pandas-stubs 3.0.5's overload for the literal
        # "pyarrow" additionally requires `to_pandas_kwargs`, which we have no use for.
        # pyarrow is the only parquet engine winder depends on, so "auto" resolves to it.
        df = pd.read_parquet(path, engine="auto")
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing manifest columns: {missing}")
        man = cls()
        for rec in df.to_dict("records"):
            # Null-safe, not a bare str(): a NULL/NaN cell in a string column (a
            # hand-edited or version-drifted parquet -- exactly the case this method's
            # own docstring is written for) must round-trip to "", this module's
            # documented convention for a missing string, not the fabricated literal
            # "nan" that str(None)/str(float("nan")) produces. For quality_flags
            # specifically, "nan" would otherwise be accepted as a real flag tag.
            flags = _null_safe_str(rec["quality_flags"])
            # status is genuinely untrusted here (a hand-edited parquet could hold
            # anything) -- RecordRow's own Literal validation is what actually enforces
            # it at runtime; the cast only satisfies the static type of Manifest.add.
            man.add(
                ecg_id=int(rec["ecg_id"]),
                status=cast(Literal["included", "excluded"], str(rec["status"])),
                reason_code=str(rec["reason_code"]),
                reason_detail=_null_safe_str(rec["reason_detail"]),
                patient_id=int(rec["patient_id"]),
                strat_fold=int(rec["strat_fold"]),
                superclass=_null_safe_str(rec["superclass"]),
                superclasses=tuple(int(rec[c]) for c in _SC_COLS),
                age=float(rec["age"]),
                sex=float(rec["sex"]),
                device=_null_safe_str(rec["device"]),
                site=_null_safe_str(rec["site"]),
                n_beats=int(rec["n_beats"]),
                phase_yield=float(rec["phase_yield"]),
                rr_mean_ms=float(rec["rr_mean_ms"]),
                rr_median_ms=float(rec["rr_median_ms"]),
                rr_sd_ms=float(rec["rr_sd_ms"]),
                rr_cv=float(rec["rr_cv"]),
                jitter_ms=float(rec["jitter_ms"]),
                quality_flags=tuple(flags.split(";")) if flags else (),
            )
        return man

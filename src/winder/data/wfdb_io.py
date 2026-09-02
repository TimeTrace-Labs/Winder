"""Minimal, dependency-free WFDB **format-16** reader (and a test-support writer).

Ported near-verbatim from ttl-phase's `src/data/wfdb_io.py` (pinned at commit
cfe2e60a5592e30a32ef1f1863ee4fb449e80714 -- see tests/fixtures/MANIFEST.json). The decode
algorithm is validated only empirically against real PTB-XL records, with no reference
library to check it against -- the `wfdb` pip package is deliberately not used here either,
so this reader is the only thing standing between raw bytes and a physical signal. Ported
with only interface changes: `read_header` returns a `WfdbHeader` dataclass instead of a
plain dict (a bare dict return trips winder's `disallow_untyped_defs`/`warn_return_any`
mypy config at every call site).

Scope, stated as a refusal list. This reader handles exactly the subset of WFDB that
PTB-XL v1.0.3 uses, and raises `ReadError` on anything else rather than guessing:

  * single-segment records only (a `name/nseg` record line is rejected);
  * every signal must live in ONE `.dat` file, interleaved sample-by-sample;
  * format `16` only (16-bit little-endian two's complement). `16x2`, `16:skew`,
    `212`, `310`, ... are rejected;
  * `samples-per-frame > 1`, non-zero skew and byte offsets are rejected.

Conventions this module fixes:

  * **Output is physical millivolts, float32**, computed as
    `physical = (raw - baseline) / gain`, then multiplied by the unit factor of the
    header's declared units (mV -> 1, uV -> 1e-3, V -> 1e3). PTB-XL declares mV, so
    the factor is 1 and the formula is literally the one in the task spec.
  * **Shape is (T, n_sig)** = (time, lead). Lead order is whatever the header says;
    `read_record` will verify it against a caller-supplied expected order but never
    reorders silently.
  * `gain == 0` in a header means "unspecified" in the WFDB spec; we substitute
    `default_gain` (200 ADU/mV, the spec's default), exposed as a named argument
    rather than buried.
  * `-32768` is WFDB's reserved "invalid sample" sentinel for format 16. It is
    treated as a hard error, not silently interpolated. Set `invalid_sentinel=None`
    to disable the check.
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ReadError",
    "WfdbHeader",
    "read_header",
    "read_record",
    "write_format16",
    "FORMAT16_DTYPE",
    "INVALID_SENTINEL_16",
    "UNIT_TO_MV",
]

#: format-16 on-disk dtype: 16-bit little-endian two's complement.
FORMAT16_DTYPE = np.dtype("<i2")

#: WFDB's reserved "sample missing" code for format 16.
INVALID_SENTINEL_16 = -32768

#: Multiplicative factor taking the header's declared units to millivolts.
UNIT_TO_MV = {"mv": 1.0, "uv": 1e-3, "v": 1e3}

#: WFDB spec default when a header declares gain 0 ("unspecified"), in ADU per unit.
DEFAULT_GAIN = 200.0

# "1000.0(0)/mV", "200/mV", "1000.0(-50)", "0"
_GAIN_RE = re.compile(
    r"^(?P<gain>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r"(?:\((?P<baseline>[-+]?[0-9]+)\))?"
    r"(?:/(?P<units>\S+))?$"
)
# leading number of an fs token like "500", "500/1000", "500/1000(0)"
_FS_RE = re.compile(r"^(?P<fs>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


class ReadError(Exception):
    """Raised on a malformed header, a short/oversized `.dat`, a failed checksum,
    an invalid-sample sentinel, or a non-finite decoded value.

    Typed on purpose: a caller can catch exactly this to log a record-level exclusion
    reason instead of dying, per the "never drop records silently" ground rule.
    """


@dataclass(frozen=True)
class WfdbHeader:
    """Parsed WFDB `.hea` header for a single-segment, single-`.dat`, format-16 record.

    Frozen, in-process only -- produced by `read_header`/`read_record` and consumed
    within the same call chain; never serialized on its own.
    """

    record_name: str
    n_sig: int
    fs: int | float
    #: samples per signal. If the header declared 0 ("unspecified"), this is inferred
    #: from the .dat size as filesize // (2 * n_sig) and `n_samp_inferred` is True.
    n_samp: int
    n_samp_inferred: bool
    hea_path: str
    dat_file: str
    dat_path: str
    fmt: int  # always 16 (anything else raises)
    gain: np.ndarray  # (n_sig,) float64, ADU per physical unit
    baseline: np.ndarray  # (n_sig,) int64, ADU value of physical zero
    adc_zero: np.ndarray  # (n_sig,) int64
    adc_res: np.ndarray  # (n_sig,) int64
    init_value: np.ndarray  # (n_sig,) int64, first raw sample (checkable)
    has_init_value: np.ndarray  # (n_sig,) bool -- field actually present in the header
    checksum: np.ndarray  # (n_sig,) int64, 16-bit sum of raw samples (checkable)
    has_checksum: np.ndarray  # (n_sig,) bool
    block_size: np.ndarray  # (n_sig,) int64
    units: list[str]  # as declared, e.g. ["mV", ...]
    unit_to_mv: np.ndarray  # (n_sig,) float64, factor taking `units` to mV
    sig_name: list[str]  # lead names, e.g. ["I", "II", ..., "V6"]


def _int(tok: str, what: str, hea_path: str) -> int:
    try:
        return int(tok)
    except ValueError as exc:
        raise ReadError(f"{hea_path}: {what} is not an integer: {tok!r}") from exc


def read_header(hea_path: str | os.PathLike, *, default_gain: float = DEFAULT_GAIN) -> WfdbHeader:
    """Parse a WFDB `.hea` header for a single-segment, single-`.dat`, format-16 record.

    Raises ReadError for every construct outside the PTB-XL subset (see module
    docstring). Comment lines (`#`) and blank lines are ignored.
    """
    hea_path = os.path.abspath(os.fspath(hea_path))
    if not os.path.isfile(hea_path):
        raise ReadError(f"header not found: {hea_path}")
    try:
        with open(hea_path, encoding="utf-8", errors="strict") as fh:
            raw_lines = fh.read().splitlines()
    except OSError as exc:
        raise ReadError(f"cannot read header {hea_path}: {exc}") from exc

    lines = [ln.strip() for ln in raw_lines]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ReadError(f"{hea_path}: empty header")

    # ---- record line: "<name> <n_sig> <fs> <n_samp> ..."
    rec = lines[0].split()
    if len(rec) < 2:
        raise ReadError(f"{hea_path}: record line has {len(rec)} fields, need >= 2: {lines[0]!r}")
    record_name = rec[0]
    if "/" in record_name:
        raise ReadError(f"{hea_path}: multi-segment records are not supported ({record_name!r})")
    n_sig = _int(rec[1], "n_sig", hea_path)
    if n_sig < 1:
        raise ReadError(f"{hea_path}: n_sig must be >= 1, got {n_sig}")

    fs: float = 250.0  # WFDB default when the field is absent
    if len(rec) >= 3:
        m = _FS_RE.match(rec[2])
        if m is None:
            raise ReadError(f"{hea_path}: cannot parse sampling frequency {rec[2]!r}")
        fs = float(m.group("fs"))
    if not fs > 0:
        raise ReadError(f"{hea_path}: non-positive fs {fs}")
    n_samp = _int(rec[3], "n_samp", hea_path) if len(rec) >= 4 else 0

    if len(lines) < 1 + n_sig:
        raise ReadError(
            f"{hea_path}: header declares {n_sig} signals but has {len(lines) - 1} signal lines"
        )

    gain = np.empty(n_sig, np.float64)
    baseline = np.empty(n_sig, np.int64)
    adc_zero = np.empty(n_sig, np.int64)
    adc_res = np.empty(n_sig, np.int64)
    init_value = np.empty(n_sig, np.int64)
    checksum = np.empty(n_sig, np.int64)
    block_size = np.empty(n_sig, np.int64)
    has_init = np.zeros(n_sig, bool)  # field actually present in the header
    has_checksum = np.zeros(n_sig, bool)  # ... so verification never invents a target
    units: list[str] = []
    sig_name: list[str] = []
    dat_files: list[str] = []

    for i in range(n_sig):
        tok = lines[1 + i].split()
        if len(tok) < 3:
            raise ReadError(
                f"{hea_path}: signal line {i} has {len(tok)} fields, need >= 3: {lines[1 + i]!r}"
            )
        fname, fmt_tok = tok[0], tok[1]
        dat_files.append(fname)
        if "+" in fname or fname == "-":
            raise ReadError(f"{hea_path}: signal {i} uses an unsupported data source {fname!r}")
        if fmt_tok != "16":
            raise ReadError(
                f"{hea_path}: signal {i} has format {fmt_tok!r}; only plain "
                f"format 16 is supported (no samples-per-frame, skew or offset)"
            )

        gm = _GAIN_RE.match(tok[2])
        if gm is None:
            raise ReadError(f"{hea_path}: signal {i} has an unparsable gain field {tok[2]!r}")
        g = float(gm.group("gain"))
        if g == 0.0:
            g = float(default_gain)  # WFDB: 0 means "unspecified"
        if not np.isfinite(g) or g <= 0.0:
            raise ReadError(f"{hea_path}: signal {i} has non-positive gain {g}")
        gain[i] = g
        unit = gm.group("units") or "mV"
        units.append(unit)

        adc_res[i] = _int(tok[3], f"signal {i} adc_res", hea_path) if len(tok) >= 4 else 16
        adc_zero[i] = _int(tok[4], f"signal {i} adc_zero", hea_path) if len(tok) >= 5 else 0
        b = gm.group("baseline")
        baseline[i] = int(b) if b is not None else adc_zero[i]
        has_init[i] = len(tok) >= 6
        init_value[i] = _int(tok[5], f"signal {i} init_value", hea_path) if has_init[i] else 0
        has_checksum[i] = len(tok) >= 7
        checksum[i] = _int(tok[6], f"signal {i} checksum", hea_path) if has_checksum[i] else 0
        block_size[i] = _int(tok[7], f"signal {i} block_size", hea_path) if len(tok) >= 8 else 0
        sig_name.append(" ".join(tok[8:]) if len(tok) >= 9 else f"sig{i}")

    if len(set(dat_files)) != 1:
        raise ReadError(
            f"{hea_path}: this reader requires one interleaved .dat for all "
            f"signals, header names {sorted(set(dat_files))}"
        )
    if np.any(block_size != 0):
        raise ReadError(f"{hea_path}: non-zero block size is not supported {block_size.tolist()}")
    bad_units = sorted({u for u in units if u.lower() not in UNIT_TO_MV})
    if bad_units:
        raise ReadError(
            f"{hea_path}: unsupported signal units {bad_units} (known: {sorted(UNIT_TO_MV)})"
        )

    dat_file = dat_files[0]
    dat_path = os.path.join(os.path.dirname(hea_path), dat_file)

    n_samp_inferred = False
    if n_samp == 0:
        if not os.path.isfile(dat_path):
            raise ReadError(f"{hea_path}: n_samp unspecified and {dat_path} is missing")
        n_bytes = os.path.getsize(dat_path)
        if n_bytes % (2 * n_sig) != 0:
            raise ReadError(
                f"{dat_path}: size {n_bytes} is not a multiple of 2*{n_sig} bytes per frame"
            )
        n_samp = n_bytes // (2 * n_sig)
        n_samp_inferred = True
    if n_samp <= 0:
        raise ReadError(f"{hea_path}: non-positive n_samp {n_samp}")

    return WfdbHeader(
        record_name=record_name,
        n_sig=n_sig,
        fs=int(fs) if float(fs).is_integer() else fs,
        n_samp=int(n_samp),
        n_samp_inferred=n_samp_inferred,
        hea_path=hea_path,
        dat_file=dat_file,
        dat_path=dat_path,
        fmt=16,
        gain=gain,
        baseline=baseline,
        adc_zero=adc_zero,
        adc_res=adc_res,
        init_value=init_value,
        has_init_value=has_init,
        checksum=checksum,
        has_checksum=has_checksum,
        block_size=block_size,
        units=units,
        unit_to_mv=np.array([UNIT_TO_MV[u.lower()] for u in units], np.float64),
        sig_name=sig_name,
    )


def read_record(
    hea_path: str | os.PathLike,
    *,
    verify_checksum: bool = False,
    invalid_sentinel: int | None = INVALID_SENTINEL_16,
    strict_length: bool = True,
    expected_sig_name: Sequence[str] | None = None,
    default_gain: float = DEFAULT_GAIN,
) -> tuple[np.ndarray, WfdbHeader]:
    """Read one format-16 record.

    Returns `(sig, header)` where `sig` is `(n_samp, n_sig)` float32 in **millivolts**.

    Decode: the `.dat` is one int16 little-endian stream with the leads interleaved
    sample-by-sample, i.e. C-order `(n_samp, n_sig)`. Then
    `sig = (raw - baseline) / gain * unit_to_mv`.

    Arguments
      verify_checksum   also check the per-signal 16-bit checksum and the first-sample
                        `init_value` from the header. Independent validation of the
                        decoder; off by default because it costs a full-array sum.
      invalid_sentinel  raw code treated as "sample missing" (WFDB uses -32768 for
                        format 16). Any occurrence raises. None disables the check.
      strict_length     require the `.dat` to hold exactly `n_samp * n_sig` samples.
                        A short file always raises regardless; this flag only governs
                        whether trailing padding is tolerated.
      expected_sig_name if given, the header's lead names must equal it (case- and
                        whitespace-insensitive). Never reorders: it raises.

    Raises ReadError on any of: malformed header, missing/short/oversized `.dat`,
    checksum or init-value mismatch, sentinel present, non-finite output.
    """
    header = read_header(hea_path, default_gain=default_gain)
    n_samp, n_sig = header.n_samp, header.n_sig
    dat_path = header.dat_path

    if expected_sig_name is not None:
        got = [s.strip().upper() for s in header.sig_name]
        want = [s.strip().upper() for s in expected_sig_name]
        if got != want:
            raise ReadError(f"{dat_path}: lead order {got} != expected {want}")

    if not os.path.isfile(dat_path):
        raise ReadError(f"signal file not found: {dat_path}")
    want_n = n_samp * n_sig
    n_bytes = os.path.getsize(dat_path)
    if n_bytes < 2 * want_n:
        raise ReadError(
            f"{dat_path}: short .dat, {n_bytes} bytes < {2 * want_n} required for "
            f"{n_samp} x {n_sig} int16"
        )
    if strict_length and n_bytes != 2 * want_n:
        raise ReadError(
            f"{dat_path}: .dat is {n_bytes} bytes, expected exactly "
            f"{2 * want_n} ({n_samp} x {n_sig} int16)"
        )

    raw = np.fromfile(dat_path, dtype=FORMAT16_DTYPE, count=want_n)
    if raw.size != want_n:  # short read despite the size check (truncated mid-read)
        raise ReadError(f"{dat_path}: read {raw.size} of {want_n} int16 samples")
    raw = raw.reshape(n_samp, n_sig)

    if invalid_sentinel is not None:
        n_bad = int(np.count_nonzero(raw == invalid_sentinel))
        if n_bad:
            raise ReadError(
                f"{dat_path}: {n_bad} samples equal the WFDB invalid-sample "
                f"sentinel {invalid_sentinel}"
            )

    if verify_checksum:
        # only signals whose header actually carried the field are checked, so a
        # minimal (but legal) header does not produce a spurious ReadError
        ck = header.has_checksum
        got_ck = raw.astype(np.int64).sum(axis=0) % 65536
        want_ck = header.checksum % 65536
        bad = np.flatnonzero(ck & (got_ck != want_ck)).tolist()
        if bad:
            raise ReadError(
                f"{dat_path}: checksum mismatch on signals {bad} "
                f"(got {got_ck[bad].tolist()}, header {want_ck[bad].tolist()})"
            )
        iv = header.has_init_value
        bad = np.flatnonzero(iv & (raw[0].astype(np.int64) != header.init_value)).tolist()
        if bad:
            raise ReadError(
                f"{dat_path}: init_value mismatch on signals {bad} "
                f"(got {raw[0][bad].tolist()}, header {header.init_value[bad].tolist()})"
            )

    scale = (header.unit_to_mv / header.gain).astype(np.float64)
    sig = ((raw.astype(np.float64) - header.baseline) * scale).astype(np.float32)
    if not np.isfinite(sig).all():
        raise ReadError(f"{dat_path}: decoded signal contains non-finite values")
    return sig, header


def write_format16(
    stem: str | os.PathLike,
    x: np.ndarray,
    fs: float,
    *,
    gain: float | Sequence[float] | np.ndarray = 1000.0,
    baseline: int | Sequence[int] | np.ndarray = 0,
    adc_zero: int | Sequence[int] | np.ndarray = 0,
    adc_res: int = 16,
    units: str = "mV",
    sig_name: Sequence[str] | None = None,
    x_is_raw: bool = False,
) -> tuple[str, str]:
    """Write `<stem>.hea` + `<stem>.dat` as a single-segment format-16 record.

    TEST SUPPORT ONLY -- winder never writes WFDB in production. It exists so the reader
    can be validated by exact round-trip without the `wfdb` package.

    `x` is `(T, n_sig)`. With `x_is_raw=True` it is int-valued ADU written verbatim
    (this is the path that makes round-trip *exact*); otherwise it is physical values
    in `units`, quantised as `round(x * gain + baseline)`. Values outside int16 range
    raise ValueError rather than wrapping. Header `checksum` and `init_value` are
    computed so that `read_record(..., verify_checksum=True)` exercises them.

    Returns `(hea_path, dat_path)`.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be (T, n_sig), got shape {x.shape}")
    T, n_sig = x.shape
    g = np.broadcast_to(np.asarray(gain, np.float64), (n_sig,)).copy()
    b = np.broadcast_to(np.asarray(baseline, np.int64), (n_sig,)).copy()
    z = np.broadcast_to(np.asarray(adc_zero, np.int64), (n_sig,)).copy()
    if sig_name is None:
        sig_name = [f"sig{i}" for i in range(n_sig)]
    if len(sig_name) != n_sig:
        raise ValueError(f"sig_name has {len(sig_name)} entries for {n_sig} signals")

    raw = x.astype(np.int64) if x_is_raw else np.rint(x.astype(np.float64) * g + b).astype(np.int64)
    if x_is_raw and not np.array_equal(raw, x):
        raise ValueError("x_is_raw=True requires integer-valued x")
    if raw.min() < -32768 or raw.max() > 32767:
        raise ValueError(f"raw ADU range [{raw.min()}, {raw.max()}] does not fit int16")

    stem = os.fspath(stem)
    hea_path, dat_path = stem + ".hea", stem + ".dat"
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    raw.astype(FORMAT16_DTYPE).tofile(dat_path)

    name = os.path.basename(stem)
    fs_tok = str(int(fs)) if float(fs).is_integer() else repr(float(fs))
    ck = raw.sum(axis=0) % 65536
    lines = [f"{name} {n_sig} {fs_tok} {T}"]
    for i in range(n_sig):
        gtok = f"{g[i]:g}({b[i]})/{units}"
        lines.append(f"{name}.dat 16 {gtok} {adc_res} {z[i]} {raw[0, i]} {ck[i]} 0 {sig_name[i]}")
    with open(hea_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return hea_path, dat_path

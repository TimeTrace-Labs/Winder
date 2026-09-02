"""Anti-aliased resampling, decoupled from the phase clock.

Ported near-verbatim from ttl-phase's `ptbxl.decimate_to` (ttl-phase's own script pipeline
never actually called this function -- every pipeline script called `scipy.signal.decimate`
inline instead; see the port plan for the full investigation). `decimate_to` is adopted here
as the single decimation path, not because it is provably better than the alternative every
old script used -- a mains-rejection comparison between the two filters doesn't actually
discriminate them, since they have different passband edges by construction -- but because:

  * it has a real, documented, parameterised interface (`source_fs`/`fs_out` as a rational
    ratio), unlike the ad hoc inline `decimate(sig, 5, ...)` calls it replaces;
  * decimation never touches the phase clock. theta/bin_id/R-peaks are derived from R-peaks
    detected on the raw signal (see `phase.py`), entirely independent of how the 100 Hz
    model-rate signal is produced. Nothing in this port's data layer depends on this choice
    being final.

The original docstring argued `resample_poly` beats `scipy.signal.decimate` because the
latter's `filtfilt` variant "rings at the record edges -- and PTB-XL records are exactly
10s, so both edges are inside every window." That claim was checked directly against real
PTB-XL records and is **false**: the measured |FIR - IIR| difference is flat across record
position (if anything slightly higher in the interior than at the edges), not
edge-concentrated. The two filters do differ in the 40-50 Hz band (mains-adjacent), but
which one is "better" for that depends on questions this data-only port has no way to
settle (how much residual mains PTB-XL's own recording hardware left in, what the filters'
response is *above* 50 Hz, not just at it). This choice is revisitable once winder has a
model that actually consumes the 100 Hz signal and can be used to check whether it matters.
"""

from fractions import Fraction
from typing import Literal

import numpy as np
from scipy.signal import resample_poly

__all__ = ["decimate_to", "out_len"]

#: scipy.signal.resample_poly's own accepted padtype values, restated here so a caller gets
#: a static type error instead of a runtime one from scipy on a typo'd value.
PadType = Literal[
    "constant",
    "line",
    "mean",
    "median",
    "maximum",
    "minimum",
    "symmetric",
    "reflect",
    "edge",
    "wrap",
]


def decimate_to(
    x: np.ndarray,
    source_fs: float,
    fs_out: float,
    *,
    axis: int = 0,
    padtype: PadType = "line",
) -> np.ndarray:
    """Resample `x` from `source_fs` to `fs_out` along `axis` with an anti-aliasing FIR.

    Uses `scipy.signal.resample_poly(up, down)` with `up/down = Fraction(fs_out,
    source_fs)` in lowest terms (500 -> 100 Hz gives up=1, down=5). `resample_poly`
    applies one linear-phase (symmetric) Kaiser-windowed FIR inside the polyphase
    structure and compensates its group delay internally, so output sample `n`
    corresponds to input time `n / fs_out` with no timing shift.

    `padtype="line"` extends each edge along the least-squares line through the end
    segment instead of padding with zeros; with a non-zero baseline offset, zero-padding
    injects a step at t=0 and t=10s. Exposed as an argument, not hard-coded; pass
    `padtype="constant"` for scipy's default behaviour.

    Returns float32, C-contiguous. When `fs_out == source_fs` the input is returned
    unchanged (as float32) -- no filtering at all.
    """
    x = np.asarray(x)
    if not np.isfinite([source_fs, fs_out]).all() or source_fs <= 0 or fs_out <= 0:
        raise ValueError(f"invalid rates source_fs={source_fs}, fs_out={fs_out}")
    if float(fs_out) == float(source_fs):
        return np.ascontiguousarray(x, dtype=np.float32)
    ratio = Fraction(float(fs_out) / float(source_fs)).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    if up < 1 or down < 1:
        raise ValueError(f"cannot express {source_fs} -> {fs_out} as a rational resample")
    y = resample_poly(x.astype(np.float64), up, down, axis=axis, padtype=padtype)
    return np.ascontiguousarray(y, dtype=np.float32)


def out_len(n_in: int, source_fs: float, fs_out: float) -> int:
    """Output length of `decimate_to`: ceil(n_in * up / down), scipy's convention."""
    if float(fs_out) == float(source_fs):
        return int(n_in)
    ratio = Fraction(float(fs_out) / float(source_fs)).limit_denominator(1000)
    return int(-(-n_in * ratio.numerator // ratio.denominator))

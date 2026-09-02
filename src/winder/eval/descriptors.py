"""Token/R-peak timestamp arithmetic shared by the phase-calibration pipeline.

Ported from ttl-phase's `src/winder/eval/descriptors.py`, scoped down to exactly the functions
`scripts/build_phase_tokens.py` (M0 calibration) needs: rescaling R-peak sample positions from the
native 500 Hz phase-clock grid (`artifacts/phase/rpeaks.npz`) onto the 100 Hz grid
`EcgWindowDataset` actually feeds the model, and mapping a token index to the raw sample whose
theta value that token is calibrated against.

Scope narrowed from the source module -- deferred, not ported (no consumer in winder-nominal
yet): `amplitude_at_tokens`, `local_rms_amplitude`, `time_since_previous_rpeak`,
`distance_to_nearest_rpeak`, `heart_rate_bucket`/`HEART_RATE_BUCKETS`. `winder.eval.robustness`
already inlines its own copy of `heart_rate_bucket` for exactly this reason (see that module's
docstring), and `winder.eval.detection` already inlines its own copy of `rpeaks_at_output_rate`
for the same reason -- both predate this module and are left as-is; this module exists so
`build_phase_tokens.py` has a properly tested, importable home for the token-timestamp arithmetic
rather than a third inlined copy.

All functions here are report/calibration-only (CON-02, CM-08): none is called anywhere on the
training path (`winder.jepa.train`, `winder.jepa.masking`, `winder.jepa.regularizers`), and none
of their outputs feed sampling, model inputs, or loss.
"""

from typing import Literal, cast

import numpy as np

from winder.data.phase import phase_from_rpeaks

__all__ = [
    "load_rpeaks_by_ecg_id",
    "rpeaks_at_output_rate",
    "token_last_sample",
    "token_centre_sample",
    "theta_at_tokens",
]


def load_rpeaks_by_ecg_id(npz_path: str) -> dict[int, np.ndarray]:
    """`ecg_id -> native (rpeaks.npz's own fs, 500 Hz) R-peak sample positions`, unpacked from
    `scripts/build_manifest.py`'s ragged `(rpeaks, offsets, ecg_ids)` archive."""
    z = np.load(npz_path)
    ecg_ids, offsets, rpeaks = z["ecg_ids"], z["offsets"], z["rpeaks"]
    return {
        int(ecg_ids[i]): rpeaks[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(ecg_ids))
    }


def rpeaks_at_output_rate(rpeaks_native: np.ndarray, decimation_factor: float) -> np.ndarray:
    """Rescale R-peak sample positions from the native acquisition rate onto the decimated
    model-input grid (native / decimation_factor).

    Exact, not an approximation: `winder.data.decimation.decimate_to`'s own contract is "output
    sample n corresponds to input time n / fs_out with no timing shift", so a native-rate sample
    index and a decimated-rate sample index describing the same instant are related by this one
    linear rescale. `winder.data.phase.phase_from_rpeaks`'s theta is itself a ratio of two
    differences of sample indices (`(t - R_i) / (R_{i+1} - R_i)`), so rescaling every index
    (query `t` and every `R`) by the same factor leaves theta exactly unchanged -- decimation
    changes the sampling grid theta is evaluated on, never the theta value at a given instant.
    """
    return np.asarray(rpeaks_native, dtype=np.float64) / float(decimation_factor)


def _patch_window(j: int, *, patch_width: int) -> tuple[int, int]:
    """Inclusive raw-sample range `(first, last)` patch `j` covers under `PatchEncoder`
    (architecture-primer.html §5-6): `[j*patch_width, (j+1)*patch_width - 1]`, exactly -- no
    receptive-field run-in beyond the patch's own samples."""
    first = j * patch_width
    last = first + patch_width - 1
    return first, last


def token_last_sample(j: int, *, patch_width: int = 8) -> int:
    """The one raw sample index token `j`'s value depends on: the last sample of its own patch,
    `(j+1)*patch_width - 1`. Used as this token's own "timestamp" for any time-indexed descriptor
    (theta lookup, R-peak distance) -- under `PatchEncoder` (architecture-primer.html §5-6) a
    token's ENTIRE dependence is its own patch (`_patch_window`), so "most recently reflects" and
    "depends on at all" coincide at this one boundary sample.
    """
    return _patch_window(j, patch_width=patch_width)[1]


def token_centre_sample(j: int, *, patch_width: int = 8) -> float:
    """The centre raw-sample index of token `j`'s own patch: `first + (patch_width - 1) / 2`
    (e.g. 3.5 samples into an 8-sample patch) -- a token's mean position in time, not its causal
    upper bound. Used only where that distinction matters: `winder.transport`'s phase
    demodulation relies on theta being each token's own MEAN phase across the patch it summarises
    (Prop 4.2 of notes/internal/phase_equivariance_notes_v13.pdf), not the causal boundary
    `token_last_sample` correctly uses for causal, time-indexed descriptors. Using
    `token_last_sample` for demodulation instead would leak a per-record, heart-rate-dependent
    rotation (offset from centre is a fixed 3.5 samples = 35 ms in absolute time, but a
    HR-dependent fraction of the cardiac cycle) into what demodulation assumes is a common
    cross-record frame -- measured at 0.056 rad standard deviation across records, growing to
    ~0.39 rad at the top retained harmonic (n=7)."""
    first, last = _patch_window(j, patch_width=patch_width)
    return (first + last) / 2.0


def theta_at_tokens(
    rpeaks_native: np.ndarray,
    n_tokens: int,
    n_samples: int,
    *,
    decimation_factor: float = 5.0,
    timestamp: Literal["last", "centre"] = "last",
) -> np.ndarray:
    """theta at every token `j`'s own timestamp sample, `(n_tokens,)` -- NaN where that sample
    falls outside every enclosing R-R interval (`winder.data.phase.phase_from_rpeaks`'s own
    convention: no theta before the first R-peak or after the last one).

    `timestamp="last"` (default, unchanged behaviour) uses `token_last_sample` -- the causally
    correct choice for a time-indexed descriptor. `timestamp="centre"` uses `token_centre_sample`
    rounded to the nearest integer sample -- see that function's docstring for why the
    transport/demodulation path needs it instead.

    `decimation_factor` defaults to 5.0 (500 Hz `rpeaks.npz` -> `EcgWindowDataset`'s 100 Hz grid,
    DATA-04's ratio) -- pass the caller's own if it differs.
    """
    rpeaks_out = rpeaks_at_output_rate(rpeaks_native, decimation_factor)
    theta_full = phase_from_rpeaks(rpeaks_out, n_samples)[:, 0]
    if timestamp == "last":
        raw_positions = [token_last_sample(j) for j in range(n_tokens)]
    elif timestamp == "centre":
        raw_positions = [round(token_centre_sample(j)) for j in range(n_tokens)]
    else:
        raise ValueError(f"timestamp must be 'last' or 'centre', got {timestamp!r}")
    positions = np.clip(np.array(raw_positions), 0, n_samples - 1)
    # cast: numpy fancy indexing is typed as Any in this stub configuration.
    return cast(np.ndarray, theta_full[positions])

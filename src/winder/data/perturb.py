"""Clinically-plausible, GROUND-TRUTHED perturbations of an ECG window, for time-localisation
evaluation.

Every function here returns the altered waveform together with an exact per-sample and per-token
mask of what it changed, so a detector's output can be scored against a known answer. Real ECG
pathology carries no such timestamp: this is the only way to ask "when did the model notice?"
rather than "did the model's record-level score go up?".

**What this measures, and what it does not.** A synthetic lesion is a controlled perturbation of a
signal, not a case of disease. A detector that localises an injected ST shift has demonstrated
that its latent responds, with known timing and known amplitude, to a change of the SHAPE that
ischemia produces. It has NOT demonstrated clinical detection performance, because real ischemia
co-occurs with rate change, axis shift, reciprocal changes in other leads, and patient-specific
baselines that no injection reproduces. Report these as sensitivity/localisation characteristics
of the representation, never as diagnostic accuracy.

**Amplitudes are specified in millivolts** and converted per-lead using the same
`winder.data.norm_stats.LeadStats.std_mv` the dataset normalised with, so a "0.1 mV ST elevation"
here is the same physical quantity as the clinical threshold it is named after, on every lead,
regardless of that lead's own scale.

The three families differ in what they localise, and the difference is the experiment:

  PHASE-restricted, time-onset  `st_shift`, `t_wave_inversion` -- confined to one arc of the
      cardiac cycle, present from a chosen beat onward. A phase-aware detector should have an
      advantage here that a phase-blind one cannot have, because "abnormal for THIS point in the
      cycle" is exactly the comparison it makes.
  BEAT-local                    `ectopic_beat` -- one whole beat replaced. Localised in time to
      ~1 RR, spans all phases.
  TIME-window, phase-blind      `lead_dropout`, `baseline_wander`, `amplitude_attenuation` --
      a contiguous stretch of samples, no phase structure at all. `baseline_wander` in particular
      is a SPECIFICITY control: it is a large, obvious signal change with no diagnostic meaning,
      so a detector that flags it as loudly as an ST shift is measuring "something changed", not
      "something clinically relevant changed".
"""

import math
from dataclasses import dataclass

import torch

__all__ = [
    "Perturbation",
    "PERTURBATIONS",
    "st_shift",
    "t_wave_inversion",
    "ectopic_beat",
    "lead_dropout",
    "baseline_wander",
    "amplitude_attenuation",
    "token_mask_from_samples",
]

TWO_PI = 2.0 * math.pi

#: ST segment and T wave as arcs of the cardiac cycle, in radians from the R-peak (theta = 0).
#: Derived from this cohort's OWN measured ensemble beat (scripts/p1_panel_numerics.py's
#: `ensemble_beat`: QRS at 43 ms, T peak at 253 ms, P at 701 ms, RR 842.6 ms), not from a textbook
#: template -- the ST arc runs from the end of the QRS complex to the foot of the T wave, and the
#: T arc brackets the measured T peak.
ST_ARC = (0.55, 1.35)  # ~74-181 ms after R
T_ARC = (1.35, 2.60)  # ~181-349 ms after R, containing the measured T peak at 1.89 rad


@dataclass(frozen=True)
class Perturbation:
    """`waveform` is `(N, n_leads, n_samples)`, altered in place of the input (never a view of
    it). `sample_mask`/`token_mask` are True exactly where a sample/token was touched;
    `onset_sample` is each record's first touched sample, or -1 for a record left unperturbed
    (e.g. one with no usable R-peaks)."""

    name: str
    waveform: torch.Tensor
    sample_mask: torch.Tensor  # (N, n_samples) bool
    token_mask: torch.Tensor  # (N, n_tokens) bool
    onset_sample: torch.Tensor  # (N,) long
    amplitude_mv: float
    leads: tuple[int, ...]


def token_mask_from_samples(
    sample_mask: torch.Tensor, n_tokens: int, patch_width: int
) -> torch.Tensor:
    """`(N, n_samples) -> (N, n_tokens)`: a token is marked perturbed if ANY sample in its own
    patch was altered.

    "Any", not "most": the encoder is causal with a receptive field WIDER than one patch
    (`winder.jepa.encoder`), so a token whose patch is only partly altered has still seen the
    alteration in full. Requiring a majority would label such tokens negative and silently
    penalise a detector for correctly firing on them.
    """
    usable = n_tokens * patch_width
    trimmed = sample_mask[:, :usable]
    return trimmed.reshape(sample_mask.shape[0], n_tokens, patch_width).any(dim=-1)


def _empty_masks(
    waveform: torch.Tensor, n_tokens: int, patch_width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, _leads, n_samples = waveform.shape
    sample_mask = torch.zeros((n, n_samples), dtype=torch.bool)
    token_mask = torch.zeros((n, n_tokens), dtype=torch.bool)
    onset = torch.full((n,), -1, dtype=torch.long)
    return sample_mask, token_mask, onset


def _finalise(
    name: str,
    waveform: torch.Tensor,
    sample_mask: torch.Tensor,
    n_tokens: int,
    patch_width: int,
    amplitude_mv: float,
    leads: tuple[int, ...],
) -> Perturbation:
    onset = torch.full((sample_mask.shape[0],), -1, dtype=torch.long)
    any_row = sample_mask.any(dim=1)
    if bool(any_row.any()):
        onset[any_row] = sample_mask[any_row].float().argmax(dim=1)
    return Perturbation(
        name=name,
        waveform=waveform,
        sample_mask=sample_mask,
        token_mask=token_mask_from_samples(sample_mask, n_tokens, patch_width),
        onset_sample=onset,
        amplitude_mv=amplitude_mv,
        leads=leads,
    )


def _z_scale(
    amplitude_mv: float, lead_std_mv: torch.Tensor, leads: tuple[int, ...]
) -> torch.Tensor:
    """mV -> the dataset's own per-lead z-scored units, `(len(leads),)`."""
    std = lead_std_mv[list(leads)].clamp_min(1e-6)
    return torch.as_tensor(amplitude_mv, dtype=torch.float32) / std


def _onset_sample_from_fraction(theta: torch.Tensor, fraction: float) -> torch.Tensor:
    """The sample index at `fraction` of the way through each record's own VALID theta span.

    Anchored to the valid span rather than to the raw window so the onset always falls where a
    cardiac phase is actually defined -- a lesion injected before the first R-peak would have no
    phase, could not be phase-restricted, and would be scored against a detector that (correctly)
    ignores phase-less tokens.
    """
    valid = torch.isfinite(theta)
    n, n_samples = theta.shape
    out = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        idx = valid[i].nonzero(as_tuple=True)[0]
        out[i] = idx[int(fraction * (len(idx) - 1))] if len(idx) > 1 else n_samples
    return out


def _arc_mask(theta: torch.Tensor, arc: tuple[float, float]) -> torch.Tensor:
    """Samples whose cardiac phase lies in `arc`, handling an arc that wraps through 0."""
    lo, hi = arc
    finite = torch.isfinite(theta)
    th = torch.where(finite, theta, torch.zeros_like(theta))
    inside = (th >= lo) & (th < hi) if lo <= hi else (th >= lo) | (th < hi)
    return inside & finite


def _taper(theta: torch.Tensor, arc: tuple[float, float]) -> torch.Tensor:
    """A raised-cosine weight that is 0 at both arc edges and 1 at its centre.

    A hard-edged rectangular injection introduces a step discontinuity, which is a
    broadband high-frequency event and is trivially detectable by ANY high-pass-sensitive
    statistic -- the detector would be scored on an artefact of the injection rather than on the
    shape change being modelled. Tapering keeps the perturbation inside the signal's own
    bandwidth.
    """
    lo, hi = arc
    span = (hi - lo) if hi > lo else (hi + TWO_PI - lo)
    th = torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta))
    rel = torch.remainder(th - lo, TWO_PI) / span
    return torch.where(
        (rel >= 0) & (rel <= 1), 0.5 - 0.5 * torch.cos(TWO_PI * rel), torch.zeros_like(rel)
    )


# ============================================================ phase-restricted, time-onset


def st_shift(
    waveform: torch.Tensor,
    theta: torch.Tensor,
    lead_std_mv: torch.Tensor,
    *,
    amplitude_mv: float = 0.1,
    leads: tuple[int, ...] = (1, 6, 7, 8),
    onset_fraction: float = 0.4,
    n_tokens: int,
    patch_width: int,
) -> Perturbation:
    """Tapered additive offset over the ST arc of every beat from `onset_fraction` onward.

    The flagship perturbation. 0.1 mV of ST deviation in two contiguous leads is the standard
    threshold for the ECG diagnosis of acute ischemia, so the amplitude axis of any sensitivity
    curve built on this function reads directly against clinical practice. Default leads are
    II, V1, V2, V3 (inferior plus anteroseptal) -- a contiguous territory, not a random set.
    """
    out = waveform.clone()
    sample_mask, _tm, _on = _empty_masks(waveform, n_tokens, patch_width)
    onset = _onset_sample_from_fraction(theta, onset_fraction)
    after = torch.arange(theta.shape[1]).unsqueeze(0) >= onset.unsqueeze(1)
    region = _arc_mask(theta, ST_ARC) & after
    weight = _taper(theta, ST_ARC) * region
    scale = _z_scale(amplitude_mv, lead_std_mv, leads)
    for k, lead in enumerate(leads):
        out[:, lead, :] = out[:, lead, :] + scale[k] * weight
    sample_mask |= region
    return _finalise("st_shift", out, sample_mask, n_tokens, patch_width, amplitude_mv, leads)


def t_wave_inversion(
    waveform: torch.Tensor,
    theta: torch.Tensor,
    lead_std_mv: torch.Tensor,
    *,
    amplitude_mv: float = 0.2,
    leads: tuple[int, ...] = (1, 6, 7, 8),
    onset_fraction: float = 0.4,
    n_tokens: int,
    patch_width: int,
) -> Perturbation:
    """Tapered SUBTRACTION over the T arc, from `onset_fraction` onward: drives an upright T wave
    down and, at sufficient amplitude, through zero into inversion.

    Implemented as a subtraction rather than a sign flip of the measured T wave because a flip
    requires knowing each record's own T amplitude, which varies by a factor of several across the
    cohort -- a fixed subtraction in mV keeps the injected quantity comparable across records,
    which is what an amplitude sensitivity curve needs.
    """
    out = waveform.clone()
    sample_mask, _tm, _on = _empty_masks(waveform, n_tokens, patch_width)
    onset = _onset_sample_from_fraction(theta, onset_fraction)
    after = torch.arange(theta.shape[1]).unsqueeze(0) >= onset.unsqueeze(1)
    region = _arc_mask(theta, T_ARC) & after
    weight = _taper(theta, T_ARC) * region
    scale = _z_scale(amplitude_mv, lead_std_mv, leads)
    for k, lead in enumerate(leads):
        out[:, lead, :] = out[:, lead, :] - scale[k] * weight
    sample_mask |= region
    return _finalise(
        "t_wave_inversion", out, sample_mask, n_tokens, patch_width, amplitude_mv, leads
    )


# ================================================================================ beat-local


def ectopic_beat(
    waveform: torch.Tensor,
    theta: torch.Tensor,
    lead_std_mv: torch.Tensor,
    *,
    amplitude_mv: float = 1.0,
    leads: tuple[int, ...] = tuple(range(12)),
    onset_fraction: float = 0.5,
    n_tokens: int,
    patch_width: int,
) -> Perturbation:
    """One beat replaced by a wide, monophasic complex -- a premature ventricular contraction
    surrogate.

    The beat containing `onset_fraction` of the valid span is overwritten across its whole cycle
    with a single broad deflection (one half-period of a cosine over the QRS-to-T portion of the
    cycle, opposite in sign to the record's own mean deflection in that lead). A real PVC also
    arrives EARLY and resets the RR sequence; that is not reproduced here, because doing so would
    also change theta itself and the detector would then be reacting to a clock change rather than
    to a morphology change. This is therefore a morphology-only PVC surrogate, and it is the
    conservative version of the test.
    """
    out = waveform.clone()
    sample_mask, _tm, _on = _empty_masks(waveform, n_tokens, patch_width)
    onset = _onset_sample_from_fraction(theta, onset_fraction)
    n, _leads, n_samples = waveform.shape
    idx = torch.arange(n_samples)
    scale = _z_scale(amplitude_mv, lead_std_mv, leads)

    for i in range(n):
        start = int(onset[i])
        if start >= n_samples:
            continue
        # This beat runs from the chosen sample's own R-peak to the next one: find where theta
        # wraps back through 0 after `start`.
        th = theta[i]
        rest = idx[(idx >= start) & torch.isfinite(th)]
        if len(rest) < 4:
            continue
        wraps = (th[rest[1:]] < th[rest[:-1]]).nonzero(as_tuple=True)[0]
        stop = int(rest[int(wraps[0]) + 1]) if len(wraps) else int(rest[-1]) + 1
        span = slice(start, stop)
        width = stop - start
        if width < 4:
            continue
        bump = torch.sin(torch.linspace(0, math.pi, width))
        for k, lead in enumerate(leads):
            sign = -torch.sign(out[i, lead, span].mean()) or 1.0
            out[i, lead, span] = out[i, lead, span] + sign * scale[k] * bump
        sample_mask[i, span] = True
    return _finalise("ectopic_beat", out, sample_mask, n_tokens, patch_width, amplitude_mv, leads)


# =================================================================== time-window, phase-blind


def lead_dropout(
    waveform: torch.Tensor,
    theta: torch.Tensor,
    lead_std_mv: torch.Tensor,
    *,
    amplitude_mv: float = 0.0,
    leads: tuple[int, ...] = (6,),
    onset_fraction: float = 0.4,
    duration_s: float = 2.0,
    fs: int = 100,
    n_tokens: int,
    patch_width: int,
) -> Perturbation:
    """One lead attenuated toward flat for `duration_s` -- a detached or saturated electrode.

    `amplitude_mv` is reinterpreted as the FRACTION of the lead removed (1.0 = full flatline),
    because a replacement has no millivolt scale. Crucially it must still be a genuine no-op at
    0.0: the driver prepends an amplitude-0 SHAM run to every perturbation to measure the
    detector's own phase-dependent baseline, and a function that flatlines regardless of
    amplitude would make its own sham indistinguishable from its real condition.
    """
    out = waveform.clone()
    sample_mask, _tm, _on = _empty_masks(waveform, n_tokens, patch_width)
    onset = _onset_sample_from_fraction(theta, onset_fraction)
    width = int(duration_s * fs)
    n_samples = waveform.shape[2]
    for i in range(waveform.shape[0]):
        start = min(int(onset[i]), max(0, n_samples - width))
        span = slice(start, min(start + width, n_samples))
        for lead in leads:
            out[i, lead, span] = out[i, lead, span] * (1.0 - amplitude_mv)
        sample_mask[i, span] = True
    return _finalise("lead_dropout", out, sample_mask, n_tokens, patch_width, amplitude_mv, leads)


def baseline_wander(
    waveform: torch.Tensor,
    theta: torch.Tensor,
    lead_std_mv: torch.Tensor,
    *,
    amplitude_mv: float = 0.3,
    leads: tuple[int, ...] = tuple(range(12)),
    onset_fraction: float = 0.4,
    duration_s: float = 2.0,
    wander_hz: float = 0.3,
    fs: int = 100,
    n_tokens: int,
    patch_width: int,
) -> Perturbation:
    """A tapered low-frequency excursion -- respiration or motion.

    THE SPECIFICITY CONTROL. At 0.3 mV this is a LARGER absolute signal change than the 0.1 mV ST
    shift, and it carries no diagnostic meaning whatsoever. A detector that ranks this at or above
    the ST shift is reporting "the signal changed", which is not the product claim. Read every
    other perturbation's detection score against this one, not against zero.
    """
    out = waveform.clone()
    sample_mask, _tm, _on = _empty_masks(waveform, n_tokens, patch_width)
    onset = _onset_sample_from_fraction(theta, onset_fraction)
    width = int(duration_s * fs)
    n_samples = waveform.shape[2]
    scale = _z_scale(amplitude_mv, lead_std_mv, leads)
    envelope = 0.5 - 0.5 * torch.cos(TWO_PI * torch.arange(width) / max(width - 1, 1))
    wave = torch.sin(TWO_PI * wander_hz * torch.arange(width) / fs) * envelope
    for i in range(waveform.shape[0]):
        start = min(int(onset[i]), max(0, n_samples - width))
        stop = min(start + width, n_samples)
        seg = wave[: stop - start]
        for k, lead in enumerate(leads):
            out[i, lead, start:stop] = out[i, lead, start:stop] + scale[k] * seg
        sample_mask[i, start:stop] = True
    return _finalise(
        "baseline_wander", out, sample_mask, n_tokens, patch_width, amplitude_mv, leads
    )


def amplitude_attenuation(
    waveform: torch.Tensor,
    theta: torch.Tensor,
    lead_std_mv: torch.Tensor,
    *,
    amplitude_mv: float = 0.5,
    leads: tuple[int, ...] = (1, 6, 7),
    onset_fraction: float = 0.4,
    duration_s: float = 2.0,
    fs: int = 100,
    n_tokens: int,
    patch_width: int,
) -> Perturbation:
    """Tapered amplitude reduction over a window -- poor electrode contact. `amplitude_mv` is
    reinterpreted as the FRACTION removed at the window's centre (0.5 = halved), since an
    attenuation is multiplicative and has no natural millivolt scale."""
    out = waveform.clone()
    sample_mask, _tm, _on = _empty_masks(waveform, n_tokens, patch_width)
    onset = _onset_sample_from_fraction(theta, onset_fraction)
    width = int(duration_s * fs)
    n_samples = waveform.shape[2]
    envelope = 0.5 - 0.5 * torch.cos(TWO_PI * torch.arange(width) / max(width - 1, 1))
    for i in range(waveform.shape[0]):
        start = min(int(onset[i]), max(0, n_samples - width))
        stop = min(start + width, n_samples)
        seg = 1.0 - amplitude_mv * envelope[: stop - start]
        for lead in leads:
            out[i, lead, start:stop] = out[i, lead, start:stop] * seg
        sample_mask[i, start:stop] = True
    return _finalise(
        "amplitude_attenuation", out, sample_mask, n_tokens, patch_width, amplitude_mv, leads
    )


#: name -> (function, default amplitude sweep, family). The sweep values are what the sensitivity
#: figure's x-axis uses; the first entry of each is chosen at or below the clinical threshold so
#: the curve shows where detection FAILS, not only where it succeeds.
PERTURBATIONS: dict[str, tuple[object, tuple[float, ...], str]] = {
    "st_shift": (st_shift, (0.025, 0.05, 0.1, 0.2, 0.4), "phase-restricted"),
    "t_wave_inversion": (t_wave_inversion, (0.05, 0.1, 0.2, 0.4, 0.8), "phase-restricted"),
    "ectopic_beat": (ectopic_beat, (0.25, 0.5, 1.0, 2.0), "beat-local"),
    "lead_dropout": (lead_dropout, (0.25, 0.5, 1.0), "time-window"),
    "baseline_wander": (baseline_wander, (0.1, 0.2, 0.3, 0.6), "time-window (control)"),
    "amplitude_attenuation": (amplitude_attenuation, (0.25, 0.5, 0.75), "time-window"),
}

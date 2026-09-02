"""A minimal synthetic waveform generator for `--synthetic` smoke runs and tests.

Not a realistic ECG model -- see `tests/_synthetic.py::synthetic_ecg` for that, used where
signal realism actually matters (e.g. `winder.data.phase`'s R-peak detector). This exists only to
exercise the training pipeline's shapes and control flow without the real corpus, at the exact,
fixed `(n_leads, n_samples)` shape a smoke run requires. Lives in `winder.jepa`, not `tests/`,
specifically so `scripts/s2_pretrain_jepa.py` (production code) can use it without a
production-depends-on-tests layering violation.
"""

import torch

__all__ = ["synthetic_waveform_batch"]


def synthetic_waveform_batch(
    batch_size: int,
    *,
    n_leads: int = 12,
    n_samples: int = 1000,
    generator: torch.Generator,
) -> torch.Tensor:
    """`(batch_size, n_leads, n_samples)` float32 -- a low-frequency sinusoid (a crude
    heartbeat-rate stand-in) plus Gaussian noise, structured enough that a JEPA predictor has
    something non-trivial to predict, unlike pure noise."""
    t = torch.linspace(0, 10, n_samples).view(1, 1, n_samples)
    freq = torch.empty(batch_size, n_leads, 1).uniform_(0.8, 1.5, generator=generator)
    phase = torch.empty(batch_size, n_leads, 1).uniform_(0, 2 * torch.pi, generator=generator)
    signal = torch.sin(2 * torch.pi * freq * t + phase)
    noise = torch.randn(batch_size, n_leads, n_samples, generator=generator) * 0.1
    return (signal + noise).float()

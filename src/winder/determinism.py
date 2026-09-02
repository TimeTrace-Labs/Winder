"""Torch seeding convention: explicit generators, never `torch.manual_seed`.

This project's numpy determinism doctrine ("explicit, never global" --
`winder.data.phase.jitter_estimate`, `winder.data.folds.calibration_subset`'s
`np.random.default_rng(seed)` passed as an argument, never a global seed) extends here to torch.
No function in `winder.jepa` calls `torch.manual_seed`; every source of randomness (parameter
init, mask sampling, SIGReg's projection directions, dropout) takes an explicit
`torch.Generator`.

Named streams (`generator(seed, stream)`) so one source of randomness cannot silently shift
another's draws -- SIGReg's directions must not depend on how many mask draws happened first, or
adding a new mask-sampler variant would change every SIGReg value downstream of it for reasons
having nothing to do with masking. Per-stream seeds are derived via blake2b, not an arithmetic
offset from the base seed: offsets collide easily (stream "mask" at seed 5 must never coincide
with an unrelated stream at some other seed for a hand-picked constant); a cryptographic hash
does not.
"""

import hashlib

import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out

__all__ = ["generator", "init_parameters"]


def generator(seed: int, stream: str, *, device: str | torch.device = "cpu") -> torch.Generator:
    """A fresh, reproducible `torch.Generator` for one named stream of one run's seed.

    Same `(seed, stream)` always gives the same generator state; different streams at the same
    seed give independent, uncorrelated states (via blake2b -- see module docstring).
    """
    digest = hashlib.blake2b(f"{stream}:{seed}".encode(), digest_size=8).digest()
    stream_seed = int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)
    gen = torch.Generator(device=device)
    gen.manual_seed(stream_seed)
    return gen


def _reset_linear_like(m: nn.Linear | nn.Conv1d, gen: torch.Generator) -> None:
    """Reproduces `nn.Linear`/`nn.Conv1d`'s own default `reset_parameters()` distribution
    (kaiming-uniform weight, fan-in-scaled uniform bias) but draws from `gen` instead of the
    global RNG."""
    nn.init.kaiming_uniform_(m.weight, a=5**0.5, generator=gen)
    if m.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / fan_in**0.5 if fan_in > 0 else 0.0
        nn.init.uniform_(m.bias, -bound, bound, generator=gen)


def init_parameters(module: nn.Module, gen: torch.Generator) -> None:
    """Re-initialize every parameter of `module` (recursively) from `gen`, reproducing each
    layer type's own default init *distribution* but drawing from an explicit generator instead
    of torch's global RNG.

    A module with bespoke parameters not covered by a known layer type (e.g. a raw
    `nn.Parameter` such as a mask-embedding or a relative-position-bias table) may implement an
    optional `reset_parameters_deterministic(self, gen)` method; this function calls it if
    present, in preference to the built-in layer-type handling below.

    Raises on any parameter-holding module with neither a recognized layer type nor that hook --
    a new primitive with learnable parameters must be wired in deliberately (mirroring
    `winder.data.manifest.REASON_CODES`'s closed-vocabulary doctrine), not silently left at
    whatever the global RNG produced at construction time.
    """
    for m in module.modules():
        reset = getattr(m, "reset_parameters_deterministic", None)
        if callable(reset):
            reset(gen)
            continue
        if isinstance(m, nn.Linear | nn.Conv1d):
            _reset_linear_like(m, gen)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif list(m.parameters(recurse=False)):
            raise ValueError(
                f"init_parameters does not know how to deterministically initialize "
                f"{type(m).__name__!r}: it owns parameters not covered by a known layer type "
                f"and defines no reset_parameters_deterministic method. Add an explicit case "
                f"here, or implement that method on the module itself, rather than letting it "
                f"silently keep global-RNG-seeded construction-time values."
            )

"""Generator-seeded dropout: an approved deviation from the design spec's plain `nn.Dropout`.

`nn.Dropout` draws its mask from torch's global RNG, with no `generator` argument -- exactly the
leak this project's determinism doctrine ("explicit, never global",
`winder.data.folds.calibration_subset`'s `np.random.default_rng(seed)`) exists to prevent. The
design spec wants `dropout=0.1` in the predictor for realistic training; this module makes that
compatible with the doctrine instead of banning dropout outright.

Each `SeededDropout` instance owns its own `torch.Generator`, seeded once at construction and
advanced on every training-mode forward call. The generator's state is exposed through
`nn.Module.get_extra_state`/`set_extra_state` (the standard PyTorch hook for exactly this kind of
non-tensor state, e.g. quantization observers), so a plain `torch.save(model.state_dict(), ...)`
/`load_state_dict(...)` checkpoints and restores it automatically -- "checkpointed RNG
continuation" comes for free from the ordinary checkpoint path, not a bespoke one.

The bernoulli draw itself always happens on CPU (a CPU-seeded `torch.Generator` cannot drive a
CUDA tensor's RNG op directly) and is moved to the input's device afterwards; for dropout-sized
tensors this cost is negligible and it avoids the separate machinery a CUDA generator would need.
"""

import torch
from torch import nn

__all__ = ["SeededDropout"]


class SeededDropout(nn.Module):
    def __init__(self, p: float, *, seed: int) -> None:
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout probability p must be in [0, 1), got {p}")
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full(x.shape, keep_prob), generator=self.generator)
        return x * mask.to(device=x.device, dtype=x.dtype) / keep_prob

    def get_extra_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_extra_state(self, state: object) -> None:
        assert isinstance(state, torch.Tensor)
        self.generator.set_state(state)

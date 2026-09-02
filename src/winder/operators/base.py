"""The learned-operator interface.

Scope note: this covers *trained* operators only (an `nn.Module` fit by the transport loss).
Gate-0's closed-form Procrustes fit is a separate, non-learned code path (numpy, fit-then-use,
no optimisation) and deliberately does not implement this interface -- forcing it in here would
give it a no-op `transport`/`closure_residual` split it doesn't need. If a closed-form fitter needs
a shared contract later, give it its own protocol rather than widening this one.

Named `transport`, not `apply`: `nn.Module.apply(fn)` already means "recursively apply fn to
submodules" -- an override with an incompatible signature under the same name is a real footgun,
not just a naming clash (mypy flags it: signature incompatible with supertype).
"""

from abc import ABC, abstractmethod

import torch


class TransportOperator(ABC):
    @abstractmethod
    def transport(self, z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """R_delta @ z, batched over delta. z: (..., K), delta: (...,) matching z's leading
        dims -> (..., K). Concrete implementations accept any number of leading dims (e.g.
        (N, K)/(N,) for a flat pair batch, or (B, T, K)/(B, T) for a per-token call) as long as
        the two shapes agree outside the trailing K axis."""

    @abstractmethod
    def closure_residual(self) -> torch.Tensor:
        """||R_{2*pi} - I||_F (or the arm's equivalent identity-return diagnostic)."""

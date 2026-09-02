"""The six single-concern JEPA primitive contracts.

Six independently swappable ingredients assemble into one JEPA: an `Encoder` maps a raw waveform
to per-token embeddings; a `ProjectionHead` reshapes those for the objective; a `MaskSampler`
splits each record's tokens into context/target; a `Predictor` guesses the target tokens'
embeddings from the context; a `PredictionLoss` scores that guess; a `Regularizer` fights
representation collapse. None of the six is where `winder.operators.TransportOperator` attaches --
that decoupling (README, CLAUDE.md) is preserved by construction: nothing here references phase,
`R_delta`, or a closure constraint.

Split by statefulness, not by convenience. `Encoder`, `ProjectionHead`, and `Predictor` hold
learned parameters and declare `forward`; every concrete implementation multiply-inherits
`(nn.Module, <ABC>)`, mirroring `winder.operators.base.TransportOperator` / `CyclicOperator`.
`MaskSampler`, `PredictionLoss`, and `Regularizer` are parameter-free and declare `__call__`
directly on a *plain* ABC, not `nn.Module`: `nn.Module.__call__` is typed `Callable[..., Any]`,
which would fight `warn_return_any` at every call site for a primitive with no parameters to
register in the first place. Concrete `Encoder`/`ProjectionHead`/`Predictor` instances are called
via `.forward(...)` explicitly throughout this package (never bare `()`), a deliberate,
documented choice to bypass `nn.Module`'s hook machinery -- this MVP registers no hooks, and the
explicit call keeps mypy able to see the ABC's own return type instead of `nn.Module.__call__`'s
`Any`.

Two contracts widen their wording slightly beyond a first sketch, both for the same reason:
`ProjectionHead` declares `input_width`/`output_width` and `Predictor` declares `width`, so
`winder.jepa.model.assemble_jepa`'s handshake can raise an actionable, named-field error before a
mismatched pair ever reaches a real forward pass. `Regularizer` deliberately declares no width at
all -- SIGReg (this MVP's only real implementation) is width-agnostic, so that link is verified by
a successful dummy call rather than a declared number; a future width-sensitive regularizer would
need one.

`Regularizer.__call__` and `MaskSampler.__call__` both take an explicit `generator` keyword. This
is the determinism doctrine ("explicit, never global", `winder.data.folds.calibration_subset`'s
`np.random.default_rng(seed)`, extended here to torch): SIGReg's random directions and each
record's mask must be resampled every call *and* be reproducible from a passed-in
`torch.Generator`, never `torch.manual_seed`'s global state.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from winder.jepa.masking import CausalMaskPlan

__all__ = [
    "Encoder",
    "ProjectionHead",
    "Predictor",
    "MaskSampler",
    "PredictionLoss",
    "Regularizer",
]


class Encoder(ABC):
    @property
    @abstractmethod
    def latent_width(self) -> int:
        """Width K of one token embedding this encoder produces."""

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, n_leads, n_samples) float32 -> (B, n_tokens, latent_width) per-token embeddings.

        Called exactly once per training step, on the unmasked waveform (`winder.jepa.train`'s
        module docstring): a causal implementation's token `j` already depends on no sample after
        its own timestamp, so there is no separate masked-context pass to run. This is not a
        target/EMA encoder -- the same instance (identical weights) produces both the context and
        target views a `MaskSampler`'s plan distinguishes downstream."""


class ProjectionHead(ABC):
    @property
    @abstractmethod
    def input_width(self) -> int:
        """Width this projector expects as input; must equal its paired encoder's
        `latent_width`."""

    @property
    @abstractmethod
    def output_width(self) -> int:
        """Width of one projected token embedding."""

    @abstractmethod
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, input_width) -> (B, n_tokens, output_width), applied per token."""


class Predictor(ABC):
    @property
    @abstractmethod
    def width(self) -> int:
        """Embedding width this predictor consumes and produces; must equal its paired
        projector's `output_width`."""

    @abstractmethod
    def forward(self, z_ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, width), (B, n_tokens) bool -> (B, n_tokens, width).

        Tokens keep their natural sequence order throughout -- there is no context/target
        concatenation and no separate index bookkeeping. `mask[b, s] = True` marks a position the
        predictor must not read its own input value at -- in this MVP, every position at or after
        a `CausalMaskPlan`'s cutoff (`~plan.context`), not only the target block itself; an
        implementation replaces that position's input with a learned embedding before predicting.
        A causal implementation must additionally ensure no masked position's own prediction is
        influenced by an unmasked position that comes after it in sequence order (CM-02). The
        caller reads back only the target positions of the output; other masked positions of the
        output are not part of this contract."""


class MaskSampler(ABC):
    @abstractmethod
    def __call__(
        self, batch_size: int, n_tokens: int, *, generator: torch.Generator
    ) -> "CausalMaskPlan":
        """-> a `CausalMaskPlan`: a sampled context cutoff, gap, and target length per record, in
        both boolean-mask and per-record scalar form (see `winder.jepa.masking`'s module
        docstring). Every target index is strictly after its own record's cutoff -- this is a
        forecasting split, not an infilling one.

        One independent draw per record in the batch (not one draw shared across the batch), and
        a fresh draw on every call -- both are load-bearing properties of the intended sampling
        distribution, not implementation details."""


class PredictionLoss(ABC):
    @abstractmethod
    def __call__(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, n_tokens, K), (B, n_tokens, K), (B, n_tokens) bool -> 0-dim scalar.

        Scored over masked positions only. `target` is never detached by anything in this
        package -- see `winder.jepa.model`'s module docstring for why."""


class Regularizer(ABC):
    @abstractmethod
    def __call__(self, z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        """(N, K) -> 0-dim scalar anti-collapse penalty, OR (T, N, K) -> the same statistic
        computed independently per leading-axis slice and averaged over T (the per-timestep
        reduction LeJEPA/LeWorldModel use, N = batch size -- architecture-primer.html §7).

        Deliberately silent on what N indexes (tokens, records, views -- the caller decides);
        this keeps the regularizer itself agnostic to that choice. Any internal randomness
        (e.g. SIGReg's projection directions) is drawn fresh from `generator` on every call,
        never from the global torch RNG -- shared across every slice of a 3-D call, not
        redrawn per slice (the shared-directions default (architecture-primer.html §9))."""

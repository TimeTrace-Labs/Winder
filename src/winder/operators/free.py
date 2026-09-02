from dataclasses import dataclass, field

from winder.operators.harmonic import HarmonicTransport

__all__ = ["FreeOperatorConfig", "FreeOperator"]


@dataclass
class FreeOperatorConfig:
    """The free arm: omega real-valued and learnable, initialised at the same integer n_j the
    cyclic arm freezes at -- makes Assumption 3 (closure) a one-parameter empirical test rather
    than an assumption. `lr`/`weight_decay` are read by the training driver to build a SEPARATE
    optimizer param group for omega (never the model's shared AdamW group): joint weight decay
    on omega is a direct force toward the trivial omega -> 0 solution -- the predecessor
    prototype's free arm collapsed to exactly this (omega ~ 0.086) under shared decay. Defaulting
    `weight_decay=0.0` here is deliberate, not an oversight.

    (k0, n_j, k_j) defaults mirror `CyclicOperatorConfig`'s own -- see that class's docstring for
    the M0-calibration derivation. Both arms start from the exact same spectrum; only
    `learnable_omega` (set by which class you instantiate, not by these fields) differs.
    """

    k0: int = 4
    n_j: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    k_j: list[int] = field(default_factory=lambda: [21, 21, 21, 21, 21, 21])
    lr: float = 1e-3
    weight_decay: float = 0.0


class FreeOperator(HarmonicTransport):
    """Unconstrained-frequency operator: omega learnable, so closure (R_{2*pi} = I) is a claim
    about the trained checkpoint, not a property of the parameterisation -- see
    `HarmonicTransport`'s module docstring for the shared maths."""

    def __init__(self, config: FreeOperatorConfig) -> None:
        n_j = [int(v) for v in config.n_j]
        k_j = [int(v) for v in config.k_j]
        super().__init__(int(config.k0), n_j, k_j, learnable_omega=True)
        self.config = config

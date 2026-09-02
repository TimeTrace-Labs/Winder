from dataclasses import dataclass, field

from winder.operators.harmonic import HarmonicTransport

__all__ = ["CyclicOperatorConfig", "CyclicOperator"]


@dataclass
class CyclicOperatorConfig:
    """The closure-constrained arm: omega frozen at n_j, so R_{2*pi} = I exactly by
    parameterisation (Assumption 3 as a design choice, not a claim to be tested on this arm).

    Defaults are `scripts/m0_phase_calibration.py`'s actual output on this cohort's full
    corpus (21577 included records; artifacts/phase/m0_calibration.json), applying that
    script's own pre-registered rounding rule to a measured sigma_theta = 0.1722 rad
    (n_max_raw = 1/sigma_theta = 5.81 -> n_max = 6, dividing the 126-dim harmonic budget evenly
    at k_j = 21 each, no remainder) -- NOT the note's own worked K0=4/n_max=7/k_j=18 example,
    and NOT the discredited ttl-phase values (n_j=[11,7,6,3,2,1,1], k0=2) this class previously
    carried: those were fitted on an untrained K=16 filterbank in a predecessor prototype, are
    gapped and repeated (violating Eq. 23's contiguity constraint), and do not correspond to any
    measurement on this cohort. Re-run that script (and update these defaults) if the phase
    pipeline, cohort, or K0 design choice ever changes.
    """

    k0: int = 4
    n_j: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    k_j: list[int] = field(default_factory=lambda: [21, 21, 21, 21, 21, 21])


class CyclicOperator(HarmonicTransport):
    """Closure-constrained operator: R_delta -> I as delta -> 2*pi (winding number 1), exactly,
    by construction -- see `HarmonicTransport`'s module docstring for the shared maths."""

    def __init__(self, config: CyclicOperatorConfig) -> None:
        n_j = [int(v) for v in config.n_j]
        k_j = [int(v) for v in config.k_j]
        super().__init__(int(config.k0), n_j, k_j, learnable_omega=False)
        self.config = config

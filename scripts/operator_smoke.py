"""P1 smoke check: instantiate every registered transport operator at its own production
defaults and report the two cheapest closed-form diagnostics available offline -- `dimension`
(expected 256, the M0-calibrated spectrum) and `closure_residual()` (||R_{2*pi} - I||_F, expected
~0 for both arms at construction time; see `winder.operators.harmonic.HarmonicTransport`).

Not a replacement for `tests/test_operators.py`'s theory-derived assertions -- this is a
human-readable spot check for `scripts/`, exercising the exact same registry the tests do.
"""

from winder.operators.harmonic import HarmonicTransport
from winder.operators.registry import OPERATOR_REGISTRY


def main() -> int:
    """Print dimension and closure residual for every operator in OPERATOR_REGISTRY at defaults."""
    print(f"{'name':>8} {'dimension':>10} {'closure_residual':>18}")
    for name, (schema_cls, operator_ctor) in OPERATOR_REGISTRY.items():
        operator = operator_ctor(schema_cls())
        residual = float(operator.closure_residual().detach())
        # `dimension` is a `HarmonicTransport`-specific property, not part of the
        # `TransportOperator` ABC (see winder.operators.base) -- narrow before reading it rather
        # than widening the shared interface just for this diagnostic script.
        dimension = operator.dimension if isinstance(operator, HarmonicTransport) else "n/a"
        print(f"{name:>8} {dimension:>10} {residual:>18.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

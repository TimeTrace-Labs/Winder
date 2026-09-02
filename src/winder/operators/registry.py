from collections.abc import Callable
from typing import Any

from winder.operators.base import TransportOperator
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig

# name -> (config schema, operator constructor). Typed as Callable rather than
# type[TransportOperator] so mypy checks each constructor's own config-argument type instead of
# collapsing them all to TransportOperator's (nonexistent) __init__ signature.
#
# OmegaConf structured configs can't express Union[FreeOperatorConfig, CyclicOperatorConfig]
# directly (unions of containers are unsupported), so arm configs carry a string tag resolved
# through this registry instead -- see src/winder/config.py.
OPERATOR_REGISTRY: dict[str, tuple[type, Callable[[Any], TransportOperator]]] = {
    "free": (FreeOperatorConfig, FreeOperator),
    "cyclic": (CyclicOperatorConfig, CyclicOperator),
}


def build_operator(name: str, operator_config: object) -> TransportOperator:
    _, operator_ctor = OPERATOR_REGISTRY[name]
    return operator_ctor(operator_config)

"""
Signal-agnostic exploration stack.

`GenericForwardValuator` + `GenericRetrospectiveValuator` perform the
paired counterfactual rollouts described in the DIAL paper (Sec 4.1):
fork the env, execute optimizer action and base action, label which
yields higher return. `GenericOracleCollector` collects ground-truth
utility on demand. `GenericVoCEstimator` aggregates signal/utility
pairs for the explore-phase dataset D.
"""
from dial.explore.valuator import (
    GenericForwardValuator,
    GenericRetrospectiveValuator,
)
from dial.explore.oracle import GenericOracleCollector
from dial.explore.voc import GenericVoCEstimator

__all__ = [
    "GenericForwardValuator",
    "GenericRetrospectiveValuator",
    "GenericOracleCollector",
    "GenericVoCEstimator",
]

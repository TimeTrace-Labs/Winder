"""Evaluation: pairwise transport gain, phase-kNN purity, closure/identity diagnostics.

Also the frozen-encoder linear probe (`metrics.py`, `probe.py`, `record.py`) added for the JEPA
MVP -- the transport-gain/kNN-purity/closure diagnostics named above are not yet implemented
(`TransportOperator` itself is still a stub); nothing here depends on them.
"""

from winder.eval.metrics import auroc_binary, macro_auroc, r_squared
from winder.eval.probe import (
    FittedProbe,
    FittedRegressionProbe,
    LinearProbeConfig,
    decision_scores,
    embed_records,
    fit_linear_probe,
    fit_linear_regression_probe,
    patient_bootstrap_ci,
    regression_predictions,
)
from winder.eval.record import EvalRecord

__all__ = [
    "auroc_binary",
    "macro_auroc",
    "r_squared",
    "LinearProbeConfig",
    "FittedProbe",
    "embed_records",
    "fit_linear_probe",
    "decision_scores",
    "FittedRegressionProbe",
    "fit_linear_regression_probe",
    "regression_predictions",
    "patient_bootstrap_ci",
    "EvalRecord",
]

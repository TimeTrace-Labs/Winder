"""Per-checkpoint operator/geometry/gain reports -- the JSON-serialisable summaries every
downstream panel figure and eval numeric reads. Promoted verbatim from
`scripts/p1_panel_numerics.py`'s own script-local functions (`operator_report`, `geometry_report`,
`gain_report`) into a real, importable, unit-tested library module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from winder.operators.harmonic import HarmonicTransport
from winder.transport.delta_gain import cluster_bootstrap_mean, delta_stratified_gain
from winder.transport.geometry import (
    harmonic_loop_projection,
    phase_resolved_trajectory,
    pooled_geometry_report,
)

__all__ = ["N_PHASE_BINS", "operator_report", "geometry_report", "gain_report"]

#: The clinical staging figure's bin count (p1_panel_numerics.py's own default): 8 bins is
#: ~105 ms at PTB-XL's cohort-median ~842.6 ms RR interval.
N_PHASE_BINS = 8


def operator_report(name: str, operator: HarmonicTransport | None) -> dict[str, Any]:
    """The operator's own spectrum, learned omega (if `learnable_omega`), and closed-form
    closure residual -- `None` if the checkpoint declares no transport arm at all."""
    if operator is None:
        return {"checkpoint": name, "has_operator": False}
    omega = operator.omega.detach().cpu().tolist()
    return {
        "checkpoint": name,
        "has_operator": True,
        "learnable_omega": operator.learnable_omega,
        "k0": operator.k0,
        "n_j": list(operator.n_j),
        "k_j": [int(v) for v in operator.k_j.tolist()],
        "dimension": operator.dimension,
        "omega": omega,
        "omega_minus_n": [w - n for w, n in zip(omega, operator.n_j, strict=True)],
        "distance_to_nearest_integer": [abs(w - round(w)) for w in omega],
        "closure_residual": float(operator.closure_residual().detach()),
    }


def geometry_report(
    z: torch.Tensor, theta: torch.Tensor, operator: HarmonicTransport
) -> dict[str, Any]:
    """Before/after-demodulation pooled geometry (`winder.transport.geometry.
    pooled_geometry_report`), the phase-resolved trajectory, and each harmonic's gauge-invariant
    loop projection -- a block with zero energy in the reference bin reports its own error string
    rather than aborting the whole report."""
    pooled = pooled_geometry_report(z, theta, operator)
    traj = phase_resolved_trajectory(z, theta, operator, n_bins=N_PHASE_BINS)
    binned = torch.tensor(traj["binned_means"], dtype=torch.float64)
    loops = {}
    for j in range(len(operator.n_j)):
        try:
            loops[str(operator.n_j[j])] = harmonic_loop_projection(
                binned, operator, harmonic_index=j
            )
        except ValueError as exc:  # a block with zero energy in the reference bin
            loops[str(operator.n_j[j])] = {"error": str(exc)}
    return {
        "pooled": pooled,
        "trajectory": {k: v for k, v in traj.items() if k != "binned_means"},
        "binned_means_invariant_block": binned[:, : operator.k0].tolist(),
        "loops": loops,
    }


def gain_report(
    z: torch.Tensor,
    theta: torch.Tensor,
    operator: HarmonicTransport,
    patient_ids: np.ndarray,
    *,
    n_strata: int,
    seed: int,
) -> dict[str, Any]:
    """Delta-stratified transport gain plus a patient-clustered bootstrap CI on the per-record
    mean gain."""
    res = delta_stratified_gain(z, theta, operator, n_strata=n_strata)
    boot = cluster_bootstrap_mean(
        np.asarray(res.per_record_mean_gain),
        patient_ids[np.asarray(res.record_index, dtype=int)],
        n_replicates=1000,
        seed=seed,
    )
    return {
        "n_strata": res.n_strata,
        "delta_centers": res.delta_centers,
        "mean_gain": res.mean_gain,
        "gain_fraction": res.gain_fraction,
        "mean_transported_cos": res.mean_transported_cos,
        "mean_identity_cos": res.mean_identity_cos,
        "n_pairs": res.n_pairs,
        "overall_mean_gain": res.overall_mean_gain,
        "overall_gain_fraction": res.overall_gain_fraction,
        "bootstrap": boot,
    }

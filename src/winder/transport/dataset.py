"""`PhaseTaggedDataset` wraps `winder.jepa.dataset.EcgWindowDataset` (never modifies it) to
additionally carry each token's own theta -- the per-token cardiac phase the transport loss
needs (`winder.transport.loss`) and mean pooling does not.

Composition, not inheritance or modification: this class holds a base `EcgWindowDataset` and a
theta lookup, and delegates every field the base dataset already produces unchanged --
`tests/test_transport_dataset.py` locks this in as a `torch.equal` identity, not an assumption.

A record whose `ecg_id` has no entry in the theta lookup (e.g. excluded from
`scripts/m0_phase_calibration.py`'s phase-QC pool -- `HIGH_RR_CV` and friends) still gets an item
here, with an all-NaN theta row: `winder.jepa.dataset`'s own module docstring is explicit that
phase-clock QC exclusions "have nothing to do with a JEPA that has no phase clock," and
`winder.transport.loss` already excludes every NaN-theta token from its own computation -- so an
all-NaN row costs that record exactly nothing beyond its own (correctly zero) contribution to
`L_trans`, while it keeps contributing to `L_pred`/`L_sig` exactly as it did before this wrapper
existed.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from winder.jepa.dataset import EcgWindowDataset, EcgWindowItem

__all__ = ["PhaseTaggedItem", "PhaseTaggedDataset", "load_theta_tokens"]


class PhaseTaggedItem(EcgWindowItem):
    theta: torch.Tensor  # (n_tokens,) float32, NaN where phase is undefined for that token


def load_theta_tokens(npz_path: str) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """`ecg_id -> (n_tokens,) float32 theta row`, plus the archive's own metadata (`patch_width`,
    `n_tokens`, `decimation_factor`, `timestamp` convention) -- unpacked from
    `scripts/m0_phase_calibration.py`'s `theta_tokens.npz`."""
    z = np.load(npz_path)
    ecg_ids, theta = z["ecg_ids"], z["theta"]
    by_id = {int(ecg_ids[i]): theta[i] for i in range(len(ecg_ids))}
    meta = {
        "patch_width": int(z["patch_width"]),
        "n_tokens": int(z["n_tokens"]),
        "decimation_factor": float(z["decimation_factor"]),
        "timestamp": str(z["timestamp"]),
    }
    return by_id, meta


class PhaseTaggedDataset(Dataset[PhaseTaggedItem]):
    """`theta_by_ecg_id`/`theta_meta` are exactly `load_theta_tokens`'s own return value -- the
    grid-agreement check (`n_tokens`/`patch_width` against what the caller's own built
    `PatchEncoder` actually uses) runs ONCE here at construction, not per `__getitem__` call, and
    raises immediately with named fields rather than silently reading a mismatched archive.
    """

    def __init__(
        self,
        base: EcgWindowDataset,
        theta_by_ecg_id: dict[int, np.ndarray],
        theta_meta: dict[str, object],
        *,
        n_tokens: int,
        patch_width: int,
    ) -> None:
        if theta_meta["n_tokens"] != n_tokens:
            raise ValueError(
                f"theta_tokens.npz was built at n_tokens={theta_meta['n_tokens']}, but this "
                f"dataset expects n_tokens={n_tokens} (the built PatchEncoder's own token "
                f"count) -- re-run scripts/m0_phase_calibration.py, or check the caller's config."
            )
        if theta_meta["patch_width"] != patch_width:
            raise ValueError(
                f"theta_tokens.npz was built at patch_width={theta_meta['patch_width']}, but "
                f"this dataset expects patch_width={patch_width} -- re-run "
                f"scripts/m0_phase_calibration.py, or check the caller's config."
            )
        self.base = base
        self.theta_by_ecg_id = theta_by_ecg_id
        self.n_tokens = n_tokens

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> PhaseTaggedItem:
        item = self.base[index]
        theta_row = self.theta_by_ecg_id.get(item["ecg_id"])
        if theta_row is None:
            theta = torch.full((self.n_tokens,), float("nan"), dtype=torch.float32)
        else:
            theta = torch.from_numpy(np.asarray(theta_row, dtype=np.float32))
        return {**item, "theta": theta}

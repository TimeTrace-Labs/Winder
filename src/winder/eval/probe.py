"""The frozen linear probe: mean-pooled predictor hidden states, projector output discarded.

`embed_records` pools every token (architecture-primer.html §5-6: no floor exclusion needed under
the non-overlapping `PatchEncoder`, unlike the retired `winder.jepa.leakage.valid_token_floor`'s
exclusion of early tokens under the overlapping `ResidualCnnEncoder`). the probe repointing: the
pooling TARGET is `JepaModel.predictor_hidden_states`, not `embed`'s encoder-only output -- under
`PatchEncoder`, `embed` is a local, context-free 80ms patch descriptor, so probing it would not
be reading the sequence model's own representation of the record so far. The encoder itself is
still frozen and still the only thing `predictor_hidden_states` derives from (no gradient
anywhere on this path); "projector AND predictor discarded" no longer holds -- only the
projector's output is discarded now, the predictor's is exactly what is probed.

`fit_linear_probe` is PyTorch AdamW, exactly as the design spec (Sec 15.4) specifies -- mini-batch
(`batch_size=256` by default), not full-batch, and NOT scikit-learn/LBFGS: that combination is
explicitly out of scope (see the plan's "Approved deviations" -- scikit-learn is a dev-only
AUROC test oracle, never a substitute for this optimizer). Mini-batch training needs a shuffle
order, which is where `seed_probe` gets a real, concrete job -- not the dead parameter an earlier
design pass worried it might be.

`patient_bootstrap_ci` resamples PATIENTS, not records, so a patient's multiple records are
always resampled as a single unit -- matching `winder.data.folds`'s own "the unit of inference is
the patient" convention. Uses `np.random.default_rng(seed_probe)` (numpy's own determinism
doctrine: explicit, never global), since this bootstrap is pure numpy, not torch.

`paired_patient_bootstrap_delta` is PRI-07's primary inference between two arms: exactly ONE
patient-level resample per replicate (via `resample_patient_indices`), reused for both arms, so
patient-level variance shared by both arms cancels out of the delta. Two separate
`patient_bootstrap_ci` calls, one per arm, compared by eye is a materially weaker and different
test -- their independent resamples don't share a patient draw, so that cancellation never
happens (CON-05).

`fit_linear_regression_probe`/`FittedRegressionProbe`/`regression_predictions` (E2-14) are
`fit_linear_probe`'s continuous-target counterpart: the identical structural pattern (train-only
standardization, mini-batch PyTorch AdamW, early stopping on a held-out metric) but MSE loss and
`winder.eval.metrics.r_squared`-based model selection instead of BCE/macro-AUROC -- a genuinely
different loss and selection metric warrants its own function rather than a mode flag bolted
onto `fit_linear_probe`'s classification-shaped one.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from winder.eval.metrics import macro_auroc, r_squared
from winder.jepa.model import JepaModel

__all__ = [
    "LinearProbeConfig",
    "FittedProbe",
    "embed_records",
    "fit_linear_probe",
    "decision_scores",
    "FittedRegressionProbe",
    "fit_linear_regression_probe",
    "regression_predictions",
    "resample_patient_indices",
    "patient_bootstrap_ci",
    "paired_patient_bootstrap_delta",
]


@dataclass
class LinearProbeConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    std_floor: float = 1e-8
    seed_probe: int = 0


@dataclass
class FittedProbe:
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray


def embed_records(
    model: JepaModel,
    waveforms: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Frozen predictor hidden state, eval mode, mean-pooled over every token -- `(n_records,
    width)`, `width` = the predictor's own width (== the projector's `output_width`, by
    `assemble_jepa`'s own handshake), not the encoder's `latent_width`. the probe repointing: the
    probe's pooling target is `JepaModel.predictor_hidden_states`, not `embed`'s local, context-free
    encoder output -- under `PatchEncoder` `embed` carries no temporal context at all, so a probe
    reading it would be scoring an 80ms patch descriptor, not the sequence model's own
    representation of the record so far."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, waveforms.shape[0], batch_size):
            batch = waveforms[start : start + batch_size].to(device)
            hidden = model.predictor_hidden_states(batch)
            pooled = hidden.mean(dim=1)
            outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: LinearProbeConfig,
    *,
    classes: tuple[str, ...],
) -> FittedProbe:
    """Feature standardization fit on `x_train` only, applied fixed to `x_val` (and, at final
    evaluation time, fold 10) -- never refit on data the probe is scored against."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    scale = np.where(std > config.std_floor, std, 1.0)

    def standardize(x: np.ndarray) -> np.ndarray:
        # cast: numpy arithmetic on ndarrays is typed as Any in this stub configuration.
        return cast(np.ndarray, (x - mean) / scale)

    x_train_t = torch.from_numpy(standardize(x_train)).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_val_t = torch.from_numpy(standardize(x_val)).float()

    n, k = x_train_t.shape
    n_classes = y_train_t.shape[1]

    gen = torch.Generator().manual_seed(config.seed_probe)
    weight = torch.zeros(n_classes, k, requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=config.lr, weight_decay=config.weight_decay)

    best_score = -float("inf")
    best_state = (weight.detach().clone(), bias.detach().clone())
    patience_counter = 0

    for _epoch in range(config.max_epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            logits = x_train_t[idx] @ weight.T + bias
            loss = F.binary_cross_entropy_with_logits(logits, y_train_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_scores = (x_val_t @ weight.T + bias).numpy()
        macro, _ = macro_auroc(y_val, val_scores)
        if macro > best_score + config.min_delta:
            best_score = macro
            best_state = (weight.detach().clone(), bias.detach().clone())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    weight_final, bias_final = best_state
    return FittedProbe(
        classes=classes,
        mean=mean,
        scale=scale,
        weight=weight_final.numpy(),
        bias=bias_final.numpy(),
    )


def decision_scores(probe: FittedProbe, x: np.ndarray) -> np.ndarray:
    """`(n_records, K) -> (n_records, n_classes)` logits -- raw scores for `macro_auroc`, which
    only needs a ranking, not calibrated probabilities."""
    x_std = (x - probe.mean) / probe.scale
    # cast: numpy arithmetic on ndarrays is typed as Any in this stub configuration.
    return cast(np.ndarray, x_std @ probe.weight.T + probe.bias)


@dataclass
class FittedRegressionProbe:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: np.ndarray


def fit_linear_regression_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: LinearProbeConfig,
) -> FittedRegressionProbe:
    """`fit_linear_probe`'s continuous-target counterpart (module docstring): feature
    standardization fit on `x_train` only, mini-batch AdamW on MSE loss, early stopping on the
    MEAN (over target columns) of `winder.eval.metrics.r_squared` computed on `(x_val, y_val)` --
    `x_val`/`y_val` should be a calibration split, never the fold a final R^2 is reported on
    (PRI-04's own "probe early stopping uses calibration split, not fold 9" convention).

    `y_train`/`y_val` are `(n, n_targets)`; call once per descriptor (`n_targets=1`) for E2-14's
    per-descriptor table, or with several columns at once if a caller wants one shared probe.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    scale = np.where(std > config.std_floor, std, 1.0)

    def standardize(x: np.ndarray) -> np.ndarray:
        # cast: numpy arithmetic on ndarrays is typed as Any in this stub configuration.
        return cast(np.ndarray, (x - mean) / scale)

    x_train_t = torch.from_numpy(standardize(x_train)).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_val_t = torch.from_numpy(standardize(x_val)).float()

    n, k = x_train_t.shape
    n_targets = y_train_t.shape[1]

    gen = torch.Generator().manual_seed(config.seed_probe)
    weight = torch.zeros(n_targets, k, requires_grad=True)
    bias = torch.zeros(n_targets, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=config.lr, weight_decay=config.weight_decay)

    best_score = -float("inf")
    best_state = (weight.detach().clone(), bias.detach().clone())
    patience_counter = 0

    for _epoch in range(config.max_epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            pred = x_train_t[idx] @ weight.T + bias
            loss = F.mse_loss(pred, y_train_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_pred = (x_val_t @ weight.T + bias).numpy()
        per_target = np.array([r_squared(y_val[:, c], val_pred[:, c]) for c in range(n_targets)])
        score = float(np.nanmean(per_target)) if not np.isnan(per_target).all() else -float("inf")
        if score > best_score + config.min_delta:
            best_score = score
            best_state = (weight.detach().clone(), bias.detach().clone())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    weight_final, bias_final = best_state
    return FittedRegressionProbe(
        mean=mean, scale=scale, weight=weight_final.numpy(), bias=bias_final.numpy()
    )


def regression_predictions(probe: FittedRegressionProbe, x: np.ndarray) -> np.ndarray:
    """`(n_records, K) -> (n_records, n_targets)` continuous predictions."""
    x_std = (x - probe.mean) / probe.scale
    # cast: numpy arithmetic on ndarrays is typed as Any in this stub configuration.
    return cast(np.ndarray, x_std @ probe.weight.T + probe.bias)


def resample_patient_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap draw: sample `len(unique_patients)` patients with replacement, then return
    the concatenated record indices of every record belonging to each sampled patient -- a
    patient's records are always included or excluded as a whole, never split across the draw.
    """
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.flatnonzero(patient_ids == p) for p in sampled])


def patient_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_replicates: int = 2000,
    seed_probe: int = 0,
) -> tuple[float, float, float]:
    """`(point, lo, hi)`: a percentile-95% CI for macro-AUROC via a patient-level bootstrap."""
    point, _ = macro_auroc(y_true, y_score)
    rng = np.random.default_rng(seed_probe)
    boot_scores = np.empty(n_replicates)
    for i in range(n_replicates):
        idx = resample_patient_indices(patient_ids, rng)
        macro, _ = macro_auroc(y_true[idx], y_score[idx])
        boot_scores[i] = macro
    lo, hi = np.nanpercentile(boot_scores, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_patient_bootstrap_delta(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_replicates: int = 2000,
    seed_probe: int = 0,
) -> tuple[float, float, float]:
    """`(point_delta, lo, hi)`: a percentile-95% CI for `macro_auroc(y_score_b) -
    macro_auroc(y_score_a)`, PAIRED across the two arms -- each replicate draws exactly one
    patient-level resample (via `resample_patient_indices`) and scores BOTH arms against that
    same resampled index set, so patient-level variance shared by both arms cancels out of the
    delta. `y_true` and `patient_ids` describe the single eval set both arms were scored on
    (e.g. the same fold-9 records under two checkpoints); `y_score_a`/`y_score_b` are each arm's
    own decision scores on that identical set.
    """
    point_a, _ = macro_auroc(y_true, y_score_a)
    point_b, _ = macro_auroc(y_true, y_score_b)
    point_delta = point_b - point_a
    rng = np.random.default_rng(seed_probe)
    boot_deltas = np.empty(n_replicates)
    for i in range(n_replicates):
        idx = resample_patient_indices(patient_ids, rng)
        macro_a, _ = macro_auroc(y_true[idx], y_score_a[idx])
        macro_b, _ = macro_auroc(y_true[idx], y_score_b[idx])
        boot_deltas[i] = macro_b - macro_a
    lo, hi = np.nanpercentile(boot_deltas, [2.5, 97.5])
    return point_delta, float(lo), float(hi)

"""M4's smoke run: `fit()` at production scale (the real K=256 config `scripts/s2_pretrain_jepa.py`
builds, the real M0-calibrated `CyclicOperator`/`FreeOperator` defaults), on synthetic data, for
a short but non-trivial number of steps -- BEFORE any real GPU-hours are spent on the 15-run
roster (M5). This is the integration-level check that dataset -> theta -> loss -> operator ->
diagnostics actually composes; the unit-level correctness of each piece is `tests/
test_transport_{loss,dataset,diagnostics}.py`'s and `tests/test_operators.py`'s job, not this
file's.
"""

import math
from collections.abc import Iterator

import pytest
import torch

from winder.determinism import generator, init_parameters
from winder.jepa.model import JepaConfig, build_jepa
from winder.jepa.synthetic import synthetic_waveform_batch
from winder.jepa.train import TrainConfig, fit
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.transport.diagnostics import (
    block_energy_participation_ratio,
    k0_energy_fraction,
    ln_gamma_cv,
    omega_summary,
)

N_TOKENS = 125
N_STEPS = 20


def _production_config() -> JepaConfig:
    """Byte-for-byte the same primitives `scripts/s2_pretrain_jepa.py::_default_config` builds
    (patch encoder, K=256) -- the config the real roster (M5) actually trains."""
    return JepaConfig(
        n_tokens=N_TOKENS,
        encoder_name="patch",
        encoder={},
        projector_name="mlp",
        projector={},
        predictor_name="transformer",
        predictor={},
        mask_sampler_name="causal_block",
        mask_sampler={},
        prediction_loss_name="mse",
        prediction_loss={},
        regularizer_name="sigreg",
        regularizer={},
    )


def _synthetic_theta_batches(n_steps: int, batch_size: int, seed: int) -> Iterator[torch.Tensor]:
    """(B, N_TOKENS) uniform-random theta, with a sprinkling of NaN tokens -- exercises the
    exclusion path under real multi-step dynamics, not just a single hand-built batch."""
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_steps):
        theta = torch.rand(batch_size, N_TOKENS, generator=gen) * 2 * math.pi
        drop = torch.rand(batch_size, N_TOKENS, generator=gen) < 0.1  # ~10% missing, like real data
        theta[drop] = float("nan")
        yield theta


def test_smoke_run_cyclic_arm_at_production_scale() -> None:
    config = _production_config()
    model = build_jepa(config, generator=generator(0, "handshake"))
    init_parameters(model, generator(0, "init"))
    operator = CyclicOperator(CyclicOperatorConfig())  # the real M0-calibrated default

    cfg = TrainConfig(n_steps=N_STEPS, lambda_sig=0.15, lambda_trans=1.0, seed_pretrain=0)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": operator.parameters(), "lr": 1e-2, "weight_decay": 0.0},
        ]
    )
    gen_data = generator(0, "synth")
    waveforms = (synthetic_waveform_batch(4, generator=gen_data) for _ in range(N_STEPS))
    thetas = _synthetic_theta_batches(N_STEPS, batch_size=4, seed=0)

    history = fit(model, waveforms, cfg, optimizer, theta_batches=thetas, operator=operator)

    assert len(history) == N_STEPS
    for m in history:
        assert math.isfinite(m.total_loss)
        assert math.isfinite(m.trans_loss)
        assert math.isfinite(m.trans_floor)
        assert math.isfinite(m.trans_gain)
        assert math.isfinite(m.closure_residual)
        assert m.closure_residual < 1e-4  # cyclic: closes by parameterisation, not by training

    # Diagnostics compute cleanly on the final trained state -- this is what M5's real driver
    # would log every N steps, exercised here once at production width/shape.
    with torch.no_grad():
        waveform = synthetic_waveform_batch(4, generator=generator(1, "smoke_check"))
        z = model.projector.forward(model.encoder.forward(waveform))
    frac = k0_energy_fraction(z, operator.k0)
    pr = block_energy_participation_ratio(z, operator)
    summary = omega_summary(operator)
    assert isinstance(model.predictor, torch.nn.Module)  # true of every registered Predictor
    cv = ln_gamma_cv(model.predictor)

    assert 0.0 <= frac <= 1.0
    assert 1.0 <= pr <= operator.k_j.numel() + 1
    assert summary.min_abs_omega == pytest.approx(1.0)  # cyclic: frozen at its integers
    assert math.isfinite(cv)


def test_smoke_run_free_arm_omega_moves_and_stays_finite() -> None:
    config = _production_config()
    model = build_jepa(config, generator=generator(0, "handshake"))
    init_parameters(model, generator(0, "init"))
    operator = FreeOperator(FreeOperatorConfig())  # same calibrated spectrum, learnable omega
    omega_before = operator.omega.detach().clone()

    cfg = TrainConfig(n_steps=N_STEPS, lambda_sig=0.15, lambda_trans=1.0, seed_pretrain=0)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": operator.parameters(), "lr": 1e-2, "weight_decay": 0.0},
        ]
    )
    gen_data = generator(0, "synth")
    waveforms = (synthetic_waveform_batch(4, generator=gen_data) for _ in range(N_STEPS))
    thetas = _synthetic_theta_batches(N_STEPS, batch_size=4, seed=0)

    history = fit(model, waveforms, cfg, optimizer, theta_batches=thetas, operator=operator)

    assert len(history) == N_STEPS
    for m in history:
        assert math.isfinite(m.total_loss)
        assert math.isfinite(m.closure_residual)
    assert not torch.equal(operator.omega.detach(), omega_before)
    assert torch.all(torch.isfinite(operator.omega))

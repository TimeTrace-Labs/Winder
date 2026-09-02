import dataclasses
import itertools
import math
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from winder.determinism import generator, init_parameters
from winder.eval.pooling import demodulated_pool
from winder.jepa import checkpoint
from winder.jepa.encoder import ResidualCnnEncoder
from winder.jepa.masking import CausalMaskPlan
from winder.jepa.model import JepaConfig, JepaModel, build_jepa
from winder.jepa.regularizers import SigReg, SigRegConfig
from winder.jepa.synthetic import synthetic_waveform_batch
from winder.jepa.train import (
    StepMetrics,
    TrainConfig,
    fit,
    lr_schedule,
    record_canonical_templates,
    train_step,
)
from winder.operators.cyclic import CyclicOperator, CyclicOperatorConfig
from winder.operators.free import FreeOperator, FreeOperatorConfig
from winder.transport.loss import transport_loss


def _tiny_config() -> JepaConfig:
    return JepaConfig(
        n_leads=12,
        n_samples=1000,
        n_tokens=250,
        encoder_name="residual_cnn",
        encoder={},
        projector_name="mlp",
        projector={"input_width": 256, "hidden_width": 32, "output_width": 32},
        predictor_name="transformer",
        predictor={"width": 32, "n_heads": 4, "feedforward_width": 64},
        mask_sampler_name="causal_block",
        mask_sampler={},
        prediction_loss_name="mse",
        prediction_loss={},
        regularizer_name="sigreg",
        regularizer={"n_directions": 8, "chunk": 8},
    )


def _build_and_init(seed: int) -> JepaModel:
    config = _tiny_config()
    model = build_jepa(config, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    return model


def test_lr_schedule_warmup_then_cosine_decay() -> None:
    cfg = TrainConfig(lr=1e-3, min_lr=1e-6, warmup_steps=5, n_steps=20)
    warmup = [lr_schedule(s, cfg) for s in range(5)]
    assert warmup == sorted(warmup)
    assert warmup[-1] == pytest.approx(cfg.lr, rel=1e-6)  # peak reached at the last warmup step
    decay = [lr_schedule(s, cfg) for s in range(5, 20)]
    assert decay == sorted(decay, reverse=True)
    assert lr_schedule(19, cfg) == pytest.approx(cfg.min_lr, abs=1e-4)


def test_lr_schedule_is_a_pure_function_of_step() -> None:
    cfg = TrainConfig()
    assert lr_schedule(7, cfg) == lr_schedule(7, cfg)


def test_train_step_losses_are_finite_and_composed_as_a_straight_sum() -> None:
    """L = L_pred + lambda_sig * L_sig -- not a convex combination."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    loss, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
    )
    assert torch.isfinite(loss)
    assert metrics.total_loss == pytest.approx(
        metrics.pred_loss + 0.1 * metrics.sigreg_loss, rel=1e-4
    )
    # Exactly one target token per record (architecture-primer.html §5-6: no gap, no length choices)
    # -- unlike the retired gap/length sampler, context+target now partitions every token EXCEPT
    # whatever sits strictly after the target, which is neither.
    assert 0 < metrics.n_context < 250 * 4
    assert metrics.n_target == 4  # one target token x 4 records
    assert metrics.n_context + metrics.n_target <= 250 * 4
    assert metrics.cutoff_mean >= 0.0
    assert torch.isfinite(torch.tensor(metrics.persistence_loss))
    assert metrics.persistence_loss >= 0.0


def test_persistence_loss_matches_repeating_the_cutoff_latent_by_hand() -> None:
    """the persistence baseline: z_{t+1} = z_t, scored with the same masked loss and
    target mask as pred_loss. Cross-checked here against a from-scratch, by-hand computation
    (encoder+projector, then repeat each record's own cutoff latent) rather than trusting
    `train_step`'s own inline copy of the same formula to be self-consistent."""
    model = _build_and_init(0)
    model.eval()
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    n_tokens = int(model.config.n_tokens)

    _, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
    )

    with torch.no_grad():
        plan = model.mask_sampler(waveform.shape[0], n_tokens, generator=generator(0, "mask"))
        z = model.projector.forward(model.encoder.forward(waveform))
        idx = torch.arange(waveform.shape[0])
        persistence_pred = z[idx, plan.cutoff, :].unsqueeze(1).expand_as(z)
        expected = model.prediction_loss(persistence_pred, z, plan.target)

    assert metrics.persistence_loss == pytest.approx(float(expected), rel=1e-5)


def test_lambda_trans_zero_is_bitwise_identical_whether_theta_and_operator_are_passed() -> None:
    """The transport path's structural skip (winder.jepa.train's module docstring): merely
    PASSING theta/operator alongside lambda_trans=0.0 (the default) must have exactly zero
    effect, proving the skip is a real branch -- not a `+ 0.0 * L_trans` term that happens to
    numerically vanish but would still poison a NaN-containing theta or advance some hidden
    state. Every pre-existing caller (which passes neither) is the special case of this."""
    model_a = _build_and_init(0)
    model_b = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    gen_mask_a, gen_sigreg_a = generator(0, "mask"), generator(0, "sigreg")
    gen_mask_b, gen_sigreg_b = generator(0, "mask"), generator(0, "sigreg")

    loss_a, metrics_a = train_step(
        model_a, waveform, lambda_sig=0.1, gen_mask=gen_mask_a, gen_sigreg=gen_sigreg_a
    )

    # model_b's projector output_width is 32 (see _tiny_config) -- build a matching operator.
    operator = CyclicOperator(CyclicOperatorConfig(k0=4, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250) * 6.28
    theta[0, 0] = float("nan")  # even a NaN present in theta must not leak through at lambda=0
    loss_b, metrics_b = train_step(
        model_b,
        waveform,
        lambda_sig=0.1,
        gen_mask=gen_mask_b,
        gen_sigreg=gen_sigreg_b,
        theta=theta,
        operator=operator,
        lambda_trans=0.0,
    )

    assert torch.equal(loss_a, loss_b)
    # dataclasses.asdict + a NaN-aware comparison: metrics_a == metrics_b would report unequal
    # on `lr`/`grad_norm`/`trans_*` purely because NaN != NaN in IEEE float semantics, not
    # because the two runs actually differ -- both sides carry the SAME "not filled in yet"
    # sentinel at the same fields (train_step, unlike fit(), never sets lr/grad_norm).
    fields_a = dataclasses.astuple(metrics_a)
    fields_b = dataclasses.astuple(metrics_b)
    for field_a, field_b in zip(fields_a, fields_b, strict=True):
        if isinstance(field_a, float) and math.isnan(field_a):
            assert math.isnan(field_b)
        else:
            assert field_a == field_b
    assert gen_mask_a.get_state().equal(gen_mask_b.get_state())
    assert gen_sigreg_a.get_state().equal(gen_sigreg_b.get_state())
    assert math.isnan(metrics_b.trans_loss)
    assert math.isnan(metrics_b.closure_residual)


def test_lambda_trans_nonzero_without_theta_or_operator_raises() -> None:
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    with pytest.raises(ValueError, match="lambda_trans"):
        train_step(
            model,
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            lambda_trans=1.0,
        )


def test_lambda_pred_omitted_reproduces_the_pre_lambda_pred_formula_bit_for_bit() -> None:
    """TrainConfig/train_step's `lambda_pred` default (1.0) must reproduce the exact formula
    `train_step` used before this field existed -- `L = pred_loss + lambda_sig * sigreg_loss`,
    with NO `lambda_pred *` multiplication anywhere in that expression, not merely a call that
    happens to be numerically close. The reference below is a from-scratch, by-hand computation
    of that literal pre-existing expression (mirroring the LEAK-01 test's `_composed` idiom),
    grad-tracked and un-detached throughout -- comparing against a `torch.no_grad()` reference
    would differ at the ULP level (this module's own LEAK-01 docstring), so this stays exactly as
    exposed to autograd as `train_step` itself is.

    `1.0 * x == x` bit-for-bit under IEEE-754 (no rounding: the exponent and mantissa are
    unchanged), so this also certifies that introducing the `lambda_pred *` multiplication for
    the nonzero branch does not perturb the default-weight case even at the bit level -- not just
    that the two code paths happen to agree in this run's specific floats.

    Two SEPARATELY built (but identically seeded) models, not one model called twice -- matching
    `test_lambda_trans_zero_is_bitwise_identical_...`'s own idiom, not LEAK-01's `model.eval()`
    idiom: `SeededDropout` (predictor.py) advances its own internal generator by call count, so
    reusing one `model` for both the hand-built reference and `train_step` would give the
    predictor's second forward pass a different dropout mask than its first, purely from being
    the second call -- a real divergence LEAK-01 avoids with `model.eval()`, not one this test
    wants to introduce."""
    model_ref = _build_and_init(0)
    model_ts = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    n_tokens = int(model_ref.config.n_tokens)
    b = waveform.shape[0]

    plan = model_ref.mask_sampler(b, n_tokens, generator=generator(0, "mask"))
    z = model_ref.projector.forward(model_ref.encoder.forward(waveform))
    sigreg_loss = model_ref.regularizer(z.transpose(0, 1), generator=generator(0, "sigreg"))
    z_hat = model_ref.predictor.forward(z, ~plan.context)
    pred_loss = model_ref.prediction_loss(z_hat, z, plan.target)
    reference_total = pred_loss + 0.1 * sigreg_loss  # the pre-lambda_pred expression, verbatim

    loss, metrics = train_step(
        model_ts,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        # lambda_pred omitted -- must default to 1.0 and take the identical code path as an
        # explicit lambda_pred=1.0 caller.
    )
    assert torch.equal(loss, reference_total)
    assert metrics.pred_loss == pred_loss.item()
    assert metrics.persistence_loss == metrics.persistence_loss  # not NaN: predictor path ran

    cfg_default = TrainConfig()
    assert cfg_default.lambda_pred == 1.0


def test_lambda_pred_zero_sets_pred_and_persistence_metrics_to_nan() -> None:
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    _, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        lambda_pred=0.0,
    )
    assert math.isnan(metrics.pred_loss)
    assert math.isnan(metrics.persistence_loss)
    # unaffected: sigreg and mask-plan bookkeeping are not predictor-derived (module docstring's
    # "mask sampling is already decoupled from the predictor call").
    assert math.isfinite(metrics.sigreg_loss)
    assert metrics.n_context > 0
    assert metrics.n_target == 4


def test_lambda_pred_zero_leaves_the_predictors_own_parameters_ungraded() -> None:
    """The discriminating proof that this is a REAL structural skip and not `0.0 * pred_loss`
    (which would still be finite, still backprop through the predictor, and leave its parameters
    with zero-but-non-None `.grad`): with the predictor forward never called, none of its
    parameters are ever part of `total_loss`'s autograd graph, so `.grad` stays `None` after
    `backward()` -- exactly the condition `winder.jepa.train`'s module docstring already relies on
    ("AdamW naturally skips params with no gradient") to justify not excluding the predictor from
    the optimizer's param groups."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    loss, _ = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        lambda_pred=0.0,
    )
    loss.backward()
    # `Predictor` (winder.jepa.base) is a plain ABC -- `.parameters()` is an nn.Module method every
    # concrete implementation also has, so this narrows the static type the same way the
    # pre-existing `isinstance(model.encoder, ResidualCnnEncoder)` check below narrows `Encoder`.
    assert isinstance(model.predictor, torch.nn.Module)
    predictor_params = list(model.predictor.parameters())
    assert len(predictor_params) > 0  # not a vacuous check
    assert all(p.grad is None for p in predictor_params)
    # Contrast: the shared encoder DOES receive a gradient (from sigreg alone), proving the loss
    # is a real, backprop-reachable tensor and not e.g. an accidental `.detach()`.
    assert isinstance(model.encoder, ResidualCnnEncoder)
    assert model.encoder.stem_conv.weight.grad is not None


def test_lambda_pred_zero_is_bitwise_identical_to_a_hand_built_transport_sigreg_only_step() -> None:
    """The transport-arm analogue of `test_lambda_trans_zero_is_bitwise_identical_...` above, for
    the OTHER structural skip this module now has: `lambda_pred=0.0` with transport active
    (`lambda_trans!=0.0`) must be bitwise identical -- total_loss AND both named generator
    streams' end states -- to a hand-built step that never constructs the predictor's input mask,
    never calls `model.predictor.forward`, and never computes a persistence baseline, only
    `mask_sampler` (still unconditional -- module docstring), `regularizer`, and `transport_loss`.

    Mask sampling still runs and still consumes `gen_mask` in both paths (decoupled from the
    predictor, not deleted), and `regularizer` still consumes `gen_sigreg` in both -- so this
    proves `lambda_pred`'s skip adds no incidental draw from either stream, the same property
    `test_lambda_trans_zero_is_bitwise_identical_...` established for the transport skip."""
    model_a = _build_and_init(0)
    model_b = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    n_tokens = int(model_a.config.n_tokens)
    b = waveform.shape[0]

    gen_mask_a, gen_sigreg_a = generator(0, "mask"), generator(0, "sigreg")
    gen_mask_b, gen_sigreg_b = generator(0, "mask"), generator(0, "sigreg")

    # projector.output_width == 32 (_tiny_config): k0=8 + 2*(4+4+4)=24 -> dimension 32, matching
    # test_fit_with_transport_trains_the_free_operators_own_omega's own K=8+24=32 comment.
    operator_a = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    operator_b = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250) * 6.28

    loss_a, metrics_a = train_step(
        model_a,
        waveform,
        lambda_sig=0.1,
        gen_mask=gen_mask_a,
        gen_sigreg=gen_sigreg_a,
        theta=theta,
        operator=operator_a,
        lambda_trans=1.0,
        lambda_pred=0.0,
    )

    # Hand-built reference: mask_sampler (unused result, matches train_step's own unconditional
    # call) -> projector(encoder(...)) -> regularizer -> transport_loss. No predictor anywhere.
    model_b.mask_sampler(b, n_tokens, generator=gen_mask_b)
    z_b = model_b.projector.forward(model_b.encoder.forward(waveform))
    sigreg_loss_b = model_b.regularizer(z_b.transpose(0, 1), generator=gen_sigreg_b)
    trans_out_b = transport_loss(z_b, theta, operator_b)
    total_loss_b = 0.1 * sigreg_loss_b + 1.0 * trans_out_b.loss

    assert torch.equal(loss_a, total_loss_b)
    assert math.isnan(metrics_a.pred_loss)
    assert math.isnan(metrics_a.persistence_loss)
    assert math.isfinite(metrics_a.trans_loss)
    assert gen_mask_a.get_state().equal(gen_mask_b.get_state())
    assert gen_sigreg_a.get_state().equal(gen_sigreg_b.get_state())


def test_transport_stop_gradient_threads_through_train_step_forward_values_unchanged() -> None:
    """`transport_stop_gradient` must reach `transport_loss` as a BACKWARD-only intervention:
    with identical seeds, every StepMetrics number (and the total loss value) is identical to
    the two-sided run's -- the flag reshapes gradients, never forward values (campaign_x2x2 X4's
    clean-paired-contrast property) -- and the flag-on backward still produces finite gradients."""
    model_a = _build_and_init(0)
    model_b = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250) * 6.28

    loss_a, metrics_a = train_step(
        model_a,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator,
        lambda_trans=1.0,
        transport_radial_weight=0.1,
    )
    loss_b, metrics_b = train_step(
        model_b,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator,
        lambda_trans=1.0,
        transport_radial_weight=0.1,
        transport_stop_gradient=True,
    )

    assert float(loss_a) == float(loss_b)
    assert metrics_a.trans_loss == metrics_b.trans_loss
    assert metrics_a.trans_directional == metrics_b.trans_directional
    assert metrics_a.trans_radial == metrics_b.trans_radial
    assert metrics_a.pred_loss == metrics_b.pred_loss
    loss_b.backward()
    grads = [p.grad for p in model_b.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_transport_radial_weight_threads_through_train_step_and_metrics() -> None:
    """`transport_radial_weight` must reach `transport_loss` (the transport total gains exactly
    the radial component) and surface as `StepMetrics.trans_directional`/`trans_radial` -- the
    campaign_x2x2 baby-run abort criterion reads their ratio from s2_history.jsonl. At the 0.0
    default `trans_radial` is NaN (the "not applicable" sentinel, this file's own trans_* NaN
    convention) and `trans_directional` equals `trans_loss` (the shipped directional-only
    formula, untouched)."""
    model_a = _build_and_init(0)
    model_b = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    # projector.output_width == 32 (_tiny_config): k0=8 + 2*(4+4+4)=24 -> dimension 32.
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250) * 6.28

    _, metrics_a = train_step(
        model_a,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator,
        lambda_trans=1.0,
    )
    _, metrics_b = train_step(
        model_b,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator,
        lambda_trans=1.0,
        transport_radial_weight=1.0,
    )

    assert math.isnan(metrics_a.trans_radial)
    assert metrics_a.trans_directional == metrics_a.trans_loss
    assert math.isfinite(metrics_b.trans_radial)
    assert metrics_b.trans_radial > 0.0  # a random encoder's token norms are not all equal
    # Identical seeds -> identical z -> the directional component is untouched by the radial knob.
    assert metrics_b.trans_directional == metrics_a.trans_directional
    assert metrics_b.trans_loss == pytest.approx(
        metrics_b.trans_directional + 1.0 * metrics_b.trans_radial, rel=1e-6
    )


def test_lambda_pred_nonzero_scales_pred_loss_in_the_straight_sum() -> None:
    """`lambda_pred` is a genuine multiplier on `pred_loss` -- symmetric with `lambda_sig` and
    `lambda_trans`, not merely an on/off gate that always applies an implicit weight of 1 once
    nonzero."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    loss, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        lambda_pred=0.3,
    )
    assert torch.isfinite(loss)
    assert metrics.total_loss == pytest.approx(
        0.3 * metrics.pred_loss + 0.1 * metrics.sigreg_loss, rel=1e-4
    )
    assert math.isfinite(metrics.persistence_loss)  # predictor path still ran at lambda_pred=0.3


def test_train_step_gradients_reach_the_shared_encoder() -> None:
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    loss, _ = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
    )
    loss.backward()
    assert isinstance(model.encoder, ResidualCnnEncoder)
    assert model.encoder.stem_conv.weight.grad is not None
    assert torch.any(model.encoder.stem_conv.weight.grad != 0)


def test_leak01_end_to_end_prefix_invariance_through_train_step() -> None:
    """LEAK-01: perturb every raw sample after context cutoff `c` and check the COMPOSED path
    `train_step` actually runs -- `mask_sampler` -> `projector(encoder(...))` -> `predictor` ->
    `prediction_loss`, driven by the real `CausalBlockMaskSampler`, not a hand-built mask -- rather
    than any one component in isolation. CM-01/CM-02/CM-04 already cover the encoder, the
    predictor, and full-vs-prefix encode equivalence separately; a bug at a *seam* between them
    (e.g. the projector's shape handling, or `train_step`'s own mask-plan wiring) would not be
    caught by any of those three on their own. This closes that composed-path gap; it does not
    re-litigate CM-01/CM-02/CM-04's own component-level claims, and it does not cover preprocessing
    (LEAK-05, gated on DATA-02; CM-08 records nothing normalizes on this path yet).

    `train_step` (train.py:85-128) returns only scalar `StepMetrics`, not the per-token latents
    this check needs, so the composed forward pass is re-run inline below, in `train_step`'s own
    line order (train.py:101-109) -- the documented fallback when `train_step` itself doesn't
    expose an intermediate tensor, not a parallel reimplementation: the cross-check right after
    `_composed` pins the two to agree bit-for-bit on the unperturbed waveform, so this test cannot
    silently drift from what `train_step` really does. That equality only holds because neither
    call is wrapped in `torch.no_grad()` -- PyTorch's no-grad and autograd-tracked paths select
    different kernels for the same op and differ at the ULP level, so `_composed` is called
    exactly the way `train_step` itself is (gradient-tracked, un-detached).

    `model.eval()` disables `SeededDropout` (predictor.py) for every call below -- its own
    per-training-forward-advancing generator is unrelated to causality, and without `eval()` two
    forward passes on one model instance would differ for a reason having nothing to do with the
    perturbation (the same reason CM-01/CM-02/CM-04's own tests call `.eval()`).

    Perturbation point: `ResidualCnnEncoder`'s own `token_window(cutoff)` last sample
    (`4*cutoff+2`, inlined here since `winder.jepa.leakage` retired -- architecture-primer.html
    §5-6), plus one -- matching CM-01's own boundary convention -- perturbing every sample of batch
    element 0 from there on, all 12 leads (mirrors CM-01's `[:, :, cutoff_sample:] += 10.0`). Other
    batch elements are left untouched throughout, so their latents/predictions are also asserted
    unchanged: a projector or predictor bug that mixed batch elements would show up there.

    Tolerance: measured maxdiff is exactly 0.0, at both `torch.set_num_threads(1)` and the default
    multi-threaded config, for every "should be invariant" quantity below -- unlike CM-04's
    prefix-vs-full-waveform comparison, this perturbation never changes input *length*, so there
    is no reassociation-order confound left to absorb. atol=1e-6 for the context-branch latents
    (CM-01's own insurance margin) and atol=1e-8 for the predictor's output (CM-02's own margin).
    The perturbation is real, not vacuous: measured maxdiff at the target block's own projected
    latent is ~0.195, and measured `|Δpred_loss|` is ~1.28e-3 -- both asserted below at thresholds
    an order of magnitude below what was measured, matching CM-01/CM-02's own "must actually move"
    checks.

    A structural note, not a bug (confirmed by hand during development, not asserted here, since
    it is a different contract than this test's own): the predictor's output is invariant to this
    perturbation at *every* token, not only tokens `<= c`. CM-05's design
    (`predictor.py`'s `q = torch.where(mask, mask_token, z_ctx)`) replaces every position at or
    after the cutoff with a fixed learned vector *before* any attention runs, so no masked
    position's own true latent ever enters the predictor's input graph at all, independent of the
    causal mask. Consequently, disabling the predictor's own causal mask (CM-02) alone does not
    make this test fail -- CM-02's own module-level test already owns that contract. What this
    test's perturbation is actually sensitive to is the encoder's own causal padding (CM-01):
    reintroducing symmetric conv padding there moves `z[0, :cutoff+1]` and fails this test
    immediately.
    """
    model = _build_and_init(0)
    model.eval()
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    n_tokens = int(model.config.n_tokens)

    _ComposedOut = tuple[
        CausalMaskPlan, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]

    def _composed(wave: torch.Tensor, *, lambda_sig: float = 0.1) -> _ComposedOut:
        """`train_step`'s own composition (train.py:101-111), inline so its per-token latents
        (`plan`, `z`, `z_hat`) are reachable -- see this test's docstring."""
        b = wave.shape[0]
        plan = model.mask_sampler(b, n_tokens, generator=generator(0, "mask"))
        z = model.projector.forward(model.encoder.forward(wave))
        z_for_sigreg = z.transpose(0, 1)
        sigreg_loss = model.regularizer(z_for_sigreg, generator=generator(0, "sigreg"))
        predictor_mask = ~plan.context
        z_hat = model.predictor.forward(z, predictor_mask)
        pred_loss = model.prediction_loss(z_hat, z, plan.target)
        total_loss = pred_loss + lambda_sig * sigreg_loss
        return plan, z, z_hat, pred_loss, sigreg_loss, total_loss

    # Pin the inline composition to train_step itself on the unperturbed waveform: same generator
    # seeds, same lambda_sig. If this drifts, the test below is no longer testing what train_step
    # actually runs.
    plan, z, z_hat, pred_loss, sigreg_loss, total_loss = _composed(waveform)
    loss_ts, metrics_ts = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
    )
    assert total_loss.item() == loss_ts.item()
    assert pred_loss.item() == metrics_ts.pred_loss
    assert sigreg_loss.item() == metrics_ts.sigreg_loss

    cutoff = int(plan.cutoff[0])
    last_sample = 4 * cutoff + 2  # ResidualCnnEncoder's token_window(cutoff)'s own last sample
    perturbed = waveform.clone()
    perturbed[0, :, last_sample + 1 :] += 10.0

    plan2, z2, z_hat2, pred_loss2, _sigreg_loss2, _total_loss2 = _composed(perturbed)
    # Plan sampling depends only on (batch_size, n_tokens, generator), never on the waveform, so
    # this must hold trivially -- asserted anyway so a future coupling regression would be caught
    # here rather than silently invalidating every assertion below it.
    assert torch.equal(plan.context, plan2.context)
    assert torch.equal(plan.target, plan2.target)

    # Invariant: context-branch projected latents at/before cutoff (CM-04's claim, through the
    # real encoder+projector composition, not the standalone encoder module).
    torch.testing.assert_close(z[0, : cutoff + 1], z2[0, : cutoff + 1], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(z[1:], z2[1:], rtol=0.0, atol=1e-6)  # untouched batch elements

    # Invariant: predictor output at context (unmasked) positions...
    torch.testing.assert_close(z_hat[0, : cutoff + 1], z_hat2[0, : cutoff + 1], rtol=0.0, atol=1e-8)
    # ...and at every masked position after the cutoff (gap and target block both) -- CM-05's
    # mask_token substitution means none of them ever see the perturbed region either.
    torch.testing.assert_close(z_hat[0, cutoff + 1 :], z_hat2[0, cutoff + 1 :], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(z_hat[1:], z_hat2[1:], rtol=0.0, atol=1e-8)  # untouched batch elems

    # Not vacuous: the perturbation must actually reach the target block's own projected latent
    # (measured ~0.195)...
    target_diff = (z2[0, plan2.target[0]] - z[0, plan.target[0]]).abs().max()
    assert target_diff > 1e-3
    # ...and therefore change pred_loss specifically (measured ~1.28e-3): z_hat, the prediction
    # side, is invariant (asserted above), so a pred_loss change can only come from the label
    # side, z at the target block, moving. Checked on pred_loss rather than total_loss because
    # sigreg_loss also moves here (its per-timestep statistic legitimately includes the perturbed
    # tokens' own timesteps), which would make a total_loss-only check ambiguous about which
    # channel changed.
    assert abs(pred_loss2.item() - pred_loss.item()) > 1e-4


def test_fit_runs_and_both_losses_are_logged_and_decrease() -> None:
    """lambda_sig=0.5, not the spec's default 0.1: the SIGReg reduction fix (per-timestep,
    N=batch size, averaged over T -- see winder.jepa.regularizers' module docstring) divides
    SIGReg's own gradient contribution by roughly T relative to the old B*T-pooled reduction, so
    the old rationale for this test's lambda (SIGReg's raw magnitude dominating pred_loss even
    after weighting) no longer holds post-fix -- measuring at this test's tiny scale (D=32),
    lambda in [0.001, 0.1] now leaves SIGReg's own loss flat-to-rising over 30 steps (too little
    gradient signal reaches it at those weights), while lambda=0.5 is the smallest value measured
    to move both losses down together. This is a mechanism sanity check (does gradient descent
    reduce loss at all), not a claim about a good lambda for real training -- the lambda sweep is
    the actual re-sweep, on real data, that answers that question."""
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = TrainConfig(n_steps=30, lambda_sig=0.5, seed_pretrain=0, warmup_steps=3)
    gen_data = generator(0, "synth")

    def batches() -> Iterator[torch.Tensor]:
        for _ in range(cfg.n_steps):
            yield synthetic_waveform_batch(4, generator=gen_data)

    history = fit(model, batches(), cfg, optimizer)
    assert len(history) == cfg.n_steps
    assert all(m.total_loss == m.total_loss for m in history)  # no NaNs (NaN != NaN)

    first_pred = sum(m.pred_loss for m in history[:5]) / 5
    last_pred = sum(m.pred_loss for m in history[-5:]) / 5
    assert last_pred < first_pred

    first_sig = sum(m.sigreg_loss for m in history[:5]) / 5
    last_sig = sum(m.sigreg_loss for m in history[-5:]) / 5
    assert last_sig < first_sig


def test_fit_is_bitwise_reproducible_at_the_same_seed() -> None:
    """True bitwise reproducibility requires pinning the CPU thread count: multi-threaded
    matmul/reduction order is not associative in floating point, so two runs can otherwise
    differ in the last ULP or two even with identical seeds -- a hardware/BLAS-threading
    artifact, not a gap in this project's own determinism doctrine (every explicit source of
    randomness IS seeded identically; verified separately by the single-generator-object test
    above)."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:

        def run() -> list[float]:
            model = _build_and_init(0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            cfg = TrainConfig(n_steps=5, lambda_sig=0.1, seed_pretrain=0, warmup_steps=1)
            gen_data = generator(0, "synth")
            batches = (synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps))
            history = fit(model, batches, cfg, optimizer)
            return [m.pred_loss for m in history] + [m.sigreg_loss for m in history]

        assert run() == run()
    finally:
        torch.set_num_threads(original_threads)


def test_fit_does_not_touch_the_global_torch_rng() -> None:
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = TrainConfig(n_steps=5, lambda_sig=0.1, seed_pretrain=0, warmup_steps=1)
    gen_data = generator(0, "synth")
    batches = [synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)]

    torch.manual_seed(12345)  # pin a known global state to then check is untouched
    before = torch.get_rng_state()
    fit(model, iter(batches), cfg, optimizer)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_fit_with_transport_trains_the_free_operators_own_omega() -> None:
    """The free arm's omega has its own SEPARATE optimizer param group here (module docstring's
    weight_decay=0.0 recommendation) -- proves optimizer.step() actually reaches
    operator.parameters() when a caller wires it in, and that fit()'s own separate grad-clip
    call on the operator (module docstring) does not block that update entirely."""
    model = _build_and_init(0)  # projector.output_width == 32 (_tiny_config)
    operator = FreeOperator(FreeOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))  # K=8+24=32
    omega_before = operator.omega.detach().clone()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 3e-4},
            {"params": operator.parameters(), "lr": 1e-2, "weight_decay": 0.0},
        ]
    )
    cfg = TrainConfig(n_steps=10, lambda_sig=0.1, lambda_trans=1.0, seed_pretrain=0, warmup_steps=1)
    gen_data = generator(0, "synth")
    n_tokens = 250

    def waveform_batches() -> Iterator[torch.Tensor]:
        for _ in range(cfg.n_steps):
            yield synthetic_waveform_batch(4, generator=gen_data)

    def theta_batches() -> Iterator[torch.Tensor]:
        gen = torch.Generator().manual_seed(0)
        for _ in range(cfg.n_steps):
            yield torch.rand(4, n_tokens, generator=gen) * 6.28

    history = fit(
        model,
        waveform_batches(),
        cfg,
        optimizer,
        theta_batches=theta_batches(),
        operator=operator,
    )
    assert len(history) == cfg.n_steps
    assert all(math.isfinite(m.trans_loss) for m in history)
    assert all(math.isfinite(m.closure_residual) for m in history)
    assert not torch.equal(operator.omega.detach(), omega_before)


def test_fit_without_transport_is_unaffected_by_the_new_optional_parameters() -> None:
    """theta_batches/operator both omitted (their defaults) -- every pre-existing call shape
    still works, and behaves exactly as it did before those parameters existed."""
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = TrainConfig(n_steps=3, lambda_sig=0.1, seed_pretrain=0, warmup_steps=1)
    gen_data = generator(0, "synth")
    batches = [synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)]

    history = fit(model, iter(batches), cfg, optimizer)
    assert len(history) == 3
    assert all(math.isnan(m.trans_loss) for m in history)


def test_fit_passes_cfg_lambda_pred_through_to_every_train_step() -> None:
    """`fit`'s own `train_step(...)` call (train.py's `for step, waveform, theta in zip(...)`
    loop) must read `cfg.lambda_pred`, not silently keep `train_step`'s own default -- the same
    kind of plumbing gap `test_fit_with_transport_trains_the_free_operators_own_omega` guards for
    `cfg.lambda_trans`."""
    model = _build_and_init(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = TrainConfig(n_steps=3, lambda_sig=0.1, lambda_pred=0.0, seed_pretrain=0, warmup_steps=1)
    gen_data = generator(0, "synth")
    batches = [synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)]

    history = fit(model, iter(batches), cfg, optimizer)
    assert len(history) == 3
    assert all(math.isnan(m.pred_loss) for m in history)
    assert all(math.isnan(m.persistence_loss) for m in history)
    assert all(math.isfinite(m.sigreg_loss) for m in history)


def test_fit_explicit_generators_match_the_default_internal_construction() -> None:
    """CKPT-01/CKPT-04 rely on this: a caller who explicitly builds `gen_mask`/`gen_sigreg` via
    `winder.determinism.generator(cfg.seed_pretrain, "mask"/"sigreg")` and passes them into `fit`
    must get bitwise-identical results to `fit`'s own default (no-kwargs) construction -- the two
    code paths must produce the SAME generator, not merely a similarly-seeded one, or a resumed
    run's saved generator state would not correspond to what an uninterrupted uses internally."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = TrainConfig(n_steps=5, lambda_sig=0.1, seed_pretrain=7, warmup_steps=1)

        def _batches() -> Iterator[torch.Tensor]:
            gen_data = generator(7, "synth")
            for _ in range(cfg.n_steps):
                yield synthetic_waveform_batch(4, generator=gen_data)

        model_a = _build_and_init(7)
        history_a = fit(model_a, _batches(), cfg, torch.optim.AdamW(model_a.parameters(), lr=3e-4))

        model_b = _build_and_init(7)
        history_b = fit(
            model_b,
            _batches(),
            cfg,
            torch.optim.AdamW(model_b.parameters(), lr=3e-4),
            gen_mask=generator(cfg.seed_pretrain, "mask"),
            gen_sigreg=generator(cfg.seed_pretrain, "sigreg"),
        )
        assert [m.total_loss for m in history_a] == [m.total_loss for m in history_b]
    finally:
        torch.set_num_threads(original_threads)


def test_ckpt04_exact_resume_matches_uninterrupted_reference(tmp_path: Path) -> None:
    """CKPT-04, the acceptance test for CKPT-01: interrupting `fit` at step K and resuming from a
    saved checkpoint must exactly match an uninterrupted run to step K+1 -- proving CKPT-01's
    bundle round-trips every source of training-run state (model, optimizer, global step, and
    every named `torch.Generator` stream `fit`/its caller draws from -- `"mask"`, `"sigreg"`, and
    the data-batch stream), not merely that save/load doesn't crash.

    `torch.set_num_threads(1)`: true bitwise reproducibility needs a pinned thread count (matches
    `test_fit_is_bitwise_reproducible_at_the_same_seed` above) -- multi-threaded matmul/reduction
    order is not associative in floating point, so two runs can otherwise differ in the last ULP
    or two even with every explicit RNG identically seeded. Tolerance: exact `==` (float and
    tensor), the same convention that test already uses at this thread count.

    `cfg.n_steps` is the SAME (K+1) across all three `fit` calls below (reference, phase 1, phase
    2): `lr_schedule` reads `cfg.n_steps` as the schedule's total length, not "how many steps
    this call runs" -- shrinking it for phase 1 would shift the learning rate at every step
    before K for a reason having nothing to do with checkpointing (see `fit`'s own updated
    docstring). What actually limits phase 1 to K steps is handing `fit` a `batches` iterator of
    length exactly K (an `itertools.islice` of the shared, infinite `_stream`), so
    `zip(range(start_step, cfg.n_steps), batches, strict=False)` stops by iterator exhaustion --
    `fit`'s own documented `strict=False` contract, not a new mechanism this test invents.
    """
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        k = 4
        cfg = TrainConfig(n_steps=k + 1, lambda_sig=0.1, seed_pretrain=0, warmup_steps=1)

        def _stream(gen_data: torch.Generator) -> Iterator[torch.Tensor]:
            while True:
                yield synthetic_waveform_batch(4, generator=gen_data)

        # ---- Reference: uninterrupted, K+1 steps in one fit() call. ----
        ref_model = _build_and_init(0)
        ref_optimizer = torch.optim.AdamW(ref_model.parameters(), lr=3e-4)
        ref_gen_data = generator(0, "ckpt04_data")
        ref_history = fit(
            ref_model, itertools.islice(_stream(ref_gen_data), k + 1), cfg, ref_optimizer
        )
        assert len(ref_history) == k + 1

        # ---- Phase 1: fresh model/optimizer/generators, run only the first K steps. ----
        model1 = _build_and_init(0)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=3e-4)
        gen_mask1 = generator(0, "mask")
        gen_sigreg1 = generator(0, "sigreg")
        gen_data1 = generator(0, "ckpt04_data")
        history1 = fit(
            model1,
            itertools.islice(_stream(gen_data1), k),
            cfg,
            optimizer1,
            gen_mask=gen_mask1,
            gen_sigreg=gen_sigreg1,
        )
        assert len(history1) == k

        ckpt_dir = os.path.join(str(tmp_path), "ckpt")
        checkpoint.save_checkpoint(
            ckpt_dir,
            model=model1,
            optimizer=optimizer1,
            step=history1[-1].step + 1,
            generators={"mask": gen_mask1, "sigreg": gen_sigreg1, "data": gen_data1},
            config_yaml="jepa: {}\ntrain: {}\n",
            meta={"note": "CKPT-04 exact-resume test"},
        )

        # ---- Phase 2: fresh model/optimizer/generators, restored purely from the checkpoint. ----
        model2 = _build_and_init(1)  # deliberately different init seed -- must be overwritten
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
        loaded = checkpoint.load_checkpoint(ckpt_dir, model=model2, optimizer=optimizer2)
        assert loaded.step == k
        assert model2.training  # load_checkpoint must not force eval() (see its own docstring)

        gen_mask2 = torch.Generator()
        gen_mask2.set_state(loaded.generator_states["mask"])
        gen_sigreg2 = torch.Generator()
        gen_sigreg2.set_state(loaded.generator_states["sigreg"])
        gen_data2 = torch.Generator()
        gen_data2.set_state(loaded.generator_states["data"])

        history2 = fit(
            model2,
            itertools.islice(_stream(gen_data2), 1),
            cfg,
            optimizer2,
            start_step=loaded.step,
            gen_mask=gen_mask2,
            gen_sigreg=gen_sigreg2,
        )
        assert len(history2) == 1  # a resume that silently ran zero steps must fail loudly here
        assert history2[0].step == k

        # The resumed single step must exactly match the reference's (K+1)-th step (index k).
        assert history2[0].pred_loss == ref_history[k].pred_loss
        assert history2[0].sigreg_loss == ref_history[k].sigreg_loss
        assert history2[0].total_loss == ref_history[k].total_loss
        assert history2[0].lr == ref_history[k].lr

        for p_ref, p_resumed in zip(ref_model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p_ref, p_resumed)
    finally:
        torch.set_num_threads(original_threads)


# ============================================================ sigreg_frame="canonical" (eq-28)
#
# _tiny_config's projector output_width == 32 -- every operator built below matches it
# (k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4] -> K = 8 + 2*(4+4+4) = 32), the same spectrum
# test_lambda_pred_zero_is_bitwise_identical_to_a_hand_built_transport_sigreg_only_step already
# uses for the same reason.


def test_sigreg_frame_canonical_without_theta_or_operator_raises() -> None:
    """The canonical-frame guard fires independent of lambda_trans (module docstring's
    `sigreg_frame` paragraph): requesting canonical framing with transport itself switched off
    (lambda_trans's own default, 0.0) must still raise -- canonical framing is a property of what
    SIGReg reads, not of whether L_trans is in total_loss."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    with pytest.raises(ValueError, match="sigreg_frame"):
        train_step(
            model,
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            sigreg_frame="canonical",
        )


def test_sigreg_frame_canonical_theta_zero_matches_raw_bitwise() -> None:
    """theory_closeout_v1.html §8.2's claim in its simplest instance: theta == 0 everywhere means
    R_0 = I, so canonical-frame demodulation is a no-op and the canonical-frame SIGReg statistic
    must equal raw-frame's own exactly -- not merely approximately close."""
    model_raw = _build_and_init(0)
    model_canon = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))  # K=32
    theta_zero = torch.zeros(4, 250)

    _, metrics_raw = train_step(
        model_raw,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
    )
    _, metrics_canon = train_step(
        model_canon,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta_zero,
        operator=operator,
        sigreg_frame="canonical",
    )
    assert metrics_canon.sigreg_loss == metrics_raw.sigreg_loss
    assert math.isnan(metrics_raw.theta_valid_frac)  # raw frame: "not applicable" convention
    assert metrics_canon.theta_valid_frac == 1.0  # every token here has a real (zero) theta


def test_sigreg_frame_canonical_uses_the_minus_theta_convention_in_train_step_itself() -> None:
    """Pins `train_step`'s OWN production code path to the `-theta` sign, at random NONZERO
    theta where sign actually matters (unlike the theta=0 test above, where `R_0 = I` makes both
    signs identical, and unlike the discriminating co-rotating-anisotropy test, which validates
    the FORMULA via a hand-built `operator.transport(z, -theta)` call of its own, never through
    `train_step`). Mirrors
    `test_lambda_pred_zero_is_bitwise_identical_to_a_hand_built_transport_sigreg_only_step`'s own
    idiom: a from-scratch reference built from the SAME formula `train_step` is supposed to run,
    compared bitwise -- not trusting `train_step`'s own inline copy to be self-consistent. A
    `+theta` regression in `train_step` itself (the exact silent failure this step's brief warns
    about) would diverge from this reference immediately at random theta, though it would still
    pass the theta=0 and K0-only-energy tests above."""
    model_ts = _build_and_init(0)
    model_ref = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator_ts = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    operator_ref = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi

    _, metrics_ts = train_step(
        model_ts,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator_ts,
        sigreg_frame="canonical",
    )

    # Hand-built reference: encoder+projector -> operator.transport(z, -theta) -> regularizer,
    # the literal formula `sigreg_frame == "canonical"` is supposed to run.
    z_ref = model_ref.projector.forward(model_ref.encoder.forward(waveform))
    u_ref = operator_ref.transport(z_ref, -theta)
    sigreg_ref = model_ref.regularizer(u_ref.transpose(0, 1), generator=generator(0, "sigreg"))

    # This equality is itself the discriminating check: a `+theta` regression in `train_step`
    # would compare against this `-theta` reference and fail here, at random nonzero theta where
    # (unlike the theta=0 test above) the two signs give genuinely different demodulated tensors.
    assert metrics_ts.sigreg_loss == float(sigreg_ref.detach())

    # Sanity: confirm the two signs actually differ on this batch (not a vacuous equality that
    # would pass regardless of which sign train_step used).
    u_wrong_sign = operator_ref.transport(z_ref, theta)
    assert not torch.equal(u_wrong_sign, u_ref)


def test_sigreg_frame_canonical_matches_raw_when_energy_only_in_invariant_block() -> None:
    """Energy confined to the k0 invariant block never rotates -- `operator.transport` passes
    `z[..., :k0]` through untouched and rotates all-zero harmonic planes to all-zero -- so
    canonical-frame SIGReg must equal raw-frame SIGReg for ANY theta, not only theta=0
    (theory_closeout_v1.html §8.2: "the invariant block doesn't rotate"). Calls
    `winder.jepa.regularizers.SigReg` directly (mirroring `winder.transport.procrustes`'s own
    `demodulated_within_record_pairs` formula) rather than through `train_step`, since a real
    encoder's own output cannot be steered to put ALL its energy in one block."""
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))  # K=32
    regularizer = SigReg(SigRegConfig(n_directions=64, chunk=64))
    gen = torch.Generator().manual_seed(0)
    n_tok, batch = 20, 32
    z = torch.zeros(batch, n_tok, operator.dimension)
    z[..., : operator.k0] = torch.randn(batch, n_tok, operator.k0, generator=gen)
    theta = torch.rand(batch, n_tok, generator=gen) * 2 * math.pi  # arbitrary, nonzero

    u = operator.transport(z, -theta)  # train_step's own canonical-branch formula
    assert torch.equal(u, z)  # the discriminating claim, at the tensor level

    raw_stat = regularizer(z.transpose(0, 1), generator=torch.Generator().manual_seed(1))
    canon_stat = regularizer(u.transpose(0, 1), generator=torch.Generator().manual_seed(1))
    assert torch.equal(raw_stat, canon_stat)


def test_sigreg_frame_canonical_sees_co_rotating_anisotropy_raw_frame_does_not() -> None:
    """The discriminating claim (theory_closeout_v1.html §8.2): raw-frame SIGReg pools over phase
    and is therefore blind to anisotropy that co-rotates with theta, while canonical-frame SIGReg
    demodulates first and sees it. Also the test that pins the `-theta` sign convention
    (`winder.transport.procrustes.demodulated_within_record_pairs`'s own): a `+theta` "canonical"
    branch reapplies the rotation instead of undoing it and stays blind (measured below, not
    merely asserted) -- exactly the failure mode the brief warns a theta=0 or K0-only-energy test
    cannot catch.

    Construction: `u` (the TRUE canonical-frame law) is iid N(0, 1) on every dimension except all
    4 planes of harmonic n_j=1 (dims k0..k0+8), which instead carry variance 2 on one in-plane
    axis and variance 0 on the other -- a real anisotropy (one axis fully collapsed) whose TOTAL
    per-plane energy (2) still matches the isotropic target (a 2-D plane's isotropic trace is
    1+1=2). `z = R_theta @ u` for theta ~ Uniform[0, 2*pi) per row: averaged over phase, a
    uniformly-rotated anisotropic-but-equal-trace covariance becomes isotropic (`R_theta`'s own
    rotation-averaging identity) -- so raw-frame SIGReg, reading `z` directly, should land near
    the isotropic anchor regardless of the collapsed direction, while canonical-frame SIGReg,
    reading `u_hat = R_{-theta} @ z` (the tokens correctly demodulated), should recover the
    collapse.

    Measured at N=16000, K=32 (this test's own construction, seed-independent order of
    magnitude): raw ~1.32 (matches winder.jepa.regularizers' own "isotropic randn lands near
    1.0-1.1" anchor), canonical ~31.3 (>20x raw), wrong-sign (`R_{+theta} @ z`, i.e. R_{2*theta}
    applied to `u`) ~1.23 -- indistinguishable from raw, confirming the margin comes from the
    SIGN, not merely from demodulating at all. Asserted margin (10x) sits comfortably below the
    measured ~24x, leaving headroom for RNG variation."""
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))  # K=32
    k0 = operator.k0
    n = 16000
    gen = torch.Generator().manual_seed(123)
    u = torch.randn(n, operator.dimension, generator=gen)
    for p in range(4):  # all 4 planes of harmonic n_j=1 (multiplicity k_j=4)
        u[:, k0 + 2 * p] = torch.randn(n, generator=gen) * math.sqrt(2.0)  # variance 2
        u[:, k0 + 2 * p + 1] = 0.0  # variance 0: fully collapsed on this axis

    theta = torch.rand(n, generator=gen) * 2 * math.pi
    z = operator.transport(u, theta)  # the raw-frame latent this anisotropic law would produce
    u_hat = operator.transport(z, -theta)  # train_step's own canonical-branch formula
    torch.testing.assert_close(u_hat, u, atol=1e-4, rtol=1e-4)  # R_{-theta} R_{theta} == I

    regularizer = SigReg(SigRegConfig(n_directions=256, chunk=64))
    raw_stat = regularizer(z, generator=torch.Generator().manual_seed(7))
    canonical_stat = regularizer(u_hat, generator=torch.Generator().manual_seed(7))

    assert 0.5 < float(raw_stat) < 3.0  # near the isotropic anchor, not elevated
    assert float(canonical_stat) > 10.0 * float(raw_stat)  # measured ~24x; asserted margin 10x


def test_sigreg_frame_canonical_nan_theta_tokens_poison_no_gradients() -> None:
    """NaN-theta tokens (module docstring's "declared, structured dilution" paragraph) must not
    poison `total_loss` or any parameter's `.grad` (part a), and must not silently dominate the
    gradient relative to an all-valid-theta control of the SAME batch (part b) -- filled tokens
    pass through UN-rotated (theta=0, R_0=I), diluting rather than corrupting the statistic."""
    model_diluted = _build_and_init(0)
    model_control = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator_diluted = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    operator_control = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))

    gen = torch.Generator().manual_seed(0)
    theta_valid = torch.rand(4, 250, generator=gen) * 2 * math.pi
    theta_diluted = theta_valid.clone()
    drop = torch.rand(4, 250, generator=gen) < 0.1  # ~10%, matching M0's own measured dilution
    theta_diluted[drop] = float("nan")
    assert bool(drop.any())  # not a vacuous check

    loss_diluted, metrics_diluted = train_step(
        model_diluted,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta_diluted,
        operator=operator_diluted,
        sigreg_frame="canonical",
    )
    loss_diluted.backward()

    def _grad_norm(grads: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for g in grads:
            total = total + (g.double() ** 2).sum()
        return torch.sqrt(total)

    # (a) no NaN/Inf anywhere in the loss or any parameter's grad.
    assert torch.isfinite(loss_diluted)
    grads_diluted = [p.grad for p in model_diluted.parameters() if p.grad is not None]
    assert len(grads_diluted) > 0  # not a vacuous check
    assert all(torch.isfinite(g).all() for g in grads_diluted)
    norm_diluted = _grad_norm(grads_diluted)

    # (b) the filled-junk ~10% must not dominate: compare the WHOLE-model gradient norm against
    # an all-valid-theta control built from the SAME waveform/generators, varying ONLY theta.
    loss_control, _metrics_control = train_step(
        model_control,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta_valid,
        operator=operator_control,
        sigreg_frame="canonical",
    )
    loss_control.backward()
    grads_control = [p.grad for p in model_control.parameters() if p.grad is not None]
    norm_control = _grad_norm(grads_control)

    ratio = float(norm_diluted / norm_control)
    assert 0.2 < ratio < 5.0  # measured ~1.0006 -- a generous sanity band, not a tight bound
    assert metrics_diluted.theta_valid_frac == pytest.approx(1.0 - float(drop.float().mean()))


def test_fit_passes_cfg_sigreg_frame_through_to_every_train_step() -> None:
    """`fit`'s own `train_step(...)` call must read `cfg.sigreg_frame`, not silently keep
    `train_step`'s own default -- the same kind of plumbing gap
    `test_fit_with_transport_trains_the_free_operators_own_omega`/
    `test_fit_passes_cfg_lambda_pred_through_to_every_train_step` already guard for
    `cfg.lambda_trans`/`cfg.lambda_pred`."""
    model = _build_and_init(0)
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = TrainConfig(
        n_steps=3,
        lambda_sig=0.1,
        lambda_trans=0.0,  # deliberately OFF: canonical framing must not need transport active
        sigreg_frame="canonical",
        seed_pretrain=0,
        warmup_steps=1,
    )
    gen_data = generator(0, "synth")
    batches = [synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)]
    theta_batches = [torch.rand(4, 250) * 2 * math.pi for _ in range(cfg.n_steps)]

    history = fit(
        model, iter(batches), cfg, optimizer, theta_batches=iter(theta_batches), operator=operator
    )
    assert len(history) == 3
    assert all(math.isfinite(m.theta_valid_frac) for m in history)
    assert all(m.theta_valid_frac == 1.0 for m in history)  # no NaNs injected in this batch


# ==================================================== sigreg_frame="record_canonical" (X6)
#
# The token-level canonical frame was falsified twice in campaign_x2x2 (gain_fraction
# -0.17..-0.38, effective rank 2.9-11.2): the arms bought token-level isotropy in the demodulated
# frame by NORM COLLAPSE WITH RESIDUAL DIRECTION SPREAD -- emptied templates, not healthy
# within-record dispersion (pre_launch_addendum.md's "Cell 3" restatement; the older "dispersion
# escape" phrasing is superseded). `record_canonical` applies SIGReg to the per-record demodulated
# MEAN instead, where BOTH failure modes violate the same unit-variance target: an emptied
# template from below, and incoherent within-record energy by shrinking the mean at fixed token
# energy. The test below measures the second, which is the one a synthetic construction can hold
# every other quantity fixed while varying.
# Same operator sizing as the canonical block above (_tiny_config's projector output_width == 32).


def test_raw_and_canonical_sigreg_frames_match_their_pre_record_canonical_golden_values() -> None:
    """Paired comparability (the X6 brief's non-negotiable invariant): `record_canonical` is a
    STRUCTURAL ADDITION, so `raw` and `canonical` must stay bitwise identical to what they
    computed before the frame existed -- the same discipline
    `test_lambda_trans_zero_is_bitwise_identical_whether_theta_and_operator_are_passed` enforces
    for the transport skip.

    The literals below were captured from the pre-`record_canonical` implementation at
    `torch.set_num_threads(1)` (pinned for the same reason the resume-determinism tests above pin
    it: CPU parallel reductions reorder with core count) and are asserted EXACTLY, not
    approximately. A failure here means an edit to `train_step` moved the default path, not that a
    tolerance is too tight -- regenerate these only to accept a reviewed, intended change to
    raw/canonical-frame behaviour."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
        theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(4242)) * 2 * math.pi
        theta[0, :7] = float("nan")  # M0's own pre-first-R-peak dilution shape
        theta[3, 100] = float("nan")

        _, m_raw = train_step(
            _build_and_init(0),
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
        )
        assert m_raw.pred_loss == 0.08325359225273132
        assert m_raw.persistence_loss == 0.0006865973700769246
        assert m_raw.sigreg_loss == 1.7095328569412231
        assert m_raw.total_loss == 0.25420689582824707
        assert (m_raw.n_context, m_raw.n_target, m_raw.cutoff_mean) == (382, 4, 94.5)

        _, m_canon = train_step(
            _build_and_init(0),
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            theta=theta,
            operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
            lambda_trans=1.0,
            sigreg_frame="canonical",
        )
        assert m_canon.pred_loss == 0.08325359225273132
        assert m_canon.persistence_loss == 0.0006865973700769246
        assert m_canon.sigreg_loss == 1.5687761306762695
        assert m_canon.total_loss == 1.0526559352874756
        assert m_canon.trans_loss == 0.812524676322937
        assert m_canon.trans_floor == 0.8138893842697144
        assert m_canon.trans_directional == 0.812524676322937
        assert m_canon.theta_valid_frac == 0.9919999837875366
        # The token-level frame reports no record count -- "not applicable", the same NaN sentinel
        # convention as the four trans_* fields.
        assert math.isnan(m_raw.sigreg_n_records)
        assert math.isnan(m_canon.sigreg_n_records)
    finally:
        torch.set_num_threads(original_threads)


def test_record_canonical_templates_match_a_loop_built_per_record_demodulated_mean() -> None:
    """The estimator itself: `templates[r] = mean over record r's VALID tokens of R_{-theta_t}
    z_t` -- `winder.eval.pooling.demodulated_pool`'s Proposition-4.2 convention (the feature the
    `z/demodulated` probe actually reads), checked against an explicit per-record, per-token
    Python loop rather than against a second vectorised rewrite of the same expression.

    This is the DELIBERATE DIVERGENCE from the token-level canonical path, which fills NaN theta
    to 0 and lets the token through un-rotated: an un-rotated token injected into a record's mean
    would contribute an arbitrary phase-zero-frame vector to the template, so here an invalid
    token is excluded from BOTH numerator and denominator."""
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    gen = torch.Generator().manual_seed(0)
    b, n_tok, k = 5, 9, operator.dimension
    z = torch.randn(b, n_tok, k, generator=gen)
    theta = torch.rand(b, n_tok, generator=gen) * 2 * math.pi
    theta[1, 2:5] = float("nan")  # record 1: partially invalid
    theta[4, :] = float("nan")  # record 4: no valid token at all

    templates, has_valid = record_canonical_templates(z, theta, operator)
    assert templates.shape == (b, k)
    assert has_valid.tolist() == [True, True, True, True, False]

    for r in range(b):
        valid_tokens = [j for j in range(n_tok) for _ in (0,) if torch.isfinite(theta[r, j])]
        if not valid_tokens:
            assert bool(torch.isnan(templates[r]).all())  # never a zero vector: a zero row would
            # be a false "collapsed template" sample in the statistic
            continue
        acc = torch.zeros(k)
        for j in valid_tokens:
            rotated = operator.transport(z[r, j].view(1, 1, k), -theta[r, j].view(1, 1))
            acc = acc + rotated.view(k)
        torch.testing.assert_close(templates[r], acc / len(valid_tokens), rtol=1e-6, atol=1e-6)

    # The count denominator is the VALID token count, not n_tok -- a filled-and-divided-by-n_tok
    # implementation would shrink record 1's template by 6/9 and pass every other assertion here.
    naive = operator.transport(
        torch.where(torch.isfinite(theta).unsqueeze(-1), z, torch.zeros_like(z)),
        -torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta)),
    ).mean(dim=1)
    assert not torch.allclose(templates[1], naive[1], rtol=1e-3, atol=1e-3)


def test_record_canonical_penalises_dispersion_while_the_token_frame_stays_flat() -> None:
    """THE mechanism this arm exists for, as a paired contrast on the SAME latents.

    Construction: each record has a template `u_r ~ N(0, I)`; every token of that record is
    `u_t = (u_r + s * eps_t) / sqrt(1 + s^2)` (norm-preserving -- token-level SIGReg pins each
    token's own scale, so within-record dispersion trades COHERENT for INCOHERENT energy rather
    than adding energy), then pre-rotated to the raw frame by its own phase, `z_t = R_theta_t u_t`.
    The demodulated record mean therefore has variance ~ 1/(1 + s^2): dispersion shrinks the
    template away from the unit-variance target.

    Measured here (seed 100, B=64 records, T=125 tokens, K=32):

        s        0.0     0.5     1.0     2.0     4.0
        record   1.02    1.44    4.18   12.38   20.34
        token    1.03    1.04    1.05    1.05    1.05

    -- the record-level statistic rises 20x while the token-level statistic is FLAT to ~2%,
    reproducing the campaign's own checkpoint-free numbers (token 1.06->1.07 while record
    1.07->11.7), and monotone across seeds {0, 7, 100}. That flatness is what the frame closes:
    within-record incoherence is invisible to a per-token statistic and expensive to a per-record
    one. Which failure mode the falsified canonical arms ACTUALLY took is a separate, measured
    question -- norm collapse with residual direction spread, see this block's header comment --
    and the record mean penalises that one too, from below, via lost template mass.

    Held fixed here on purpose: total token energy (the `1/sqrt(1 + s^2)` factor). The frame is
    INDIFFERENT along the scale-compensated direction instead -- inflating tokens by
    `sqrt(1 + s^2/T)` keeps the record statistic at floor while the coherent share of template
    energy falls to 0.33 (module docstring's measured-indifference paragraph)."""
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    regularizer = SigReg(SigRegConfig(n_directions=256, chunk=64))
    b, n_tok, k = 64, 125, operator.dimension

    record_stats: list[float] = []
    token_stats: list[float] = []
    for dispersion in (0.0, 0.5, 1.0, 2.0, 4.0):
        gen = torch.Generator().manual_seed(100)
        u_r = torch.randn(b, 1, k, generator=gen)  # the population-compatible ideal template
        eps = torch.randn(b, n_tok, k, generator=gen)
        u_t = (u_r + dispersion * eps) / math.sqrt(1.0 + dispersion**2)
        theta = torch.rand(b, n_tok, generator=gen) * 2 * math.pi
        z = operator.transport(u_t, theta)  # the raw-frame latents this law would produce

        templates, has_valid = record_canonical_templates(z, theta, operator)
        assert bool(has_valid.all())
        record_stats.append(
            float(regularizer(templates[has_valid], generator=torch.Generator().manual_seed(200)))
        )
        token_stats.append(
            float(regularizer(z.transpose(0, 1), generator=torch.Generator().manual_seed(200)))
        )

    assert record_stats[0] < 2.0  # s=0 (u_r exactly the ideal) sits at the N=64 floor (~0.9-1.3)
    assert record_stats == sorted(record_stats)  # monotone in dispersion -- the whole mechanism
    assert all(b_ > a for a, b_ in zip(record_stats, record_stats[1:], strict=False))  # strictly
    assert record_stats[-1] > 10.0 * record_stats[0]  # measured ~20x

    assert all(0.7 < s < 1.6 for s in token_stats)  # every point near the isotropic anchor
    assert max(token_stats) / min(token_stats) < 1.5  # measured 1.02: FLAT, not merely bounded


def test_sigreg_frame_record_canonical_wires_the_templates_into_train_step_itself() -> None:
    """`train_step`'s own production path (not a reference formula): its `sigreg_loss` must be
    SIGReg applied to the `(N_valid, K)` stack of per-record templates -- the regularizer's own
    documented `T=1` case, N = number of contributing records, not `B*T` pooled and not `(T, B,
    K)`. Compared bitwise against a from-scratch rebuild, the same idiom
    `test_sigreg_frame_canonical_uses_the_minus_theta_convention_in_train_step_itself` uses."""
    model_ts = _build_and_init(0)
    model_ref = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator_ts = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    operator_ref = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi
    theta[2, :20] = float("nan")

    _, metrics = train_step(
        model_ts,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator_ts,
        sigreg_frame="record_canonical",
    )

    z_ref = model_ref.projector.forward(model_ref.encoder.forward(waveform))
    templates_ref, has_valid_ref = record_canonical_templates(z_ref, theta, operator_ref)
    assert templates_ref[has_valid_ref].shape == (4, 32)  # N = 4 records, K = 32 -- the T=1 case
    sigreg_ref = model_ref.regularizer(
        templates_ref[has_valid_ref], generator=generator(0, "sigreg")
    )
    assert metrics.sigreg_loss == float(sigreg_ref.detach())
    assert metrics.sigreg_n_records == 4.0
    assert metrics.theta_valid_frac == pytest.approx(1.0 - 20.0 / (4 * 250))

    # Discriminating: the token-level canonical frame on the SAME batch is a different number, so
    # this is not an equality that any frame would satisfy.
    _, metrics_token = train_step(
        _build_and_init(0),
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
        sigreg_frame="canonical",
    )
    assert metrics_token.sigreg_loss != metrics.sigreg_loss


def test_record_canonical_templates_are_the_probes_own_demodulated_pool_feature() -> None:
    """Cross-module drift guard on the claim `record_canonical_templates`' docstring makes: the
    quantity SIGReg constrains in this frame IS the feature the `z/demodulated` readout consumes
    (`winder.eval.pooling.demodulated_pool`, Proposition 4.2), not a look-alike. `train.py` cannot
    import `winder.eval` (core training path), so the arithmetic is duplicated there -- this test
    is what keeps the duplicate honest if either side is ever edited.

    Equality is exact-to-`allclose` on the SAME `(z, theta, operator)`, all-NaN rows included:
    both sides emit an all-NaN row for a record with zero valid tokens, hence `equal_nan=True`."""
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(5, 9, operator.dimension, generator=gen)
    theta = torch.rand(5, 9, generator=gen) * 2 * math.pi
    theta[1, 2:5] = float("nan")
    theta[4, :] = float("nan")

    templates, has_valid = record_canonical_templates(z, theta, operator)
    pooled = demodulated_pool(z, theta, operator)
    torch.testing.assert_close(templates, pooled, rtol=0.0, atol=0.0, equal_nan=True)
    # And the mask is the pool's own "which rows must a caller drop" predicate, stated positively.
    assert has_valid.tolist() == (~torch.isnan(pooled).any(dim=1)).tolist()


def test_record_canonical_frame_consumes_the_same_rng_draws_as_the_token_level_frames() -> None:
    """The X6 brief's second non-negotiable invariant (no perturbation of the RNG streams): a
    `record_canonical` step must leave `gen_sigreg` AND `gen_mask` in exactly the state a
    raw/canonical step leaves them in, so a record-frame arm stays draw-for-draw aligned with the
    paired arms it is compared against.

    Mechanically this holds because `SigReg.__call__` draws exactly one `randn(K, n_directions)`
    per call and `K` is the projector width in every frame -- the frames differ in the SHAPE of
    the statistic's input, never in the direction draw. Pinned as a test rather than left as an
    argument, because a future frame that called the regularizer twice (say, to log a second
    statistic) would silently desynchronise every subsequent draw in the run."""
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi
    theta[2, :20] = float("nan")  # dilution present, but no record loses all its tokens

    states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for frame in ("raw", "canonical", "record_canonical"):
        gen_sigreg = generator(0, "sigreg")
        gen_mask = generator(0, "mask")
        train_step(
            _build_and_init(0),
            waveform,
            lambda_sig=0.1,
            gen_mask=gen_mask,
            gen_sigreg=gen_sigreg,
            theta=theta,
            operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
            sigreg_frame=frame,
        )
        states[frame] = (gen_sigreg.get_state(), gen_mask.get_state())

    for frame in ("canonical", "record_canonical"):
        assert torch.equal(states[frame][0], states["raw"][0]), f"{frame} moved gen_sigreg"
        assert torch.equal(states[frame][1], states["raw"][1]), f"{frame} moved gen_mask"
    # Non-vacuous: the streams did advance, so this is not three untouched generators matching.
    assert not torch.equal(states["raw"][0], generator(0, "sigreg").get_state())
    assert not torch.equal(states["raw"][1], generator(0, "mask").get_state())


def test_sigreg_frame_record_canonical_nan_theta_poisons_no_gradients() -> None:
    """The `0 * NaN` gradient-poisoning class this repo has been bitten by before
    (`winder.transport.loss`'s own `z_filled` comment): a batch carrying BOTH partially-invalid
    and fully-invalid theta rows must produce a finite loss AND finite gradients, with exactly
    zero gradient at every excluded token (never merely 'finite'), and real gradient at the valid
    tokens of the same records."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    gen = torch.Generator().manual_seed(0)
    theta = torch.rand(4, 250, generator=gen) * 2 * math.pi
    theta[1, 5:30] = float("nan")  # partially invalid
    theta[3, :] = float("nan")  # fully invalid: excluded from the statistic entirely

    z_leaf = model.projector.forward(model.encoder.forward(waveform)).detach().requires_grad_(True)
    templates, has_valid = record_canonical_templates(z_leaf, theta, operator)
    assert has_valid.tolist() == [True, True, True, False]
    model.regularizer(templates[has_valid], generator=generator(0, "sigreg")).backward()
    assert z_leaf.grad is not None
    assert bool(torch.isfinite(z_leaf.grad).all())
    assert bool((z_leaf.grad[3] == 0.0).all())  # the fully-invalid record: excluded, not zeroed
    assert bool((z_leaf.grad[1, 5:30] == 0.0).all())  # its invalid tokens contribute nothing
    assert bool((z_leaf.grad[1, 30:] != 0.0).any())  # its valid tokens still carry gradient

    # And end to end through train_step + backward, where a NaN would land in the parameters.
    loss, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator,
        sigreg_frame="record_canonical",
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(loss)
    assert metrics.sigreg_n_records == 3.0  # record 3 dropped: N is data-dependent here
    assert metrics.theta_valid_frac == pytest.approx(1.0 - (25 + 250) / (4 * 250))


def test_sigreg_frame_record_canonical_all_invalid_theta_batch_is_a_finite_zero_loss() -> None:
    """The documented zero-record decision (mirroring `winder.transport.loss`'s zero-pair
    safety): a batch in which NO record has a single valid theta contributes exactly zero SIGReg
    loss and zero SIGReg gradient -- never a NaN, and never a stack of zero vectors passed off to
    the regularizer as genuine 'collapsed template' samples. `sigreg_n_records == 0.0` marks the
    step in the history file."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.full((4, 250), float("nan"))

    gen_sigreg = generator(0, "sigreg")
    loss, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=gen_sigreg,
        theta=theta,
        operator=operator,
        lambda_pred=0.0,  # isolate the SIGReg term: the total loss must then be exactly zero
        sigreg_frame="record_canonical",
    )
    assert metrics.sigreg_loss == 0.0
    assert metrics.total_loss == 0.0
    assert metrics.sigreg_n_records == 0.0
    # The regularizer is never CALLED on such a step, so it draws no directions: gen_sigreg is
    # left exactly where it was (module docstring's degenerate-batch paragraph -- how much RNG a
    # call consumes is the regularizer's own property, so train_step must not fake a draw here).
    assert torch.equal(gen_sigreg.get_state(), generator(0, "sigreg").get_state())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(bool((g == 0.0).all()) for g in grads)


def test_sigreg_frame_record_canonical_without_theta_or_operator_raises() -> None:
    """Same guard as the token-level canonical frame, and for the same reason: the frame is a
    property of what SIGReg reads, not of whether L_trans is in total_loss -- so it fires at
    lambda_trans's own 0.0 default."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.zeros(4, 250)
    for kwargs in ({}, {"theta": theta}, {"operator": operator}):
        with pytest.raises(ValueError, match="record_canonical"):
            train_step(
                model,
                waveform,
                lambda_sig=0.1,
                gen_mask=generator(0, "mask"),
                gen_sigreg=generator(0, "sigreg"),
                sigreg_frame="record_canonical",
                **kwargs,  # type: ignore[arg-type]
            )


def test_unknown_sigreg_frame_still_raises_and_names_every_supported_frame() -> None:
    """The closed set stays enforced at runtime (`sigreg_frame` is typed `str`, not a Literal --
    module docstring's OmegaConf paragraph), and the message must name the new frame too, so a
    typo'd arm in a launcher file is diagnosable from the traceback alone."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    with pytest.raises(ValueError) as excinfo:
        train_step(
            model,
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            sigreg_frame="record-canonical",  # a plausible typo: hyphen, not underscore
        )
    message = str(excinfo.value)
    assert "record-canonical" in message  # the offending value, echoed back
    for frame in ("raw", "canonical", "record_canonical"):
        assert frame in message


def test_sigreg_frame_record_canonical_gradients_reach_encoder_and_projector() -> None:
    """Gradient sanity through the record mean, with the predictor structurally skipped
    (lambda_pred=0.0, the eq-28 configuration this arm will actually train in): the SIGReg term
    alone must deliver finite, nonzero gradient to BOTH the encoder and the projector."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi

    loss, _ = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=operator,
        lambda_pred=0.0,
        sigreg_frame="record_canonical",
    )
    loss.backward()

    def _check(params: Iterator[torch.nn.Parameter], name: str) -> None:
        grads = [p.grad for p in params if p.grad is not None]
        assert len(grads) > 0, f"{name} received no gradient at all"
        assert all(torch.isfinite(g).all() for g in grads), f"{name} gradient is non-finite"
        assert any(bool((g != 0.0).any()) for g in grads), f"{name} gradient is identically zero"

    # Both submodules are nn.Modules in every registered variant, but the Encoder/ProjectionHead
    # protocols themselves do not declare .parameters() -- pull them off the model's own
    # nn.Module view by name rather than annotating the protocol away.
    _check(model.get_submodule("encoder").parameters(), "encoder")
    _check(model.get_submodule("projector").parameters(), "projector")


def test_fit_passes_record_canonical_sigreg_frame_through_to_every_train_step() -> None:
    """The plumbing sibling of `test_fit_passes_cfg_sigreg_frame_through_to_every_train_step`:
    `cfg.sigreg_frame="record_canonical"` must reach `train_step` (a `fit` that silently kept the
    `"raw"` default would still train, just not the arm the config claims) -- detected here via
    `sigreg_n_records`, which only the record frame ever fills in."""
    model = _build_and_init(0)
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = TrainConfig(
        n_steps=3,
        lambda_sig=0.1,
        lambda_trans=0.0,  # deliberately OFF, as for the token-level canonical frame
        sigreg_frame="record_canonical",
        seed_pretrain=0,
        warmup_steps=1,
    )
    gen_data = generator(0, "synth")
    batches = [synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)]
    theta_batches = [torch.rand(4, 250) * 2 * math.pi for _ in range(cfg.n_steps)]

    history = fit(
        model, iter(batches), cfg, optimizer, theta_batches=iter(theta_batches), operator=operator
    )
    assert len(history) == 3
    assert all(m.sigreg_n_records == 4.0 for m in history)
    assert all(math.isfinite(m.sigreg_loss) for m in history)


# ================================================ lambda_sig_record: the X7 two-term repair
#
# X6 (sigreg_frame="record_canonical") REPLACED the token-level statistic with the record-level
# one and collapsed at all three doses: transport gain +0.0000, trans_floor == trans_loss in every
# arm (the latents fell into the non-rotating k0 invariant block, where the objective is trivially
# satisfiable), pred_loss ~0.0001 at the two low doses. Diagnosis: a record-level statistic
# constrains only the ~63 per-record MEANS and is therefore structurally BLIND to within-record
# degeneracy -- if every token of a record is identical, ubar_r is just that vector and the
# template distribution can still look isotropic. The token-level term had been doing anti-collapse
# work across 125 timesteps x 64 records that no record-level statistic can do.
#
# X7 therefore KEEPS the token-level term (in the RAW frame, the working incumbent W3's own) and
# ADDS the record-level one:
#
#     L = lambda_pred*L_pred + lambda_sig*S({z_t}) + lambda_sig_record*S({ubar_r})
#         + lambda_trans*L_trans
#
# -- the record term supplies template isotropy, the token term supplies anti-collapse. The
# non-negotiable invariants the tests below pin: (a) lambda_sig_record == 0.0 is a STRUCTURAL SKIP
# leaving every existing caller bitwise identical, and (b) the record term's SECOND regularizer
# call draws from its OWN generator stream, so `gen_sigreg` is left draw-for-draw where the paired
# arms leave it.
# Same operator sizing as the blocks above (_tiny_config's projector output_width == 32).


def test_lambda_sig_record_zero_reproduces_the_golden_raw_and_canonical_values_bitwise() -> None:
    """The X7 structural-skip invariant, against the SAME literals
    `test_raw_and_canonical_sigreg_frames_match_their_pre_record_canonical_golden_values` pins --
    captured from the pre-`lambda_sig_record` implementation, asserted EXACTLY. Passing
    `lambda_sig_record=0.0` AND a `gen_sigreg_record` generator explicitly must move nothing: no
    second regularizer call, no extra draw, no perturbation of the two token-level frames.

    A failure here means the record term is not a real branch (e.g. a `+ 0.0 * S_rec` term, which
    would also poison `0 * NaN` on an all-invalid-theta batch). Regenerate these literals only to
    accept a reviewed, intended change to raw/canonical-frame behaviour."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
        theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(4242)) * 2 * math.pi
        theta[0, :7] = float("nan")  # M0's own pre-first-R-peak dilution shape
        theta[3, 100] = float("nan")

        gen_record_raw = generator(0, "sigreg_record")
        _, m_raw = train_step(
            _build_and_init(0),
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            gen_sigreg_record=gen_record_raw,
            theta=theta,
            operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
            lambda_sig_record=0.0,
        )
        assert m_raw.pred_loss == 0.08325359225273132
        assert m_raw.persistence_loss == 0.0006865973700769246
        assert m_raw.sigreg_loss == 1.7095328569412231
        assert m_raw.total_loss == 0.25420689582824707
        assert (m_raw.n_context, m_raw.n_target, m_raw.cutoff_mean) == (382, 4, 94.5)
        # "Not applicable", the same NaN sentinel convention as the four trans_* fields.
        assert math.isnan(m_raw.sigreg_record_loss)
        assert math.isnan(m_raw.sigreg_n_records)
        # The record stream was never drawn from: the branch does not exist at lambda 0.0.
        assert torch.equal(gen_record_raw.get_state(), generator(0, "sigreg_record").get_state())

        gen_record_canon = generator(0, "sigreg_record")
        _, m_canon = train_step(
            _build_and_init(0),
            waveform,
            lambda_sig=0.1,
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            gen_sigreg_record=gen_record_canon,
            theta=theta,
            operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
            lambda_trans=1.0,
            sigreg_frame="canonical",
            lambda_sig_record=0.0,
        )
        assert m_canon.pred_loss == 0.08325359225273132
        assert m_canon.persistence_loss == 0.0006865973700769246
        assert m_canon.sigreg_loss == 1.5687761306762695
        assert m_canon.total_loss == 1.0526559352874756
        assert m_canon.trans_loss == 0.812524676322937
        assert m_canon.trans_floor == 0.8138893842697144
        assert m_canon.theta_valid_frac == 0.9919999837875366
        assert math.isnan(m_canon.sigreg_record_loss)
        assert torch.equal(gen_record_canon.get_state(), generator(0, "sigreg_record").get_state())
    finally:
        torch.set_num_threads(original_threads)


def test_lambda_sig_record_zero_is_bitwise_identical_to_never_passing_the_new_arguments() -> None:
    """The same discipline `test_lambda_trans_zero_is_bitwise_identical_whether_theta_and_operator
    _are_passed` enforces for the transport skip, for X7's own two new arguments: a caller that
    passes NEITHER `lambda_sig_record` NOR `gen_sigreg_record` (i.e. every arm already launched)
    and a caller that passes both at their inert values must agree on the loss TENSOR and on every
    StepMetrics field, with both named streams left in the same state."""
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    gen_mask_a, gen_sigreg_a = generator(0, "mask"), generator(0, "sigreg")
    gen_mask_b, gen_sigreg_b = generator(0, "mask"), generator(0, "sigreg")

    loss_a, metrics_a = train_step(
        _build_and_init(0), waveform, lambda_sig=0.1, gen_mask=gen_mask_a, gen_sigreg=gen_sigreg_a
    )
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi
    theta[1, :5] = float("nan")  # a NaN present in theta must not leak through at lambda 0.0
    loss_b, metrics_b = train_step(
        _build_and_init(0),
        waveform,
        lambda_sig=0.1,
        gen_mask=gen_mask_b,
        gen_sigreg=gen_sigreg_b,
        gen_sigreg_record=generator(0, "sigreg_record"),
        theta=theta,
        operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
        lambda_sig_record=0.0,
    )

    assert torch.equal(loss_a, loss_b)
    # dataclasses.astuple + NaN-aware comparison (the transport-skip test's own idiom): a plain
    # `metrics_a == metrics_b` reports unequal purely because NaN != NaN in IEEE semantics.
    for field_a, field_b in zip(
        dataclasses.astuple(metrics_a), dataclasses.astuple(metrics_b), strict=True
    ):
        if isinstance(field_a, float) and math.isnan(field_a):
            assert math.isnan(field_b)
        else:
            assert field_a == field_b
    assert gen_mask_a.get_state().equal(gen_mask_b.get_state())
    assert gen_sigreg_a.get_state().equal(gen_sigreg_b.get_state())


def test_gen_sigreg_stream_is_untouched_by_an_active_record_term() -> None:
    """X7's OTHER non-negotiable invariant, and the subtle one: the record term needs a SECOND
    `model.regularizer` call, and `SigReg.__call__` draws `randn(K, n_directions)` from whatever
    generator it is handed. A naive second call on the shared `gen_sigreg` would advance that
    stream and desynchronise every subsequent draw in the run, breaking comparability with the
    paired arms (W3 and the X ladder) this arm is read against.

    So: after one step, `gen_sigreg` and `gen_mask` must be bitwise where a `lambda_sig_record=0`
    step leaves them, whatever the record dose -- while `gen_sigreg_record` itself advances only
    when the term is active."""
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi
    theta[2, :20] = float("nan")

    states: dict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for dose in (0.0, 0.3, 5.0):
        gen_mask, gen_sigreg = generator(0, "mask"), generator(0, "sigreg")
        gen_sigreg_record = generator(0, "sigreg_record")
        train_step(
            _build_and_init(0),
            waveform,
            lambda_sig=0.1,
            gen_mask=gen_mask,
            gen_sigreg=gen_sigreg,
            gen_sigreg_record=gen_sigreg_record,
            theta=theta,
            operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
            lambda_sig_record=dose,
        )
        states[dose] = (
            gen_sigreg.get_state(),
            gen_mask.get_state(),
            gen_sigreg_record.get_state(),
        )

    fresh_sigreg = generator(0, "sigreg").get_state()
    fresh_mask = generator(0, "mask").get_state()
    fresh_record = generator(0, "sigreg_record").get_state()
    for dose in (0.3, 5.0):
        assert torch.equal(states[dose][0], states[0.0][0]), f"dose {dose} moved gen_sigreg"
        assert torch.equal(states[dose][1], states[0.0][1]), f"dose {dose} moved gen_mask"
        # ...and the record stream DID advance, so the invariant above is not two idle generators
        # matching (the failure mode a `pass`-shaped implementation would sail through).
        assert not torch.equal(states[dose][2], fresh_record), f"dose {dose} drew no directions"
    # Non-vacuity on the other side: the shared streams advanced at all this step.
    assert not torch.equal(states[0.0][0], fresh_sigreg)
    assert not torch.equal(states[0.0][1], fresh_mask)
    assert torch.equal(states[0.0][2], fresh_record)  # inert at dose 0.0


def test_lambda_sig_record_composes_with_the_raw_token_term_component_by_component() -> None:
    """THE composition claim, against INDEPENDENTLY computed components (never against itself):
    with `sigreg_frame="raw"` the token term keeps the raw frame and the record term is added on
    top, so

        total = lambda_pred*L_pred + lambda_sig*S(raw tokens)
                + lambda_sig_record*S(templates) + lambda_trans*L_trans

    The reference below is a from-scratch rebuild on a separately built, identically seeded model
    (the `test_lambda_pred_zero_is_bitwise_identical_to_a_hand_built_...` idiom -- one model called
    twice would give `SeededDropout` a different mask on its second predictor call), accumulated in
    `train_step`'s own order so the comparison can be exact rather than approximate."""
    model_ts, model_ref = _build_and_init(0), _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(11)) * 2 * math.pi
    theta[0, :9] = float("nan")
    operator_ts = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    operator_ref = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    lam_sig, lam_rec, lam_trans = 0.1, 0.3, 1.0

    loss, metrics = train_step(
        model_ts,
        waveform,
        lambda_sig=lam_sig,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        gen_sigreg_record=generator(0, "sigreg_record"),
        theta=theta,
        operator=operator_ts,
        lambda_trans=lam_trans,
        lambda_sig_record=lam_rec,
        sigreg_frame="raw",
    )

    b, n_tokens = waveform.shape[0], int(model_ref.config.n_tokens)
    plan = model_ref.mask_sampler(b, n_tokens, generator=generator(0, "mask"))
    z = model_ref.projector.forward(model_ref.encoder.forward(waveform))
    s_token = model_ref.regularizer(z.transpose(0, 1), generator=generator(0, "sigreg"))
    templates, has_valid = record_canonical_templates(z, theta, operator_ref)
    s_record = model_ref.regularizer(templates[has_valid], generator=generator(0, "sigreg_record"))
    z_hat = model_ref.predictor.forward(z, ~plan.context)
    pred = model_ref.prediction_loss(z_hat, z, plan.target)
    trans = transport_loss(z, theta, operator_ref).loss
    reference = 1.0 * pred + (lam_sig * s_token + lam_rec * s_record) + lam_trans * trans

    # Component-exact: each statistic is the independently computed one, not merely close.
    assert metrics.sigreg_loss == float(s_token.detach())
    assert metrics.sigreg_record_loss == float(s_record.detach())
    assert metrics.sigreg_n_records == 4.0  # every record kept at least one valid token
    assert torch.equal(loss, reference)
    # Discriminating: the two statistics are genuinely different numbers on this batch, so the
    # equality above is not one any single-term implementation would satisfy.
    assert metrics.sigreg_loss != metrics.sigreg_record_loss
    assert metrics.total_loss == pytest.approx(
        metrics.pred_loss
        + lam_sig * metrics.sigreg_loss
        + lam_rec * metrics.sigreg_record_loss
        + lam_trans * metrics.trans_loss,
        rel=1e-6,
    )
    # And the record term really is IN the total: the same step at dose 0.0 is a different loss.
    _, metrics_off = train_step(
        _build_and_init(0),
        waveform,
        lambda_sig=lam_sig,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        theta=theta,
        operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
        lambda_trans=lam_trans,
    )
    assert metrics_off.sigreg_loss == metrics.sigreg_loss  # the token term is untouched...
    assert metrics_off.total_loss != metrics.total_loss  # ...and only the record term was added


def test_lambda_sig_record_nan_theta_poisons_no_gradients() -> None:
    """The `0 * NaN` gradient-poisoning class this repo has been bitten by before
    (`winder.transport.loss`'s own `z_filled` comment): with the record term ACTIVE, a batch
    carrying both partially-invalid and fully-invalid theta rows must give a finite loss AND finite
    gradients everywhere -- forward finiteness alone has hidden this bug here before."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi
    theta[1, 5:30] = float("nan")  # partially invalid
    theta[3, :] = float("nan")  # fully invalid: dropped from the record statistic entirely

    loss, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.1,
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        gen_sigreg_record=generator(0, "sigreg_record"),
        theta=theta,
        operator=operator,
        lambda_sig_record=0.5,
    )
    assert torch.isfinite(loss)
    assert math.isfinite(metrics.sigreg_record_loss)
    assert metrics.sigreg_n_records == 3.0  # record 3 dropped: N is data-dependent here
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_lambda_sig_record_all_invalid_theta_batch_is_a_finite_zero_record_term() -> None:
    """The degenerate batch, mirroring the record_canonical FRAME's own documented decision: no
    record has a single valid theta, so the record statistic is undefined -- it contributes exactly
    0.0 (the empty selection's own graph-attached `.sum()`, a well-defined backward no-op, never a
    NaN and never a stack of zero vectors passed off as genuine collapsed-template samples), the
    token term is untouched, and `gen_sigreg_record` is NOT advanced (how much RNG a call consumes
    is the regularizer's own property -- `NoRegularizer` consumes none -- not something `train_step`
    may fake)."""
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    gen_sigreg_record = generator(0, "sigreg_record")

    loss, metrics = train_step(
        model,
        waveform,
        lambda_sig=0.0,  # isolate the record term: total_loss must then be exactly zero
        gen_mask=generator(0, "mask"),
        gen_sigreg=generator(0, "sigreg"),
        gen_sigreg_record=gen_sigreg_record,
        theta=torch.full((4, 250), float("nan")),
        operator=operator,
        lambda_pred=0.0,
        lambda_sig_record=1.0,
    )
    assert metrics.sigreg_record_loss == 0.0  # computed-and-zero, NOT the NaN sentinel
    assert metrics.sigreg_n_records == 0.0
    assert metrics.total_loss == 0.0
    assert torch.equal(gen_sigreg_record.get_state(), generator(0, "sigreg_record").get_state())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(bool((g == 0.0).all()) for g in grads)


def test_lambda_sig_record_guards_theta_operator_generator_and_the_double_dose() -> None:
    """Fail-fast guards, all four at the shape the frames' own guards already take (a raise the
    traceback alone diagnoses, not a silently different arm):

    - no `theta` / no `operator`: the record term demodulates, so it needs both -- independent of
      `lambda_trans`, exactly like the canonical frames.
    - no `gen_sigreg_record`: `train_step` must not invent one (it has no seed, and a per-step
      fresh generator would freeze the direction draw across the whole run, silently deleting the
      per-call resampling LeJEPA Sec 4.3 makes load-bearing).
    - `sigreg_frame="record_canonical"` together with a nonzero dose: that is the SAME statistic
      twice with two independent direction draws -- a silently double-dosed record term, which is
      exactly the arm X7 is not.
    """
    model = _build_and_init(0)
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
    theta = torch.zeros(4, 250)
    base: dict[str, object] = {
        "lambda_sig": 0.1,
        "gen_mask": generator(0, "mask"),
        "gen_sigreg": generator(0, "sigreg"),
        "lambda_sig_record": 0.5,
    }

    for kwargs in ({}, {"theta": theta}, {"operator": operator}):
        with pytest.raises(ValueError, match="lambda_sig_record"):
            train_step(
                model,
                waveform,
                gen_sigreg_record=generator(0, "sigreg_record"),
                **{**base, **kwargs},  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="gen_sigreg_record"):
        train_step(
            model,
            waveform,
            theta=theta,
            operator=operator,
            **base,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="record_canonical"):
        train_step(
            model,
            waveform,
            theta=theta,
            operator=operator,
            gen_sigreg_record=generator(0, "sigreg_record"),
            sigreg_frame="record_canonical",
            **base,  # type: ignore[arg-type]
        )


def test_lambda_sig_record_gradients_reach_encoder_and_projector_from_the_record_branch() -> None:
    """Gradient flow attributable to the record branch ALONE: the token term is weighted to 0.0
    (still computed, so its own `0.0 * S_tok` contributes exactly zero gradient) and the predictor
    is structurally skipped, leaving the record term as the only live path to the parameters.

    The control below is what makes this non-vacuous: the identical call at dose 0.0 leaves every
    parameter gradient exactly zero, so the nonzero gradients above come from the record term and
    not from some other term leaking in."""
    waveform = synthetic_waveform_batch(4, generator=generator(0, "synth"))
    theta = torch.rand(4, 250, generator=torch.Generator().manual_seed(0)) * 2 * math.pi

    def _step(dose: float) -> JepaModel:
        model = _build_and_init(0)
        loss, _ = train_step(
            model,
            waveform,
            lambda_sig=0.0,  # the token term weighted out
            gen_mask=generator(0, "mask"),
            gen_sigreg=generator(0, "sigreg"),
            gen_sigreg_record=generator(0, "sigreg_record"),
            theta=theta,
            operator=CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4])),
            lambda_pred=0.0,  # the predictor structurally skipped
            lambda_sig_record=dose,
        )
        loss.backward()
        return model

    active = _step(1.0)
    for name in ("encoder", "projector"):
        grads = [p.grad for p in active.get_submodule(name).parameters() if p.grad is not None]
        assert len(grads) > 0, f"{name} received no gradient from the record term"
        assert all(torch.isfinite(g).all() for g in grads), f"{name} gradient is non-finite"
        assert any(bool((g != 0.0).any()) for g in grads), f"{name} gradient is identically zero"

    control = _step(0.0)
    for name in ("encoder", "projector"):
        grads = [p.grad for p in control.get_submodule(name).parameters() if p.grad is not None]
        assert all(bool((g == 0.0).all()) for g in grads), f"{name} got gradient with no live term"


def test_fit_passes_cfg_lambda_sig_record_through_to_every_train_step() -> None:
    """The plumbing sibling of `test_fit_passes_cfg_sigreg_frame_through_to_every_train_step`: a
    `fit` that silently kept `train_step`'s own 0.0 default would still train, just not the arm the
    config claims -- detected here via `sigreg_record_loss`, which only the active term fills in,
    and via a `total_loss` that differs from the same run at dose 0.0."""
    operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))

    def _run(dose: float) -> list[StepMetrics]:
        model = _build_and_init(0)
        cfg = TrainConfig(
            n_steps=3,
            lambda_sig=0.1,
            lambda_trans=0.0,  # deliberately OFF: the record term must not need transport active
            lambda_sig_record=dose,
            seed_pretrain=0,
            warmup_steps=1,
        )
        gen_data = generator(0, "synth")
        batches = [synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)]
        theta_batches = [
            torch.rand(4, 250, generator=torch.Generator().manual_seed(s)) * 2 * math.pi
            for s in range(cfg.n_steps)
        ]
        return fit(
            model,
            iter(batches),
            cfg,
            torch.optim.AdamW(model.parameters(), lr=3e-4),
            theta_batches=iter(theta_batches),
            operator=operator,
        )

    history = _run(0.4)
    assert len(history) == 3
    assert all(math.isfinite(m.sigreg_record_loss) for m in history)
    assert all(m.sigreg_record_loss > 0.0 for m in history)
    assert all(m.sigreg_n_records == 4.0 for m in history)
    off = _run(0.0)
    assert all(math.isnan(m.sigreg_record_loss) for m in off)
    assert history[0].total_loss != off[0].total_loss


def test_fit_default_sigreg_record_generator_matches_the_explicit_construction() -> None:
    """The X7 twin of `test_fit_explicit_generators_match_the_default_internal_construction`, and
    what CKPT-01 rests on: `fit`'s own default construction of the record stream must be the SAME
    generator a resuming caller builds by hand as `generator(cfg.seed_pretrain, "sigreg_record")`
    -- not merely a similarly-seeded one, or a saved state would not correspond to what an
    uninterrupted run used internally."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
        cfg = TrainConfig(
            n_steps=4,
            lambda_sig=0.1,
            lambda_sig_record=0.4,
            seed_pretrain=7,
            warmup_steps=1,
        )

        def _inputs() -> tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]]:
            gen_data = generator(7, "synth")
            waveforms = [
                synthetic_waveform_batch(4, generator=gen_data) for _ in range(cfg.n_steps)
            ]
            thetas = [
                torch.rand(4, 250, generator=torch.Generator().manual_seed(s)) * 2 * math.pi
                for s in range(cfg.n_steps)
            ]
            return iter(waveforms), iter(thetas)

        model_a = _build_and_init(7)
        waveforms_a, thetas_a = _inputs()
        history_a = fit(
            model_a,
            waveforms_a,
            cfg,
            torch.optim.AdamW(model_a.parameters(), lr=3e-4),
            theta_batches=thetas_a,
            operator=operator,
        )

        model_b = _build_and_init(7)
        waveforms_b, thetas_b = _inputs()
        history_b = fit(
            model_b,
            waveforms_b,
            cfg,
            torch.optim.AdamW(model_b.parameters(), lr=3e-4),
            gen_mask=generator(cfg.seed_pretrain, "mask"),
            gen_sigreg=generator(cfg.seed_pretrain, "sigreg"),
            gen_sigreg_record=generator(cfg.seed_pretrain, "sigreg_record"),
            theta_batches=thetas_b,
            operator=operator,
        )
        assert [m.sigreg_record_loss for m in history_a] == [
            m.sigreg_record_loss for m in history_b
        ]
        assert [m.total_loss for m in history_a] == [m.total_loss for m in history_b]
    finally:
        torch.set_num_threads(original_threads)


def test_ckpt04_exact_resume_with_the_record_term_needs_the_sigreg_record_stream(
    tmp_path: Path,
) -> None:
    """CKPT-01/CKPT-04 extended to X7's third stream: the record term draws fresh directions every
    active step, so `"sigreg_record"` is training-run state a resume must restore -- not an
    optional extra. Structure mirrors
    `test_ckpt04_exact_resume_matches_uninterrupted_reference` (same pinned thread count, same
    `cfg.n_steps` across all three fits, phase 1 limited by its own iterator length).

    The negative control at the end is the load-bearing half: a resume that restores mask/sigreg
    but REPLAYS the record stream from seed diverges from the reference, which is what makes
    "checkpoint this stream" a requirement rather than a nicety."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        k = 3
        operator = CyclicOperator(CyclicOperatorConfig(k0=8, n_j=[1, 2, 3], k_j=[4, 4, 4]))
        cfg = TrainConfig(
            n_steps=k + 1,
            lambda_sig=0.1,
            lambda_sig_record=0.4,
            seed_pretrain=0,
            warmup_steps=1,
        )

        def _waveforms(gen_data: torch.Generator) -> Iterator[torch.Tensor]:
            while True:
                yield synthetic_waveform_batch(4, generator=gen_data)

        def _thetas(gen_theta: torch.Generator) -> Iterator[torch.Tensor]:
            while True:
                yield torch.rand(4, 250, generator=gen_theta) * 2 * math.pi

        # ---- Reference: uninterrupted, K+1 steps, fit()'s own default generator construction. ----
        ref_model = _build_and_init(0)
        ref_data, ref_theta = generator(0, "ckpt04x7_data"), generator(0, "ckpt04x7_theta")
        ref_history = fit(
            ref_model,
            itertools.islice(_waveforms(ref_data), k + 1),
            cfg,
            torch.optim.AdamW(ref_model.parameters(), lr=3e-4),
            theta_batches=itertools.islice(_thetas(ref_theta), k + 1),
            operator=operator,
        )
        assert len(ref_history) == k + 1
        assert all(math.isfinite(m.sigreg_record_loss) for m in ref_history)

        # ---- Phase 1: explicit generators, only the first K steps, then checkpoint. ----
        model1 = _build_and_init(0)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=3e-4)
        gen_mask1, gen_sigreg1 = generator(0, "mask"), generator(0, "sigreg")
        gen_record1 = generator(0, "sigreg_record")
        gen_data1, gen_theta1 = generator(0, "ckpt04x7_data"), generator(0, "ckpt04x7_theta")
        history1 = fit(
            model1,
            itertools.islice(_waveforms(gen_data1), k),
            cfg,
            optimizer1,
            gen_mask=gen_mask1,
            gen_sigreg=gen_sigreg1,
            gen_sigreg_record=gen_record1,
            theta_batches=itertools.islice(_thetas(gen_theta1), k),
            operator=operator,
        )
        assert len(history1) == k

        ckpt_dir = os.path.join(str(tmp_path), "ckpt")
        checkpoint.save_checkpoint(
            ckpt_dir,
            model=model1,
            optimizer=optimizer1,
            step=history1[-1].step + 1,
            generators={
                "mask": gen_mask1,
                "sigreg": gen_sigreg1,
                "sigreg_record": gen_record1,
                "data": gen_data1,
                "theta": gen_theta1,
            },
            config_yaml="jepa: {}\ntrain: {}\n",
            meta={"note": "CKPT-04 x X7 record-term resume test"},
        )

        def _resume(replay_record_stream: bool) -> StepMetrics:
            model2 = _build_and_init(1)  # a different init seed: must be overwritten by the load
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
            loaded = checkpoint.load_checkpoint(ckpt_dir, model=model2, optimizer=optimizer2)
            assert loaded.step == k
            streams: dict[str, torch.Generator] = {}
            for name in ("mask", "sigreg", "sigreg_record", "data", "theta"):
                gen = torch.Generator()
                gen.set_state(loaded.generator_states[name])
                streams[name] = gen
            if replay_record_stream:  # the negative control: reseeded, not restored
                streams["sigreg_record"] = generator(0, "sigreg_record")
            history2 = fit(
                model2,
                itertools.islice(_waveforms(streams["data"]), 1),
                cfg,
                optimizer2,
                start_step=loaded.step,
                gen_mask=streams["mask"],
                gen_sigreg=streams["sigreg"],
                gen_sigreg_record=streams["sigreg_record"],
                theta_batches=itertools.islice(_thetas(streams["theta"]), 1),
                operator=operator,
            )
            assert len(history2) == 1  # a resume that ran zero steps must fail loudly here
            return history2[0]

        resumed = _resume(replay_record_stream=False)
        assert resumed.step == k
        assert resumed.sigreg_loss == ref_history[k].sigreg_loss
        assert resumed.sigreg_record_loss == ref_history[k].sigreg_record_loss
        assert resumed.total_loss == ref_history[k].total_loss

        replayed = _resume(replay_record_stream=True)
        assert replayed.sigreg_record_loss != ref_history[k].sigreg_record_loss
    finally:
        torch.set_num_threads(original_threads)

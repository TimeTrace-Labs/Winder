"""`JepaModel`'s own eval-surface accessors: `embed` (the local, context-free encoder output) and
`predictor_hidden_states` (the accessor shared by the probe repointing and the anomaly score for the
predictor's causal hidden state at every position, from a single unmasked forward pass).
"""

import torch

from winder.determinism import generator, init_parameters
from winder.jepa.model import JepaConfig, JepaModel, build_jepa


def _tiny_config() -> JepaConfig:
    return JepaConfig(
        n_leads=12,
        n_samples=1000,
        n_tokens=125,
        encoder_name="patch",
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


def _build_and_init(seed: int = 0) -> JepaModel:
    config = _tiny_config()
    model = build_jepa(config, generator=generator(seed, "handshake"))
    init_parameters(model, generator(seed, "init"))
    return model


def test_predictor_hidden_states_shape() -> None:
    model = _build_and_init()
    model.eval()
    waveform = torch.randn(3, 12, 1000)
    with torch.no_grad():
        hidden = model.predictor_hidden_states(waveform)
    assert hidden.shape == (3, 125, 32)


def test_predictor_hidden_states_matches_an_unmasked_predictor_call() -> None:
    """Cross-check against the composition by hand: encoder -> projector -> predictor with an
    all-False mask -- confirms the method is exactly that, not a reimplementation that happens
    to agree."""
    model = _build_and_init()
    model.eval()
    waveform = torch.randn(2, 12, 1000)
    with torch.no_grad():
        hidden = model.predictor_hidden_states(waveform)
        projected = model.projector.forward(model.encoder.forward(waveform))
        no_mask = torch.zeros(2, 125, dtype=torch.bool)
        expected = model.predictor.forward(projected, no_mask)
    torch.testing.assert_close(hidden, expected)


def test_predictor_hidden_states_is_causal() -> None:
    """CM-02's own guarantee, exercised through this accessor specifically: perturbing every
    sample strictly after token c's own patch must leave hidden states at tokens `<= c`
    unchanged."""
    model = _build_and_init()
    model.eval()
    torch.manual_seed(0)
    waveform = torch.randn(1, 12, 1000)
    cutoff_token = 60
    perturb_from_sample = (cutoff_token + 1) * 8  # PatchEncoderConfig's own default patch_width
    with torch.no_grad():
        baseline = model.predictor_hidden_states(waveform)
        perturbed_waveform = waveform.clone()
        perturbed_waveform[:, :, perturb_from_sample:] += 1000.0
        perturbed = model.predictor_hidden_states(perturbed_waveform)
    torch.testing.assert_close(
        baseline[:, : cutoff_token + 1], perturbed[:, : cutoff_token + 1], rtol=0.0, atol=1e-6
    )
    # Not vacuous: later tokens must actually move.
    assert not torch.allclose(baseline[:, cutoff_token + 1 :], perturbed[:, cutoff_token + 1 :])


def test_predictor_hidden_states_never_substitutes_a_mask_token() -> None:
    """Distinguishes this accessor from `predictor.forward` under a real (some-True) mask: with
    every position unmasked, no position's own input is ever replaced -- confirmed by checking
    the accessor's result differs from calling the predictor with an arbitrary nonzero mask on
    the SAME projected tokens."""
    model = _build_and_init()
    model.eval()
    waveform = torch.randn(2, 12, 1000)
    with torch.no_grad():
        hidden = model.predictor_hidden_states(waveform)
        projected = model.projector.forward(model.encoder.forward(waveform))
        some_masked = torch.zeros(2, 125, dtype=torch.bool)
        some_masked[:, 100:] = True
        masked_result = model.predictor.forward(projected, some_masked)
    assert not torch.allclose(hidden[:, 100:], masked_result[:, 100:])


def test_embed_and_predictor_hidden_states_have_different_widths() -> None:
    """The probe repointing's own point: `embed`'s width is the ENCODER's own latent_width (256 by
    default, local/context-free); `predictor_hidden_states`'s width is the PREDICTOR's own width
    (32 in this tiny config, but the point holds generally) -- distinct eval surfaces, not two
    names for the same tensor."""
    model = _build_and_init()
    model.eval()
    waveform = torch.randn(2, 12, 1000)
    with torch.no_grad():
        embedded = model.embed(waveform)
        hidden = model.predictor_hidden_states(waveform)
    assert embedded.shape == (2, 125, model.encoder.latent_width)
    assert hidden.shape == (2, 125, model.predictor.width)

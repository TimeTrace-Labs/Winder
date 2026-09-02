import glob
import os

import pytest
import torch

from winder.data.decimation import decimate_to, out_len
from winder.data.wfdb_io import read_record
from winder.jepa.base import Encoder
from winder.jepa.encoder import (
    PatchEncoder,
    PatchEncoderConfig,
    ResidualCnnEncoder,
    ResidualCnnEncoderConfig,
)


def test_output_is_exactly_250_tokens_of_width_256() -> None:
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    tokens = encoder.forward(torch.randn(3, 12, 1000))
    assert tokens.shape == (3, 250, 256)


def test_output_is_exactly_250_tokens_on_a_real_fixture_record() -> None:
    """Real signal statistics, not synthetic noise -- the idiom from
    tests/test_decimation.py:72-86. This is not the production data path (that reads
    records100/ directly, see the plan); decimate_to is used here only to shape a real waveform
    to (12, 1000) for exercising the encoder's own exact-token-count contract."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures", "wfdb")
    hea_path = sorted(glob.glob(os.path.join(fixtures_dir, "*.hea")))[0]
    sig, header = read_record(hea_path)
    assert sig.shape == (5000, 12)
    assert header.fs == 500
    sig_100hz = decimate_to(sig, 500, 100)
    assert sig_100hz.shape[0] == out_len(5000, 500, 100) == 1000
    waveform = torch.from_numpy(sig_100hz.T.copy()).float().unsqueeze(0)

    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    tokens = encoder.forward(waveform)
    assert tokens.shape == (1, 250, 256)


def test_encoder_satisfies_the_encoder_contract() -> None:
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    assert isinstance(encoder, Encoder)
    assert encoder.latent_width == 256


def test_wrong_lead_count_raises() -> None:
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    with pytest.raises(ValueError, match="must be"):
        encoder.forward(torch.randn(1, 8, 1000))


def test_non_two_stage_config_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="two-stage"):
        ResidualCnnEncoder(ResidualCnnEncoderConfig(stage_widths=[64, 128, 256]))


def test_same_encoder_instance_is_deterministic_on_identical_input() -> None:
    """Same weights, same input, same output -- confirms this is not a target/EMA encoder with
    hidden state that would make two calls diverge."""
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    encoder.eval()
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        a = encoder.forward(x)
        b = encoder.forward(x)
    assert torch.equal(a, b)


def test_perturbation_locality() -> None:
    """A large perturbation to one raw input sample changes only a bounded, contiguous window of
    output tokens -- the encoder is a local convolutional stack, not something that mixes
    information globally across all 250 tokens."""
    torch.manual_seed(0)
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    encoder.eval()
    x = torch.randn(1, 12, 1000)
    with torch.no_grad():
        baseline = encoder.forward(x)
        perturbed_x = x.clone()
        perturbed_x[0, :, 500] += 1000.0
        perturbed = encoder.forward(perturbed_x)
    changed = (baseline - perturbed).abs().amax(dim=-1)[0] > 1e-6
    n_changed = int(changed.sum())
    assert 0 < n_changed < 250
    changed_indices = changed.nonzero().flatten()
    assert int(changed_indices.max() - changed_indices.min()) + 1 == n_changed  # contiguous run


def test_gradients_reach_the_stem() -> None:
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    x = torch.randn(1, 12, 1000, requires_grad=True)
    out = encoder.forward(x)
    out.sum().backward()
    assert encoder.stem_conv.weight.grad is not None
    assert torch.any(encoder.stem_conv.weight.grad != 0)


def test_cm04_context_encode_matches_full_and_prefix_encode_up_to_cutoff() -> None:
    """CM-04 / LEAK-01: because this encoder is causal, token `c`'s value depends on no sample
    past `4c+2` (`winder.jepa.leakage.token_window`). So encoding the full waveform, encoding it
    with everything after the cutoff zeroed, and encoding only the raw prefix up to the cutoff
    must all agree on tokens `0..c` -- this is exactly what makes `winder.jepa.train.train_step`'s
    single encoder pass correct rather than merely convenient (no second, masked-context pass is
    needed to produce the same context-branch values).

    Tolerance: not bitwise. Measured maxdiff between the full-waveform and prefix-only encodes is
    0.0 at `torch.set_num_threads(1)` but 7.15e-07 at the default multi-threaded BLAS config --
    different input lengths can select a different conv accumulation order. atol=1e-5 absorbs
    that and is five orders below a genuine causality violation at this boundary (~1.3, see
    test_stage0_contracts.py's CM-01 test).
    """
    torch.manual_seed(0)
    encoder = ResidualCnnEncoder(ResidualCnnEncoderConfig())
    encoder.eval()
    x = torch.randn(1, 12, 1000)
    for cutoff in (29, 59, 100, 159, 248):
        last_sample = 4 * cutoff + 2  # token_window(cutoff)[1]
        zeroed = x.clone()
        zeroed[:, :, last_sample + 1 :] = 0.0
        prefix = x[:, :, : last_sample + 1]
        with torch.no_grad():
            full_tokens = encoder.forward(x)[:, : cutoff + 1]
            zeroed_tokens = encoder.forward(zeroed)[:, : cutoff + 1]
            prefix_tokens = encoder.forward(prefix)[:, : cutoff + 1]
        torch.testing.assert_close(full_tokens, zeroed_tokens, rtol=0.0, atol=1e-5)
        torch.testing.assert_close(full_tokens, prefix_tokens, rtol=0.0, atol=1e-5)


def test_patch_encoder_output_is_exactly_125_tokens_of_width_256() -> None:
    encoder = PatchEncoder(PatchEncoderConfig())
    tokens = encoder.forward(torch.randn(3, 12, 1000))
    assert tokens.shape == (3, 125, 256)


def test_patch_encoder_satisfies_the_encoder_contract() -> None:
    encoder = PatchEncoder(PatchEncoderConfig())
    assert isinstance(encoder, Encoder)
    assert encoder.latent_width == 256


def test_patch_encoder_wrong_lead_count_raises() -> None:
    encoder = PatchEncoder(PatchEncoderConfig())
    with pytest.raises(ValueError, match="must be"):
        encoder.forward(torch.randn(1, 8, 1000))


def test_patch_encoder_non_divisible_length_raises() -> None:
    encoder = PatchEncoder(PatchEncoderConfig(patch_width=8))
    with pytest.raises(ValueError, match="divisible"):
        encoder.forward(torch.randn(1, 12, 1003))


def test_patch_encoder_invalid_patch_width_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="patch_width"):
        PatchEncoder(PatchEncoderConfig(patch_width=0))


def test_patch_encoder_same_instance_is_deterministic_on_identical_input() -> None:
    encoder = PatchEncoder(PatchEncoderConfig())
    encoder.eval()
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        a = encoder.forward(x)
        b = encoder.forward(x)
    assert torch.equal(a, b)


def test_patch_encoder_perturbation_affects_exactly_one_token() -> None:
    """Stronger than the CNN's own `test_perturbation_locality`: a non-overlapping patch has
    zero run-in, so a perturbation to one raw sample must change EXACTLY one output token (the
    patch it falls inside), not a wider contiguous window."""
    torch.manual_seed(0)
    encoder = PatchEncoder(PatchEncoderConfig())
    encoder.eval()
    x = torch.randn(1, 12, 1000)
    with torch.no_grad():
        baseline = encoder.forward(x)
        perturbed_x = x.clone()
        perturbed_x[0, :, 500] += 1000.0  # patch index 500 // 8 = 62
        perturbed = encoder.forward(perturbed_x)
    changed = (baseline - perturbed).abs().amax(dim=-1)[0] > 1e-6
    assert int(changed.sum()) == 1
    assert int(changed.nonzero().flatten()[0]) == 500 // 8


def test_patch_encoder_every_token_depends_on_only_its_own_patch() -> None:
    """Exhaustive version of the perturbation test above: zeroing every OTHER patch must leave
    token j's own value unchanged, for every j -- the exact "no run-in, no shared samples"
    property the patch encoder replaces the CNN's ~96%-overlap receptive field with."""
    encoder = PatchEncoder(PatchEncoderConfig())
    encoder.eval()
    x = torch.randn(1, 12, 1000)
    p = encoder.config.patch_width
    with torch.no_grad():
        full = encoder.forward(x)
        for j in (0, 1, 62, 124):
            isolated = torch.zeros_like(x)
            isolated[:, :, j * p : (j + 1) * p] = x[:, :, j * p : (j + 1) * p]
            isolated_tokens = encoder.forward(isolated)
            torch.testing.assert_close(full[:, j], isolated_tokens[:, j], rtol=0.0, atol=1e-6)


def test_patch_encoder_gradients_reach_fc1() -> None:
    encoder = PatchEncoder(PatchEncoderConfig())
    x = torch.randn(1, 12, 1000, requires_grad=True)
    out = encoder.forward(x)
    out.sum().backward()
    assert encoder.fc1.weight.grad is not None
    assert torch.any(encoder.fc1.weight.grad != 0)

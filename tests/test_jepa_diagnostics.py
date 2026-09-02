import numpy as np
import pytest
import torch

from winder.jepa.diagnostics import (
    RunningMoments,
    covariance,
    effective_rank,
    spectrum_report,
    stable_rank,
)


def test_effective_rank_of_identity_covariance_equals_dimension() -> None:
    k = 16
    cov = torch.eye(k, dtype=torch.float64)
    assert effective_rank(cov) == pytest.approx(k, rel=1e-6)


def test_stable_rank_of_identity_covariance_equals_dimension() -> None:
    k = 16
    cov = torch.eye(k, dtype=torch.float64)
    assert stable_rank(cov) == pytest.approx(k, rel=1e-6)


def test_rank_one_covariance_gives_rank_near_one() -> None:
    v = torch.randn(8, dtype=torch.float64)
    cov = torch.outer(v, v)
    assert effective_rank(cov) == pytest.approx(1.0, abs=1e-6)
    assert stable_rank(cov) == pytest.approx(1.0, abs=1e-6)


def test_covariance_matches_numpy_population_covariance() -> None:
    torch.manual_seed(0)
    z = torch.randn(500, 12, dtype=torch.float64)
    cov = covariance(z)
    expected = np.cov(z.numpy().T, bias=True)
    np.testing.assert_allclose(cov.numpy(), expected, rtol=1e-8, atol=1e-8)


def test_covariance_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        covariance(torch.randn(4, 5, 6))


def test_running_moments_matches_numpy_across_multiple_batches() -> None:
    torch.manual_seed(0)
    dim = 6
    batches = [torch.randn(37, dim, dtype=torch.float64) for _ in range(5)]
    rm = RunningMoments(dim)
    for b in batches:
        rm.update(b)
    all_x = torch.cat(batches, dim=0)
    expected_mean = all_x.mean(dim=0)
    expected_cov = np.cov(all_x.numpy().T, bias=True)

    assert rm.n == sum(b.shape[0] for b in batches)
    torch.testing.assert_close(rm.mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rm.covariance.numpy(), expected_cov, rtol=1e-10, atol=1e-10)


def test_running_moments_wrong_dim_raises() -> None:
    rm = RunningMoments(4)
    with pytest.raises(ValueError, match="must be"):
        rm.update(torch.randn(3, 5))


def test_running_moments_covariance_before_enough_samples_raises() -> None:
    rm = RunningMoments(4)
    with pytest.raises(ValueError, match="at least 2 samples"):
        _ = rm.covariance


def test_pooled_effective_rank_is_capped_by_batch_size() -> None:
    """8 samples in a 256-dim space cannot have effective rank much above 7 (centering removes
    one degree of freedom) regardless of collapse -- the batch-size artifact the module
    docstring warns about."""
    torch.manual_seed(0)
    z = torch.randn(8, 256, dtype=torch.float64)
    cov = covariance(z)
    assert effective_rank(cov) <= 7.5


def test_spectrum_report_shape_and_keys() -> None:
    z = torch.randn(100, 16)
    report = spectrum_report(z)
    assert set(report) == {"n", "k", "mean_norm", "stable_rank", "effective_rank"}
    assert report["n"] == 100
    assert report["k"] == 16

import numpy as np
import pytest
from numpy.typing import NDArray
from sbbe.benford_mixture import BenfordMixtureEstimator
from sbbe.distributions import make_benford
from sbbe.distributions import make_uniform
from sbbe.simulation.sample_from_mixture import sample_from_mixture


@pytest.fixture
def mix_ratios():
    """Mixing ratios fixture."""
    return np.arange(0, 1.01, 0.02)


@pytest.fixture
def estimator(mix_ratios: NDArray):
    """BF Mixture estimator fixture."""
    return BenfordMixtureEstimator("log_BF", mixing_ratios=mix_ratios)


def test_single_sample_estimation(
    estimator: BenfordMixtureEstimator,
    mix_ratios: NDArray,
):
    """Test that a single sample returns valid mixture estimate and simulation."""
    size = 2000
    true_ratio = 0.6
    sample = sample_from_mixture(
        dist_a=make_benford(),
        dist_b=make_uniform(),
        size=size,
        mix_ratio=true_ratio,
    )
    rng = np.random.default_rng()
    sample = sample * 10 ** rng.uniform(0, 5, size=len(sample))
    _, probs, m_vals = estimator(sample, n_replicas=1000)

    # Basic output checks
    assert np.isclose(np.sum(probs), 1), "Probabilities should sum to 1"
    assert len(m_vals) == len(mix_ratios), "Length of m_vals should match mixing_ratios"

    # Simulation length
    expected_rows = 1000 * len(mix_ratios)
    assert len(estimator.simulation) == expected_rows, "Simulation length mismatch"


def test_multiple_calls_same_sample(
    estimator: BenfordMixtureEstimator,
    mix_ratios: NDArray,
):
    """Test that repeated calls with same sample properly reuse or extend simulation."""
    size = 2000
    true_ratio = 0.6
    sample = sample_from_mixture(
        dist_a=make_benford(),
        dist_b=make_uniform(),
        size=size,
        mix_ratio=true_ratio,
    )
    rng = np.random.default_rng()
    sample = sample * 10 ** rng.uniform(0, 5, size=len(sample))
    # First call
    _ = estimator(sample, n_replicas=1000)
    expected_rows1 = 1000 * len(mix_ratios)
    assert len(estimator.simulation) == expected_rows1

    # Second call with larger n_replicas
    _ = estimator(sample, n_replicas=2000)
    expected_rows2 = 2000 * len(mix_ratios)
    assert len(estimator.simulation) == expected_rows2

    # Third call with smaller n_replicas than available
    _ = estimator(sample, n_replicas=1500)
    # Simulation should not increase
    assert len(estimator.simulation) == expected_rows2


def test_multiple_samples(estimator: BenfordMixtureEstimator, mix_ratios: NDArray):
    """Test that estimator can handle different sample sizes and merges simulations."""
    size = 2000
    true_ratio = 0.6

    sample1 = sample_from_mixture(
        dist_a=make_benford(),
        dist_b=make_uniform(),
        size=size,
        mix_ratio=true_ratio,
    )
    sample2 = sample_from_mixture(
        dist_a=make_benford(),
        dist_b=make_uniform(),
        size=size // 2,
        mix_ratio=true_ratio,
    )

    rng1 = np.random.default_rng()
    sample1 = sample1 * 10 ** rng1.uniform(0, 5, size=len(sample1))

    rng2 = np.random.default_rng()
    sample2 = sample2 * 10 ** rng2.uniform(0, 5, size=len(sample2))

    # Call with first sample
    estimator(sample1, n_replicas=1000)
    expected_rows1 = 1000 * len(mix_ratios)
    assert len(estimator.simulation) == expected_rows1

    # Call with second, smaller sample
    estimator(sample2, n_replicas=1000)
    expected_rows_total = 2 * 1000 * len(mix_ratios)
    assert len(estimator.simulation) == expected_rows_total

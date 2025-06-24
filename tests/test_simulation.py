import numpy as np
import pandas as pd
import pytest
from scipy.stats import rv_discrete
from chemford.benford.benford import benford_first_digit_distribution
from chemford.simulation.distributions import make_multinomial
from chemford.simulation.distributions import sample_from_mixture
from chemford.simulation.estimate_mixture_ratio import (
    estimate_mixture_ratio_from_simulation,
)
from chemford.simulation.simulate_BF import simulate_benford_and_uniform_mixture
from chemford.statistics.bayes_factor import bayes_factor_dirichlet_multinomial
from chemford.statistics.utils import observed_frequencies


def test_make_multinomial_basic():
    """Test return type and sampling.

    Verify the distribution object type
    and that samples lie within the defined outcomes.
    """
    outcomes = [1, 2, 3]
    probs = [0.2, 0.3, 0.5]
    dist = make_multinomial(outcomes, probs, random_state=42)
    assert isinstance(dist, rv_discrete)
    samples = dist.rvs(size=1000)
    assert np.all(np.isin(samples, outcomes))


def test_make_multinomial_invalid_lengths():
    """Test ValueError.

    Ensure a ValueError is raised when outcomes
    and probabilities have mismatched lengths.
    """
    with pytest.raises(ValueError, match="Length of outcomes and probs must match."):
        make_multinomial([1, 2], [0.5], random_state=0)


def test_make_multinomial_probs_not_summing_to_one():
    """Ensure a ValueError is raised when probabilities do not sum to 1."""
    with pytest.raises(ValueError, match="Probabilities must sum to 1."):
        make_multinomial([1, 2, 3], [0.1, 0.2, 0.6], random_state=0)


def test_sample_from_mixture_basic():
    """Check validity of mixture sampling.

    Verify that sampling from a mixture
    returns the correct sample size and only valid outcomes.
    """
    dist1 = make_multinomial([0, 1], [0.9, 0.1], random_state=1)
    dist2 = make_multinomial([0, 1], [0.1, 0.9], random_state=2)

    test_size_1000 = 1000
    samples = sample_from_mixture(
        dist1,
        dist2,
        size=test_size_1000,
        mix_ratio=0.5,
        random_state=3,
    )
    assert len(samples) == test_size_1000
    assert np.all(np.isin(samples, [0, 1]))


def test_sample_from_mixture_ratio_bounds():
    """Test edge cases.

    Ensure that mix_ratio of 0.0 or 1.0 yields
    samples exclusively from one distribution.
    """
    dist_a = make_multinomial([1, 2], [1.0, 0.0])  # Always 1
    dist_b = make_multinomial([1, 2], [0.0, 1.0])  # Always 2

    samples_a = sample_from_mixture(dist_a, dist_b, size=100, mix_ratio=1.0)
    samples_b = sample_from_mixture(dist_a, dist_b, size=100, mix_ratio=0.0)

    sample_value_dist_a = 1
    sample_value_dist_b = 2

    assert np.all(samples_a == sample_value_dist_a)
    assert np.all(samples_b == sample_value_dist_b)


def test_sample_from_mixture_invalid_ratio():
    """Raise ValueError if mix_ratio is outside the range [0, 1]."""
    dist = make_multinomial([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError, match="mix_ratio must be between 0 and 1."):
        sample_from_mixture(dist, dist, size=10, mix_ratio=-0.5)

    with pytest.raises(ValueError, match="mix_ratio must be between 0 and 1."):
        sample_from_mixture(dist, dist, size=10, mix_ratio=1.5)


def test_sample_from_mixture_invalid_size():
    """Raise ValueError if sample size is negative."""
    dist = make_multinomial([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError, match="size must be non-negative."):
        sample_from_mixture(dist, dist, size=-1, mix_ratio=0.5)


def test_sample_from_mixture_outcome_space_mismatch():
    """Raise ValueError when distributions have different outcome spaces."""
    dist1 = make_multinomial([1, 2], [0.5, 0.5])
    dist2 = make_multinomial([2, 3], [0.5, 0.5])

    with pytest.raises(ValueError, match="different outcome spaces"):
        sample_from_mixture(dist1, dist2, size=10, mix_ratio=0.5)


@pytest.fixture
def simulation_data():
    """Create synthetic test data."""
    rows = []
    rng = np.random.default_rng(42)
    for mix in [0.0, 0.5, 1.0]:
        for _ in range(20):
            bf = rng.normal(loc=mix * 2, scale=0.5)
            rows.append((100, mix, bf))
    return pd.DataFrame(rows, columns=["n_samples", "mixing_ratio", "log_bf10"])


def test_output_shapes(simulation_data: pd.DataFrame):
    """Test estimator shape and types."""
    logBF = 1.0
    n_samples = 100
    M_CI, probs, m_vals = estimate_mixture_ratio_from_simulation(
        simulation_data,
        logBF,
        n_samples,
    )

    assert isinstance(M_CI, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert isinstance(m_vals, np.ndarray)
    assert probs.shape == m_vals.shape
    assert np.isclose(probs.sum(), 1.0)


def test_missing_columns():
    """Test ValueError for missing columns."""
    test = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match="Simulation DataFrame must contain"):
        estimate_mixture_ratio_from_simulation(test, 0.0, 100)


def test_no_matching_sample_size(simulation_data: pd.DataFrame):
    """Test if n_samples was not simulated."""
    with pytest.raises(ValueError, match="No data found for n_samples"):
        estimate_mixture_ratio_from_simulation(
            simulation_data,
            logBF=1.0,
            n_samples=999,
        )


def test_insufficient_kde_data():
    """Test RuntimeError for to little data."""
    test = pd.DataFrame(
        {
            "n_samples": [100],
            "mixing_ratio": [0.5],
            "log_bf10": [1.0],  # only one data point for this mixing ratio
        },
    )
    with pytest.raises(RuntimeError, match="No valid likelihood estimates"):
        estimate_mixture_ratio_from_simulation(test, logBF=1.0, n_samples=100)


def test_simulation_basic():
    """Test basic functionality of simulation."""
    n_replicas = 10
    dataset_sizes = np.logspace(2, 4)
    mixing_ratios = np.arange(0.0, 1.01, 0.1)

    simulation = simulate_benford_and_uniform_mixture(
        n_replicas=n_replicas,
        sizes=dataset_sizes,
        mixing_ratios=mixing_ratios,
    )

    expected_rows = n_replicas * len(dataset_sizes) * len(mixing_ratios)
    assert len(simulation) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(simulation)}"
    )
    assert set(simulation.columns) == {
        "iteration",
        "n_samples",
        "mixing_ratio",
        "log_bf10",
    }


def test_interplay_simulation_estimation():
    """Test interplay between simulation and estimation.

    Ensure that
    """
    n_replicas = 10
    dataset_sizes = np.logspace(2, 4)
    mixing_ratios = np.arange(0.0, 1.01, 0.1)

    simulation = simulate_benford_and_uniform_mixture(
        n_replicas=n_replicas,
        sizes=dataset_sizes,
        mixing_ratios=mixing_ratios,
    )

    benford_outcome = benford_first_digit_distribution()
    benford = make_multinomial(
        list(benford_outcome.keys()),
        list(benford_outcome.values()),
    )

    uniform = make_multinomial(range(1, 10, 1), [1 / 9] * 9)
    sample = sample_from_mixture(
        dist_a=benford,
        dist_b=uniform,
        size=int(dataset_sizes[0]),
        mix_ratio=0.5,
    )
    counts = observed_frequencies(sample)
    log_bf10 = bayes_factor_dirichlet_multinomial(
        counts,
        list(benford_outcome.values()),
        alpha=1,
    )
    M_CI, probs, m_vals = estimate_mixture_ratio_from_simulation(
        simulation,
        log_bf10,
        dataset_sizes[0],
    )

    assert isinstance(M_CI, np.ndarray), "M_CI should be a NumPy array."
    assert isinstance(probs, np.ndarray), "probs should be a NumPy array."
    assert isinstance(m_vals, np.ndarray), "m_vals should be a NumPy array."

    assert len(probs) == len(m_vals), "probs and m_vals must be the same length."
    assert np.isclose(probs.sum(), 1.0, atol=1e-2), (
        f"Posterior probs must sum to ~1. Got {probs.sum()}"
    )

    for val in M_CI:
        assert val in mixing_ratios, (
            f"Value {val} in M_CI not in simulated mixing_ratios"
        )

    mode_idx = np.argmax(probs)
    mode_m = m_vals[mode_idx]
    assert mode_m in M_CI, "MAP estimate should lie within the credible interval"

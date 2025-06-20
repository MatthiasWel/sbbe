import numpy as np
import pytest
from scipy.stats import rv_discrete

from chemford.simulation.distributions import make_multinomial, sample_from_mixture

def test_make_multinomial_basic():
    """Verify the distribution object type and that samples lie within the defined outcomes."""
    outcomes = [1, 2, 3]
    probs = [0.2, 0.3, 0.5]
    dist = make_multinomial(outcomes, probs, random_state=42)
    assert isinstance(dist, rv_discrete)
    samples = dist.rvs(size=1000)
    assert np.all(np.isin(samples, outcomes))

def test_make_multinomial_invalid_lengths():
    """Ensure a ValueError is raised when outcomes and probabilities have mismatched lengths."""
    with pytest.raises(ValueError, match="Length of outcomes and probs must match."):
        make_multinomial([1, 2], [0.5], random_state=0)

def test_make_multinomial_probs_not_summing_to_one():
    """Ensure a ValueError is raised when probabilities do not sum to 1."""
    with pytest.raises(ValueError, match="Probabilities must sum to 1."):
        make_multinomial([1, 2, 3], [0.1, 0.2, 0.6], random_state=0)

def test_sample_from_mixture_basic():
    """Verify that sampling from a mixture returns the correct sample size and only valid outcomes."""
    dist1 = make_multinomial([0, 1], [0.9, 0.1], random_state=1)
    dist2 = make_multinomial([0, 1], [0.1, 0.9], random_state=2)

    samples = sample_from_mixture(dist1, dist2, size=1000, mix_ratio=0.5, random_state=3)
    assert len(samples) == 1000
    assert np.all(np.isin(samples, [0, 1]))

def test_sample_from_mixture_ratio_bounds():
    """Ensure that mix_ratio of 0.0 or 1.0 yields samples exclusively from one distribution."""
    dist_a = make_multinomial([1, 2], [1.0, 0.0])  # Always 1
    dist_b = make_multinomial([1, 2], [0.0, 1.0])  # Always 2

    samples_a = sample_from_mixture(dist_a, dist_b, size=100, mix_ratio=1.0)
    samples_b = sample_from_mixture(dist_a, dist_b, size=100, mix_ratio=0.0)

    assert np.all(samples_a == 1)  # All samples from dist_a are 1
    assert np.all(samples_b == 2)  # All samples from dist_b are 2

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
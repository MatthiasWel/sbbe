import pytest
import numpy as np

from chemford.distributions import make_multinomial, make_benford, make_uniform
from scipy.stats import rv_discrete

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



def test_make_benford():
    """Test that the output has 9 digits (1-9) as support."""
    dist = make_benford(random_state=42)
    assert len(dist.xk) == 9, "Supported on 9 elements"
    assert all(d in range(1, 10) for d in dist.xk), "Supported on 1-9"
    assert isinstance(dist, rv_discrete), "Is a rv_discrete"



def test_make_uniform():
    """Test that the output has 9 digits (1-9) as support."""
    dist = make_uniform(random_state=42)
    assert len(dist.xk) == 9, "Supported on 9 elements"
    assert all(d in range(1, 10) for d in dist.xk), "Supported on 1-9"
    assert isinstance(dist, rv_discrete), "Is a rv_discrete"

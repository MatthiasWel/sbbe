from typing import Sequence, Optional
from scipy.stats import rv_discrete
from numpy.random import Generator, default_rng
import numpy as np
from numpy.typing import NDArray

def make_multinomial(
    outcomes: Sequence,
    probs: Sequence[float],
    random_state: Optional[int | Generator] = None
) -> rv_discrete:
    """
    Create a multinomial distribution from outcomes and probabilities.

    Args:
        outcomes: Possible discrete outcomes.
        probs: Probabilities associated with each outcome.
        random_state: Optional seed or Generator for reproducibility.

    Returns:
        A scipy.stats.rv_discrete distribution instance.
    """
    if len(outcomes) != len(probs):
        raise ValueError("Length of outcomes and probs must match.")
    if not np.isclose(sum(probs), 1.0, atol=1e-8):
        raise ValueError("Probabilities must sum to 1.")
    
    return rv_discrete(values=(outcomes, probs), seed=random_state)

def sample_from_mixture(
    dist_a: rv_discrete,
    dist_b: rv_discrete,
    size: int,
    mix_ratio: float,
    random_state: Optional[int | Generator] = None
) -> NDArray:
    """
    Sample from a mixture of two discrete distributions.

    Args:
        dist_a: First rv_discrete distribution.
        dist_b: Second rv_discrete distribution.
        size: Total number of samples to draw.
        mix_ratio: Fraction of samples from dist_a (between 0 and 1).
        random_state: Optional seed or Generator for reproducibility.

    Returns:
        A shuffled NumPy array of samples from the mixture.
    """
    if not (0 <= mix_ratio <= 1):
        raise ValueError("mix_ratio must be between 0 and 1.")
    if size < 0:
        raise ValueError("size must be non-negative.")
    outcomes_a = set(dist_a.xk)
    outcomes_b = set(dist_b.xk)
    if outcomes_a != outcomes_b:
        raise ValueError(f"Distributions have different outcome spaces: {outcomes_a} vs {outcomes_b}")

    rng = default_rng(random_state)
    n_a = int(round(size * mix_ratio))
    n_b = size - n_a

    samples_a = dist_a.rvs(size=n_a, random_state=rng)
    samples_b = dist_b.rvs(size=n_b, random_state=rng)

    mixed_samples = np.concatenate([samples_a, samples_b])
    rng.shuffle(mixed_samples)
    return mixed_samples
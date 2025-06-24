from collections.abc import Sequence
import numpy as np
from numpy.random import Generator
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.stats import rv_discrete


def make_multinomial(
    outcomes: Sequence,
    probs: Sequence[float],
    random_state: int | Generator | None = None,
) -> rv_discrete:
    """Create a multinomial distribution from outcomes and probabilities.

    Args:
        outcomes: Possible discrete outcomes.
        probs: Probabilities associated with each outcome.
        random_state: Optional seed or Generator for reproducibility.

    Returns:
        A scipy.stats.rv_discrete distribution instance.
    """
    if len(outcomes) != len(probs):
        msg = "Length of outcomes and probs must match."
        raise ValueError(msg)
    if not np.isclose(sum(probs), 1.0, atol=1e-8):
        msg = "Probabilities must sum to 1."
        raise ValueError(msg)

    return rv_discrete(values=(outcomes, probs), seed=random_state)


def sample_from_mixture(
    dist_a: rv_discrete,
    dist_b: rv_discrete,
    size: int,
    mix_ratio: float,
    random_state: int | Generator | None = None,
) -> NDArray:
    """Sample from a mixture of two discrete distributions.

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
        msg = "mix_ratio must be between 0 and 1."
        raise ValueError(msg)
    if size < 0:
        msg = "size must be non-negative."
        raise ValueError(msg)
    outcomes_a = set(dist_a.xk)
    outcomes_b = set(dist_b.xk)
    if outcomes_a != outcomes_b:
        msg = (
            f"Distributions have different outcome spaces: {outcomes_a} vs {outcomes_b}"
        )
        raise ValueError(msg)

    rng = default_rng(random_state)
    n_a = round(size * mix_ratio)
    n_b = size - n_a

    samples_a = dist_a.rvs(size=n_a, random_state=rng)
    samples_b = dist_b.rvs(size=n_b, random_state=rng)

    mixed_samples = np.concatenate([samples_a, samples_b])
    rng.shuffle(mixed_samples)
    return mixed_samples

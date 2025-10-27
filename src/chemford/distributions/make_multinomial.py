from collections.abc import Sequence
import numpy as np
from numpy.random import Generator
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

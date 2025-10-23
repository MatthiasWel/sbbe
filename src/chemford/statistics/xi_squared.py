import numpy as np

from collections.abc import Iterable

def xi_squared(
    observed: Iterable[int],
    expected: Iterable[float],
) -> float:
    return np.sum(
        (observed - expected) ** 2 / expected
    )
    
def xi_squared_counts(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    counts = np.array(counts)
    n = np.sum(counts)
    expected_probs = np.array(expected_probs)
    expected_counts = n * expected_probs
    return xi_squared(observed=counts, expected=expected_counts)
    

def xi_squared_proportions(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    counts = np.array(counts)
    n = np.sum(counts)
    observed_proportions = counts / n
    return xi_squared(observed=observed_proportions, expected=expected_probs)
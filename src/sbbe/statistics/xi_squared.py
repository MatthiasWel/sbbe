from collections.abc import Iterable
import numpy as np
from numpy.typing import NDArray


def xi_squared(
    observed: NDArray[int],
    expected: NDArray[float],
) -> float:
    """Compute the chi-squared statistic between observed and expected values.

    Args:
        observed: Observed frequencies or proportions.
        expected: Expected frequencies or proportions.

    Returns:
        float: Chi-squared statistic measuring deviation from expected values.
    """
    return np.sum((observed - expected) ** 2 / expected)


def xi_squared_counts(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute the chi-squared statistic using observed counts.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Chi-squared statistic based on counts and expected probabilities.
    """
    counts = np.array(counts)
    n = np.sum(counts)
    expected_probs = np.array(expected_probs)
    expected_counts = n * expected_probs
    return xi_squared(observed=counts, expected=expected_counts)


def xi_squared_proportions(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute the chi-squared statistic using observed proportions.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Chi-squared statistic based on proportions and expected probabilities.
    """
    counts = np.array(counts)
    n = np.sum(counts)
    observed_proportions = counts / n
    return xi_squared(observed=observed_proportions, expected=expected_probs)

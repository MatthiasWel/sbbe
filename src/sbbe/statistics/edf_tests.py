from collections.abc import Iterable
import numpy as np


def cdf_difference(counts: Iterable[int], expected_probs: Iterable[float]) -> float:
    """Compute the difference between empirical and theoretical CDFs.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Array of cumulative distribution differences (empirical - theoretical).
    """
    counts = np.array(counts)
    n = np.sum(counts)
    expected_probs = np.array(expected_probs)
    expected_counts = n * expected_probs
    ecdf = np.cumsum(counts) / n
    tcdf = np.cumsum(expected_counts) / n
    return ecdf - tcdf


def ks_d(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Maximum absolute difference between empirical and theoretical CDFs.
    """
    cdf_diff = cdf_difference(counts, expected_probs)
    return np.max(np.abs(cdf_diff))


def kuipers_v(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute the Kuiper's V statistic.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Sum of the maximum positive and negative deviations between empirical
        and theoretical CDFs.
    """
    cdf_diff = cdf_difference(counts, expected_probs)
    return np.max(cdf_diff) - np.min(cdf_diff)

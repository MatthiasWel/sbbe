from collections.abc import Iterable
import numpy as np


def max_l1_distance_leemis(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute Leemis' maximum L1 distance between observed and expected proportions.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Maximum absolute difference between observed and expected proportions.
    """
    counts = np.array(counts)
    n = np.sum(counts)
    observed_proportions = counts / n
    return np.max(np.abs(observed_proportions - expected_probs))


def max_l1_distance_morrow(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute Morrow's scaled L1 distance statistic.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Scaled maximum L1 distance (Leemis' D multiplied by sqrt(n)).
    """
    counts = np.array(counts)
    n = np.sum(counts)
    return np.sqrt(n) * max_l1_distance_leemis(counts, expected_probs)


def euclidean_distance_cho_gains(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    """Compute Cho-Gaines Euclidean distance statistic.

    Args:
        counts: Observed counts for each category or bin.
        expected_probs: Expected probabilities for each corresponding category.

    Returns:
        float: Euclidean distance between observed and expected proportions,
        scaled by sample size n.
    """
    counts = np.array(counts)
    n = np.sum(counts)
    observed_proportions = counts / n
    return np.sqrt(n * np.sum((observed_proportions - expected_probs) ** 2))

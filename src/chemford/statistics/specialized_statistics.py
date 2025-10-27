from collections.abc import Iterable
import numpy as np


def max_l1_distance_leemis(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    counts = np.array(counts)
    n = np.sum(counts)
    observed_proportions = counts / n
    return np.max(np.abs(observed_proportions - expected_probs))


def max_l1_distance_morrow(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    counts = np.array(counts)
    n = np.sum(counts)
    return np.sqrt(n) * max_l1_distance_leemis(counts, expected_probs)


def euclidean_distance_cho_gains(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    counts = np.array(counts)
    n = np.sum(counts)
    observed_proportions = counts / n
    return np.sqrt(n * np.sum((observed_proportions - expected_probs) ** 2))

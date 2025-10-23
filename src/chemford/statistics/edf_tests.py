import numpy as np

from collections.abc import Iterable



def cdf_difference(counts, expected_probs):
    counts = np.array(counts)
    n = np.sum(counts)
    expected_probs = np.array(expected_probs)
    expected_counts = n * expected_probs
    ecdf = np.cumsum(counts) / n
    tcdf = np.cumsum(expected_counts) / n
    cdf_diff = ecdf - tcdf
    return cdf_diff

def ks_d(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    cdf_diff = cdf_difference(counts, expected_probs)
    return np.max(np.abs(cdf_diff))

def kuipers_v(
    counts: Iterable[int],
    expected_probs: Iterable[float],
) -> float:
    cdf_diff = cdf_difference(counts, expected_probs)
    return np.max(cdf_diff) - np.min(cdf_diff)
from collections.abc import Sequence
import numpy as np


def benford_first_digit_distribution() -> dict[int, float]:
    """Returns Benford's Law probability distribution for the first digit (1-9).

    Returns:
        dict: Mapping from digits 1 through 9 to their Benford probabilities.

    Example:
        >>> benford_first_digit_distribution()
        {1: 0.301, 2: 0.176, ..., 9: 0.046}
    """
    return {d: float(np.log10(1 + 1 / d)) for d in range(1, 10, 1)}


def benford_first_two_digit_distribution() -> dict[int, float]:
    """Returns Benford's Law probability distribution for the first two digits (10-99).

    Returns:
        dict: Mapping from integers 10 through 99 to their Benford probabilities.

    Notes:
        According to Benford's Law, the distribution of the
        first two digits in naturally occurring datasets
        follows the formula: P(d) = log10(1 + 1/d), where d is an integer from 10 to 99.

    Example:
        >>> benford_first_two_digit_distribution()
        {10: 0.07918, 11: 0.07297, ..., 99: 0.00461}
    """
    return {d: float(np.log10(1 + 1 / d)) for d in range(10, 100, 1)}


def benford_n_digit_distribution(n: int) -> dict[int, float]:
    """Returns Benford's Law probability distribution for the first two digits (10-99).

    Returns:
        dict: Mapping from integers 10 through 99 to their Benford probabilities.

    Notes:
        According to Benford's Law, the distribution of the first
        two digits in naturally occurring datasets
        follows the formula: P(d) = log10(1 + 1/d), where d is an integer from 10 to 99.

    Example:
        >>> benford_n_digit_distribution(2)
        {0: 0.119, 1: 0.113, 2: 0.1088, ...}
    """
    if not n > 1:
        message = f"n must be bigger than 1 ({n} currently)"
        raise ValueError(message)
    return {
        d: float(
            np.sum(
                [
                    np.log10(1 + 1 / (10 * k + d))
                    for k in range(10 ** (n - 2), 10 ** (n - 1))
                ],  # 10**(n - 1) - 1 + 1
            ),
        )
        for d in range(10)
    }


def has_sufficient_log_scale_coverage(
    data: Sequence,
    log_scale_coverage: float = 2.0,
) -> bool:
    """Check whether data spans a log scale range.

    Args:
            data: A sequence of positive numeric values
            (e.g., list, numpy array, or pandas Series).

            log_scale_coverage: The maximum allowed log
            scale range (in orders of magnitude).

    Returns:
        True if data covers less than the specified log scale range, False otherwise.
    """
    data = np.asarray(data)
    if np.any(data) <= 0:
        msg = "All values in data must be positive to compute log scale coverage."
        raise ValueError(msg)

    min_val = np.min(data)
    max_val = np.max(data)
    return max_val / min_val > 10**log_scale_coverage


def has_sufficient_data(data: Sequence, threshold: int = 100) -> bool:
    """Check if the input data has a length greater than or equal to a threshold.

    Args:
        data: Any object with a defined length (e.g., list, numpy array, pandas Series).
        threshold: Minimum number of elements required.

    Returns:
        True if length of data >= threshold, False otherwise.
    """
    return len(data) >= threshold

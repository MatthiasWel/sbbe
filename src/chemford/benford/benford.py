import numpy as np

from typing import Dict

def benford_first_digit_distribution() -> Dict[int, float]:
    """
    Returns the Benford's Law probability distribution for the first digit (1–9).

    Returns:
        dict: Mapping from digits 1 through 9 to their Benford probabilities.

    Example:
        >>> benford_first_digit_distribution()
        {1: 0.301, 2: 0.176, ..., 9: 0.046}
    """
    return {
        d: float(np.log10(1 + 1 /d))\
        for d in range(1, 10, 1)
    }

def benford_first_two_digit_distribution() -> Dict[int, float]:
    """
    Returns the Benford's Law probability distribution for the first two digits (10–99).

    Returns:
        dict: Mapping from integers 10 through 99 to their Benford probabilities.

    Notes:
        According to Benford's Law, the distribution of the first two digits in naturally occurring datasets
        follows the formula: P(d) = log10(1 + 1/d), where d is an integer from 10 to 99.

    Example:
        >>> benford_first_two_digit_distribution()
        {10: 0.07918, 11: 0.07297, ..., 99: 0.00461}
    """
    return {
        d: float(np.log10(1 + 1 /d))\
        for d in range(10, 100, 1)
    }

def benford_n_digit_distribution(n) -> Dict[int, float]:
    """
    Returns the Benford's Law probability distribution for the first two digits (10–99).

    Returns:
        dict: Mapping from integers 10 through 99 to their Benford probabilities.

    Notes:
        According to Benford's Law, the distribution of the first two digits in naturally occurring datasets
        follows the formula: P(d) = log10(1 + 1/d), where d is an integer from 10 to 99.

    Example:
        >>> benford_n_digit_distribution(2)
        0: np.float64(0.11967926859688073), 1: np.float64(0.1138901034075564), 2: np.float64(0.10882149900550825)
    """
    assert n > 1, "n must be bigger than 1"
    return {
        d: float(np.sum(
            [np.log10(1 + 1 / (10 * k + d)) for k in range(10**(n - 2), 10**(n - 1))] # 10**(n - 1) - 1 + 1
        )) for d in range(10)
    }


if __name__ == '__main__':
    print(benford_first_digit_distribution())
    print(benford_first_two_digit_distribution())
    print(benford_n_digit_distribution(2))
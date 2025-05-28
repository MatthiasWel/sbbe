import numpy as np

def observed_frequencies(first_digits, mini: int=1, maxi: int=9):
    """
    Compute observed frequencies of first digits in a dataset.

    Parameters
    ----------
    first_digits : array-like
        Sequence of first digits (integers).
    mini : int, optional
        Minimum digit to include (default is 1).
    maxi : int, optional
        Maximum digit to include (default is 9).

    Returns
    -------
    counts : ndarray
        Array of counts for digits from `mini` to `maxi`.
    """

    first_digits = np.array(first_digits)

    if np.any((first_digits < mini) | (first_digits > maxi)):
        msg = f"All digits must be in range [{mini}, {maxi}]"
        raise ValueError(msg)
    
    counts = np.bincount(first_digits, minlength=1 + maxi)[mini:maxi + 1]
    return counts
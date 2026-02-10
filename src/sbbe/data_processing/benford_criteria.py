from collections.abc import Sequence
import numpy as np


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
    data_arr = np.asarray(data)  # type: np.ndarray

    # Remove NaN and non-positive values
    data_arr = data_arr[(data_arr > 0) & ~np.isnan(data_arr)]

    if data_arr.size == 0:
        msg = "Data must contain at least one positive numeric value."
        raise ValueError(msg)

    # Compute log10 range
    log_range = np.log10(np.max(data_arr)) - np.log10(np.min(data_arr))
    return log_range > log_scale_coverage


def has_sufficient_data(data: Sequence, threshold: int = 100) -> bool:
    """Check if the input data has a length greater than or equal to a threshold.

    Args:
        data: Any object with a defined length (e.g., list, numpy array, pandas Series).
        threshold: Minimum number of elements required.

    Returns:
        True if length of data >= threshold, False otherwise.
    """
    return len(data) >= threshold

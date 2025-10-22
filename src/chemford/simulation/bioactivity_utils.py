from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray


def convert_to_p_activity(
    nM_values: Sequence[float] | NDArray,
) -> NDArray:
    """Convert nanomolar (nM) activity values (e.g., IC50, Ki, EC50) to pActivity.

    pActivity = -log10(value in molar units)

    Args:
        nM_values: Activity values in nanomolar units (list, tuple, or NumPy array).

    Returns:
        A NumPy array of pActivity values.
    """
    arr = np.asarray(nM_values, dtype=float)
    if np.any(arr <= 0):
        msg = "All activity values must be positive."
        raise ValueError(msg)

    return -np.log10(arr * 1e-9)


def round_to_decimal(
    values: Sequence[float] | NDArray,
    decimals: int = 1,
    is_log: bool = False,
) -> NDArray:
    """Round numeric values to a fixed number of decimal places.

    Args:
        values: Input numeric values (list, tuple, or NumPy array).
        decimals: Number of decimal places to round to (default is 1).
        is_log: If True, assumes values are already in log scale (pActivity).
                If False, converts from nM to log scale first.

    Returns:
        A NumPy array of rounded values.
    """
    arr = np.asarray(values, dtype=float)

    if not is_log:
        arr = convert_to_p_activity(arr)

    return np.round(arr, decimals=decimals)


def round_to_step(
    values: Sequence[float] | NDArray,
    step: float = 0.05,
    is_log: bool = False,
) -> NDArray:
    """Round numeric values to the nearest fixed step size.

    Args:
        values: Input numeric values (list, tuple, or NumPy array).
        step: Step size to round to (default is 0.05).
        is_log: If True, assumes values are already in log scale (pActivity).
                If False, converts from nM to log scale first.

    Returns:
        A NumPy array of values rounded to the nearest step.
    """
    if step <= 0:
        msg = "Step size must be positive."
        raise ValueError(msg)

    arr = np.asarray(values, dtype=float)

    if not is_log:
        arr = convert_to_p_activity(arr)

    factor = 1 / step
    return np.round(arr * factor) / factor


def empirical_distribution_round_to_one_significant_digit() -> dict[int, float]:
    """Return the empirical probability distribution of first digits
    after rounding bioactivity values to one significant digit in the
    log scale.

    Returns:
        dict: Mapping of first digit (1 to 9) to its probability (0.0 to 1.0).
    """
    return {
        1: 0.39329,
        2: 10.207,
        3: 20.218,
        4: 0.0,
        5: 10.101,
        6: 10.138,
        7: 10.006,
        8: 0.0,
        9: 0.0,
    }


def empirical_distribution_round_to_step_0_05() -> dict[int, float]:
    """Return the empirical probability distribution of first digits
    after rounding bioactivity values to 0.05 increments in the log
    scale.

    Returns:
        dict: Mapping of first digit (1 to 9) to its probability (0.0 to 1.0).
    """
    return {
        1: 0.34659,
        2: 0.15355,
        3: 0.14855,
        4: 0.04986,
        5: 0.10176,
        6: 0.04918,
        7: 0.10038,
        8: 0.05013,
        9: 0.0,
    }

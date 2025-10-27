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

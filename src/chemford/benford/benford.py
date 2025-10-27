from collections.abc import Sequence
from decimal import Decimal
from decimal import InvalidOperation
from decimal import getcontext
import numpy as np






def extract_significant_digits(
    num: float,
    start: int = 1,
    length: int = 1,
) -> int | None:
    # TODO: How to handle negative values?
    """Extract a segment of significant digits from a positive number.

    Parameters
    ----------
    num : float
        The input number to extract digits from.
    start : int, default=1
        The position (1-based index) of the first digit to extract.
    length : int, default=1
        The number of digits to extract starting from `start`.

    Returns:
    -------
    Optional[int]
        The extracted digits as an integer, or None if the input is invalid
        or does not contain enough significant digits.

    Notes:
    -----
    - The function operates on significant digits only.
    - Leading zeros before the first non-zero digit are ignored.
    - Negative numbers and zero return None.
    - Handles scientific notation safely using the `decimal` module.

    Examples:
    --------
    >>> extract_significant_digits(1.40, 1, 2) # ignores decimal point
    14
    >>> extract_significant_digits(0.00456, 1, 2) # ignores leading zeros
    45
    >>> extract_significant_digits(1e-5, 1, 1) # ignores scientific notation
    1
    >>> extract_significant_digits(123.45, 2, 1)
    2
    >>> extract_significant_digits(-100.0, 1, 1) # ignores negative values
    None
    """
    if not isinstance(num, (int, float)) or num <= 0:
        return None

    try:
        # Use high precision and convert to Decimal
        getcontext().prec = 30
        d = Decimal(str(num)).normalize()

        # Remove sign, decimal point, and leading zeros
        digits = "".join(c for c in format(d, "f") if c.isdigit()).lstrip("0")
    except InvalidOperation:
        return None

    if len(digits) < start + length - 1:
        return None

    return int(digits[start - 1 : start - 1 + length])

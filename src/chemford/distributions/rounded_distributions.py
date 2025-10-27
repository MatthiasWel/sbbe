def empirical_distribution_round_to_one_significant_digit() -> dict[int, float]:
    """Return the empirical probability distribution of first digits.

    This distribution is computed after rounding bioactivity values
    to one significant digit in the log scale.

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
    """Return the empirical probability distribution of first digits.

    This distribution is computed after rounding bioactivity values
    to 0.05 increments in the log scale.

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

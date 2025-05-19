import pandas as pd
import numpy as np

from chemford.benford.benford import benford_first_digit_distribution
from chemford.benford.benford import benford_first_two_digit_distribution
from chemford.benford.benford import benford_n_digit_distribution

from chemford.benford.benford import has_sufficient_log_scale_coverage, has_sufficient_data

def test_benford_distributions():
    """Test benford distributions."""
    first = benford_first_digit_distribution()
    assert isinstance(first, dict), "First digit distribution is no dict"
    expected_value = 9
    assert len(first) == expected_value, (
        "First digit distribution does not have appropriate number of entries"
    )

    first_two = benford_first_two_digit_distribution()
    assert isinstance(first_two, dict), "First digit distribution is no dict"
    expected_value = 90
    assert len(first_two) == expected_value, (
        "First two digits distribution does not have appropriate number of entries"
    )

    for n in range(2, 7):  # very long runtimes for high n
        n_th = benford_n_digit_distribution(n)
        assert isinstance(n_th, dict), f"{n}th digit distribution is no dict"
        expected_value = 10
        assert len(n_th) == expected_value, (
            f"{n}th digit distribution does not have appropriate number of entries"
        )

def test_sufficiency_checks():
    """Test sufficiency checks"""
    short = np.arange(1, 100, 10)
    long = np.arange(1, 1000, 0.01)
    transforms = (pd.Series, np.array, list)
    for transform in transforms:
        assert not has_sufficient_data(transform(short)), f"{transform} did not fail with short test data"
        assert has_sufficient_data(transform(long)), f"{transform} failed with long test data"

    no_coverage = np.arange(1, 10, 0.01) 
    good_coverage = np.arange(1, 1000, 0.01)
    for transform in transforms:
        assert not has_sufficient_log_scale_coverage(transform(no_coverage)), f"{transform} did not fail with little coverage"
        assert has_sufficient_log_scale_coverage(transform(good_coverage)), f"{transform} failed with enough coverage"
    

    
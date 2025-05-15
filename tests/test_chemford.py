"""Tests for the chemford.my_module module."""

from src.chemford.benford.benford import (
    benford_first_digit_distribution,
    benford_first_two_digit_distribution,
    benford_n_digit_distribution,
)

def test_benford():
    """Test benford distributions"""
    first = benford_first_digit_distribution()
    assert isinstance(first, dict), "First digit distribution is no dict"
    assert len(first) == 9, (
        "First digit distribution does not have appropriate number of entries"
    )

    first_two = benford_first_two_digit_distribution()
    assert isinstance(first_two, dict), "First digit distribution is no dict"
    assert len(first_two) == 90, (
        "First two digits distribution does not have appropriate number of entries"
    )

    for n in range(2, 7): # very long runtimes for high n
        n_th = benford_n_digit_distribution(n)
        assert isinstance(n_th, dict), f"{n}th digit distribution is no dict"
        assert len(n_th) == 10, (
            f"{n}th digit distribution does not have appropriate number of entries"
        )

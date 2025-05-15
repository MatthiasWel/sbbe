from chemford.benford.benford import benford_first_digit_distribution
from chemford.benford.benford import benford_first_two_digit_distribution
from chemford.benford.benford import benford_n_digit_distribution


def test_benford():
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

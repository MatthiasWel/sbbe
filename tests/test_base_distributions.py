from chemford.distributions.benford import benford_first_digit_distribution, benford_first_two_digit_distribution, benford_n_digit_distribution
from chemford.distributions.rounded_distributions import empirical_distribution_round_to_one_significant_digit, empirical_distribution_round_to_step_0_05

def assess_basic_distribution_correctness(distribution, distribution_name, expected_value_of_entries):
    assert isinstance(distribution, dict), f"{distribution_name} is no dict"
    assert len(distribution) == expected_value_of_entries, (
        f"{distribution_name} does not have appropriate number of entries"
    )

def test_benford_distributions():
    """Test benford distributions."""
    first = benford_first_digit_distribution()
    name = "First digit distribution"
    expected_value = 9
    assess_basic_distribution_correctness(first, name, expected_value)

    first_two = benford_first_two_digit_distribution()
    name = "First two digit distribution"
    expected_value = 90
    assess_basic_distribution_correctness(first_two, name, expected_value)

    expected_value = 10
    for n in range(2, 7):  # very long runtimes for high n
        n_th = benford_n_digit_distribution(n)
        name = f"{n}th digit distribution"
        assess_basic_distribution_correctness(n_th, name, expected_value)


def test_rounded_distributions():
    """Test rounded distributions"""
    expected_value = 9

    to_one = empirical_distribution_round_to_one_significant_digit()
    name = "Rounded to one digit distribution"
    
    assess_basic_distribution_correctness(to_one, name, expected_value)

    to_half_step = empirical_distribution_round_to_step_0_05()
    name = "Rounded distribution to half-step"
    assess_basic_distribution_correctness(to_half_step, name, expected_value)


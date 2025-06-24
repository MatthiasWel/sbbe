import numpy as np
import pytest
from chemford.statistics.bayes_factor import bayes_factor_dirichlet_multinomial
from chemford.statistics.utils import observed_frequencies


def test_basic_counts():
    """Check counts for simple digit list."""
    digits = [1, 2, 2, 3, 3, 3]
    expected = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0])
    result = observed_frequencies(digits)
    assert all(result == expected), f"Expected {expected}, got {result}"


def test_full_range():
    """Verify counts for full digit range 10-99."""
    digits = list(range(10, 100))
    result = observed_frequencies(digits, mini=10, maxi=99)
    expected = [1] * 90
    assert all(result == expected), f"Expected {expected}, got {result}"


def test_value_error_on_out_of_range():
    """Ensure ValueError is raised for out-of-range digits."""
    digits = [0, 1, 2, 10]
    with pytest.raises(ValueError) as exc_info:
        observed_frequencies(digits)
    assert "range" in str(exc_info.value), f"Unexpected error message: {exc_info.value}"


def test_bayes_factor_favors_H0_when_data_matches():
    """BF < 1 when counts match expected_probs (H0 is supported)."""
    counts = np.array([10, 5, 5]) * 100
    expected_probs = np.array([0.5, 0.25, 0.25])
    bf = bayes_factor_dirichlet_multinomial(counts, expected_probs)
    assert bf < 1, (
        f"BF (currently {bf}) should favor H0 when data matches expected_probs"
    )


def test_bayes_factor_favors_H1_when_data_differs():
    """BF > 1 when counts differ from expected_probs (H1 is supported)."""
    counts = np.array([10, 10, 10]) * 100
    expected_probs = np.array([0.5, 0.25, 0.25])
    bf= bayes_factor_dirichlet_multinomial(counts, expected_probs)
    assert bf > 1, (
        f"BF (currently {bf}) should favor H1 when data differs from expected_probs"
    )


def test_scalar_and_vector_alpha_consistency():
    """Scalar and vector alpha should yield the same BF."""
    counts = np.array([5, 5, 5]) * 100
    expected_probs = [1 / 3, 1 / 3, 1 / 3]
    bf1 = bayes_factor_dirichlet_multinomial(counts, expected_probs, alpha=1.0)
    bf2 = bayes_factor_dirichlet_multinomial(
        counts,
        expected_probs,
        alpha=[1.0, 1.0, 1.0],
    )
    assert np.isclose(bf1, bf2), (
        f"Expected same BF for scalar and vector alpha, got {bf1} vs {bf2}"
    )

import numpy as np
import pandas as pd
from chemford.data_processing.benford_criteria import has_sufficient_data
from chemford.data_processing.benford_criteria import has_sufficient_log_scale_coverage


def test_sufficiency_checks():
    """Test sufficiency checks."""
    short = np.arange(1, 100, 10)
    long = np.arange(1, 1000, 0.01)
    transforms = (pd.Series, np.array, list)
    for transform in transforms:
        assert not has_sufficient_data(transform(short)), (
            f"{transform} did not fail with short test data"
        )
        assert has_sufficient_data(transform(long)), (
            f"{transform} failed with long test data"
        )

    no_coverage = np.arange(1, 10, 0.01)
    good_coverage = np.arange(1, 1000, 0.01)
    for transform in transforms:
        assert not has_sufficient_log_scale_coverage(transform(no_coverage)), (
            f"{transform} did not fail with little coverage"
        )
        assert has_sufficient_log_scale_coverage(transform(good_coverage)), (
            f"{transform} failed with enough coverage"
        )

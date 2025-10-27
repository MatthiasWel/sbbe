from numpy.random import Generator
from scipy.stats import rv_discrete
from chemford.distributions.benford import benford_first_digit_distribution
from chemford.distributions.make_general_multinomial import make_multinomial


def make_benford(random_state: int | Generator | None = None) -> rv_discrete:
    """Create a multinomial according to first digit Benford's law."""
    benford = benford_first_digit_distribution()
    return make_multinomial(
        list(benford.keys()),
        list(benford.values()),
        random_state=random_state,
    )

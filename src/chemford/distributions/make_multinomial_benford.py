from chemford.distributions.benford import benford_first_digit_distribution
from chemford.distributions.make_general_multinomial import make_multinomial


def make_benford(random_state=None):
    benford = benford_first_digit_distribution()
    return make_multinomial(
        list(benford.keys()),
        list(benford.values()),
        random_state=random_state,
    )

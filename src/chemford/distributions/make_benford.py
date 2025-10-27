from chemford.distributions.make_multinomial import make_multinomial
from chemford.distributions.benford import benford_first_digit_distribution

def make_benford(random_state=None):
    benford = benford_first_digit_distribution()
    return make_multinomial(benford.keys(), benford.values(),random_state=random_state)
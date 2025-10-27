from numpy.random import Generator
from scipy.stats import rv_discrete
from chemford.distributions.make_general_multinomial import make_multinomial


def make_uniform(random_state: int | Generator | None = None) -> rv_discrete:
    """Create a multinomial distribution with uniform probs on 1-9."""
    return make_multinomial(
        list(range(1, 10, 1)),
        [1 / 9] * 9,
        random_state=random_state,
    )

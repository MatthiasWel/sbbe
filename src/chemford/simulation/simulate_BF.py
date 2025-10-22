from collections.abc import Sequence
import pandas as pd
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from chemford.benford.benford import benford_first_digit_distribution
from chemford.simulation.distributions import make_multinomial
from chemford.simulation.distributions import sample_from_mixture
from chemford.statistics.bayes_factor import bayes_factor_dirichlet_multinomial
from chemford.statistics.utils import observed_frequencies


def simulate_benford_and_uniform_mixture(
    n_replicas: int,
    sizes: Sequence[int],
    mixing_ratios: Sequence[float],
) -> pd.DataFrame:
    """Simulate samples from a mixture of Benford's and uniform distributions,
    compute Bayes factors against Benford's law using the Dirichlet-multinomial model.

    Parameters:
    - n_replicas: Number of repetitions for each configuration
    - sizes: List of sample sizes
    - mixing_ratios: Mixture weights for Benford
        (1.0 = pure Benford, 0.0 = pure uniform)

    Returns:
    - pd.DataFrame with columns: ['iteration', 'n_samples', 'mixing_ratio', 'log_bf10']
    """
    benford_outcome = benford_first_digit_distribution()
    labels = list(benford_outcome.keys())
    benford_probs = list(benford_outcome.values())
    benford = make_multinomial(labels, benford_probs)

    uniform_probs = [1 / 9] * 9
    uniform = make_multinomial(labels, uniform_probs)

    result = []

    for replica in tqdm(range(n_replicas), desc="Simulating"):
        for size in sizes:
            for mixing_ratio in mixing_ratios:
                sample = sample_from_mixture(
                    dist_a=benford,
                    dist_b=uniform,
                    size=int(size),
                    mix_ratio=mixing_ratio,
                )
                counts = observed_frequencies(sample)
                log_bf10 = bayes_factor_dirichlet_multinomial(
                    counts,
                    benford_probs,
                    alpha=1,
                )
                result.append((replica, size, mixing_ratio, log_bf10))

    return pd.DataFrame(
        result,
        columns=["iteration", "n_samples", "mixing_ratio", "log_bf10"],
    )


def simulate_single_replica(
    replica: int,
    sizes: Sequence[int],
    mixing_ratios: Sequence[float],
    benford: object,
    uniform: object,
    benford_probs: list[float],
) -> list[tuple]:
    """Helper function to simulate a single replica."""
    results = []
    for size in sizes:
        for mixing_ratio in mixing_ratios:
            sample = sample_from_mixture(
                dist_a=benford,
                dist_b=uniform,
                size=int(size),
                mix_ratio=mixing_ratio,
            )
            counts = observed_frequencies(sample)
            log_bf10 = bayes_factor_dirichlet_multinomial(
                counts,
                benford_probs,
                alpha=1,
            )
            results.append((replica, size, mixing_ratio, log_bf10))
    return results


def simulate_benford_and_uniform_mixture_parallel(
    n_replicas: int,
    sizes: Sequence[int],
    mixing_ratios: Sequence[float],
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Parallel version of the Benford-uniform mixture simulation.

    Parameters:
    - n_replicas: Number of repetitions for each configuration
    - sizes: List of sample sizes
    - mixing_ratios: Mixture weights for Benford
    - n_jobs: Number of CPU cores (-1 = all available)
    """
    benford_outcome = benford_first_digit_distribution()
    labels = list(benford_outcome.keys())
    benford_probs = list(benford_outcome.values())
    benford = make_multinomial(labels, benford_probs)

    uniform_probs = [1 / 9] * 9
    uniform = make_multinomial(labels, uniform_probs)

    # # Parallel execution
    results = Parallel(n_jobs=n_jobs)(  # , batch_size=1
        delayed(simulate_single_replica)(
            replica,
            sizes,
            mixing_ratios,
            benford,
            uniform,
            benford_probs,
        )
        for replica in tqdm(range(n_replicas), desc="Simulating: ")
    )

    # Flatten results
    flat_results = [item for sublist in results for item in sublist]

    return pd.DataFrame(
        flat_results,
        columns=["iteration", "n_samples", "mixing_ratio", "log_bf10"],
    )

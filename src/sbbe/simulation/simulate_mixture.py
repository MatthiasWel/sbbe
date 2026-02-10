from collections.abc import Callable
from collections.abc import Sequence
import pandas as pd
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from sbbe.data_processing.observed_frequencies import observed_frequencies
from sbbe.distributions import make_benford
from sbbe.distributions import make_uniform
from sbbe.distributions.benford import benford_first_digit_distribution
from sbbe.simulation.sample_from_mixture import sample_from_mixture


def simulate_single_replica(
    replica: int,
    statistic: Callable,
    sizes: Sequence[int],
    mixing_ratios: Sequence[float],
) -> list[tuple]:
    """Helper function to simulate a single replica."""
    benford_probs = list(benford_first_digit_distribution().values())

    benford = make_benford()
    uniform = make_uniform()
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
            stat = statistic(counts, benford_probs)
            results.append((replica, size, mixing_ratio, stat))
    return results


def simulate_benford_and_uniform_mixture(
    n_replicas: int,
    statistic: Callable,
    sizes: Sequence[int],
    mixing_ratios: Sequence[float],
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Parallel version of the Benford-uniform mixture simulation.

    Parameters:
    - n_replicas: Number of repetitions for each configuration
    - statistic: Function to calculate the statistic
    - sizes: List of sample sizes
    - mixing_ratios: Mixture weights for Benford
    - n_jobs: Number of CPU cores (-1 = all available)
    """
    results = Parallel(n_jobs=n_jobs)(  # , batch_size=1
        delayed(simulate_single_replica)(replica, statistic, sizes, mixing_ratios)
        for replica in tqdm(range(n_replicas), desc="Simulating: ")
    )

    # Flatten results
    flat_results = [item for sublist in results for item in sublist]

    return pd.DataFrame(
        flat_results,
        columns=["iteration", "n_samples", "mixing_ratio", "value"],
    )

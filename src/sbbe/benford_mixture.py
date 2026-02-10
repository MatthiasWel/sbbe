import warnings
from collections.abc import Callable
from typing import ClassVar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import beta
from sbbe.data_processing.benford_criteria import has_sufficient_data
from sbbe.data_processing.benford_criteria import has_sufficient_log_scale_coverage
from sbbe.data_processing.extract_significant_digit import extract_significant_digits
from sbbe.data_processing.observed_frequencies import observed_frequencies
from sbbe.distributions import make_benford
from sbbe.distributions.benford import benford_first_digit_distribution
from sbbe.simulation.estimate_mixture_ratio import (
    estimate_mixture_ratio_from_simulation,
)
from sbbe.simulation.simulate_mixture import simulate_benford_and_uniform_mixture
from sbbe.statistics.bayes_factor import bayes_factor_dirichlet_multinomial
from sbbe.statistics.edf_tests import ks_d
from sbbe.statistics.edf_tests import kuipers_v
from sbbe.statistics.specialized_statistics import euclidean_distance_cho_gains
from sbbe.statistics.specialized_statistics import max_l1_distance_leemis
from sbbe.statistics.specialized_statistics import max_l1_distance_morrow
from sbbe.statistics.xi_squared import xi_squared_counts
from sbbe.statistics.xi_squared import xi_squared_proportions


class InsufficientDataError(Exception):
    """Raised when the dataset is too small or has insufficient log coverage."""


class BenfordMixtureEstimator:
    """Estimate the mixture ratio of Benford-conforming vs. uniform data in a dataset.

    The estimator maintains a cumulative simulation DataFrame and reuses
    simulations where possible to avoid unnecessary recalculation.
    """

    STAT_MAP: ClassVar[dict[str, Callable[..., float]]] = {
        "log_BF": bayes_factor_dirichlet_multinomial,
        "ks": ks_d,
        "kuipers": kuipers_v,
        "cho_gains": euclidean_distance_cho_gains,
        "leemis": max_l1_distance_leemis,
        "morrow": max_l1_distance_morrow,
        "xi_squared_counts": xi_squared_counts,
        "xi_squared_proportions": xi_squared_proportions,
    }

    def __init__(
        self,
        statistic: str | Callable[..., float],
        mixing_ratios: NDArray | None = None,
        ignore_invalid: bool = False,
        a: float = 1,
        b: float = 1,
    ):
        """Initialize a BenfordMixtureEstimator.

        Parameters
        ----------
        statistic : str or callable
            The statistic to use for goodness-of-fit. If a string, it must
            be a key in STAT_MAP; otherwise, it should be a callable.
        mixing_ratios : NDArray, optional
            Array of mixing ratios to simulate. Default is np.arange(0, 1.01, 0.02).
        """
        if mixing_ratios is None:
            mixing_ratios = np.arange(0, 1.01, 0.02)
        if isinstance(statistic, str):
            self.statistic = self.STAT_MAP[statistic]
        else:
            self.statistic = statistic
        self.mixing_ratios = mixing_ratios
        self.simulation = pd.DataFrame()
        self.benford = make_benford()
        self.benford_probs = self.benford.pk
        self.ignore_invalid = ignore_invalid
        self.prior = beta(a, b)

    def __call__(self, data: NDArray, n_replicas: int = 1000):
        """Estimate the Benford-uniform mixture ratio for the given dataset.

        Parameters
        ----------
        data : NDArray
            The observed dataset to analyze.
        n_replicas : int
            Number of simulation replicates to use for estimating mixture ratios.

        Returns:
        -------
        tuple
            M_CI : float
                Estimated mixture ratio with confidence interval.
            probs : NDArray
                Probabilities for each simulated mixing ratio.
            m_vals : NDArray
                The corresponding mixing ratio values.
        """
        first_digits = self._prepare_first_digits(data)
        n = len(first_digits)
        counts = observed_frequencies(first_digits)
        stat = self.statistic(counts=counts, expected_probs=self.benford_probs)

        sim = self._prepare_simulation(n_replicas, n)

        return estimate_mixture_ratio_from_simulation(
            sim,
            stat=stat,
            n_samples=n,
            prior=self.prior,
        )

    def _prepare_simulation(self, n_replicas: int, n: int) -> pd.DataFrame:
        """Ensure the simulation contains enough replicates for a given sample size.

        If the current simulation is empty or contains fewer replicates than requested,
        it generates additional simulations and appends them to the internal DataFrame.
        Finally, it samples the requested number of replicates for use in estimation.

        Parameters
        ----------
        n : int
            The sample size of the data for which simulations are needed.
        n_replicas : int
            The number of simulation replicates required.

        Returns:
        -------
        pd.DataFrame containing `n_replicas` simulations for the given sample size.
        """
        if self.simulation.empty:
            difference = n_replicas
        else:
            available_replicas = len(
                self.simulation[self.simulation.n_samples == n],
            ) / len(self.mixing_ratios)
            difference = int(n_replicas - available_replicas)
        if difference > 0:
            # no enough replicates
            simulation = simulate_benford_and_uniform_mixture(
                n_replicas=difference,
                statistic=self.statistic,
                sizes=[n],
                mixing_ratios=self.mixing_ratios,
            )
            self.simulation = pd.concat([self.simulation, simulation])
        return self.simulation.sample(n_replicas)

    def _prepare_first_digits(self, data: NDArray) -> list[int]:
        """Validate the input data and extract first significant digits.

        Performs the following checks:
        1. Ensures there are enough data points for meaningful estimation.
        2. Ensures sufficient log-scale coverage.
        3. Extracts first digits from each data point.
        4. Optionally removes invalid entries (None) and issues a warning.

        Parameters
        ----------
        data
            The numeric dataset to analyze.

        Returns:
        -------
        List[int]
            List of valid first significant digits extracted from the data.

        Raises:
        ------
        InsufficientDataError
            If the data is too small or lacks log-scale coverage.
        ValueError
            If no valid digits could be extracted from the data.
        """
        if not has_sufficient_data(data, threshold=80):
            msg = "Not enough data points to make meaningful estimates."
            raise InsufficientDataError(msg)

        if not has_sufficient_log_scale_coverage(data):
            msg = "Insufficient log scale coverage."
            raise InsufficientDataError(msg)

        first_digits = [extract_significant_digits(ele) for ele in data]
        if self.ignore_invalid:
            invalid_count = sum(d is None for d in first_digits)
            first_digits = [d for d in first_digits if d is not None]
            if invalid_count > 0:
                warnings.warn(
                    f"{invalid_count} entries are considered invalid",
                    UserWarning,
                )
        if not any(first_digits):
            msg = "Unable to process some entries to digits"
            raise ValueError(msg)
        return [d for d in first_digits if d is not None]

    def visualize(self, data: NDArray) -> None:
        """Plot the FSD distribution and Estimated Benfordness (SBBE).

        Parameters
        ----------
        data: numeric dataset.

        Returns:
        -------
        None
        """
        M_hdi, probs, m_vals = self(data)
        first_digits = self._prepare_first_digits(data)
        counts = observed_frequencies(first_digits)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        ax[0].bar(
            range(1, 10),
            counts,
            color="darkblue",
            alpha=0.8,
            width=0.8,
            label=f"Observed (n={len(data):,})",
        )
        ax[0].plot(
            range(1, 10),
            [i * len(data) for i in list(benford_first_digit_distribution().values())],
            color="darkred",
            alpha=0.8,
            linewidth=2,
            marker="o",
            label="Benford's law",
        )
        ax[0].grid(True, alpha=0.3, linestyle=":")
        ax[0].legend()
        ax[0].set_xticks(range(1, 10))
        ax[0].set_xlabel("First significant digit")
        ax[0].set_ylabel("Count")
        ax[0].set_title("Distribution of first significant digits")
        ax[1].plot(m_vals, probs, color="darkblue", alpha=0.8, linewidth=2)
        m_vals_step = 0.02
        ax[1].axvspan(
            max(M_hdi) + m_vals_step,
            min(M_hdi) - m_vals_step,
            color="darkblue",
            alpha=0.2,
            label="95% Highest density interval",
        )
        ax[1].axvline(
            x=m_vals[np.argmax(probs)],
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Estimated Benfordness: {m_vals[np.argmax(probs)]:.2f}",
        )
        ax[1].grid(True, alpha=0.3, linestyle=":")
        ax[1].set_xlabel("Benfordness")
        ax[1].set_ylabel("Density")
        ax[1].set_title("SBBE Benfordness Estimation")
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].legend()
        plt.tight_layout()
        plt.show()

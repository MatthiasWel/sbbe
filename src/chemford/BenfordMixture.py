import pandas as pd
import numpy as np
from numpy.typing import NDArray

from chemford.statistics.bayes_factor import bayes_factor_dirichlet_multinomial
from chemford.statistics.edf_tests import ks_d
from chemford.statistics.edf_tests import kuipers_v
from chemford.statistics.specialized_statistics import euclidean_distance_cho_gains
from chemford.statistics.specialized_statistics import max_l1_distance_leemis
from chemford.statistics.specialized_statistics import max_l1_distance_morrow
from chemford.statistics.xi_squared import xi_squared_counts
from chemford.statistics.xi_squared import xi_squared_proportions
from abc import ABC

from chemford.simulation.simulate_mixture import simulate_benford_and_uniform_mixture
from chemford.data_processing.observed_frequencies import observed_frequencies
from chemford.distributions import make_benford
from chemford.simulation.estimate_mixture_ratio import estimate_mixture_ratio_from_simulation

from typing import Callable, Iterable, Dict

class BenfordMixtureEstimator(ABC):
    STAT_MAP: Dict[str, Callable[..., float]] = {
        'log_BF': bayes_factor_dirichlet_multinomial,
        'ks': ks_d,
        'kuipers': kuipers_v,
        'cho_gains': euclidean_distance_cho_gains,
        'leemis': max_l1_distance_leemis,
        'morrow': max_l1_distance_morrow,
        'xi_squared_counts': xi_squared_counts,
        'xi_squared_proportions': xi_squared_proportions
    }

    def __init__(self, statistic : str | Callable[..., float], mixing_ratios: NDArray = np.arange(0,1.01,0.02)):
        if isinstance(statistic, str):
            self.statistic = self.STAT_MAP[statistic]
        else:
            self.statistic = statistic
        self.mixing_ratios = mixing_ratios
        self.simulation = pd.DataFrame()
        self.benford = make_benford()
        self.benford_probs = self.benford.pk


    def __call__(self, data: NDArray, n_replicas: int = 1000):
        n = len(data)
        if self.simulation.empty:
            difference = n_replicas
        else:
            available_replicas = len(self.simulation[self.simulation.n_samples == n]) / len(self.mixing_ratios)
            difference = int(n_replicas - available_replicas)
        if difference > 0:
            # no enough replicates
            simulation = simulate_benford_and_uniform_mixture(
                n_replicas=difference,
                statistic=self.statistic,
                sizes=[n],
                mixing_ratios=self.mixing_ratios
            )
            self.simulation = pd.concat([self.simulation, simulation])
        sim = self.simulation.sample(n_replicas)
        counts = observed_frequencies(data)
        stat = self.statistic(
            counts=counts,
            expected_probs=self.benford_probs
        )
        return estimate_mixture_ratio_from_simulation(
            sim,
            stat=stat,
            n_samples=n,
        )


    def plot(self):
        raise NotImplementedError
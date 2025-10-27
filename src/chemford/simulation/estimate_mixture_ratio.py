import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def estimate_mixture_ratio_from_simulation(
    simulation: pd.DataFrame,
    stat: float,
    n_samples: int,
    ci_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the posterior over mixture ratios given a Bayes factor from simulation data.

    Parameters:
    - simulation: DataFrame with columns ['n_samples', 'mixing_ratio', 'log_bf10']
    - logBF: Observed log Bayes factor
    - n_samples: Sample size for which the estimation is performed
    - ci_level: Confidence level for the credible interval (default 0.95)

    Returns:
    - M_CI: Array of mixing ratios within the credible interval
    - probs: Posterior probabilities for each mixture ratio
    - m_vals: Corresponding mixing ratio values
    """
    if not {"n_samples", "mixing_ratio", "log_bf10"}.issubset(simulation.columns):
        msg = "Simulation DataFrame must contain 'n_samples', 'mixing_ratio', 'log_bf10' columns."
        raise ValueError(msg)

    df_sub = simulation[simulation["n_samples"] == n_samples]
    if df_sub.empty:
        msg = f"No data found for n_samples = {n_samples}."
        raise ValueError(msg)

    likelihoods = {}
    for m_val, group in df_sub.groupby("mixing_ratio"):
        if len(group) < 2:
            continue
        kde = gaussian_kde(group["log_bf10"])
        likelihoods[m_val] = kde(stat)[0]

    if not likelihoods:
        msg = "No valid likelihood estimates; not enough data per mixture ratio."
        raise RuntimeError(msg)

    m_vals = np.array(sorted(likelihoods.keys()))
    probs = np.array([likelihoods[m] for m in m_vals])
    probs_sum = probs.sum()
    if probs_sum == 0:
        msg = "All likelihoods are zero. Check input Bayes factor or simulation data."
        raise RuntimeError(msg)

    probs /= probs_sum

    sorted_idx = np.argsort(probs)[::-1]
    cum_prob = np.cumsum(probs[sorted_idx])
    ci_mask = cum_prob <= ci_level
    ci_idx = sorted_idx[ci_mask]
    M_CI = m_vals[ci_idx]

    return M_CI, probs, m_vals

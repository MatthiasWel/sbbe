from collections.abc import Callable
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.stats import gaussian_kde

PriorPDF = Callable[[np.ndarray], np.ndarray]


def estimate_mixture_ratio_from_simulation(
    simulation: pd.DataFrame,
    stat: float,
    n_samples: int,
    prior: PriorPDF | None = None,
    ci_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the posterior from simulation data.

    Parameters:
    - simulation: DataFrame with columns ['n_samples', 'mixing_ratio', 'value']
    - stat: Observed statistic:
            Note make sure that the statistic is the same as in the simulation
    - n_samples: Sample size for which the estimation is performed
    - ci_level: Confidence level for the credible interval (default 0.95)

    Returns:
    - M_CI: Array of mixing ratios within the credible interval
    - probs: Posterior probabilities for each mixture ratio
    - m_vals: Corresponding mixing ratio values
    """
    if not {"n_samples", "mixing_ratio", "value"}.issubset(simulation.columns):
        msg = (
            "Simulation DataFrame must contain "
            "'n_samples', 'mixing_ratio', 'value' columns."
        )
        raise ValueError(msg)

    df_sub = simulation[simulation["n_samples"] == n_samples]
    if df_sub.empty:
        msg = f"No data found for n_samples = {n_samples}."
        raise ValueError(msg)

    prior_pdf = construct_prior(prior)

    likelihoods = calculate_likelihoods(stat, df_sub)

    m_vals = np.array(sorted(likelihoods.keys()))
    probs = np.array([likelihoods[m] * prior_pdf(m) for m in m_vals])
    probs_sum = probs.sum()
    if probs_sum == 0:
        msg = (
            "All likelihoods are zero. Simulation does not exhibit density in "
            "the region of the observed statistic. Observed data likely not from"
            "a mixture in the simulation."
        )
        raise RuntimeError(msg)

    probs /= probs_sum

    sorted_idx = np.argsort(probs)[::-1]
    cum_prob = np.cumsum(probs[sorted_idx])
    ci_mask = cum_prob <= ci_level
    if not any(ci_mask):
        ci_mask[0] = True
    ci_idx = sorted_idx[ci_mask]
    M_CI = m_vals[ci_idx]

    return M_CI, probs, m_vals


def calculate_likelihoods(stat: float, df_sub: pd.DataFrame) -> dict[float, float]:
    """Compute KDE-based likelihoods of the observed statistic for each mixing ratio.

    Parameters:
    - stat: Observed statistic at which to evaluate the KDE likelihood, e.g., a summary
      statistic computed on the observed data that matches the simulated 'value'.
    - df_sub: Subset of the simulation DataFrame containing at least the columns
      ['mixing_ratio', 'value'].

    Returns:
    - likelihoods: Dict mapping mixing_ratio (keys) to likelihood values
      L(stat | mixing_ratio) estimated via Gaussian KDE.

    Raises:
    - RuntimeError: If no valid likelihoods could be computed (e.g., all groups had
      fewer than the minimum number of samples).
    """
    likelihoods = {}
    min_samples_for_kde = 2
    for m_val, group in df_sub.groupby("mixing_ratio"):
        if len(group) < min_samples_for_kde:
            continue
        kde = gaussian_kde(group["value"])
        likelihoods[m_val] = kde(stat)[0]

    if not likelihoods:
        msg = "No valid likelihood estimates; not enough data per mixture ratio."
        raise RuntimeError(msg)
    return likelihoods


def construct_prior(prior: PriorPDF | None = None) -> PriorPDF:
    """Construct a prior PDF callable over mixing ratios in [0, 1].

    Accepts either:
    - None: defaults to the uniform prior Beta(1, 1).
    - A callable that takes an array of m values and returns the prior density
      at those points.

    Parameters:
    - prior: One of None, or a callable rior_pdf. If None, uses Beta(1, 1).

    Returns:
    - prior_pdf: a function that returns the prior
      density evaluated at given mixing ratio values.

    Raises:
    - TypeError: If `prior` is not None, not a frozen SciPy distribution with .pdf,
      and not a callable.
    """
    if prior is None:
        _prior = beta(1, 1)
        prior_pdf = _prior.pdf
    elif hasattr(prior, "pdf"):
        prior_pdf = prior.pdf
    elif callable(prior):
        prior_pdf = prior
    else:
        msg = (
            "Prior must be None, a frozen scipy "
            "distribution with .pdf, or a callable pdf(m)."
        )
        raise TypeError(msg)
    return prior_pdf

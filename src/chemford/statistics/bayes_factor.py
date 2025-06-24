from collections.abc import Iterable
import numpy as np
from scipy.special import gammaln


def bayes_factor_dirichlet_multinomial(
    counts: Iterable[int],
    expected_probs: Iterable[float],
    alpha: Iterable[float] | float = 1.0,
) -> tuple[float, float]:
    """Compute the Bayes Factor comparing two models for observed count data.

    - H0: A multinomial model with fixed probabilities (`expected_probs`).
    - H1: A multinomial model with a Dirichlet prior
          over probabilities (`Dirichlet(alpha)`).

    Parameters
    ----------
    counts : array-like
        Observed counts for each category.
    expected_probs : array-like
        Fixed probabilities under H0 (should sum to 1).
    alpha : int or array-like, optional
        Dirichlet prior parameter(s) for H1.
        If scalar, a symmetric Dirichlet prior is used.

    Returns:
    -------
    bf10 : float
        Bayes Factor in favor of H1 over H0.
    log_bf10 : float
        Natural logarithm of the Bayes Factor.
    """
    counts = np.array(counts)
    n = np.sum(counts)

    alpha_np = np.ones_like(counts) * alpha if np.isscalar(alpha) else np.array(alpha)

    log_likelihood_H0 = np.sum(counts * np.log(expected_probs))

    # Closed form of log integral TODO: double check math
    log_marginal_H1 = (
        gammaln(np.sum(alpha_np))
        - gammaln(n + np.sum(alpha_np))
        + np.sum(gammaln(counts + alpha_np) - gammaln(alpha_np))
    )

    return log_marginal_H1 - log_likelihood_H0


def bayes_factor_fixed_probas(
    counts: Iterable[int],
    expected_d1: Iterable[float],
    expected_d2: Iterable[float],
) -> tuple[float, float]:
    """Compute the Bayes Factor comparing two fixed-probability multinomial models.

    - H0: A multinomial model with fixed probabilities (`expected_d1`).
    - H1: A multinomial model with fixed probabilities (`expected_d2`).

    Parameters
    ----------
    counts : array-like
        Observed counts for each category.
    expected_d1 : array-like
        Fixed probabilities under H0 (should sum to 1).
    expected_d2 : array-like
        Fixed probabilities under H1 (should sum to 1).

    Returns:
    -------
    bf10 : float
        Bayes Factor in favor of H1 over H0.
    log_bf10 : float
        Natural logarithm of the Bayes Factor.

    """
    counts = np.array(counts)

    log_likelihood_H0 = np.sum(counts * np.log(expected_d1))
    log_likelihood_H1 = np.sum(counts * np.log(expected_d2))

    log_bf10 = log_likelihood_H1 - log_likelihood_H0
    return np.exp(log_bf10), log_bf10

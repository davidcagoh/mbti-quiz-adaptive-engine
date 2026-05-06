"""
Domain-agnostic Bayesian update for Gaussian prior and linear observation model.
"""

import numpy as np


def bayesian_update(
    mu_prior: np.ndarray,
    Sigma_prior: np.ndarray,
    w_t: np.ndarray,
    y_t: float,
    sigma2_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Bayesian update for Gaussian prior and likelihood.

    Observation model: y_t = w_t^T theta + noise, noise ~ N(0, sigma2_t).

    Parameters
    ----------
    mu_prior : np.ndarray
        Prior mean vector (d,).
    Sigma_prior : np.ndarray
        Prior covariance matrix (d, d).
    w_t : np.ndarray
        Question weight vector (d,).
    y_t : float
        Observed response.
    sigma2_t : float
        Observation noise variance.

    Returns
    -------
    mu_post : np.ndarray
        Posterior mean (d,).
    Sigma_post : np.ndarray
        Posterior covariance (d, d).
    """
    Sigma_inv_prior = np.linalg.inv(Sigma_prior)
    Sigma_post = np.linalg.inv(Sigma_inv_prior + np.outer(w_t, w_t) / sigma2_t)
    mu_post = Sigma_post @ (Sigma_inv_prior @ mu_prior + w_t * y_t / sigma2_t)
    return mu_post, Sigma_post

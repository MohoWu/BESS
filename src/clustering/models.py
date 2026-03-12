"""Clustering model fitting: KMeans and GMM."""
import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def fit_kmeans(
    X_scaled: np.ndarray,
    k_values: list[int] | None = None,
    n_init: int = 20,
    seeds: list[int] | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Fit KMeans for each combination of k and random seed.

    Parameters
    ----------
    X_scaled:
        Scaled feature matrix (n_days x n_features).
    k_values:
        List of cluster counts to try. Default: [3, 4, 5, 6].
    n_init:
        Number of centroid initialisations per KMeans run.
    seeds:
        List of random seeds. Default: [42, 123, 456].

    Returns
    -------
    dict keyed by (k, seed) with values:
        ``labels``, ``inertia``, ``model`` (fitted KMeans object).
    """
    if k_values is None:
        k_values = [3, 4, 5, 6]
    if seeds is None:
        seeds = [42, 123, 456]

    results: dict[tuple[int, int], dict[str, Any]] = {}

    for k in k_values:
        for seed in seeds:
            km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
            km.fit(X_scaled)
            results[(k, seed)] = {
                "labels": km.labels_,
                "inertia": km.inertia_,
                "model": km,
            }
            logger.debug("KMeans k=%d seed=%d  inertia=%.2f", k, seed, km.inertia_)

    return results


def fit_gmm(
    X_scaled: np.ndarray,
    k_values: list[int] | None = None,
    seeds: list[int] | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Fit GaussianMixture models for each combination of k and seed.

    Parameters
    ----------
    X_scaled:
        Scaled feature matrix (n_days x n_features).
    k_values:
        List of component counts to try. Default: [3, 4, 5, 6].
    seeds:
        List of random seeds. Default: [42, 123, 456].

    Returns
    -------
    dict keyed by (k, seed) with values:
        ``labels``, ``bic``, ``aic``, ``model`` (fitted GaussianMixture object).
    """
    if k_values is None:
        k_values = [3, 4, 5, 6]
    if seeds is None:
        seeds = [42, 123, 456]

    results: dict[tuple[int, int], dict[str, Any]] = {}

    for k in k_values:
        for seed in seeds:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=5,
                random_state=seed,
            )
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)
            bic = gmm.bic(X_scaled)
            aic = gmm.aic(X_scaled)
            results[(k, seed)] = {
                "labels": labels,
                "bic": bic,
                "aic": aic,
                "model": gmm,
            }
            logger.debug("GMM k=%d seed=%d  BIC=%.2f  AIC=%.2f", k, seed, bic, aic)

    return results

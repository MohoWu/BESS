"""Cluster evaluation metrics and visualisation functions."""
import logging
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.clustering.config import NEGATIVE_PRICE_THRESHOLD, SPIKE_PRICE_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def evaluate_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute standard internal cluster validity indices.

    Parameters
    ----------
    X_scaled:
        Scaled feature matrix.
    labels:
        Integer cluster assignments (length n_days).

    Returns
    -------
    dict with keys: silhouette, davies_bouldin, calinski_harabasz,
                    cluster_sizes, cluster_fractions.
    """
    unique, counts = np.unique(labels, return_counts=True)
    fractions = counts / len(labels)

    metrics = {
        "silhouette": silhouette_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
        "cluster_fractions": dict(zip(unique.tolist(), fractions.tolist())),
    }

    logger.info(
        "Silhouette=%.3f  DB=%.3f  CH=%.1f",
        metrics["silhouette"],
        metrics["davies_bouldin"],
        metrics["calinski_harabasz"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Profile plots
# ---------------------------------------------------------------------------


def plot_cluster_profiles(
    daily_df: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """2-row grid: average price shape and net-load shape by cluster.

    Each sub-plot shows the 48-SP mean profile ± 1 std for each cluster.
    """
    if figsize is None:
        figsize = (14, 7)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    titles = ["Price shape (z-score)", "Net-load shape (z-score)"]
    prefixes = ["price_shape", "netload_shape"]
    palette = sns.color_palette("tab10", n_clusters)

    sp_labels = list(range(1, 49))

    df = daily_df.copy()
    df["cluster"] = labels

    for ax, prefix, title in zip(axes, prefixes, titles):
        cols = [f"{prefix}_{sp}" for sp in sp_labels]
        available = [c for c in cols if c in df.columns]
        if not available:
            ax.set_title(f"{title} (data unavailable)")
            continue

        for k in range(n_clusters):
            sub = df[df["cluster"] == k][available]
            mean = sub.mean()
            std = sub.std()
            x = list(range(len(available)))
            ax.plot(x, mean.values, color=palette[k], label=f"Cluster {k}", linewidth=2)
            ax.fill_between(
                x,
                (mean - std).values,
                (mean + std).values,
                color=palette[k],
                alpha=0.15,
            )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Settlement period")
    axes[-1].set_xticks(range(0, 48, 4))
    axes[-1].set_xticklabels([str(s) for s in range(1, 49, 4)])

    fig.suptitle(f"Cluster profiles (k={n_clusters})", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------


def plot_summary_distributions(
    daily_df: pd.DataFrame,
    labels: np.ndarray,
    cols: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Box plots of key summary statistics by cluster."""
    if cols is None:
        cols = [
            "price_mean", "price_std", "price_range",
            "netload_mean", "netload_std",
            "evening_peak_minus_midday_trough",
        ]
    cols = [c for c in cols if c in daily_df.columns]

    n_cols = min(4, len(cols))
    n_rows = (len(cols) + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    df = daily_df.copy()
    df["cluster"] = labels

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x="cluster", y=col, ax=axes[i], palette="tab10")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Cluster")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Seasonality plot
# ---------------------------------------------------------------------------


def plot_seasonality(
    daily_df: pd.DataFrame,
    labels: np.ndarray,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap showing cluster assignment frequency by month and weekday."""
    df = daily_df.copy()
    df["cluster"] = labels

    if figsize is None:
        figsize = (14, 5)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Month heatmap
    month_ct = df.groupby(["month", "cluster"]).size().unstack(fill_value=0)
    month_ct_pct = month_ct.div(month_ct.sum(axis=1), axis=0)
    sns.heatmap(
        month_ct_pct,
        ax=axes[0],
        cmap="YlOrRd",
        annot=True,
        fmt=".0%",
        cbar=False,
    )
    axes[0].set_title("Cluster frequency by month")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Month")

    # Weekday heatmap
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_ct = df.groupby(["weekday", "cluster"]).size().unstack(fill_value=0)
    day_ct_pct = day_ct.div(day_ct.sum(axis=1), axis=0)
    day_ct_pct.index = [day_names[i] for i in day_ct_pct.index]
    sns.heatmap(
        day_ct_pct,
        ax=axes[1],
        cmap="YlOrRd",
        annot=True,
        fmt=".0%",
        cbar=False,
    )
    axes[1].set_title("Cluster frequency by weekday")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Weekday")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Negative/spike price plot
# ---------------------------------------------------------------------------


def plot_negative_and_spike_days(
    daily_df: pd.DataFrame,
    labels: np.ndarray,
    thresholds: dict[str, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Bar chart showing % of negative-price and spike-price days by cluster."""
    if thresholds is None:
        thresholds = {
            "negative": NEGATIVE_PRICE_THRESHOLD,
            "spike": SPIKE_PRICE_THRESHOLD,
        }

    df = daily_df.copy()
    df["cluster"] = labels
    df["is_negative"] = df["price_min"] < thresholds["negative"]
    df["is_spike"] = df["price_max"] > thresholds["spike"]

    stats = df.groupby("cluster")[["is_negative", "is_spike"]].mean() * 100

    if figsize is None:
        figsize = (8, 4)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(stats))
    width = 0.35
    ax.bar(x - width / 2, stats["is_negative"], width, label="% days with negative price", color="steelblue")
    ax.bar(x + width / 2, stats["is_spike"], width, label=f"% days with spike >{thresholds['spike']} £/MWh", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {k}" for k in stats.index])
    ax.set_ylabel("% of days")
    ax.set_title("Negative and spike price day frequency by cluster")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------


def stability_analysis(
    X_scaled: np.ndarray,
    method: str = "kmeans",
    k: int = 4,
    n_bootstrap: int = 20,
    sample_frac: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap stability analysis via Adjusted Rand Index.

    Repeatedly fits the model on random subsamples and computes pairwise ARI
    between all pairs of label assignments. High mean ARI (>0.7) indicates
    a stable solution.

    Parameters
    ----------
    method:
        ``"kmeans"`` or ``"gmm"``.
    k:
        Number of clusters / components.
    n_bootstrap:
        Number of bootstrap fits.
    sample_frac:
        Fraction of days to sample per bootstrap iteration.
    seed:
        Base random seed.

    Returns
    -------
    dict with keys: ``ari_values`` (list of pairwise ARIs), ``mean_ari``,
                    ``std_ari``, ``fig`` (ARI distribution plot).
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    n = len(X_scaled)
    sample_size = int(n * sample_frac)
    all_labels = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sub = X_scaled[idx]

        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1000)))
        else:
            model = GaussianMixture(n_components=k, n_init=3, random_state=int(rng.integers(1000)))

        model.fit(X_sub)
        # Predict labels on the *full* dataset for comparability
        if method == "kmeans":
            full_labels = model.predict(X_scaled)
        else:
            full_labels = model.predict(X_scaled)
        all_labels.append(full_labels)

    # Pairwise ARI on full dataset labels
    ari_values = []
    for i in range(n_bootstrap):
        for j in range(i + 1, n_bootstrap):
            ari_values.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    mean_ari = float(np.mean(ari_values))
    std_ari = float(np.std(ari_values))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ari_values, bins=20, edgecolor="black", color="steelblue", alpha=0.8)
    ax.axvline(mean_ari, color="red", linestyle="--", label=f"Mean ARI = {mean_ari:.3f}")
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap stability ({method.upper()}, k={k}, n={n_bootstrap} runs)")
    ax.legend()
    fig.tight_layout()

    logger.info("Stability mean ARI = %.3f ± %.3f", mean_ari, std_ari)
    return {
        "ari_values": ari_values,
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "fig": fig,
    }

"""Clustering metrics: regime change detection, trajectory clustering.

New metrics for E5 (state detection) and E8 (token type analysis).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def detect_regime_changes(
    trajectory: np.ndarray,
    window_size: int = 10,
    threshold: float = 2.0,
) -> List[int]:
    """Detect regime changes in a trajectory using a sliding-window distance metric.

    Parameters
    ----------
    trajectory:
        Shape ``(num_steps, n_features)`` — e.g., PCA coordinates over decode steps.
    window_size:
        Size of the sliding window for computing local statistics.
    threshold:
        Number of standard deviations above the mean step-distance to flag as a change.

    Returns
    -------
    List of step indices where regime changes are detected.
    """
    if len(trajectory) < 2 * window_size:
        return []

    # Compute step-to-step distances
    diffs = np.diff(trajectory, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)

    # Moving average and std
    mean_dist = np.mean(step_dists)
    std_dist = np.std(step_dists)

    if std_dist < 1e-10:
        return []

    changepoints = []
    for i in range(window_size, len(step_dists) - window_size):
        local_dist = step_dists[i]
        if local_dist > mean_dist + threshold * std_dist:
            # Check it's a local maximum
            if local_dist >= max(step_dists[max(0, i - 3):i + 4]):
                changepoints.append(i + 1)  # +1 because diff shifts by 1

    return changepoints


def cluster_trajectories(
    trajectories: np.ndarray,
    min_cluster_size: int = 5,
    method: str = "hdbscan",
) -> Tuple[np.ndarray, int]:
    """Cluster activation trajectories.

    Parameters
    ----------
    trajectories:
        Shape ``(num_samples, n_features)``.
    min_cluster_size:
        Minimum cluster size for HDBSCAN.
    method:
        Clustering method: "hdbscan" or "kmeans".

    Returns
    -------
    (labels, n_clusters) — cluster assignments (-1 for noise) and number of clusters.
    """
    if method == "hdbscan":
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(trajectories)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            return labels, n_clusters
        except ImportError:
            logger.warning("hdbscan not installed; falling back to kmeans.")
            method = "kmeans"

    if method == "kmeans":
        from sklearn.cluster import KMeans
        # Use silhouette score to pick k
        best_k = 2
        best_score = -1
        for k in range(2, min(10, len(trajectories) // 2)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(trajectories)
            from sklearn.metrics import silhouette_score
            score = silhouette_score(trajectories, labels)
            if score > best_score:
                best_score = score
                best_k = k

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(trajectories)
        return labels, best_k

    raise ValueError(f"Unknown clustering method: {method}")


def compute_fisher_discriminant(
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Fisher's linear discriminant ratio for class separability.

    Parameters
    ----------
    features:
        Shape ``(n_samples, n_features)``.
    labels:
        Shape ``(n_samples,)`` — integer class labels.

    Returns
    -------
    Fisher discriminant ratio (higher = more separable).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    overall_mean = features.mean(axis=0)
    n_features = features.shape[1]

    S_b = np.zeros((n_features, n_features))  # Between-class scatter
    S_w = np.zeros((n_features, n_features))  # Within-class scatter

    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        n_k = len(class_features)
        if n_k == 0:
            continue

        class_mean = class_features.mean(axis=0)
        diff = (class_mean - overall_mean).reshape(-1, 1)
        S_b += n_k * (diff @ diff.T)

        centered = class_features - class_mean
        S_w += centered.T @ centered

    # Fisher criterion: trace(S_b) / trace(S_w)
    trace_sw = np.trace(S_w)
    if trace_sw < 1e-10:
        return float('inf')
    return np.trace(S_b) / trace_sw

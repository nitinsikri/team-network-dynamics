"""Interaction patterns: K-means on player connection profiles, dyads and triads."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.cluster import KMeans

from .synth import POSITIONS


def cluster_players(A: np.ndarray, k: int = 3, seed: int = 0) -> dict:
    """Group players whose connection profiles (normalised rows of A) look alike."""
    prof = A / np.maximum(A.sum(axis=1, keepdims=True), 1)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(prof)
    groups: dict[int, list[str]] = {}
    for p, l in zip(POSITIONS, labels):
        groups.setdefault(int(l), []).append(p)
    return groups


def strong_dyads(A: np.ndarray, top: int = 5):
    pairs = [(A[i, j], POSITIONS[i], POSITIONS[j])
             for i, j in itertools.combinations(range(len(A)), 2)]
    return [(a, b, int(w)) for w, a, b in sorted(pairs, reverse=True)[:top]]


def strong_triads(A: np.ndarray, top: int = 5):
    """Triangles ranked by their weakest edge (all three links must be strong)."""
    tri = [(min(A[i, j], A[j, k], A[i, k]), POSITIONS[i], POSITIONS[j], POSITIONS[k])
           for i, j, k in itertools.combinations(range(len(A)), 3)]
    return [(a, b, c, int(w)) for w, a, b, c in sorted(tri, reverse=True)[:top]]


def configuration_variety(windows: list[np.ndarray], k: int = 4, seed: int = 0) -> dict:
    """Cluster 10-minute networks into tactical configurations; more distinct,
    evenly used configurations suggest tactical flexibility."""
    X = np.array([w[np.triu_indices(len(w), 1)] / max(w.sum(), 1) for w in windows])
    k = min(k, len(X))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    p = np.bincount(labels) / len(labels)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {"labels": labels.tolist(), "entropy": entropy}

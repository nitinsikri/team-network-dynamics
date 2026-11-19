"""Passing network construction and structural metrics (paper Eqs. 1-7)."""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd

from .synth import POSITIONS


def adjacency(passes: pd.DataFrame, players=POSITIONS) -> np.ndarray:
    """Undirected weighted adjacency: A_ij = passes exchanged between i and j (Eq. 1)."""
    k = {p: i for i, p in enumerate(players)}
    A = np.zeros((len(players), len(players)))
    for s, r in zip(passes["sender"], passes["receiver"]):
        if s != r:
            A[k[s], k[r]] += 1
    return A + A.T


def total_links(A: np.ndarray) -> float:          # Eq. 2
    return A.sum() / 2


def density(A: np.ndarray) -> float:              # Eq. 3, as defined in the paper (L / C(11,2))
    n = len(A)
    return total_links(A) / (n * (n - 1) / 2)


def diameter(A: np.ndarray) -> float:             # Eqs. 4-5, distance = 1 / A_ij on paths
    G = nx.Graph()
    n = len(A)
    for i, j in itertools.combinations(range(n), 2):
        if A[i, j] > 0:
            G.add_edge(i, j, d=1.0 / A[i, j])
    if G.number_of_nodes() < n or not nx.is_connected(G):
        return float("inf")
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="d"))
    return max(max(v.values()) for v in lengths.values())


def weighted_clustering(A: np.ndarray) -> np.ndarray:   # Eq. 6
    n = len(A)
    C = np.zeros(n)
    for i in range(n):
        num = den = 0.0
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                num += A[i, j] * A[j, k] * A[i, k]
                den += A[i, j] * A[i, k]
        C[i] = num / den if den else 0.0
    return C


def team_clustering(A: np.ndarray) -> float:             # Eq. 7
    return float(weighted_clustering(A).mean())


def summary(A: np.ndarray) -> dict:
    return {"total_links": float(total_links(A)), "density": float(density(A)),
            "diameter": float(diameter(A)), "clustering": float(team_clustering(A))}


def windowed_networks(passes: pd.DataFrame, minutes: int = 10):
    """Adjacency matrices for consecutive time windows (tactical changes in a match)."""
    out = []
    for start in range(0, int(passes["t"].max()) + 1, minutes * 60):
        w = passes[(passes["t"] >= start) & (passes["t"] < start + minutes * 60)]
        if len(w):
            out.append(adjacency(w))
    return out

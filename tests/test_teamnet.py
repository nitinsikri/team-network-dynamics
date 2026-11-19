import numpy as np
import pandas as pd

from teamnet import (adjacency, summary, weighted_clustering, simulate_season, season_table,
                     AdversarialRegression, strong_triads)
from teamnet.synth import POSITIONS


def _passes(pairs):
    return pd.DataFrame({"t": range(len(pairs)), "sender": [a for a, _ in pairs],
                         "receiver": [b for _, b in pairs], "x": 50, "y": 50})


def test_adjacency_symmetric_and_counts():
    A = adjacency(_passes([("D1", "D2"), ("D2", "D1"), ("M1", "F1")]))
    assert np.allclose(A, A.T)
    assert summary(A)["total_links"] == 3


def test_clustering_triangle():
    A = adjacency(_passes([("D1", "D2"), ("D2", "D3"), ("D3", "D1")]))
    C = weighted_clustering(A)
    assert C[POSITIONS.index("D1")] == 1.0
    assert strong_triads(A, 1)[0][:3] == ("D1", "D2", "D3")


def test_adversarial_weights_on_simplex():
    t = season_table(simulate_season(12, seed=1))
    m = AdversarialRegression().fit(t)
    assert abs(m.beta.sum() - 1) < 1e-9
    assert set(m.predict(t)) <= {"win", "loss", "draw"}

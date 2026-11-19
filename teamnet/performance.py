"""Performance indicators and the adversarial regression model (paper Eqs. 8-11)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .network import adjacency, team_clustering

INDICATORS = ["coordination", "distribution", "tempo", "flexibility", "pressing"]


def tempo_seconds(passes: pd.DataFrame, block: int = 50) -> float:
    """Mean time taken to complete ``block`` consecutive passes (lower = faster)."""
    t = passes["t"].to_numpy()
    if len(t) <= block:
        return float(t[-1] - t[0]) if len(t) else np.nan
    return float(np.mean(t[block:] - t[:-block]))


def indicators(passes: pd.DataFrame, stats: dict, block: int = 50) -> dict:
    f = stats["goals"] / max(stats["shots"], 1)
    d = stats["recoveries"] / max(stats["losses"], 1)
    return {"coordination": team_clustering(adjacency(passes)),   # gamma = C
            "distribution": f * d,                                   # phi = f x d   (Eq. 8)
            "tempo": -tempo_seconds(passes, block),                  # negated: higher = faster
            "flexibility": float(passes["y"].std()),                 # sigma_y
            "pressing": float(passes["x"].mean())}                   # x-bar


def season_table(season, block: int = 50) -> pd.DataFrame:
    rows = []
    for m in season:
        a = indicators(*m["A"], block=block)
        b = indicators(*m["B"], block=block)
        rows.append({"match": m["match"], "result": m["result"],
                     **{f"A_{k}": v for k, v in a.items()}, **{f"B_{k}": v for k, v in b.items()}})
    return pd.DataFrame(rows)


class AdversarialRegression:
    """Find weights beta (on the simplex, step 0.1) so that S_A - S_B agrees with
    the match outcome within tolerance delta (Eq. 11); ties are broken by the
    lowest variance of scores, as described in the paper."""

    def __init__(self, delta: float = 0.25, step: float = 0.1):
        self.delta, self.step = delta, step

    def _scale(self, df):
        A = df[[f"A_{k}" for k in INDICATORS]].to_numpy()
        B = df[[f"B_{k}" for k in INDICATORS]].to_numpy()
        if not hasattr(self, "mu"):
            both = np.vstack([A, B])
            self.mu, self.sd = both.mean(0), both.std(0) + 1e-9
        return (A - self.mu) / self.sd, (B - self.mu) / self.sd

    def _consistent(self, diff, results):
        return np.where(results == "win", diff > self.delta,
                        np.where(results == "loss", diff < -self.delta, np.abs(diff) <= self.delta))

    def fit(self, df: pd.DataFrame) -> "AdversarialRegression":
        A, B = self._scale(df)
        res = df["result"].to_numpy()
        n = round(1 / self.step)
        best = None
        for c in itertools.product(range(n + 1), repeat=len(INDICATORS)):
            if sum(c) != n:
                continue
            beta = np.array(c) / n
            sa, sb = A @ beta, B @ beta
            acc = self._consistent(sa - sb, res).mean()
            var = np.var(np.concatenate([sa, sb]))
            key = (acc, -var)
            if best is None or key > best[0]:
                best = (key, beta)
        self.beta = best[1]
        self.train_accuracy = best[0][0]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        A, B = self._scale(df)
        diff = (A - B) @ self.beta
        return np.where(diff > self.delta, "win", np.where(diff < -self.delta, "loss", "draw"))

    def weights(self) -> dict:
        return {k: round(float(v), 2) for k, v in zip(INDICATORS, self.beta)}

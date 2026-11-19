"""Synthetic match event data (passes, shots, defensive actions).

Each team has a latent style (coordination, width, pressing, tempo) that shapes
its passing events; match outcomes depend on the same latent qualities, so the
analysis pipeline has a real signal to find.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITIONS = ["GK", "D1", "D2", "D3", "D4", "M1", "M2", "M3", "F1", "F2", "F3"]
BASE_X = np.array([5, 20, 20, 20, 20, 45, 45, 45, 75, 75, 75], float)
BASE_Y = np.array([50, 15, 38, 62, 85, 25, 50, 75, 20, 50, 80], float)


@dataclass
class TeamStyle:
    name: str
    cohesion: float = 1.0    # concentration of passes inside sub-groups
    width: float = 1.0       # lateral spread
    press: float = 0.0       # shift of play up the pitch
    tempo: float = 1.0       # passes per minute multiplier
    finishing: float = 0.12  # goals per shot
    defending: float = 0.55  # successful defensive actions per loss of possession


def simulate_team_events(style: TeamStyle, rng: np.random.Generator, minutes: int = 90):
    n_passes = int(rng.normal(420, 40) * style.tempo)
    # preference matrix: neighbours in formation pass more, cohesion sharpens it
    d = np.hypot(BASE_X[:, None] - BASE_X[None], BASE_Y[:, None] - BASE_Y[None])
    pref = np.exp(-d / (25 / style.cohesion))
    np.fill_diagonal(pref, 0)
    pref /= pref.sum(axis=1, keepdims=True)
    times = np.sort(rng.uniform(0, minutes * 60, n_passes))
    sender = rng.integers(1, 11, n_passes)
    receiver = np.array([rng.choice(11, p=pref[s]) for s in sender])
    x = np.clip(BASE_X[sender] + style.press * 15 + rng.normal(0, 8, n_passes), 0, 100)
    y = np.clip(50 + (BASE_Y[sender] - 50) * style.width + rng.normal(0, 8, n_passes), 0, 100)
    shots = rng.poisson(12 * (0.7 + 0.3 * style.press + 0.2 * style.cohesion))
    goals = rng.binomial(shots, style.finishing)
    losses = rng.poisson(90)
    recoveries = rng.binomial(losses, style.defending)
    passes = pd.DataFrame({"t": times, "sender": [POSITIONS[i] for i in sender],
                           "receiver": [POSITIONS[i] for i in receiver], "x": x, "y": y})
    return passes, {"shots": shots, "goals": goals, "losses": losses, "recoveries": recoveries}


def random_style(name: str, rng: np.random.Generator) -> TeamStyle:
    return TeamStyle(name, cohesion=rng.uniform(0.7, 1.5), width=rng.uniform(0.7, 1.3),
                     press=rng.uniform(-0.3, 0.6), tempo=rng.uniform(0.8, 1.25),
                     finishing=rng.uniform(0.06, 0.18), defending=rng.uniform(0.4, 0.7))


def simulate_season(n_matches: int = 38, seed: int = 0, team: TeamStyle | None = None):
    """Return a list of matches: dicts with both teams' events and the result for team A."""
    rng = np.random.default_rng(seed)
    team = team or TeamStyle("Huskies", cohesion=1.2, width=1.1, press=0.2, finishing=0.11)
    season = []
    for m in range(n_matches):
        opp = random_style(f"Opponent{m}", rng)
        a_pass, a_stats = simulate_team_events(team, rng)
        b_pass, b_stats = simulate_team_events(opp, rng)
        diff = a_stats["goals"] - b_stats["goals"]
        result = "win" if diff > 0 else "loss" if diff < 0 else "draw"
        season.append({"match": m + 1, "A": (a_pass, a_stats), "B": (b_pass, b_stats),
                       "result": result, "opponent": opp.name})
    return season

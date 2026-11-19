"""Network-centric framework for evaluating team dynamics (reference implementation)."""
from .synth import simulate_season, TeamStyle, POSITIONS
from .network import adjacency, summary, windowed_networks, weighted_clustering
from .patterns import cluster_players, strong_dyads, strong_triads, configuration_variety
from .performance import indicators, season_table, AdversarialRegression, INDICATORS

__all__ = ["simulate_season", "TeamStyle", "POSITIONS", "adjacency", "summary",
           "windowed_networks", "weighted_clustering", "cluster_players", "strong_dyads",
           "strong_triads", "configuration_variety", "indicators", "season_table",
           "AdversarialRegression", "INDICATORS"]

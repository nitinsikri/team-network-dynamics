"""Full pipeline on a synthetic 38-match season: networks, patterns, adversarial
regression (33 training / 5 validation matches) and a sensitivity check."""
import numpy as np

from teamnet import (simulate_season, adjacency, summary, windowed_networks, cluster_players,
                     strong_dyads, strong_triads, configuration_variety, season_table,
                     AdversarialRegression)

season = simulate_season(38, seed=3)
first = season[0]
A = adjacency(first["A"][0])
print("Match 1 network metrics:", {k: round(v, 3) for k, v in summary(A).items()})
print("Player groups (K-means):", cluster_players(A))
print("Strongest dyads :", strong_dyads(A, 3))
print("Strongest triads:", strong_triads(A, 3))
print("Tactical variety (10-min windows):",
      round(configuration_variety(windowed_networks(first["A"][0]))["entropy"], 3))

table = season_table(season)
train, valid = table.iloc[:33], table.iloc[33:]
model = AdversarialRegression(delta=0.25).fit(train)
print("\nEstimated weights:", model.weights())
print(f"Training consistency  : {model.train_accuracy:.1%}")
print(f"Validation accuracy   : {(model.predict(valid) == valid.result).mean():.1%}")

print("\nSensitivity to the draw threshold delta:")
for d in [0.1, 0.25, 0.5, 1.0]:
    m = AdversarialRegression(delta=d).fit(train)
    print(f"  delta={d:<4}  train={m.train_accuracy:.1%}  "
          f"valid={(m.predict(valid) == valid.result).mean():.1%}  beta={m.weights()}")

print("\nSensitivity to the tempo block size:")
for block in [25, 50, 100]:
    t = season_table(season, block=block)
    m = AdversarialRegression().fit(t.iloc[:33])
    print(f"  block={block:<3} train={m.train_accuracy:.1%}")

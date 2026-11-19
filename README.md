# Network-Centric Analysis of Team Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24380915.svg)](https://doi.org/10.5281/zenodo.24380915)

Reference implementation of the method described in:

**Nitin Sikri.** *A Network-Centric Framework for Evaluating Team Dynamics: From Sports to Organizational Success.* Proceedings of the 5th International Conference on Big Data and Artificial Intelligence Applications (ICBDAIA'26), Springer, Lecture Notes in Networks and Systems, 2026. *(In press.)*

> **Reference implementation.** This repository was written in November 2026, after the paper was accepted, to illustrate the method described in the paper and to make it easy to reuse. It is **not** the original experimental code, it runs on **synthetic data**, and its numbers will **differ** from the results reported in the paper. Please cite the paper for the method and its reported results.

## What the paper proposes

Team play is modelled as a weighted passing network (players = nodes, passes = edges). Structural metrics (total links, density, diameter, weighted clustering) and five indicators – coordination, distribution, tempo, flexibility, pressing – feed an **adversarial regression** in which a team's score must beat, tie or lose to the opponent's score consistently with the match result. K-means on player connection profiles reveals dyads, triads and tactical configurations; the paper then generalises the indicators to organisational teams.

## What this repository contains

- `teamnet/synth.py` – synthetic match events with latent team styles (the original match data are not redistributable)
- `teamnet/network.py` – adjacency (Eq. 1), total links (2), density (3), diameter with distance 1/A<sub>ij</sub> (4–5), weighted clustering (6–7), 10-minute windows
- `teamnet/patterns.py` – K-means player groups, strongest dyads and triads, configuration variety
- `teamnet/performance.py` – indicators (Eq. 8 etc.) and the adversarial regression (Eqs. 9–11) with the paper's simplex grid search (step 0.1)
- `examples/season_analysis.py` – 33 training / 5 validation matches plus sensitivity analysis
- `tests/`

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python examples/season_analysis.py
python -m pytest -q
```



## How it maps to the paper

| Paper | Code |
|---|---|
| Passing network (3.1) | `adjacency`, `windowed_networks` |
| Network metrics (3.2) | `summary`, `weighted_clustering` |
| Interaction patterns (3.3) | `cluster_players`, `strong_dyads`, `strong_triads` |
| Performance indicators (4.1) | `indicators` |
| Adversarial regression (4.2–4.3) | `AdversarialRegression` |
| Sensitivity analysis (7.1–7.2) | end of `examples/season_analysis.py` |

## Limitations

- Synthetic matches; weights and accuracies differ from the paper's Huskies analysis.
- Indicators are standardised before fitting, so the draw tolerance δ is expressed in standard-deviation units (default 0.25) rather than the raw value used in the paper.
- Density and clustering follow the paper's formulas (unnormalised weights), so their values can exceed 1.

## How to cite

If you use this code, please cite the paper:

```bibtex
@inproceedings{sikri2026team,
  author    = {Sikri, Nitin},
  title     = {A Network-Centric Framework for Evaluating Team Dynamics: From Sports to Organizational Success},
  booktitle = {Proceedings of the 5th International Conference on Big Data and Artificial Intelligence Applications (ICBDAIA'26)},
  publisher = {Springer, Lecture Notes in Networks and Systems},
  year      = {2026}
}
```

To cite this software itself, use its Zenodo DOI: [10.5281/zenodo.24380915](https://doi.org/10.5281/zenodo.24380915).

GitHub also shows a **"Cite this repository"** button generated from `CITATION.cff`.

## Licence

MIT – see [LICENSE](LICENSE).

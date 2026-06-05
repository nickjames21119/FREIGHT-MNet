# FREIGHT-MNet

**A Public Multimodal Benchmark for Cold-Start Freight OD Travel-Time Risk**

FREIGHT-MNet aligns fragmented public U.S. freight signals into a supervised
origin–destination (OD)-year graph task: given a Freight Analysis Framework (FAF)
origin `o`, destination `d`, and year `t`, predict the annual truck travel-time
risk vector `(q25, q50, q75)` (in minutes) over a typed freight graph. It pairs the
data with evaluation protocols for temporal recurrence, **Cold-OD** unseen-relation
transfer, sparse / high-risk slices, calibration diagnostics, and CVaR-style
top-risk screening, together with a controlled family of historical, tabular,
identity, graph, topology, and disruption predictors.

## Data

All inputs are public. To download everything the benchmark needs, run:

```
Code/DataCollection.ipynb
```

This notebook fetches the raw public sources used to build the benchmark:

- **FMI** (Freight Mobility Indicators) — county-pair truck travel-time quantiles (labels)
- **FAF** (Freight Analysis Framework) — commodity / mode demand flows
- **NTAD** (National Transportation Atlas Database) — FAF regions, truck topology, rail topology, intermodal terminals
- **NOAA** — weather / disruption signals
- **STB** — rail service signals

After collection, the construction notebooks produce the model-ready supervised
tables: an all-valid-FAF artifact with 112,675 OD-year rows over 16,469 FAF OD
pairs (2018–2024), and the primary **E+G** benchmark used in the experiments with
73,972 OD-year rows over 10,762 FAF OD pairs and 64 public covariates.

**Label-leakage policy:** the truck quantiles `q25 / q50 / q75` are used only as
prediction targets, never as model inputs.

> The raw and processed data are large (tens of GB) and are reproduced locally
> from the public sources above rather than redistributed in this repository.

## Environment

- Python 3.10+
- PyTorch and PyTorch Geometric (GraphSAGE / HGT models)
- LightGBM (tabular baselines)
- pandas, numpy, scikit-learn
- geopandas, shapely (NTAD geospatial layers)
- jupyter

Install dependencies with `pip install -r requirements.txt`, or your preferred
environment manager.

## Repository structure

```
Code/
  DataCollection.ipynb                    # download all public source data
  Cold_OD_Split_and_Baselines.ipynb       # build supervised tables + Cold-OD split
  FREIGHT_MNet_M0_Baselines_Improved.ipynb   # temporal historical/tabular baselines
  FREIGHT_MNet_M05_Hybrid_Baselines.ipynb     # temporal hybrid baselines
  MLP_Residual.ipynb                      # residual MLP baseline
  Strategic_Baselines_ColdOD_Comparison.ipynb # Numeric / STID / ZoneID controls
  GraphSAGE_HGT_ColdOD_Baselines.ipynb    # GraphSAGE / HGT Cold-OD predictors
  GraphSAGE_HGT_ColdOD_Baselines_v2.ipynb
  Graph_ColdOD_Ablation_v2.ipynb          # graph-construction ablation
  DCQHGT_Topology_ColdOD_v2.ipynb         # topology-aware HGT
  DCQHGT_Topology_Ablation_v3.ipynb       # topology ablation
  DCQHGT_Disruption_ColdOD.ipynb          # disruption-gated HGT
  Disruption_Gate_Stabilization.ipynb     # stabilized disruption-gate recipe
  Calibration_and_CVaR_Evaluation.ipynb   # calibration + CVaR top-risk screening
  Final_Results_Consolidation.ipynb       # consolidate all outputs into paper tables/figures
```

## Reproducing the paper

Run the notebooks in this order:

1. `DataCollection.ipynb` — download the public data.
2. `Cold_OD_Split_and_Baselines.ipynb` — build the supervised tables and the strict Cold-OD split.
3. The baseline / model notebooks below (in any order).
4. `Final_Results_Consolidation.ipynb` — regenerate the paper tables and figures.

### Notebook → paper result

| Paper table / figure | Notebook(s) |
| --- | --- |
| Table II (temporal split) | `FREIGHT_MNet_M0_Baselines_Improved`, `FREIGHT_MNet_M05_Hybrid_Baselines`, `MLP_Residual` |
| Table III (Cold-OD main) | `Strategic_Baselines_ColdOD_Comparison`, `GraphSAGE_HGT_ColdOD_Baselines(_v2)` |
| Table IV (strategic non-graph controls) | `Strategic_Baselines_ColdOD_Comparison` |
| Table V (graph ablation) | `Graph_ColdOD_Ablation_v2` |
| Table VI (topology) | `DCQHGT_Topology_ColdOD_v2`, `DCQHGT_Topology_Ablation_v3` |
| Table VII (disruption) | `DCQHGT_Disruption_ColdOD`, `Disruption_Gate_Stabilization` |
| Tables VIII–IX, Fig. 3 (calibration, CVaR screening) | `Calibration_and_CVaR_Evaluation` |

Neural models are trained with five seeds `{7, 42, 123, 2026, 535}`;
seed-ensemble predictions are reported where available.

## License and data sources

Code is released under the MIT License (add a `LICENSE` file if you adopt it).
The underlying public data remain subject to the terms of their respective
providers: FMI, FAF, NTAD, NOAA, and STB.

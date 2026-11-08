# Topological optimisation of cycling networks — T-ITS 2025

Implementation of the methodology of

> M. Campero Jurado, C. Canudas-de-Wit, G. De Nunzio,
> *A Graph-Based Approach to the Topological Optimization of Cycling Networks for the
> Improvement of Safety and Comfort of Cyclists*, IEEE T-ITS (`paper/`).

The package computes the graph metrics of 14 French cycling networks, fits the prediction model of
perceived safety and comfort, and solves the budget-constrained upgrade problem with MLCC (the proposed
heuristic), brute force, a genetic algorithm (SGA), SGA seeded with MLCC and Bayesian optimisation.
All numbers and figures are produced by the notebooks from the data in `data/`.

## Quick start

```bash
conda env create -f environment.yml        # once (~10 min)
conda activate cycling-tits
jupyter lab notebooks/                      # run the notebooks 01 → 06 in order
```

Without opening Jupyter, `python run_all.py --skip 5` executes notebooks 01–04 and 06 in place (`--list` shows the steps; notebook 05 alone takes 1–2 days).

> **Windows:** start Jupyter from the *activated* environment (or pick the kernel
> “Python (cycling-tits)” in VS Code). Running the environment's `python.exe` without activation can
> crash matplotlib, because other programs on the PATH (for example MiKTeX) ship DLLs that shadow the
> environment's own. `tools/run_in_env.ps1` runs any command with the right PATH, e.g.
> `powershell -ExecutionPolicy Bypass -File tools/run_in_env.ps1 python -m jupyter lab`.
>
> Figures use LaTeX fonts if a LaTeX distribution is installed; otherwise matplotlib's built-in
> Computer Modern is used. With MiKTeX, enable automatic package installation once:
> `initexmf --set-config-value [MPM]AutoInstall=1`.

## What to run

| Step | Notebook | What it computes (section of the paper) | Runtime |
|---|---|---|---|
| 1 | `01_network_metrics` | metrics s, c, l of the 14 cities (Sec. II-C, Table II); size, length, B_max (Table VI); maps (Figs. 3, 8) | ~10 min the first time (closeness of Paris), then cached |
| 2 | `02_model_regression_sdv` | OLS significance test (Table III), SDV enrichment, NNLS model θ̃ of Eq. (10), test-city predictions (Fig. 6) | seconds |
| 3 | `03_benchmark_example` | all methods incl. brute force on the 10-node benchmark (Sec. V, Figs. 5, 7, Table V) | ~20 min |
| 4 | `04_sioux_falls` | all methods on Sioux Falls (Appendix C, Fig. 11, Table VIII) | ~15 min |
| 5 | `05_grenoble` | all methods on Grenoble (Sec. VI, Fig. 9, Table VII) | **1–2 days** with full settings; `QUICK = True` ≈ 2.5 h test run |
| 6 | `06_fourteen_cities_mlcc` | MLCC on the 14 cities (Sec. VII, Figs. 10, 12) | ~1 h (Paris alone ~25 min) |

Notebook 2 needs the output of notebook 1 (`results/metrics_cities.csv`); notebooks 3–6 are independent.
Every expensive result is cached in `results/<preset>/*.json` and figures go to `figures/<preset>/`;
delete a cached file to recompute it. Notebooks 01–04 and 06 are stored with their outputs; 05 was not
executed in this package (it takes 1–2 days).

## Folder layout

```
T_ITS_2025/
├── README.md                 this file
├── environment.yml           conda environment (Python 3.11, pygmo, scikit-optimize, SDV, ...)
├── notebooks/                01 … 06, run in this order
├── src/cycling_tits/         the code
│   ├── data.py               loading of networks and survey data
│   ├── metrics.py            node strength s_G, closeness c_G, largest connected component l_G
│   ├── model.py              prediction models, upgrade costs, the two presets
│   ├── mlcc.py               MLCC heuristic (Sec. IV-A)
│   ├── optimizers.py         brute force, SGA, SGA-MLCC, Bayesian optimisation
│   ├── experiments.py        runs and caches the comparisons
│   └── plotting.py           figures
├── data/
│   ├── networks/             14 city networks + Benchmark + Sioux Falls (CSV: nodes, edges, DoS)
│   └── model/                survey data (Baromètre Vélo), SDV synthetic samples, Auto-ML predictions
├── results/                  cached results written by the notebooks (safe to delete)
├── figures/                  figures written by the notebooks
├── tools/                    notebook generator, data conversion script, Windows runner
└── paper/                    LaTeX sources and PDF of the paper
```

## Method

Each cycling network is a weighted graph whose edges carry a *Degree of Separation* (DoS 1–6), mapped
to the CycleRAP Safety and Comfort Score (CSCS 1, 1.55, 2.18, 2.99, 4.64, 5.23). Three graph metrics
(node strength `s_G`, closeness centrality `c_G`, largest connected component `l_G`) are computed for
14 French cities and regressed against the perceived safety and comfort of the Baromètre Vélo survey.
Only `l_G` is significant (OLS). The training set (10 cities) is enriched with Synthetic Data Vault
samples and an NNLS fit gives `y = θ0 + θ1·l_G`. The upgrade problem (Problem 2) maximises the
predicted safety/comfort under a budget `B = α·B_max` (α on a log grid 0.01…1) and is solved with
MLCC, brute force (benchmark only), pygmo's SGA, SGA seeded with MLCC, and Bayesian optimisation
(scikit-optimize).

## Two model variants (`PRESET` in notebooks 03–06)

| | `dos` (default) | `cscs` |
|---|---|---|
| edge weights | integer DoS 1…6 | CSCS 1…5.23 |
| safety model | 3.0024 + 0.2125·s_G + 2.8675·l_G | 0.41 + 0.58·l_G (Eq. 10) |
| comfort model | 3.3364 + 2.8084·l_G | 0.45 + 0.57·l_G (Eq. 10) |
| cost per metre | exp(b−4) − exp(a−4) | Cerema (150…750 €/m) |

`dos` uses the linear models calibrated in the ITSC-2024 set-up; `cscs` uses the NNLS model of
notebook 02 and Cerema costs. With `cscs` the prediction only depends on `l_G`, so the percentage of
achievable improvement is the same for safety and comfort. (The names `published` and `paper_text`
used by an earlier version are still accepted.)

## Implementation choices

* **l_G threshold weights**: thresholds k = 2…6 weighted by exp(k), components by exp(n − j).
* **Closeness**: node closeness on G with the inverted weight d_e·w(7 − DoS_e).
* **Weighted lengths**: edges merged during the network simplification keep the weighted length
  Σ(segment length × weight) of their segments; it is stored in `data/networks/*_edges.csv`
  (`wlen_dos`, `wlen_cscs`).
* **MLCC shortest paths** use the weight length × DoS; each edge on a path or in a component is upgraded
  to the highest level the remaining budget allows.
* **θ̃** of Eq. (10) is the fitted NNLS vector truncated to two decimals.
* **Stochastic methods** (SGA, SGA-MLCC, Bayesian, new SDV samples, AutoGluon) are seeded, but results
  vary with library versions and hardware. MLCC and brute force are deterministic.
* **Auto-ML**: AutoGluon is not in `environment.yml` (heavy, conflicts with pygmo); its predictions are
  stored in `data/model/automl_predictions.csv` and an optional cell in notebook 02 re-runs it.

## Data provenance

* Networks: OpenStreetMap (2024 extraction), DoS classification from the ITSC-2024 paper
  (https://github.com/leunamCampero/ITSC2024), simplified graphs. `tools/build_data_from_legacy.py`
  converted the original pickles (legacy folder `CODE/2024/2024/First_project_2024`, not shipped) to CSV;
  it does not need to be run.
* Survey: Baromètre Vélo (FUB), safety and comfort grades scaled as (grade − 1)/5.
* Synthetic samples: SDV Gaussian copula fitted on the 10 training cities (2024 run).
* Auto-ML predictions: AutoGluon `TabularPredictor` (2024 run).

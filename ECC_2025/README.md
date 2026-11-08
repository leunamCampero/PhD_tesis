# Bi-level upgrade of Grenoble's cycling network: methodology and results

Code and data implementing the methodology of

> M. Campero Jurado et al., *A bi-level optimization framework to upgrade cycling
> infrastructure by degree of separation (DoS)*, ECC 2025 (`paper/ECC_2025_final_submission.pdf`,
> LaTeX source `paper/root.tex`).

The framework:
1. **Lower level:** predicts cyclist flows in Grenoble with an all-or-nothing (AON)
   assignment on a **cyclist distance** (length × slope factor × **traffic factor**^ρ + turns);
2. **Upper level:** upgrades bike lanes to a higher DoS within a budget to minimise the
   cyclists' **exposure to traffic** J_E. Two algorithms are compared: **GAB** (genetic
   algorithm with efficient budget spending) and **MLCC** (maximise the largest connected
   DoS-6 component).

This package implements the equations and algorithms as written in the paper. It is a
clean implementation, not a bit-for-bit re-run of the 2024 notebooks, so the numbers
differ from the printed ones, but the conclusion is the same: **GAB reduces cyclist
exposure much more than MLCC for the same budget.**

## Quick start

From the activated environment, run everything with **`python run_all.py`** (`--list` shows the steps, `--from N` resumes at step N, `--only N` runs one step), or run the steps one by one, in this order:

```bash
conda env create -f environment.yml          # once, ~5 min
conda activate cycling-ecc

python 01_cyclist_distance_and_traffic_factor.py   # model tables, demand, worked example   (~10 s)
python 02_flows_original_network.py                # flows + J_E of today's network         (~10 s)
python 03_run_mlcc.py                              # MLCC, budget = 50 % of B_max           (~20 s)
python 04_run_gab.py                               # GAB (genetic algorithm)                (~40 min)
python 05_compare_and_figures.py                   # Table III, Fig. 2, Fig. 3              (~20 s)
```

Options: `--beta 0.5` (budget as a fraction of B_max) in every script;
`04_run_gab.py --pop 20 --generations 20 --seed 0`, or `--quick` for a 3-minute test.
The GA is seeded, so the same settings always give the same solution.
`--demand historical_unit` uses the newer OD file instead of the EMD 2020 demand (see *Demand*).

> **Windows:** start the scripts from the *activated* environment. If matplotlib
> crashes with a DLL error, another program on the PATH (e.g. MiKTeX) is shadowing the
> environment's libraries; `conda activate cycling-ecc` fixes it.

## Folder layout

```
ECC_2025/
├── README.md, environment.yml
├── 01_ … 05_*.py             the five steps (run in order)
├── src/ecc_cycling/           the code, every function commented
│   ├── config.py              ALL parameters of the paper
│   ├── model.py               traffic factor, slope factor, cyclist distance, AON flows, J_E
│   ├── costs.py               Cerema costs, B_max, km per DoS
│   ├── mlcc.py                Algorithm 2 (MLCC)
│   ├── gab.py                 pygmo SGA + Algorithm 1 (GAB)
│   └── plotting.py            Fig. 1, 2, 3
├── data/
│   ├── raw/                   G_pygmo.pkl (network with DoS), grenoble_historical_od_2017_2022.pkl
│   ├── network/               nodes / edges CSV (DoS, length, slope), IRIS polygons (GeoJSON)
│   └── demand/                70×70 IRIS OD (EMD 2020), IRIS names, optional newer OD
├── results/                   written by the scripts (CSV / JSON); solutions/ = DoS of every arc
├── figures/                   written by the scripts
├── paper/                     root.tex, bib, class, images, final PDF
├── tools/build_data.py        how data/ was built from the 2024 pickles (no need to run it)
└── _archive_old_files/        the previous contents of this folder (old notebooks, pickles,
                               DoS_Category/) + recovered_legacy_code/ (the 2024 GA script)
```

## Methodology

### Network
`data/raw/G_pygmo.pkl`: the cycling network of the city of Grenoble, 3,452 nodes,
4,993 arcs, 308 km, each arc with its length, slope and DoS (Table I of the paper).

| DoS | Description | AADT (veh/day) | km today |
|---|---|---|---|
| 6 | Dedicated bike lanes, fully separated | 14,400 | 31.0 |
| 5 | Shared with pedestrians, separated from cars | 16,000 | 43.9 |
| 4 | Mixed traffic, 30 km/h, no bike lane | 960 | 147.0 |
| 3 | Shared roads with cycling pictograms | 3,120 | 36.3 |
| 2 | Painted bike lanes | 6,912 | 43.8 |
| 1 | Unmarked shared car-bike lanes | 8,448 | 6.0 |

### Lower level: cyclist flows

| Element | Formula / value | Code |
|---|---|---|
| traffic factor f_a (Eq. 2) | ψ·exp(AADT/1000) for DoS ≤ 4, ψ·2.03 for DoS 5, ψ for DoS 6; ψ = 0.84 | `model.traffic_factor` |
| slope factor s_a (Table IV) | 1 / 1.371 / 2.203 / 4.239 for slopes < 2 / < 4 / < 6 / ≥ 6 % uphill; reciprocal downhill | `model.directional_slope_factor` |
| cyclist distance b (Eq. 1) | Σ d_a·s_a·f_a^ρ + κ·Σ turns; ρ = 0.04, κ = 0.14 | `Network.cyclist_distance` |
| flows x_a (Eqs. 5-6) | AON: all demand q_uv on the b-shortest path, summed over all node pairs | `Network.flows` |

The arc part d_a·s_a·f_a^ρ is the Dijkstra weight. The turn term is added when the
cyclist distance of a whole path is reported (script 01 shows a worked example).

### Demand
* **Source:** EMD 2020 IRIS × IRIS daily trip matrix (all modes), restricted to the
  **70 IRIS of the city of Grenoble** (`data/demand/od_iris_grenoble.csv`).
* **× 0.5:** the matrix is symmetric, i.e. every trip is stored once as (r, s) and once as
  (s, r). Halving removes this double count: 887,975 → **443,988 trips/day**.
  This factor is **not** the bike share.
* **× 0.05:** bike modal share → **22,199 bike trips/day**.
* Each IRIS pair's demand is spread over the network nodes of the two IRIS:
  q_uv = 0.025 · OD[r,s] · a_u · a_v, where a_u is the node's share of its IRIS
  (`zone_weight` in `grenoble_nodes.csv`). That gives 11.7 M OD node pairs, all routed.
* *Optional:* `grenoble_historical_od_2017_2022.pkl` (68 Grenoble IRIS → 248 facilities) is a
  **boolean** matrix: each value counts existing IRIS → facility pairs, not trips, so
  × 0.5 does not convert it into bike trips. It is snapped to this network in
  `data/demand/historical_unit_od.csv` and can be used with `--demand historical_unit`.

### Upper level: exposure and budget
* Exposure of an arc: E_a = d̂_a · f̂_a (Eq. 9); total exposure **J_E = Σ_a d̂_a · f̂_a · x̂_a**,
  with min-max normalised length, traffic factor and flow (`Network.evaluate`).
  Normalised traffic factor f̂: DoS 1 = 1, DoS 2 = 0.215, DoS 3 = 0.005, DoS 4-6 ≈ 0.
* Upgrade costs (Cerema, €/m), e.g. DoS 4→6: 450, DoS 1→6: 750 (`config.COST`).
* B_max (whole network to DoS 6) = **€126.08 M**; budget B = ½ B_max = **€63.04 M**.

### GAB (Algorithm 1, `gab.py`)
* chromosome: the DoS of every arc, between its current DoS and 6
* initial population: today's network with each arc raised by one level with probability ½
* fitness: over-budget configurations are discarded; otherwise the remaining budget is
  spent with Algorithm 1 (lowest-DoS arcs first, raised to DoS 6), then J_E is computed
  with a full traffic assignment
* search: pygmo standard genetic algorithm (`sga`): tournament selection, exponential
  crossover (0.9), uniform integer mutation; 20 individuals × 20 generations, seed 0

### MLCC (Algorithm 2, `mlcc.py`)
Grow one connected DoS-6 network: start from the largest DoS-6 component, then attach the
other DoS-6 components and then the components of DoS 5, 4, 3, 2, 1 (largest first). Each
attachment follows the shortest DoS-path (arc weight b_a·(7−w_a)), and the path and
component are upgraded to DoS 6, until the budget is spent. MLCC does not use the flows.

## Results (budget 50 %, `results/05_table3_b0.5.csv`)

| Solution | J_E | exposure reduction | budget used | DoS6 | DoS5 | DoS4 | DoS3 | DoS2 | DoS1 (km) |
|---|---|---|---|---|---|---|---|---|---|
| Original | 0.515 | – | – | 31.0 | 43.9 | 147.0 | 36.3 | 43.8 | 6.0 |
| **GAB** | **0.002** | **99.6 %** | 50.0 % | 127.3 | 91.6 | 89.2 | 0.0 | 0.0 | 0.0 |
| MLCC | 0.350 | 32.1 % | 50.0 % | 189.1 | 0.2 | 38.7 | 33.9 | 40.6 | 5.8 |

Figures: `figures/fig1_dos_original.png` (Fig. 1), `figures/fig2_exposure_b0.5.png` (Fig. 2),
`figures/fig3_maps_b0.5.png` (Fig. 3: flows and DoS before/after), `figures/gab_convergence_b0.5.png`.

**Interpretation (same conclusion as the paper):**
* **GAB is the better method to reduce cyclist exposure.** It uses the flows, so it spends
  the budget on the arcs that cyclists actually ride and that are most exposed. It removes
  every DoS 1-3 arc and spreads the money over DoS 5 and DoS 6, the cheaper DoS 5 standing in for 6 wherever that suffices.
* **MLCC builds more DoS-6 km (189 vs 127 km)**, a large connected safe network that
  improves perceived safety and comfort. But it ignores flows: 46 km of DoS 1-2 remain on
  busy routes, so its exposure reduction is smaller.
* The two methods are complementary: GAB for exposure, MLCC for network connectivity.

Numbers differ from the printed paper (0.423 / 0.177 / 0.207) because this package follows
the paper's equations: all OD pairs routed, downhill slope factor, Algorithm 1 as written.
The 2024 notebooks instead used a random sample of 2000 OD pairs and code variants. The ranking
GAB < MLCC < Original is the same.

## Rebuilding `data/` (optional)

`python tools/build_data.py` regenerates `data/network` and `data/demand` from the 2024
pickles in `Second_project_2024/` (paths at the top of the script, ~2 min, ~6 GB RAM).

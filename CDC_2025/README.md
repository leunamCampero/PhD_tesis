# Utilitarian vs sufficientarian upgrades of Grenoble's cycling network

Code and data implementing the methodology of

> M. Campero Jurado et al., *Optimal bike network design in line with principles of justice*,
> CDC 2025 (`paper/CDC_2025__Optimal_bike_network_design__Manuel_et_al.pdf`, source `paper/root.tex`).

A bi-level problem: the **lower level** routes cyclists (all-or-nothing on a cyclist distance
that includes slope and car traffic); the **upper level** upgrades the degree of separation
(DoS) of bike lanes within a budget, following one of two principles of justice:

* **utilitarian** (Problem 1): minimise the total exposure to traffic J_E;
* **sufficientarian** (Problem 2): make as many cyclists as possible reach a sufficient safety
  level: minimise the safety insufficiency J_I (+ η·J_E).

Three algorithms are compared: **MLCC** (connectivity), **GA-U** and **GA-S**.

This package implements the methodology with the same network and demand as `ECC_2025`,
and the 70 IRIS of `iris.gpkg` as regions. The numbers are not those of the paper, but the
conclusion is the same: **GA-S reduces safety insufficiency more than the other methods, at
almost no cost in overall exposure** (see *Results*).

## Quick start

From the activated environment, run everything with **`python run_all.py`** (`--list` shows the steps, `--from N` resumes at step N, `--only N` runs one step), or run the steps one by one, in this order:

```bash
conda activate cycling-ecc            # same environment as ECC_2025 (environment.yml)

python 01_network_and_regions.py      # network, IRIS, demand, budgets (Fig. 1)          ~5 s
python 02_original_network.py         # today: J_E, s_G, thresholds, J_I                 ~5 s
python 03_budget_sensitivity.py       # 50 GA-S runs, budget sensitivity (Fig. 2)        ~15 min
python 04_run_algorithms.py           # MLCC, GA-U, GA-S (5 lambda) at B = 5.7 % B_max   ~15 min
python 05_figures_and_table.py        # Figs. 3-7 and results/05_summary.csv             ~30 s
```

Steps 03 and 04 run several optimisations in parallel (all CPU cores but one). Options:
`03 --pop --generations`, `04 --beta --pop --generations`. GA runs are seeded, so results are
reproducible.

## Folder layout

```
CDC_2025/
├── README.md, environment.yml
├── 01_ … 05_*.py              the five steps (run in order)
├── src/cdc_cycling/            the code, every function commented
│   ├── config.py               ALL parameters (λ, η, ε, ρ, κ, ψ, costs, budgets, nodes per IRIS)
│   ├── model.py                lower level: cyclist distance, AON flows, path & total exposure
│   ├── justice.py              sufficientarian measures: s_p, s_G, s_thr, I_p, I_r, J_I
│   ├── ga.py                   GA-U / GA-S (Algorithm 1)
│   ├── mlcc.py, costs.py       MLCC (Algorithm 2) and Cerema costs (same as ECC_2025)
│   ├── runner.py               one optimisation per call (for parallel runs)
│   └── plotting.py, utils.py
├── data/
│   ├── network/                nodes (with IRIS zone_id) and edges, same network as ECC_2025
│   ├── regions/                the 70 IRIS of Grenoble (GeoJSON, from iris.gpkg)
│   ├── demand/                 EMD 2020 IRIS × IRIS trips
│   └── raw/iris_grenoble.gpkg  the 70 Grenoble IRIS extracted from the national iris.gpkg
├── results/, figures/          written by the scripts
├── paper/                      root.tex, bib, images, PDF
├── tools/build_data.py         how data/ was built (from ECC_2025/data and iris.gpkg)
└── _archive_old_files/         previous contents: notebooks (incl. pygmo_test_clean_fair_CDC),
                                national iris.gpkg (160 MB), iris.geojson
```

## Methodology

**Network and regions.** `G_pygmo.pkl` (via `ECC_2025`): 3,452 nodes, 4,993 arcs, 308 km, DoS 1–6.
Every node is assigned to one of the 70 IRIS of `iris.gpkg` by point in polygon (all nodes fall
inside the city; 104 nodes changed IRIS compared with the older polygons used in 2024).

**Demand.** EMD 2020 IRIS × IRIS trips × 0.5 (the matrix stores every trip in both directions)
× 0.05 (bike share) = **22,199 bike trips/day**, spread over the nodes of each IRIS.

**Lower level** (`model.py`). Cyclist distance b = Σ d_a·s_a·f_a^ρ + κ·Σ∡ (Eq. 1), traffic factor
f_a (Eq. 2), slope factor uphill / reciprocal downhill (Table III); all-or-nothing assignment on
the b-shortest paths (Eqs. 3–4). The same pass gives the arc flows and the exposure of every
OD path, E_p = Σ_{a∈p} d̂_a f̂_a (Eq. 5).

**Upper level, utilitarian** (Problem 1). J_E = Σ_a d̂_a f̂_a x̂_a.

**Upper level, sufficientarian** (Problem 2, `justice.py`).

| Quantity | Definition | Value (today) |
|---|---|---|
| path safety | s_p = 1 / (E_p + ε), ε = 0.01 (so s_p ≤ 100) | – |
| average safety | s_G = mean of s_p over the OD paths of today's network | 36.8 |
| threshold | s_thr = λ·s_G, λ ∈ {10, 20, 30, 40, 50} %; fixed on today's network | 3.7 … 18.4 |
| path insufficiency | I_p = max(0, (s_thr − s_p)/s_thr) | – |
| regional | I_r = demand-weighted mean of I_p over the trips leaving IRIS r | map in Fig. 3 |
| network | J_I = Σ_r n_r I_r / Σ_r n_r, n_r = cyclists of IRIS r | 0.015 … 0.219 |
| GA-S objective | J_I + η·J_E, η = 0.05 | – |

**Budget.** Cerema costs (matrix Π), B_max = €126.08 M; budgets B_k log-spaced from 1 % to
50 % of B_max (0.010, 0.015, 0.024, 0.037, 0.057, 0.088, 0.136, 0.210, 0.324, 0.500).

**Algorithms.** MLCC grows one connected DoS-6 network (Algorithm 2). GA-U and GA-S use
pygmo's standard GA with the budget spent by Algorithm 1 (see the encoding change below).

## Results (budget B_5 = 5.7 % of B_max = €7.2 M, `results/05_summary.csv`)

Insufficiency J_I and % of cyclists below the threshold; GA-S is the run made for that λ:

| λ | Original | MLCC | GA-U | **GA-S** | GA-S vs GA-U |
|---|---|---|---|---|---|
| 10 % | 0.0148 / 5.1 % | 0.0141 / 5.0 % | 0.0009 / 0.21 % | **0.0006 / 0.13 %** | J_I −40 % |
| 20 % | 0.0585 / 17.2 % | 0.0530 / 14.6 % | 0.0037 / 1.57 % | **0.0008 / 1.22 %** | J_I −79 % |
| 30 % | 0.1151 / 28.6 % | 0.1019 / 24.7 % | 0.0118 / 4.26 % | **0.0075 / 4.03 %** | J_I −36 % |
| 40 % | 0.1700 / 38.1 % | 0.1512 / 34.6 % | 0.0253 / 8.99 % | **0.0207 / 8.66 %** | J_I −18 % |
| 50 % | 0.2193 / 44.8 % | 0.1975 / 41.6 % | 0.0424 / 13.6 % | **0.0393** / 14.1 % | J_I −7 % |

Utilitarian exposure J_E: Original 0.527, MLCC 0.488, **GA-U 0.115**, GA-S 0.118–0.143
(+3 % to +24 % vs GA-U; +4 % at λ = 40 %). Sufficientarian objective J_I + η·J_E: GA-S is the
lowest of all methods at every λ (e.g. λ = 40 %: Original 0.196, MLCC 0.176, GA-U 0.031, GA-S 0.027).

Figures: `fig1_network_and_iris.png`, `fig2_marginal_improvements.png`,
`fig3_dos_and_insufficiency.png` (λ = 40 %), `fig4_safety_histograms.png` (λ = 40 %),
`fig5_threshold_distribution.png`, `fig6_sufficientarian_objective.png`, `fig7_exposure.png`.

**Interpretation (same conclusion as the paper):**
* **GA-S always gives the lowest safety insufficiency J_I** (7–79 % lower than GA-U, 82–96 %
  lower than today). For λ ≤ 40 % it also leaves fewer cyclists below the threshold. At λ = 50 %
  the share below is 0.5 points higher than GA-U while J_I is lower: GA-S lifts the *worst*
  paths further rather than pushing marginal ones just above the line.
* **GA-U gives the lowest overall exposure J_E**, but GA-S is close (+4 % at λ = 40 %):
  *sufficientarian planning improves the situation of the least safe cyclists without
  significantly affecting overall (utilitarian) safety*, which is the paper's conclusion.
* **MLCC** builds connected DoS-6 corridors but ignores where cyclists ride; with this small
  budget it improves J_E by ~7 % and J_I by ~5–11 % only.
* Fig. 3: the insufficiency of today's network is concentrated in the south and north-west
  IRIS; GA-S removes almost all of it.

**Budget sensitivity (Fig. 2).** J_I + η·J_E falls quickly with the first millions of euros and
is almost flat above ~14 % of B_max (diminishing returns), for every λ.

## Suggested changes to the methodology (implemented here)

1. **GA encoding.** With the 2024 encoding (DoS of each arc, then Algorithm 1 spends the rest of
   the budget on the lowest-DoS arcs), the fill decides almost everything at small budgets:
   GA-U and GA-S returned *the same network* in a test. Here the chromosome is a priority per
   arc, decoded by Algorithm 1 in priority order instead of DoS order. Every individual is
   feasible and spends the whole budget, and Algorithm 1 itself is one of the individuals.
2. **Objective-aware seeding.** GA-U starts from "exposure × flow saved per €"; GA-S from the
   same score computed with the flow of the trips *below the threshold* only (one extra traffic
   assignment), plus the utilitarian ordering, so GA-S is never worse than GA-U on Problem 2.
3. **η = 0.05, not 0.5.** The paper text says η = 0.5, but its Fig. 6 values (e.g. Original,
   λ = 10 %: 0.014 + 0.05 × 0.42 ≈ 0.035) and the 2024 code use 0.05. With 0.5 the exposure
   term is ~10× larger than J_I and GA-S becomes a copy of GA-U. Please correct the text.
4. **Budget selection.** The budget formula in the paper (B_min + (B_max/2 − B_min)·log k / log 10)
   does not give the printed values; the printed values are a log-spaced sequence (used here).
   Also, "largest marginal improvement per €" always picks the smallest budget when J(B) is
   concave, as it is here; the paper's peak at B_5 is probably GA noise. This package therefore
   reports the criterion (`results/03_budget_star.json`) but compares the algorithms at the
   paper's budget B_5. Better options: a knee (elbow) criterion, or the smallest budget that
   brings J_I below a target.
5. **Representative trip ends.** Trips start/end at the 10 heaviest nodes of each IRIS
   (697 nodes, 0.49 M OD pairs instead of 11.7 M), 5× faster with almost the same indicators
   (J_E 0.527 vs 0.521, J_I(40 %) 0.170 vs 0.165). Set `NODES_PER_IRIS = None` in `config.py`
   for all nodes.
6. **Thresholds from all OD paths.** s_G and s_thr are computed once on today's network from
   every OD path with trips (the 2024 code used 1000 random node pairs), so all solutions are
   judged against the same, non-random "sufficient" level.
7. **Same budget for all algorithms**, so that the comparison is fair.

## Rebuilding `data/` (optional)

`python tools/build_data.py` (needs `../ECC_2025/data` and `_archive_old_files/iris.gpkg`).

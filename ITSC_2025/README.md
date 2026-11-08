# Safety-oriented bike-lane upgrades and car travel time — ITSC 2025

Code and data implementing the methodology of

> M. Campero Jurado, M. Salazar, G. De Nunzio, C. Canudas-de-Wit, *Safety-Oriented Design of
> Cycling Infrastructure Minimizing Impact on Vehicular Traffic*, ITSC 2025
> (`paper/root.tex`, `paper/ITSC_2025.pdf`).

A bi-level problem on a two-layer (bike + car) network of Grenoble:

* **lower level** – cyclists choose the shortest path for a *cyclist distance* that includes
  slope and adjacent car traffic (all-or-nothing assignment);
* **upper level** – within a budget B = α·B_max, raise the degree of separation (DoS) of bike
  arcs to minimise ε·(cyclist exposure) + (1−ε)·(car total travel time). A bike arc upgraded to a
  separated lane (DoS ≥ 5) **halves the capacity of the car road it runs along**; car flows are the
  pre-upgrade user equilibrium (UE) φ\* (Problem 1), kept fixed during the optimisation;
* **afterwards** – a post-upgrade UE lets drivers re-route.

Same bicycle network, demand and costs as `ECC_2025` / `CDC_2025`. The numbers differ from the
paper, but the conclusion is the same: **cyclist exposure can be cut by ~90 % with a car
travel-time increase of only 1.5–2 % once drivers re-route** (see *Results*).

## Quick start

From the activated environment, run everything with **`python run_all.py`** (`--list` shows the steps, `--from N` resumes at step N, `--only N` runs one step), or run the steps one by one, in this order:

```bash
conda activate cycling-ecc        # same environment as ECC_2025 / CDC_2025 (environment.yml)

python 01_networks_and_demand.py  # both layers, coupling, budgets (Fig. 1)              ~10 s
python 02_car_user_equilibrium.py # demand calibration + pre-upgrade UE (Problem 1, Fig. 2)  ~3 min
python 03_pareto_trade_off.py     # GA: 3 budgets x 11 eps (Problem 2, Figs. 3-4)       ~60 min
python 04_post_upgrade_ue.py      # post-upgrade UE for every solution (Fig. 5)          ~10 min
python 05_figures.py              # Figs. 3-6 and results/05_summary.csv                 ~1 min
python tests/run_tests.py         # unit tests (trees, UE vs analytic, decoding, coupling) ~5 min
```

On Windows put the environment on the PATH first (`conda activate`), otherwise the MiKTeX DLLs
crash matplotlib. Steps 03-04 use all cores but one. GA runs are seeded (reproducible).

## Folder layout

```
ITSC_2025/
├── README.md, environment.yml
├── 01_ … 05_*.py                the five steps (run in order)
├── src/itsc_cycling/            the code, every function commented
│   ├── config.py                ALL parameters
│   ├── trees.py                 fast shortest-path-tree loading / path sums (both layers)
│   ├── bike.py                  bike layer: cyclist distance, AON flows, exposure J_E
│   ├── cars.py                  car layer: BPR, AON, bi-conjugate Frank-Wolfe UE
│   ├── coupling.py              which car roads lose capacity
│   ├── problem.py               objective of Problem 2, normalisation, GA seeds
│   ├── ga.py, costs.py          GA (priority encoding + Algorithm 1), Cerema costs
│   └── runner.py, plotting.py, utils.py
├── tests/run_tests.py
├── data/network/                bike_{nodes,edges}.csv (+ car_edge), car_{nodes,edges}.csv
├── data/demand, data/regions    EMD 2020 IRIS x IRIS trips, the 70 IRIS
├── tools/build_data.py          how data/ was built (CDC_2025 data + Grenoble_H_simplified.pkl)
├── results/, figures/           written by the scripts
├── paper/                       root.tex, bib, images, PDF
└── _archive_old_files/          the 2024 notebooks (car_ue_network, pareto_front, ...)
```

## Methodology

| Element | Implementation | Code |
|---|---|---|
| Bike layer | ECC/CDC graph: 3,452 nodes, 4,993 arcs, 308 km, DoS 1–6, slope | `bike.py` |
| Cyclist distance (Eq. 1–2) | d·s·f^ρ, f = ψ·exp(AADT/1000) (DoS ≤ 4), ψ·2.03 (DoS 5), ψ (DoS 6); ρ = 0.04, ψ = 0.84 | `bike.arc_costs` |
| Bike demand | EMD 2020 × 0.5 × 5 % = 22,199 trips/day, 10 trip ends per IRIS | `bike._build_demand` |
| Exposure (Eq. 4) | J_E = Σ d̂·f̂·φ̂ (min-max normalised) | `bike.evaluate` |
| Car layer | OSM drive network: 4,720 nodes, 6,175 roads (both directions), 377 km; capacity per lane and speed by road class, lanes and maxspeed from OSM | `cars.py`, `config.ROAD_CLASSES` |
| Car demand | morning peak hour = EMD × 0.5 × car share 25.3 % × 10 % × **m**, 4 trip ends per IRIS (k-means) | `cars._build_demand` |
| BPR (Eq. 6) | t0·(1 + 0.15 (φ/θ)^4) | `cars.time` |
| UE (Problem 1) | Beckmann program, bi-conjugate Frank–Wolfe, exact line search, relative gap 10⁻⁴ | `cars.solve_ue` |
| Coupling (Eq. 9) | bike arc DoS ≤ 4 → DoS ≥ 5 along a car road ⇒ that road's capacity × ½ (both directions) | `coupling.py` |
| Problem 2 | ε·Ê + (1−ε)·T̂, T = Σ φ\*·t(φ\*; θ̃), budget ≤ α·B_max | `problem.py` |
| GA (Algorithm 1) | pygmo sga, 16 individuals × 12 generations, priority encoding decoded by Algorithm 1 | `ga.py` |
| ε\* | closest point to the utopia point of the min-max normalised front | `03_pareto_trade_off.py` |
| Post-upgrade UE | same solver, warm-started at φ\*, reduced capacities | `04_post_upgrade_ue.py` |

**Calibration (step 02).** The EMD matrix only contains trips made inside the city; roads also
carry exchange and through traffic. With m = 1 the modelled flows next to DoS 1–3 bike lanes are
only 29 % of the GTLVille traffic of Table I. m = **3.49** makes them equal (length-weighted
two-way peak flow = 10 % × AADT): 37,369 cars in the peak hour, UE TTT = 2,850 veh·h, +7 % delay
over free flow, 3.5 % of the arcs above v/c = 0.8.

## Results (`results/03_pareto.csv`, `04_post_upgrade.csv`, `05_summary.csv`)

B_max = €126.1 M; α = 1/3, 2/5, 1/2 → €42.0 M, €50.4 M, €63.0 M. Today: J_E = 0.526, TTT = 2,850 veh·h.

| α | ε\* | exposure J_E | car TTT, routes fixed | car TTT, after re-routing (UE) | roads narrowed |
|---|---|---|---|---|---|
| 1/3 | 0.6 | 0.054 (**−89.8 %**) | +3.0 % | **+1.5 %** | 641 (45 km) |
| 2/5 | 0.6 | 0.053 (**−89.9 %**) | +3.0 % | **+1.5 %** | 731 (52 km) |
| 1/2 | 0.7 | 0.045 (**−91.4 %**) | +4.3 % | **+2.0 %** | 945 (67 km) |

The whole front (α = 1/3, car TTT after re-routing):

| ε | 0 | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|
| J_E / J_E⁰ | 0.64 | 0.26 | 0.24 | 0.12 | 0.10 | 0.06 | 0.01 | 0.01 |
| car TTT fixed / UE | 0 / 0 % | +0.4 / +0.2 % | +0.8 / +0.4 % | +2.4 / +1.2 % | +3.0 / +1.5 % | +8.6 / +2.7 % | +25.5 / +6.4 % | +33.9 / +7.7 % |

Interpretation (same direction as the paper):
* **Large safety gains for small car costs.** Only a few car roads matter; avoiding them
  (ε ≈ 0.1–0.6) keeps 75–90 % of the exposure reduction at +0.2 to +1.5 % car time.
* **Re-routing halves the car impact** (Fig. 5): e.g. +3.0 % → +1.5 %. Even the pure-safety designs
  (ε = 1) stay below +10.3 % after re-routing (paper: "maximum 11.1 %"). Mean car trip time rises
  from 4.58 to 4.65 min at ε\*; about 10–12 % of the used roads change flow by more than 20 %
  (Fig. 6): traffic moves from the narrowed corridors to parallel arterials.
* **Higher budgets** give slightly lower exposure but more narrowed roads and a higher car cost;
  the three fronts almost coincide because with ≥ 1/3 of B_max the exposure is nearly saturated.
* ε = 0 gives the best *car-neutral* design (−36 % exposure using only arcs off busy roads).

## Changes to the methodology (implemented here) — please review

1. **Car network.** The 2024 car graph (`Grenoble_car_capacity.pkl`) re-used the bicycle topology and
   gave a **random** highway type (hence capacity and speed) to every road not found in OSM. Here the
   car layer is the real OSM drive network (`Grenoble_H_simplified.pkl`), and bike arcs are matched to
   the road they run along by geometry (8 m tolerance; 2,496 on-road arcs, 135 km).
2. **Demand calibration** against the GTLVille AADT (m = 3.49, see above) and a **peak hour** instead
   of the daily/24 average of the 2024 code; with daily/24 the network is uncongested and the car term
   only moves by 0.15 % (the 2024 results), which made the trade-off visible only after min-max
   rescaling.
3. **Objective.** The 2024 code minimised the *Beckmann integral* of the fixed flows; the paper (and this
   code) uses the car **total travel time** Σ φ t(φ).
4. **Normalisation** (`problem.py`): E and T are scaled between their ideal and nadir values
   (from the ε = 1 and ε = 0 runs). Dividing by today's values makes (1−ε)·T almost constant (T moves by
   a few %, E by 90 %), so ε would only matter near 0.
5. **Choice of ε\*.** Eq. (15) of the paper, √((ε·E)² + ((1−ε)·T)²), is always minimised by ε = 0
   (both weighted terms vanish there), which is what the code finds (`eps_star_paper_eq15` in
   `results/03_pareto.json`). We use the closest point to the utopia point of the *unweighted*
   min-max-normalised front. Please correct Eq. (15).
6. **GA encoding** as in CDC_2025 (priority per arc decoded by Algorithm 1), plus a "skip" range so that
   the GA may leave budget unspent instead of being forced to narrow a busy road (B is a "≤" constraint);
   seeds: Algorithm 1, exposure saved per euro, the car-free variant and a car-aware blend.
7. **Faster UE.** Bi-conjugate Frank–Wolfe with exact line search and warm starts, and an all-or-nothing
   loader that processes shortest-path trees level by level (`trees.py`): one UE takes 30–70 s from
   scratch and 1–2 min after an upgrade, instead of external solver runs.
8. **Turns** (κ term of Eq. 1) are reported but not used in routing, as in ECC/CDC_2025.

## Rebuilding `data/` (optional)

`python tools/build_data.py` (needs `../CDC_2025/data` and `../../Grenoble_H_simplified.pkl`).

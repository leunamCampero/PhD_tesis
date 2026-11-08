# Justice-informed cycling infrastructure upgrade — T-ITS Special Issue

Code and data implementing the methodology of

> M. Campero Jurado, C. Canudas-de-Wit, G. De Nunzio, M. Salazar, *Justice-informed Cycling
> Infrastructure Upgrade via Multi-modal Bi-level Optimization* (`paper/main.tex`).

It extends `ITSC_2025` (bike lanes vs. car travel time, two-layer network) and `CDC_2025`
(sufficientarian GA) with

* a **mode-choice model**: the bike share depends on the mean cyclist distance, so a better bike
  network moves trips from car to bike;
* a **capacity rule** that depends on the number of lanes and their width;
* a **surrogate of the car user equilibrium (UE)** trained on exact UE samples, followed by a
  feasibility projection, so that the GA can evaluate thousands of designs;
* **utilitarian** (Problem 2: exposure vs. total travel time) and **sufficientarian** (Problem 3:
  path insufficiency vs. trip insufficiency of cyclists and drivers) formulations.

Same bicycle network, demand and costs as `ECC_2025` / `CDC_2025`; same car network, calibration and
UE solver as `ITSC_2025`. The numbers differ from the paper; the conclusions point the same way
(see *Results*).

## Quick start

From the activated environment, run everything with **`python run_all.py`** (`--list` shows the steps, `--from N` resumes at step N, `--only N` runs one step), or run the steps one by one, in this order:

```bash
conda activate cycling-ecc            # same environment as ECC/CDC/ITSC (environment.yml)

python 01_networks_and_mode_choice.py # layers, capacity rule, mode-choice calibration (Fig. 1-2)   ~20 s
python 02_ue_and_samples.py           # demand calibration, today's UE, 1000 exact UE samples      ~4 h
python 03_train_surrogate.py          # surrogate (PCA + ridge / MLP) + projection, test metrics   ~15 min
python 03b_train_gnn_surrogate.py     # GNN surrogate vs ridge vs ensemble, picks the model  ~2 h
python 04_run_optimisation.py         # GA: 11 utilitarian + 55 sufficientarian runs               ~15 h
python 05_validate_exact_ue.py        # exact UE for the designs, sensitivity analysis             ~20 min
python 05b_exact_all_designs.py       # exact UE of all 66 designs + user shares (figures)   ~20 min
python 06_figures.py                  # Figs. 4-11 and results/06_summary.csv                      ~1 min
python tests/run_tests.py             # unit tests                                                 ~5 min
```

Steps 02, 04 and 05 use all cores but one (8-core PC). Step 02 can be stopped and restarted: it
continues from the samples already saved. `--n`, `--workers` (step 02) and `--pop`,
`--generations` (step 04) change the sizes. Everything is seeded.

## Folder layout

```
T_ITS_Special_Issue/
├── README.md, environment.yml
├── 01_ … 06_*.py                the six steps (run in order)
├── src/si_cycling/
│   ├── config.py                ALL parameters (with the changes w.r.t. the paper marked CHANGE)
│   ├── trees.py                 fast shortest-path-tree loading / path sums
│   ├── bike.py                  bike layer: cyclist distance, AON flows, exposure J_E
│   ├── cars.py                  car layer: BPR, bi-conjugate Frank-Wolfe UE
│   ├── coupling.py              capacity rule theta(w) (Eq. 17)
│   ├── modeshift.py             binary logit mode choice, bike / car demand scales
│   ├── surrogate.py             UE surrogate + projection on {B phi = d, phi >= 0}
│   ├── justice.py               path safety S_p, insufficiencies, regional aggregation
│   ├── problems.py              evaluation chain of a design, Problems 2 and 3, GA seeds
│   ├── ga.py, costs.py, mlcc.py GA (priority encoding + Algorithm 1), Cerema costs, MLCC seed
│   └── runner.py, plotting.py, utils.py
├── tests/run_tests.py
├── data/                        copy of ITSC_2025/data (built by ITSC_2025/tools/build_data.py)
├── results/, figures/           written by the scripts
├── paper/                       main.tex, bib, images, PDFs
└── _archive_old_files/          the 2025 notebooks
```

## Methodology

| Element | Implementation | Code |
|---|---|---|
| Bike layer, cyclist distance, exposure | as ECC/CDC/ITSC: d·s·f^ρ, AON assignment, J_E = Σ d̂·f̂·φ̂ | `bike.py` |
| Car layer, UE | OSM drive network (6,175 roads), BPR, bi-conjugate Frank–Wolfe, peak-hour demand calibrated to GTLVille (m = 3.49, 37,369 cars) | `cars.py` |
| Mode choice (Eq. 18–20) | P_b = 1 / (1 + exp(−(α + β·D̄))), β = −0.886 /km, D̄ = demand-weighted mean cyclist distance; α calibrated so P_b = 16.49 % today | `modeshift.py` |
| Demand update | bike demand × P_b / P_b⁰; the extra cyclists are removed from the car demand | `modeshift.ModeChoice.scales` |
| Capacity rule (Eq. 17) | bike arc upgraded from DoS < 5 to DoS ≥ 5 along a car road with lane width ≤ 4.5 m ⇒ θ = θ₀ (1 − ω/n), n = lanes per direction, **ω = 0.5** | `coupling.py` |
| UE surrogate (Eq. 26–27) | features: capacity reduction of the 1,985 candidate roads + demand scale; target: φ − s·φ₀ → 60 PCA components; ridge vs. MLP (best on validation); then exact projection on {Bφ = d} (graph Laplacian, closed form) alternated with φ ≥ 0; **then 3 Frank–Wolfe iterations** | `surrogate.py`, `problems.car_flows` |
| Total travel time (Eq. 28) | T_tot = T_c (Σ φ·t) + T_b (cyclists × cyclist distance / 15 km/h), peak hour | `problems.evaluate` |
| Path sufficiency (Eq. 30–35) | S_p = 1 / (Σf·d / Σd + 1), S_thr = λ·S_G, I_p = max(0, (S_thr − S_p)/S_thr), regional (70 IRIS) demand-weighted → J_SI | `justice.py` |
| Trip insufficiency (Eq. 36–39) | bike: 20 min; car: 1.25 × free-flow OD time → J_TI = J_TI^b + J_TI^c | `justice.py` |
| Problem 2 (utilitarian) | γ·Ĵ_E + (1 − γ)·T̂_tot | `problems.objective` |
| Problem 3 (sufficientarian) | γ·Ĵ_SI(λ) + (1 − γ)·η·Ĵ_TI, λ ∈ {10, …, 50 %} | `problems.objective` |
| GA | budget 0.15·B_max = €18.9 M, pygmo sga 16 × 12, Algorithm-1 decoding, seeds incl. MLCC | `ga.py`, `problems.seeds` |
| Validation | every design re-evaluated with the exact UE (gap 10⁻⁴) | `05_validate_exact_ue.py` |

## Results

**Today (step 01–02).** P_b = 16.49 %, mean cyclist distance D̄ = 2.80 km; car UE TTT = 2,850 veh·h
(peak hour), bikes 408 h, T_tot = 3,258 traveller-hours. Path safety: S_G = 0.067; at λ = 40 %,
**66 % of the cyclists are below the threshold** (J_SI = 0.443) and 5.0 % of the bike trips
exceed 20 min. B = 0.15·B_max = €18.9 M. 1,985 roads (138 km) can lose capacity.

**Surrogate (step 03, `results/03_surrogate_metrics.json`, Fig. 3).** 1,000 exact UE samples
(27 CPU-hours instead of the paper's 5,928 for 912 samples): 850 for training and validation,
150 for testing. Ridge regression on 60 PCA components beat the MLP. R² on the link flows = 0.992,
but R² on the *deviation* from today's flows is only 0.46 (still rising with more samples), and the
car TTT was overestimated by +2.2 % on average (max 9 %), which is as large as the effects being
compared. With 3 warm-started Frank–Wolfe iterations after the projection, the error falls to
**0.26 % (max 1.05 %)**; at the optimised designs, the error vs. the exact UE is 0.10 % (max 0.2 %).

**Representative designs, exact UE** (`results/06_summary.csv`; γ\* = 0.5 for both, following the paper's
rule γ\*_λ = λ + 10 %, λ = 40 %):

| | today | utilitarian | sufficientarian |
|---|---|---|---|
| exposure J_E | 0.526 | 0.068 (−87 %) | **0.065 (−88 %)** |
| path insufficiency J_SI (λ = 40 %) | 0.443 | 0.087 | **0.079 (−10 % vs util.)** |
| cyclists below the safety threshold | 66.2 % | 18.3 % | **16.9 %** |
| bike trips > 20 min (J_TI^b) | 5.0 % (0.0079) | 3.6 % (0.0055) | 3.6 % (0.0055) |
| drivers above 1.25 × free flow (J_TI^c) | 1.1 % (0.00036) | 1.2 % (0.00040) | 1.3 % (0.00041) |
| bike share P_b | 16.49 % | 18.21 % | 18.21 % |
| car TTT T_c | 2,850 | −0.23 % | −0.20 % |
| total travel time T_tot | 3,258 | +0.35 % | +0.37 % |
| roads that lose capacity | 0 | 276 | 284 |

**Fronts (Fig. 4, `results/04_runs.csv`, `05_exact.csv`).**
* γ = 0 gives a **car-neutral** design (−39 to −42 % exposure, T_tot ± 0 %, car TTT −0.4 % thanks to
  mode shift). Up to γ ≈ 0.8, T_tot rises by less than 1 % while exposure falls by 78–90 %. Only
  γ ≥ 0.9 (pure safety) costs +2 to +3.4 % (car TTT +2 to +3 %).
* At equal T_tot, the sufficientarian front (λ = 40 %) has lower path insufficiency at almost every
  point (e.g. 0.079 vs 0.087 at +0.36 %, 0.011 vs 0.015 at +3.1 %). At γ = 0.9, it also has a
  smaller car impact (+1.9 % vs +2.8 %, driver insufficiency 0.0018 vs 0.0032).
* Both fronts are close to each other, because S_p and J_E both depend on the same traffic factor f:
  the arcs that remove exposure are also the arcs that lift the worst paths above the threshold.
* γ\* from the utopia rule: 0.5 (util.) and 0.8–0.9 (suff.), see `results/04_runs.json`.

**Car impact of the representative designs (exact UE, `results/05_sensitivity.csv`).**

| | this model | without mode shift | paper's capacity rule (ω = 1) |
|---|---|---|---|
| utilitarian | −0.23 % | +0.47 % | +3.45 % |
| sufficientarian | −0.20 % | +0.50 % | +3.12 % |

Conclusions (same direction as the paper):
1. Upgrades cut cyclist exposure and path insufficiency by close to 90 % with **T_tot increases
   below 0.4 %**. The mode shift (+1.7 points of bike share) offsets the car capacity lost.
2. The two justice principles lead to different priorities. The sufficientarian GA targets the
   cyclists below the threshold and gives the most balanced result: −10 % path insufficiency and
   −8 % fewer cyclists below the threshold than the utilitarian design, with the same bike trip
   insufficiency and practically the same T_tot.
3. Driver insufficiency stays very low (≈ 1 % of the car trips) unless γ → 1.

## Step 3b — graph neural network surrogate (added; the ridge model is kept)

`03b_train_gnn_surrogate.py` + `src/si_cycling/gnn.py` (pure PyTorch, CPU). `results/03_surrogate.pkl`
and every result of steps 04–06 are unchanged.

* **Model**: edge-centric message passing on the car graph (8 layers, 48 hidden units, residual +
  LayerNorm, node mean aggregation + a global mean term). Arcs that never carry > 1 veh/h in the samples
  (~5,000 of 12,350) are pruned and keep s·φ0. Features per arc: log t0, log capacity, lanes, v/c and
  BPR marginal today, candidate flag, capacity reduction r, demand scale, new v/c, log time shock.
  Target (φ − s·φ0)/σ; AdamW + cosine schedule, 40 epochs (~2.4 min each), early stopping on the
  validation R² of the deviation.
* **Data**: the same 1,000 UE samples of step 2; the same 150 test samples as step 3 (seed 0).

| test set (150 UE) | R² flows | R² of φ − sφ0 | within 10 % | car TTT error, 0 / 1 / 3 FW it. (mean abs) | max abs |
|---|---|---|---|---|---|
| ridge (step 3)  | 0.9916 | 0.455 | 79.6 % | 2.16 / 0.84 / **0.26** % (bias +) | 9.35 / 3.29 / 1.05 % |
| GNN             | **0.9970** | **0.803** | **84.0 %** | **0.35** / 0.33 / 0.39 % (unbiased) | 1.49 % |
| ensemble (mean) | 0.9960 | 0.740 | 83.3 % | 0.65 / **0.28** / **0.15** % | 2.34 / 0.87 / **0.54** % |

* The GNN explains 80 % of the variance of the UE re-routing vs 46 % for ridge, and without any
  correction its TTT error is 6× smaller and unbiased. FW iterations do not improve it further
  (the remaining error is not along the descent direction).
* Ridge and GNN errors have opposite signs, so the **ensemble** is the most accurate:
  1 FW iteration already reaches the 0.3 % tolerance, 3 iterations give 0.15 % (max 0.54 %).
* **Choice** (`results/03b_surrogate_choice.json`, rule: fewest iterations with mean |TTT err| ≤ 0.3 %):
  **ensemble + 1 FW iteration**. Check on a 40-upgrade design: 3.7 s per evaluation vs 39 s exact UE,
  T_c error +0.15 %.
* `config.SURROGATE_MODEL`: `"auto"` (default, uses the choice above), `"ridge"` (exactly the model used
  for the results of steps 04–06 above), `"gnn"`, `"ensemble"`. To reproduce the published-here numbers,
  set `"ridge"`; rerunning step 04 with `"auto"` would overwrite `results/04_*` (back them up first).
* Proposals: (i) TTT-weighted / physics-informed loss (penalise the BPR-weighted error, or the
  UE gap of the predicted flow) to remove the residual 0.35 %; (ii) active learning — add exact UE
  samples around the GA designs (the optimiser explores budgets near 15 %, the samples span 2–30 %);
  (iii) ensemble + 3 iterations when accuracy matters more than run time (e.g. final fronts);
  (iv) train the GNN on the residual of ridge instead of averaging.

## Figures v2 — paper style and fair choice of the compared designs (step 5b + new step 6)

Old figures and the old `06_figures.py`: `figures/v1_before_restyle/`.  Steps 01–05 and their results are unchanged.

**Problems in v1**
* Fig. 5 mixed the sufficientarian runs of **all** λ (19 bars, so γ appeared twice) and plotted each indicator
  divided by today's value. Today's car-delay insufficiency is tiny (J_TI^c = 0.0004), so its ratio jumped to ×9–10
  at γ ≥ 0.9.  That jump is real: with weight only on safety, the GA takes car lanes on busy roads (7 % of drivers
  above the delay threshold vs 1.1 % today).  But the ratio scale made it look absurd.
* Fig. 10: GA-S(λ) used γ = λ + 0.1 and GA-U used γ = 0.5.  For λ = 20 % and 30 %, GA-S therefore put *less*
  weight on safety than GA-U and came out worse.  This was an artefact of the γ choice, not of the method.
* Fig. 6 only showed two colours in the legend, and the panels had different y scales.  Figs. 7 and 9 showed the
  three designs side by side, so a change of 1 % was invisible.

**What changed**
* `05b_exact_all_designs.py` (new, ~20 min): exact UE for all 66 designs (11 utilitarian + 5 × 11
  sufficientarian), plus the shares of users below each threshold.
  Outputs: `results/05b_all_designs.csv` and `05b_arrays.npz` (~240 MB, too large for GitHub, so it is not in the repository: run step 5b before step 6).
* `src/si_cycling/selection.py`, set by `config.REPRESENTATIVE_RULE` (**CHANGE**, default `"no_harm"`):
  * GA-U is the utopia point of the utilitarian front (γ = 0.5, unchanged).
  * GA-S(λ) is the design of the λ-front with the **lowest J_SI(λ)** among those that leave no more cyclists
    above 20 min and no more drivers above the delay threshold than GA-U (± 0.02 pp), and cost at most +0.5 %
    total travel time.
  * In words: the most sufficient network that harms no traveller group more than GA-U, at almost the same
    system cost.
  * `"paper"` (the old rule) and `"utopia"` are still available.
* Paper style (`plotting.paper_style`): serif text, dotted grid, Okabe-Ito colours (blue J_SI, green bike,
  pink car), wheat for Original, orange for GA-U, blue for GA-S, viridis maps.
* Fig. 5: same layout as the paper's figure.  Each indicator is min-max normalised over Original + 11 designs on a
  log axis.  Each panel shows one front only, and the compared γ is in red.
* Fig. 6: shared y axis, full legend in every panel, today's distribution as a dashed outline, and a fourth panel
  with the per-bin difference GA-S − GA-U (+913 cyclists/day above the threshold).
* Fig. 7: the three distributions on the same axes, a log-scale zoom on the tail around the threshold, and a
  second row with the change per time band with respect to today.
* Fig. 9: the same colour scale for the three maps in each row (square-root scale, so low values stay visible).
  A fourth column shows GA-S − GA-U (blue = GA-S better) and the links where the two networks differ.
* Fig. 10: GA-S(λ) is compared with GA-U for every λ, with the relative difference labelled.

**Results (exact UE, λ = 40 %: GA-U = util γ 0.5, GA-S = suff γ 0.7)**

| | Original | GA-U | GA-S |
|---|---|---|---|
| J_SI(0.4) | 0.443 | 0.087 | **0.066 (−24 %)** |
| cyclists below S_thr | 66.2 % | 18.3 % | **14.7 %** |
| exposure J_E | 0.526 | 0.068 | **0.059 (−14 %)** |
| cyclists > 20 min | 5.04 % | 3.61 % | 3.60 % |
| drivers above delay thr. | 1.11 % | 1.219 % | 1.221 % |
| T_tot vs today | — | +0.35 % | +0.66 % |
| bike share | 16.49 % | 18.21 % | 18.24 % |

J_SI of GA-S(λ) vs GA-U: −44 % (λ = 10 %), −34 % (20 %), −30 % (30 %), −24 % (40 %), −22 % (50 %).
The price is about +0.3 % total travel time compared with GA-U, mostly for drivers in the city centre (Fig. 9,
last row).  The trip-time shares of cyclists and drivers are the same as with GA-U.

**Caveat**: the selection is made after the optimisation, from designs the GA already produced.  The fronts
themselves are unchanged.  In the J_SI–T_tot plane the two fronts almost coincide (Fig. 4, right): the sufficientarian
design is a different point on a very similar trade-off curve, not a strictly better network.

## Changes to the methodology (implemented here) — please review

1. **Car network, demand, UE** as in `ITSC_2025` (OSM drive network instead of the 2024 graph with
   random highway types; peak-hour demand calibrated on the GTLVille AADT; bi-conjugate Frank–Wolfe
   and fast tree loading).
2. **Capacity rule ω = 0.5** (`config.OMEGA`). Eq. 17 with ω = 1 gives θ = 0 for single-lane roads
   (88 % of the road length of Grenoble), i.e. the road is closed; the 2025 code silently replaced
   that 0 by 600 veh/h. With ω = 0.5, a single-lane road keeps half of its capacity. The paper's rule
   is evaluated in the sensitivity analysis (+3.1 to +3.5 % car TTT).
3. **Mode-choice constant α = 0.861.** With the paper's α = 0.128 and β = −0.886/km, the model gives
   P_b = 8.7 % today instead of 16.49 %; α is recalibrated so the logit reproduces 16.49 % at D̄₀.
4. **Car trip threshold = 1.25 × free-flow OD time** (`config.CAR_DELAY_TOLERANCE`). With an absolute
   20 min, no car trip is insufficient (the longest peak-hour trip takes 14 min) and J_TI^c ≡ 0.
5. **Surrogate.** The 2025 code trained the MLP on randomly perturbed flows, not on UE solutions. Here:
   1,000 exact UE samples (random budgets from 2 to 30 % of B_max; 60 % of them near the greedy orderings
   the GA starts from), PCA + ridge/MLP, exact projection, and **3 Frank–Wolfe iterations warm-started
   at the projected flows** (`config.SURROGATE_UE_ITERS`; 0 = paper). A GCN was not possible (no
   deep-learning library in the environment), but ridge on PCA scores already reaches the paper's
   accuracy level on link flows.
6. **Normalisation and γ\*** as in `ITSC_2025`: ideal/nadir values from the γ = 1 and γ = 0 runs;
   representative γ by the paper's illustrative rule (default) or by the utopia point
   (`config.GAMMA_STAR_RULE`).
7. **GA**: Algorithm-1 decoding with a skip range (the budget is "≤"), seeded with Algorithm 1,
   exposure-per-euro, car-free and car-aware variants, the MLCC solution and, for Problem 3, the flow
   of today's insufficient trips.
8. **T_tot** adds the bike time of the peak hour (cyclist distance at 15 km/h) to the car TTT, so
   cars moving to bikes are counted in both modes.

## Proposals

* **More samples / active learning.** The R² of the deviation still rises at 850 samples (0.31 →
  0.46). Adding samples around the GA designs (e.g. 200 more from the final populations, ~1 h)
  would let the correction iterations be reduced or dropped and would halve the 15 h of step 04.
* **Fewer GA evaluations.** Each GA run takes ~45 min (≈ 9 s per evaluation with 7 parallel runs).
  Phase 2 could warm-start each γ from the population of the neighbouring γ.
* **Paper text**: correct α (point 3), state ω or the minimum capacity (point 2), and use a relative
  car-time threshold (point 4); the Euclidean-distance rule for γ\* (Eq. 15 in ITSC) is minimised by
  γ = 0 and should not be used.

## Run times (8-core PC)

Step 02: 4 h (1,000 UE samples, 97 s each on average, 53 BFW iterations). Step 03: 15 min.
Step 04: 15 h (66 GA runs, 16 × 12). Step 05: 14 min. Tests: 11/11 pass.

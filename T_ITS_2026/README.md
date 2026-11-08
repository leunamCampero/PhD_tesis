# Comparison and optimisation of urban cycling networks via safety-aware clustering — T-ITS 2026

Code and data implementing the methodology of

> M. Campero Jurado, C. Canudas-de-Wit, G. De Nunzio, *Mathematical Comparison and Optimization of Urban
> Cycling Networks via Safety-Aware Clustering* (`paper/main.tex`, `paper/T_ITS_2026.pdf`).

Grenoble (6 773 nodes, reference network H), Lyon (11 857) and Strasbourg (11 417):

1. **Clustering** of Lyon and Strasbourg to |V_Grenoble| nodes with the **CC-Algorithm** (proposed) and with
   spectral **k-means** (baseline);
2. **Comparison** with Grenoble: Gromov–Wasserstein (GW) distance of the exposure measures, kernel distance;
3. **Optimisation** of DoS upgrades under a budget B = β B_max: GA on the original network (Problem 1), GA on
   the clustered network lifted back (Problem 2), for both clusterings;
4. **Transfer** of the optimised Grenoble strategy through the GW plan (Problem 3).

The conference version (clustering + comparison only) is in `../ITSC_2026` with the same library.
The numbers are not the printed ones (the notebooks' inputs were incomplete, see *Changes*); the proposed
algorithm is better than k-means in every comparison (see *Results*).

## Quick start

From the activated environment, run everything with **`python run_all.py`** (`--list` shows the steps, `--from N` resumes at step N, `--only N` runs one step), or run the steps one by one, in this order:

```bash
conda activate cycling-ecc              # environment.yml

python 01_prepare_networks.py           # CSV networks, demand, today's exposure, Fig. 1            ~30 s
python 02_toy_example.py                # toy example (Fig. 2)                                      ~5 s
python 03_clustering.py                 # k-means vs CC, 13 values of k per city (Figs. 4-5)        ~10 min
python 04_compare_networks.py           # GW / kernel distances to Grenoble                         ~20 min
python 05_optimisation.py               # 28 GA runs (Grenoble + 2 cities x 3 methods x 4 budgets)   ~35 min (7 cores)
python 05_optimisation.py --random-init # paper-like GA from a random population (Fig. 7)           ~25 min
python 06_transfer.py                   # GW transfer of the Grenoble strategy (Problem 3)          ~30 s
python 07_figures.py                    # all figures + results/07_summary.csv                      ~2 min
python tests/run_tests.py               # 21 unit tests                                             ~3 min
```

On Windows run them from an activated conda shell (scipy's ARPACK / MKL crash otherwise).
`--workers`, `--pop`, `--generations`, `--betas` change step 5. Everything is seeded.

## Folder layout

```
T_ITS_2026/
├── 01_ … 07_*.py               the steps (run in order)
├── src/cyclenet/
│   ├── config.py               ALL parameters (changes w.r.t. the paper marked CHANGE)
│   ├── network.py              original network: f(w), cyclist distance, AON flows, exposure, costs (Sec. II)
│   ├── clustered.py            clustered network G~: objects, clustered distance / demand / flows / exposure,
│   │                           object costs Pi_g, lifting (Sec. II-D/E, III-B, V-A)
│   ├── clustering.py           safety-aware adjacency + Laplacian, k-means baseline, CC-Algorithm (Alg. 1)
│   ├── compare.py              exposure measures, entropic GW, exposure kernels / graphons (Sec. IV)
│   ├── ga.py                   GA for Problems 1 and 2 (Algorithms 2-3)
│   ├── transfer.py             GW transfer, Problem 3 (Algorithm 4)
│   └── plotting.py, utils.py
├── data/raw/                   input pickles + city boundaries;  data/networks/  CSVs of step 1
├── results/                    CSV tables, partitions, GW plans, GA runs (05_runs*/), transfers (06_runs/)
├── figures/                    PNG + PDF
├── tests/run_tests.py
├── paper/                      LaTeX sources, images and PDF (moved here, unchanged)
└── _archive_old_files/         the four notebooks that were in the folder (moved, not modified)
```

The notebooks load `*_bike_network_with_slope.pkl`, `*_node_demand_dic.pkl`, `*_sample_pairs.pkl`, which were
not in the folder; they were copied from `CODE/2024/2024/Third_project_2024/` to `data/raw/`.

## Changes to the methodology (and why)

1. **CC-Algorithm.** The notebooks' "constrained" clustering was k-means on spectral + geographic features
   with no constraint. Algorithm 1 is implemented with its hard constraints: the assignment step is a
   *constrained greedy normalised-cut contraction* on the same Laplacian (merge the two adjacent clusters with
   the largest A_rs/√(vol_r vol_s) minus λ₁ × the exposure distortion of the merge) subject to
   **DoS homogeneity** of every clustered object (max − min DoS ≤ ε_max = 0) and a diameter ≤ δ_max; when stuck
   ε_max / δ_max are relaxed; the outer loop checks |ΔE| ≤ τ_E, |ΔΦ| ≤ τ_Φ on the clustered network, tries
   the next (λ₁, δ_max) pair and relaxes the tolerances by α_relax; fallback k-means. Clusters are connected.
2. **Demand.** The notebooks' 130 OD pairs with U(0,1) weights touch ~5 % of the arcs; a gravity demand
   (150 origins × 10 destinations, same generator for the three cities) is used. `config.DEMAND = "paper130"`
   restores the notebooks' pairs.
3. **Slope.** The stored slopes are uniform in [−8, 8] % in all three cities (median 6.4 % in flat
   Strasbourg): placeholders, not used (s_a = 1; `SLOPE_SOURCE = "file"` restores them). No angles (η = 0).
4. **Budgets.** With the paper's β ∈ {1/5, …, 4/5} (notebooks: −95 % at 40 %) every method removes almost all
   the exposure: the traffic factor of DoS 1 arcs (3 926) is 4 700 × that of DoS 6, so a few km of upgrades
   suffice. β ∈ {0.1, 0.25, 0.5, 1} % spans the range of the paper's Fig. 6 (E/E0 ≈ 0.8 → 0.2).
5. **GA.** Random-key GA with a budget-greedy decoder (= repair: every design is feasible), level gene
   (DoS 5 or 6). Candidates = objects used by cyclists today or when every object is DoS 6. The reported
   designs come from a population seeded around the "exposure reduction per euro" greedy design, computed
   on each problem's own model (fair between methods): random populations never reach it in 60 generations.
   The paper-like GA (random population) is run separately for the convergence figure (Fig. 7).
6. **Clustered model.** Totals E^clust, E^orig compared (the |Ã|+k and |A| normalisations change the ratio
   by themselves); intra flow only for clusters that contain arcs (k = |V| gives G exactly, tested). Theorem 2
   holds for the OD demand (Σ q̃ = Σ q, tested), not for Σ_a φ_a, which depends on the path lengths.
7. **GW.** Each term of D scaled to [0, 1], coordinates standardised per city, top-M most exposed nodes
   (M = 200 / 400 / 800; plan at M = 400 used for the transfer), entropic GW with log-domain Sinkhorn
   (no POT); GW(Grenoble, Grenoble) = 0.055 is the entropic floor. Kernel distance = mean absolute difference
   of degree-sorted graphons (label-invariant).
8. **Transfer.** Clusters outside the GW subset receive the IDW score of their 5 nearest subset clusters;
   Problem 3 solved by rank-and-select on S_g (L − w0)/Π_g(L). **Transfer+** (proposal, not in the paper):
   S_g multiplied by the target's own cyclist flow on g (one AON on the target, no optimisation).

## Results

**Clustering (k = 6 773; Figs. 4-5)**

| | Lyon k-means | Lyon CC | Strasbourg k-means | Strasbourg CC |
|---|---|---|---|---|
| E^clust vs E^orig | −15.5 % | **−1.5 %** | +14.4 % | **−3.0 %** |
| Φ^clust vs Φ^orig | −1.9 % | +0.6 % | +5.7 % | +4.0 % |
| arcs in mixed-DoS objects | 14.3 % | **0 %** | 20.1 % | **0 %** |
| disconnected clusters | 1 070 | **0** | 874 | **0** |

Sweep k = |V| … 5 000: mean |ΔE| 0.6 % / 1.0 % with CC vs 7.5 % / 18.4 % with k-means (max 16 % / 68 %).

**Comparison with Grenoble (Fig. 9)** — GW at M = 200 / 400 / 800:
CC: Lyon 0.285 / 0.301 / 0.289, **Strasbourg 0.246 / 0.224 / 0.221** → Strasbourg closer (paper's conclusion,
Baromètre Vélo); kernel distance 0.558 vs **0.451**, same ranking.
k-means: Lyon 0.229 / 0.203 / 0.197, Strasbourg 0.265 / 0.229 / 0.228 → the ranking **flips**.

**Optimisation and transfer (normalised exposure E/E0 on the original network, Fig. 6, `results/07_summary.csv`)**

| β | 0.1 % | 0.25 % | 0.5 % | 1 % |
|---|---|---|---|---|
| Lyon GA (original network) | 0.593 | 0.410 | 0.275 | 0.152 |
| Lyon GA-Cluster (CC, lifted) | 0.604 | 0.434 | 0.299 | 0.186 |
| Lyon GA-Cluster (k-means, lifted) | 0.883 | 0.743 | 0.648 | 0.519 |
| Lyon Transfer (CC) / (k-means) | 0.985 / 0.990 | 0.958 / 0.978 | 0.873 / 0.951 | 0.737 / 0.900 |
| Lyon Transfer+ | 0.693 | 0.476 | 0.283 | 0.198 |
| Strasbourg GA | 0.827 | 0.659 | 0.451 | 0.270 |
| Strasbourg GA-Cluster (CC) | 0.828 | 0.658 | 0.471 | 0.280 |
| Strasbourg GA-Cluster (k-means) | 0.947 | 0.781 | 0.656 | 0.494 |
| Strasbourg Transfer (CC) / (k-means) | 0.955 / 0.997 | 0.949 / 0.956 | 0.917 / 0.942 | 0.887 / 0.916 |
| Strasbourg Transfer+ | 0.854 | 0.765 | 0.665 | 0.524 |
| Grenoble GA (source of the transfer) | 0.714 | 0.569 | 0.401 | 0.204 |

* **GA-Cluster on the CC network ≈ GA on the original network** (within 0.00–0.03) in **about half the
  time** (5–7 vs 10–14 min per run; 1.7 × faster per generation, Fig. 7). On the k-means network the lifted
  designs are 0.12–0.37 worse, and the k-means model is misleading: it predicts 0.70 where the real value is
  0.88 (Lyon, 0.1 %).
* **Paper-like GA (random population, Fig. 7)**: after 60 generations GA-Cluster is better than or equal to
  the original GA in 7 of 8 cases (equal within 0.002 for Strasbourg 0.1 % and 1 %; e.g. Strasbourg 0.5 %: 0.805 vs
  0.883), worse only for Lyon 1 % (0.707 vs 0.687),
  in 0.4–0.6 × the time.
* **Transfer**: CC target better than k-means target at every budget in both cities. The pure GW transfer
  (Problem 3) recovers only 4–31 % of the GA's exposure reduction: it ignores where the target's cyclists ride.
  Transfer+ recovers 75–99 % in Lyon (0.198 vs 0.152 at 1 %) and 61–84 % in Strasbourg.
* Not reproduced: the paper's claim that the transfer works better for the GW-closer city (Strasbourg).
  In relative terms it works better for Lyon here.

## Caveats

* Synthetic demand and no slopes (no OD survey / DEM for Lyon and Strasbourg).
* The greedy design is nearly optimal in this model (AON routing reacts weakly to upgrades: f^0.04), so the
  GA refines it by < 1 %; the GA-vs-GA-Cluster comparison mostly measures the fidelity of the clustered model.
* CC misses τ_E = 1 % for Strasbourg at k = 6 773 (−3.0 %); its flow error there is +4.0 %.
* GW uses the 200–800 most exposed nodes; the GW ranking is consistent for the three M.

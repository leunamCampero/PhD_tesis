# Safety-aware clustering and comparison of cycling networks — ITSC 2026

Code and data implementing the methodology of

> M. Campero Jurado, C. Canudas-de-Wit, G. De Nunzio, *Mathematical comparison of urban cycling
> networks via safety-aware clustering* (ITSC 2026, `paper/main.tex`, `paper/Manuel_Carlos_Giovanni_ITSC_2026.pdf`).

Three cities (Grenoble 6 773 nodes, Lyon 11 857, Strasbourg 11 417). Lyon and Strasbourg are clustered
down to |V_Grenoble| = 6 773 nodes with the safety-aware **CC-Algorithm** (Algorithm 1) and with plain
spectral **k-means** (the baseline). The clustered networks are then compared with Grenoble through a
Gromov–Wasserstein (GW) distance between exposure measures and an exposure-kernel (graphon) distance.

The same library is used, and extended with the optimisation and the strategy transfer, in `../T_ITS_2026`
(journal version). The numbers are not the printed ones; the conclusions point the same way, and the
proposed algorithm beats k-means on every criterion (see *Results*).

## Quick start

From the activated environment, run everything with **`python run_all.py`** (`--list` shows the steps, `--from N` resumes at step N, `--only N` runs one step), or run the steps one by one, in this order:

```bash
conda activate cycling-ecc           # environment.yml (numpy, scipy, pandas, networkx, scikit-learn, matplotlib)

python 01_prepare_networks.py        # CSV networks, demand, today's flows / exposure, Fig. 1        ~30 s
python 02_toy_example.py             # toy example of the clustering and of the kernels (Figs. 2-3)  ~5 s
python 03_clustering.py              # k-means vs CC for 13 numbers of clusters per city (Figs. 4-5)  ~10 min
python 04_compare_networks.py        # GW and kernel distances to Grenoble                            ~20 min
python 05_figures.py                 # all figures                                                    ~3 min
python tests/run_tests.py            # 21 unit tests                                                  ~3 min
```

On Windows run them from an activated conda shell (scipy's ARPACK crashes otherwise). Steps 3 and 4
were run once in `../T_ITS_2026` and their outputs copied to `results/` (identical code, configuration and
seeds); steps 1, 2, 5 and the tests were run in this folder.

## Folder layout

```
ITSC_2026/
├── 01_ … 05_*.py             the steps (run in order)
├── src/cyclenet/
│   ├── config.py              ALL parameters (changes w.r.t. the paper marked CHANGE)
│   ├── network.py             original network: traffic factor, cyclist distance, AON flows, exposure
│   ├── clustered.py           clustered network G~ (objects, clustered distance / demand / flows / exposure)
│   ├── clustering.py          safety-aware adjacency + Laplacian (Eqs. 19-20), k-means baseline, CC-Algorithm
│   ├── compare.py             exposure measures, entropic Gromov-Wasserstein, exposure kernels, graphons
│   ├── ga.py, transfer.py     used by T_ITS_2026 only
│   └── plotting.py, utils.py
├── data/raw/                  input pickles (networks with DoS and slope, the notebooks' OD pairs), city boundaries
├── data/networks/             CSV networks + demand written by step 1
├── results/, figures/         outputs (PNG + PDF)
├── tests/run_tests.py
├── paper/                     LaTeX sources, images and PDF of the paper (moved here, unchanged)
└── _archive_old_files/        the four notebooks that were in the folder (moved, not modified)
```

`data/raw/*_with_slope.pkl`, `*_node_demand_dic.pkl` and `*_sample_pairs.pkl` are the files the notebooks load;
they were missing from this folder and were copied from `CODE/2024/2024/Third_project_2024/`.

## Methodology (what each step computes)

| Paper | Code |
|---|---|
| DoS, traffic factor f(w) (Table I, Eq. 2), cyclist distance (Eq. 1) | `network.py` |
| AON flows (Eqs. 3-5), arc exposure E_a = φ_a f_a d_a, E^orig | `Network.evaluate` (one multi-source Dijkstra + vectorised walk of the trees) |
| clustered network: min-DoS objects, flow-weighted d, s, angle (Eqs. 6-14) | `ClusteredNetwork` |
| clustered demand, inter / intra flows, E^clust, Φ^clust (Eqs. 15-19) | `ClusteredNetwork.evaluate` |
| safety-aware adjacency and normalised Laplacian (Eqs. 19-20, Theorem 1) | `clustering.safety_adjacency / laplacian` |
| Algorithm 1 (CC-Algorithm) | `clustering.cc_algorithm` |
| baseline k-means (paper's "K-mean") | `clustering.kmeans_baseline` = notebooks' spectral + geographic k-means |
| exposure measures μ, dissimilarities D, GW distance (Eqs. 28-37) | `compare.py` |
| exposure kernel distance, top-100 kernels, node strength (Figs. 3, 6, 7) | `compare.kernel_*`, `graphon`, `top_kernel` |

## Changes to the methodology (and why)

1. **CC-Algorithm** – the notebooks' "constrained" clustering was k-means on spectral + geographic
   features (no constraint was enforced). Here Algorithm 1 is implemented with real hard constraints:
   the assignment step is a *constrained greedy normalised-cut contraction* on the same Laplacian: the two
   adjacent clusters with the largest normalised association A_rs/√(vol_r vol_s) (what spectral clustering
   maximises) minus λ₁ × the exposure distortion of the merge are merged, as long as
   (i) every clustered object it creates is **DoS-homogeneous** (max − min DoS ≤ ε_max = 0) and
   (ii) the cluster's diameter is ≤ δ_max. Stuck → ε_max / δ_max relaxed. The outer loop checks
   |ΔE| ≤ τ_E and |ΔΦ| ≤ τ_Φ on the clustered network, tries the next (λ₁, δ_max) pair and relaxes the
   tolerances by α_relax (as Algorithm 1); fallback k-means. Clusters are connected by construction.
2. **Demand** – the notebooks used 130 OD pairs with U(0,1) weights; these touch ~5 % of the arcs. A
   gravity demand (150 origins × 10 destinations, density-weighted, exp(−d/2.5 km)) generated the same way
   for the three cities is used (`config.DEMAND = "paper130"` restores the notebooks' pairs).
3. **Slope** – the stored slopes are uniformly spread in [−8, 8] % in all three cities (median |slope|
   6.4 % even in flat Strasbourg): placeholders. Slope factor = 1 (`config.SLOPE_SOURCE = "file"` restores
   them). Angular change is not in the graphs (η = 0, as in the notebooks).
4. **Exposure totals** – E^clust and E^orig are compared as totals: dividing by |Ã|+k and |A| (paper)
   changes the ratio with k even when nothing else changes.
5. **Intra flow** only for clusters that contain at least one arc, so that k = |V| gives exactly G
   (tested). **Theorem 2** as printed is not true for arc flows (Σ_a φ_a = Σ q × path length, which changes
   with the clustering); what holds exactly is conservation of the OD demand (Σ q̃ = Σ q, tested). Φ^clust is
   therefore reported as a measured distortion.
6. **GW** – each term of D scaled to [0, 1] (raw |μ_i − μ_j| ~ 1e-4 vanishes against coordinates);
   coordinates standardised per city; problem solved on the M nodes with the largest exposure mass
   (M = 200 / 400 / 800, the notebooks used 800); entropic GW with log-domain Sinkhorn (no POT needed);
   GW(Grenoble, Grenoble) = 0.055 is the entropic floor.
7. **Kernel distance** – the paper's value (4.36e-8 vs 4.365e-8) depends on node labels. Here: mean absolute
   difference of the degree-sorted empirical graphons (50 × 50 blocks), label-invariant and size-independent.

## Results

**Clustering to k = |V_Grenoble| = 6 773 (Fig. 4-5, `results/03_clusters_summary.csv`)**

| | Lyon k-means | Lyon CC | Strasbourg k-means | Strasbourg CC |
|---|---|---|---|---|
| exposure E^clust vs E^orig | −15.5 % | **−1.5 %** | +14.4 % | **−3.0 %** |
| flow Φ^clust vs Φ^orig | −1.9 % | +0.6 % | +5.7 % | +4.0 % |
| arcs in mixed-DoS objects | 14.3 % | **0 %** | 20.1 % | **0 %** |
| disconnected clusters | 1 070 | **0** | 874 | **0** |

Over the whole sweep (12 values of k from |V| to 5 000) the mean |ΔE| is 0.6 % (Lyon) and 1.0 %
(Strasbourg) with CC against 7.5 % and 18.4 % with k-means (max 16 % and 68 %); the mean |ΔΦ| is 1.4–1.6 %
with CC against 3.4–5.3 %. k-means mixes DoS 1 arcs (f = 3 926) with DoS 4–6 arcs (f ≤ 2.2) in the same
object, and the conservative min-DoS rule then inflates (or, through shortcuts of disconnected clusters,
deflates) the exposure. CC takes 1–7 s per clustering, k-means 5–20 s.

**Comparison with Grenoble (`results/04_comparison.csv`, Fig. 9)**

| GW to Grenoble (M = 200 / 400 / 800) | Lyon | Strasbourg |
|---|---|---|
| CC | 0.285 / 0.301 / 0.289 | **0.246 / 0.224 / 0.221** |
| k-means | 0.229 / 0.203 / 0.197 | 0.265 / 0.229 / 0.228 |
| kernel distance, CC | 0.558 | **0.451** |
| kernel distance, k-means | 0.362 | 0.632 |

With the CC clustering, **Strasbourg is closer to Grenoble than Lyon** for every M and for both
distances — the paper's conclusion, consistent with the Baromètre Vélo ranking. With k-means the ranking
**flips** (Lyon appears closer): the comparison is only trustworthy if the clustering preserves the
exposure structure, which is the point of the CC-Algorithm. Reference values: GW(Grenoble, Grenoble) =
0.055 (entropic floor), GW(Lyon CC, Strasbourg CC) = 0.254.

**Toy example (Figs. 2-3)**: with k = 4 the clustered exposure equals the original one (3 303k vs 3 301k);
k = 3 and 2 require mixing DoS (ε relaxed to 4–5) and the min-DoS rule multiplies the exposure by 3–8.

## Caveats

* The demand is synthetic (no OD survey for Lyon / Strasbourg); the slopes are not used.
* CC does not meet τ_E = 1 % for Strasbourg at k = 6 773 (−3.0 %, best of the 8 (λ₁, δ_max) pairs).
* GW is computed on the 200–800 most exposed nodes, not on all 6 773.

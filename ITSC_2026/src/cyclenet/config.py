"""
All parameters of the methodology in one place.

Values from paper/main.tex (T-ITS 2026, "Mathematical Comparison and Optimization of Urban Cycling
Networks via Safety-Aware Clustering") unless a comment says CHANGE (the reasons are in README.md).
"""
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
NETWORKS = DATA / "networks"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CITIES = ("Grenoble", "Lyon", "Strasbourg")
REFERENCE = "Grenoble"            # smallest network = H (source of the transfer, Sec. VII)
TARGETS = ("Lyon", "Strasbourg")  # clustered to |V_Grenoble| nodes
SEED = 42

# ----------------------------------------------------------------------------------------------
# Bicycle layer (Section II, Table I, Eqs. 1-7)
# ----------------------------------------------------------------------------------------------
AADT = {6: 14_400, 5: 16_000, 4: 960, 3: 3_120, 2: 6_912, 1: 8_448}   # Table I (GTLVille)
PSI = 0.84
RHO = 0.04
PEDESTRIAN_FACTOR = 2.03
SLOPE_TABLE = ((2.0, 1.0), (4.0, 1.371), (6.0, 2.203), (np.inf, 4.239))
# CHANGE: the slopes stored in *_bike_network_with_slope.pkl are uniformly spread in [-8, 8] %
# in the three cities (median |slope| 6.4 % even in flat Strasbourg): they are placeholders, not
# terrain.  "flat" (s_a = 1) keeps the three cities comparable; "file" uses the stored values.
SLOPE_SOURCE = "flat"
ETA_ANGLE = 0.0                   # weight of angular change: not available in the graphs (= notebooks)

# ----------------------------------------------------------------------------------------------
# Demand
# ----------------------------------------------------------------------------------------------
# "paper130": the 130 OD pairs with U(0, 1) weights used by the notebooks (*_node_demand_dic.pkl)
# "gravity" (CHANGE, default): same generator for the three cities: N_ORIGINS origins and
#            DEST_PER_ORIGIN destinations per origin drawn proportionally to the local network
#            density, gravity weight exp(-distance / GRAVITY_LENGTH_M).  130 OD pairs only touch
#            ~5 % of the arcs, so any budget above a few percent of B_max upgrades all of them.
DEMAND = "gravity"
N_ORIGINS = 150
DEST_PER_ORIGIN = 10
GRAVITY_LENGTH_M = 2500.0
DENSITY_RADIUS_M = 300.0
TRIPS_PER_CITY = 10_000.0         # total demand (scales flows only; exposure is reported normalised)

# ----------------------------------------------------------------------------------------------
# Upgrade costs (EUR per metre) - same table as the notebooks / previous papers
# ----------------------------------------------------------------------------------------------
COST = {
    1: {1: 0, 2: 150, 3: 250, 4: 300, 5: 500, 6: 750},
    2: {2: 0, 3: 100, 4: 150, 5: 350, 6: 600},
    3: {3: 0, 4: 50, 5: 250, 6: 500},
    4: {4: 0, 5: 200, 6: 450},
    5: {5: 0, 6: 250},
    6: {6: 0},
}

# ----------------------------------------------------------------------------------------------
# Clustering (Section II-F, Algorithm 1 = CC-Algorithm)
# ----------------------------------------------------------------------------------------------
GAMMA_ADJ_M = 250.0               # gamma of Eq. (19): spatial decay exp(-d / gamma)
ALPHA_FLOW = 0.0                  # alpha of Eq. (19) (0 as in the notebooks: most arcs carry no flow)
N_EIGEN = 2                       # spectral embedding dimension (notebooks)
KMEANS_GEO_WEIGHT = 10.0          # k-means baseline = notebooks' spectral + geographic k-means
KMEANS_N_INIT = 1
KMEANS_MAX_ITER = 60
# CC-Algorithm (CHANGE: the assignment step is a constrained greedy normalised-cut contraction on
# the same safety-aware Laplacian instead of Lloyd iterations; constraints, penalties, relaxation
# and k-means fallback as in Algorithm 1)
CC_EPS_MAX = 0                    # epsilon_max: DoS range allowed inside a clustered object
CC_DELTA_MAX_M = 400.0            # delta_max: diameter (bounding-box diagonal) of a cluster
CC_TAU_E0 = 0.01                  # tau_E^(0): relative exposure distortion accepted
CC_TAU_PHI0 = 0.02                # tau_Phi^(0): relative flow distortion accepted
CC_ALPHA_RELAX = 1.25             # alpha_relax (tolerances; also epsilon / delta when stuck)
CC_LAMBDA_E0 = 0.5                # lambda_1: weight of the exposure-distortion penalty of a merge
# (lambda_1, delta_max) tried by the outer loop of Algorithm 1 (one pair per iteration t)
CC_SCHEDULE = ((0.5, 400.0), (4.0, 400.0), (64.0, 400.0), (4.0, 150.0), (64.0, 150.0), (1e4, 150.0),
               (0.5, 150.0), (1e4, 400.0))
CC_T_MAX = 8

# number of clusters of the sweep of Fig. 4 (from |V| down to 5 000, |V_Grenoble| included)
SWEEP_POINTS = 12
SWEEP_MIN_K = 5000

# ----------------------------------------------------------------------------------------------
# Network comparison (Section IV)
# ----------------------------------------------------------------------------------------------
GW_TOP_M = 400                    # nodes with the largest exposure mass used in the GW problem
GW_GAMMA = 1.0                    # gamma of Eqs. (36)-(37) (both terms scaled to [0, 1], CHANGE)
GW_EPSILON = 5e-3                 # entropic regularisation
GW_OUTER_ITERS = 40
GW_SIZES = (200, 400, 800)       # robustness check of the city ranking
KERNEL_BLOCKS = 50                # blocks of the degree-sorted empirical graphon (kernel distance)
TOP_KERNEL_NODES = 100            # ITSC Fig. 6

# ----------------------------------------------------------------------------------------------
# Optimisation (Section III, VI) and transfer (Section V)
# ----------------------------------------------------------------------------------------------
# CHANGE: beta of B = beta * B_max.  The traffic factor of DoS 1-2 arcs (f = 845-3 926) is 1 000-4 700
# times that of DoS 6 (0.84), so a few kilometres of upgrades remove most of the exposure: with the
# paper's beta in {1/5,...,4/5} (or even 2 %) every method reaches E/E0 < 0.1 (notebooks: -95 % at 40 %).
# beta in {0.1, 0.25, 0.5, 1} % spans E/E0 ~ 0.8 -> 0.2, the range shown in the paper's Fig. 6.
BUDGET_FRACTIONS = (0.001, 0.0025, 0.005, 0.01)
BUDGET_FIGURE = 0.005             # beta of the maps (Fig. 8)
GA_POP = 20
GA_GENERATIONS = 60
GA_ELITE = 2
GA_TOURNAMENT = 3
GA_CROSSOVER = 0.9
GA_MUTATION = 0.02                # per-gene probability (Gaussian step on the priority key / level flip)
GA_MUTATION_SIGMA = 0.01          # keys are ranks in [0, 1]: 0.01 = ~1 % of the candidates
GA_INIT_SIGMA = (0.002, 0.05)     # noise of the initial copies of the greedy individual
TRANSFER_EPS = 1e-12
TRANSFER_NEIGHBOURS = 5           # clusters outside the GW subset get the IDW score of their 5 nearest

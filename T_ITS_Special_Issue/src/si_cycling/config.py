"""
All parameters of the methodology in one place (values from the T-ITS special-issue paper,
paper/main.tex, unless a comment says otherwise).  The bicycle and car layers, the costs and
the UE solver are the same as in ITSC_2025.
"""
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ----------------------------------------------------------------------------------------
# Bicycle layer: DoS (Table I) and cyclist distance (Section II-C, Cooper 2017)
# ----------------------------------------------------------------------------------------
DOS_NAMES = {
    6: "Dedicated bike lanes, fully separated",
    5: "Shared with pedestrians, separated from cars",
    4: "Mixed traffic, 30 km/h, no bike lane",
    3: "Shared roads with cycling pictograms",
    2: "Painted bike lanes",
    1: "Unmarked shared car-bike lanes",
}
AADT = {6: 14_400, 5: 16_000, 4: 960, 3: 3_120, 2: 6_912, 1: 8_448}   # F(a), veh/day, GTLVille
RHO = 0.04
PSI = 0.84
KAPPA = 0.14               # weight of turns (reported, not used for routing; as ECC/CDC)
PEDESTRIAN_FACTOR = 2.03
SLOPE_TABLE = ((2.0, 1.0), (4.0, 1.371), (6.0, 2.203), (float("inf"), 4.239))

SYMMETRIC_MATRIX_FACTOR = 0.5
BIKE_SHARE = 0.05                   # today's cyclists = 5 % of all trips (as ECC/CDC_2025)
BIKE_NODES_PER_IRIS = 10
V_BIKE_KMH = 15.0                   # v_b, Eq. (14): bike travel time = cyclist distance / v_b

# ----------------------------------------------------------------------------------------
# Upgrade costs (EUR per metre, Cerema) and budget
# ----------------------------------------------------------------------------------------
COST = {
    1: {1: 0, 2: 150, 3: 250, 4: 300, 5: 500, 6: 750},
    2: {2: 0, 3: 100, 4: 150, 5: 350, 6: 600},
    3: {3: 0, 4: 50, 5: 250, 6: 500},
    4: {4: 0, 5: 200, 6: 450},
    5: {5: 0, 6: 250},
    6: {6: 0},
}
BUDGET_FRACTION = 0.15             # Grenoble cycling plan 2025 (24.5 M EUR) ~ 0.15 B_max
GAMMAS = tuple(np.round(np.linspace(0, 1, 11), 2))     # trade-off weights (set Lambda)
LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5)                    # path-sufficiency thresholds
LAMBDA_FIGURES = 0.4                                   # threshold used in the figures
# representative weight of every front used in steps 05-06:
#   "paper"  : gamma*_lambda = lambda + 10 % (illustrative choice of the paper); the utilitarian
#              design uses the same gamma as the sufficientarian one at LAMBDA_FIGURES, so both are
#              compared at equal weight
#   "utopia" : closest point to the utopia point of the min-max normalised front
# both values are stored in results/04_runs.json
GAMMA_STAR_RULE = "paper"
# designs compared in the figures of step 6 (src/si_cycling/selection.py): "paper" | "utopia" | "no_harm"
# "no_harm" (CHANGE): GA-S(lambda) = lowest J_SI(lambda) of its front among the designs that leave no more
# cyclists above 20 min and no more drivers above the delay threshold than GA-U (+ NO_HARM_TOL_PP) and cost
# at most NO_HARM_MAX_TTOT more total travel time than GA-U
REPRESENTATIVE_RULE = "no_harm"
NO_HARM_TOL_PP = 0.02              # percentage points (shares of users), below the model resolution
NO_HARM_MAX_TTOT = 0.005           # +0.5 % total travel time

# ----------------------------------------------------------------------------------------
# Car layer (same as ITSC_2025)
# ----------------------------------------------------------------------------------------
ROAD_CLASSES = {
    "motorway":       dict(speed=90, cap_per_lane=1800),
    "motorway_link":  dict(speed=50, cap_per_lane=1500),
    "trunk":          dict(speed=70, cap_per_lane=1600),
    "trunk_link":     dict(speed=50, cap_per_lane=1200),
    "primary":        dict(speed=50, cap_per_lane=1000),
    "primary_link":   dict(speed=50, cap_per_lane=1000),
    "secondary":      dict(speed=50, cap_per_lane=900),
    "secondary_link": dict(speed=50, cap_per_lane=900),
    "tertiary":       dict(speed=50, cap_per_lane=700),
    "tertiary_link":  dict(speed=50, cap_per_lane=700),
    "unclassified":   dict(speed=30, cap_per_lane=600),
    "minor":          dict(speed=30, cap_per_lane=600),
    "residential":    dict(speed=30, cap_per_lane=600),
    "living_street":  dict(speed=20, cap_per_lane=300),
    "service":        dict(speed=20, cap_per_lane=300),
}
DEFAULT_LANE_WIDTH_M = 3.25
BPR_MU = 0.15
BPR_SIGMA = 4.0
CAR_NODES_PER_IRIS = 4
CAR_SHARE = 0.253                  # bike / (bike + car) = 0.05 / 0.303 = 16.5 % today
PEAK_HOUR_FACTOR = 0.10
DEMAND_MULTIPLIER_BOUNDS = (1.0, 20.0)
UE_METHOD = "bfw"         # bi-conjugate Frank-Wolfe
UE_REL_GAP = 1e-4
UE_MAX_ITER = 400

# ----------------------------------------------------------------------------------------
# Bike lanes vs car capacity (Eq. 17):  theta = theta0 * (1 - omega * z / n)
# ----------------------------------------------------------------------------------------
SEPARATED_DOS = 5
LANE_WIDTH_THRESHOLD_M = 4.5       # z = 1 only if the car lanes are not wider than this
OMEGA = 0.5                        # share of a lane taken by a separated bike lane (paper: 1)
MIN_CAPACITY_SHARE = 0.10          # floor used only with omega = 1 on single-lane roads

# ----------------------------------------------------------------------------------------
# Mode choice (Section II-F)
# ----------------------------------------------------------------------------------------
P_BIKE_2025 = 0.1649               # bike share of the bike + car trips in 2025 (paper)
BETA_PER_KM = -0.886               # beta of the logit (paper, 2015-2025 Grenoble series)

# ----------------------------------------------------------------------------------------
# Sufficientarian measures (Section II-H)
# ----------------------------------------------------------------------------------------
T_THR_BIKE_MIN = 20.0              # bike trip-time threshold (paper: 20 min)
CAR_DELAY_TOLERANCE = 0.25         # car threshold = 1.25 x free-flow OD time (CHANGE, README)
ETA = 1.0                          # weight of trip insufficiency (after normalisation)

# ----------------------------------------------------------------------------------------
# Surrogate and samples
# ----------------------------------------------------------------------------------------
N_SAMPLES = 1000                   # exact UE samples (paper: 912)
SAMPLE_BUDGETS = (0.02, 0.30)      # budget fractions of the random sample configurations
SAMPLE_UE_GAP = 2e-4               # well below the surrogate error
# CHANGE: after the surrogate + feasibility projection, a few bi-conjugate Frank-Wolfe
# iterations warm-started at the projected flows move them towards the equilibrium.
# Surrogate alone: car TTT error +2.6 % on average (max 9 %), as large as the effects we
# measure; + 3 iterations (~3 s): ~0.3 %.  0 = pure surrogate as in the paper.
SURROGATE_UE_ITERS = 3
# which surrogate the optimiser uses:
#   "ridge"    PCA + ridge of step 3 (results/03_surrogate.pkl) with SURROGATE_UE_ITERS
#   "gnn"      graph neural network of step 3b (results/03b_surrogate_gnn.pt)
#   "ensemble" average of both
#   "auto"     model AND number of correction iterations chosen by step 3b on the test samples
#              (results/03b_surrogate_choice.json); falls back to "ridge" if step 3b was not run
SURROGATE_MODEL = "auto"
SURROGATE_TTT_TOLERANCE = 0.003    # step 3b: accepted mean |car TTT error| (0.3 %)

# ----------------------------------------------------------------------------------------
# Genetic algorithm
# ----------------------------------------------------------------------------------------
GA_POP = 16
GA_GENERATIONS = 12

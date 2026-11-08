"""
All parameters of the methodology in one place (values from the ITSC 2025 paper unless a
comment says otherwise).
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
AADT = {6: 14_400, 5: 16_000, 4: 960, 3: 3_120, 2: 6_912, 1: 8_448}   # veh/day, GTLVille
RHO = 0.04                 # exponent of the traffic factor, Eq. (1)
PSI = 0.84                 # psi, Eq. (2)
KAPPA = 0.14               # weight of turns (reported, not used for routing; as ECC/CDC)
PEDESTRIAN_FACTOR = 2.03   # DoS 5: shared with pedestrians (Mesimaki 2021)
SLOPE_TABLE = ((2.0, 1.0), (4.0, 1.371), (6.0, 2.203), (float("inf"), 4.239))  # Table III

# Cyclist demand: EMD 2020 IRIS x IRIS (all modes, symmetric matrix) x 0.5 x bike share
SYMMETRIC_MATRIX_FACTOR = 0.5
BIKE_SHARE = 0.05                   # same as ECC_2025 / CDC_2025 -> 22,199 bike trips/day
BIKE_NODES_PER_IRIS = 10            # representative trip ends per IRIS (as CDC_2025)

# ----------------------------------------------------------------------------------------
# Upgrade costs (EUR per metre, Cerema) and budgets
# ----------------------------------------------------------------------------------------
COST = {
    1: {1: 0, 2: 150, 3: 250, 4: 300, 5: 500, 6: 750},
    2: {2: 0, 3: 100, 4: 150, 5: 350, 6: 600},
    3: {3: 0, 4: 50, 5: 250, 6: 500},
    4: {4: 0, 5: 200, 6: 450},
    5: {5: 0, 6: 250},
    6: {6: 0},
}
ALPHAS = (1 / 3, 2 / 5, 1 / 2)                       # B = alpha * B_max  (set Phi)
EPSILONS = tuple(np.round(np.linspace(0, 1, 11), 2))  # epsilon-constraint weights (set Lambda)

# ----------------------------------------------------------------------------------------
# Car layer (Section II-D): OSM drive network, BPR, user equilibrium
# ----------------------------------------------------------------------------------------
# free-flow speed (km/h, if no maxspeed tag) and capacity per lane (veh/h) by road class
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
BPR_MU = 0.15              # mu of the BPR function, Eq. (6)
BPR_SIGMA = 4.0            # sigma
CAR_NODES_PER_IRIS = 4     # representative car trip ends per IRIS (k-means spread)

# Car demand of the modelled period = morning peak hour (veh/h):
#   q_c = OD trips x 0.5 x CAR_SHARE x PEAK_HOUR_FACTOR x m
# CAR_SHARE: car-driver share chosen so that bike / (bike + car) = 16.5 %, the 2025 bicycle
#            share used in the T-ITS special-issue paper (0.05 / 0.303).
# m:         demand multiplier for the traffic NOT in the intra-city EMD matrix (exchange and
#            through traffic).  Calibrated by 02_car_user_equilibrium.py so that the modelled
#            flows on the roads next to DoS 1-3 bike lanes match the GTLVille AADT of Table I.
CAR_SHARE = 0.253
PEAK_HOUR_FACTOR = 0.10
DEMAND_MULTIPLIER_BOUNDS = (1.0, 20.0)

# user-equilibrium solver (bi-conjugate Frank-Wolfe)
UE_METHOD = "bfw"          # 'bfw', 'cfw' or 'fw'
UE_REL_GAP = 1e-4          # stop when the relative gap is below this value
UE_MAX_ITER = 400

# ----------------------------------------------------------------------------------------
# Interaction bikes <-> cars (Section II-E)
# ----------------------------------------------------------------------------------------
# A bike arc of DoS <= 4 that is upgraded to DoS >= 5 and runs along a car road reduces the
# capacity of that road (both directions): theta_new = CAPACITY_FACTOR * theta (paper: 1/2).
CAPACITY_FACTOR = 0.5
SEPARATED_DOS = 5

# ----------------------------------------------------------------------------------------
# Genetic algorithm
# ----------------------------------------------------------------------------------------
GA_POP = 16
GA_GENERATIONS = 12

"""
All parameters of the methodology in one place (values from the CDC 2025 paper).
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
# Degree of separation (DoS) - Table I
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

# ----------------------------------------------------------------------------------------
# Lower level: cyclist distance (Section II-C, Cooper 2017)
# ----------------------------------------------------------------------------------------
RHO = 0.04                 # exponent of the traffic factor, Eq. (1)
PSI = 0.84                 # psi in Eq. (2)
KAPPA = 0.14               # weight of the angular change, Eq. (1)
PEDESTRIAN_FACTOR = 2.03   # DoS 5: shared with pedestrians (Mesimaki 2021)
SLOPE_TABLE = ((2.0, 1.0), (4.0, 1.371), (6.0, 2.203), (float("inf"), 4.239))  # Table III

# ----------------------------------------------------------------------------------------
# Upgrade costs (EUR per metre, Cerema) - matrix Pi of Section III-B
# ----------------------------------------------------------------------------------------
COST = {
    1: {1: 0, 2: 150, 3: 250, 4: 300, 5: 500, 6: 750},
    2: {2: 0, 3: 100, 4: 150, 5: 350, 6: 600},
    3: {3: 0, 4: 50, 5: 250, 6: 500},
    4: {4: 0, 5: 200, 6: 450},
    5: {5: 0, 6: 250},
    6: {6: 0},
}

# Budgets of the sensitivity analysis: 10 values, logarithmically spaced between
# B_max/100 and B_max/2 (the values printed in the paper: 0.01, 0.015, ..., 0.323, 0.5).
BUDGET_FRACTIONS = tuple(np.round(np.logspace(np.log10(0.01), np.log10(0.5), 10), 5))

# ----------------------------------------------------------------------------------------
# Demand (EMD 2020, IRIS x IRIS, all modes, symmetric)
# ----------------------------------------------------------------------------------------
SYMMETRIC_MATRIX_FACTOR = 0.5      # every trip is stored as (r, s) and (s, r)
BIKE_SHARE = 0.05                  # bike modal share
DEMAND_FACTOR = SYMMETRIC_MATRIX_FACTOR * BIKE_SHARE
# Representative nodes per IRIS used as trip origins/destinations (speed-up, see README).
# None = every node with a demand weight (11.7 M OD pairs, ~10x slower).
NODES_PER_IRIS = 10

# ----------------------------------------------------------------------------------------
# Sufficientarian upper level (Section II-E)
# ----------------------------------------------------------------------------------------
EPSILON = 0.01                     # safety score s_p = 1 / (E_p + epsilon)  -> s_p <= 100
LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5)   # threshold s_thr = lambda * s_G
# weight of J_E in the GA-S objective J_I + eta * J_E.  The paper text says 0.5, but its
# Fig. 6 values and the 2024 code use 0.05 (with 0.5 the exposure term is ~10x larger than
# J_I and GA-S becomes a copy of GA-U).
ETA = 0.05

"""
All parameters of the methodology in one place (values from the paper).
"""
from pathlib import Path

# ----------------------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# ----------------------------------------------------------------------------------------
# Degree of separation (DoS) classes - Table I of the paper
# ----------------------------------------------------------------------------------------
DOS_LEVELS = (1, 2, 3, 4, 5, 6)
DOS_NAMES = {
    6: "Dedicated bike lanes, fully separated",
    5: "Shared with pedestrians, separated from cars",
    4: "Mixed traffic, 30 km/h, no bike lane",
    3: "Shared roads with cycling pictograms",
    2: "Painted bike lanes",
    1: "Unmarked shared car-bike lanes",
}
# AADT of the car lanes adjacent to each DoS class (veh/day), Table I (GTLVille, Grenoble)
AADT = {6: 14_400, 5: 16_000, 4: 960, 3: 3_120, 2: 6_912, 1: 8_448}

# ----------------------------------------------------------------------------------------
# Cyclist-distance parameters - Section IV-A (calibration of Cooper 2017)
# ----------------------------------------------------------------------------------------
RHO = 0.04                 # exponent of the traffic factor in Eq. (1)
PSI = 0.84                 # psi in Eq. (2)
KAPPA = 0.14               # weight of the angular change (degrees) in Eq. (1)
PEDESTRIAN_FACTOR = 2.03   # DoS 5 (shared with pedestrians): 2.03x crash risk (Mesimaki 2021)

# Slope factor s_a (Table IV): (upper bound of the slope in %, uphill factor).
# Downhill arcs use the reciprocal of the uphill factor.
SLOPE_TABLE = ((2.0, 1.0), (4.0, 1.371), (6.0, 2.203), (float("inf"), 4.239))

# ----------------------------------------------------------------------------------------
# Upgrade costs (EUR per metre) from Cerema, COST[from DoS][to DoS]
# ----------------------------------------------------------------------------------------
COST = {
    1: {1: 0, 2: 150, 3: 250, 4: 300, 5: 500, 6: 750},
    2: {2: 0, 3: 100, 4: 150, 5: 350, 6: 600},
    3: {3: 0, 4: 50, 5: 250, 6: 500},
    4: {4: 0, 5: 200, 6: 450},
    5: {5: 0, 6: 250},
    6: {6: 0},
}
BETA = 0.5                 # budget B = BETA * B_max (Section V: 50 % of the full-upgrade budget)

# ----------------------------------------------------------------------------------------
# Demand (see README, section "Demand")
# ----------------------------------------------------------------------------------------
# od_iris_grenoble.csv holds the symmetric EMD-2020 IRIS x IRIS matrix of ALL modes.
# Each trip appears in both (r, s) and (s, r) -> x 0.5;  bike modal share 5 % -> x 0.05.
SYMMETRIC_MATRIX_FACTOR = 0.5
BIKE_SHARE = 0.05
DEMAND_FACTOR = SYMMETRIC_MATRIX_FACTOR * BIKE_SHARE

# "full_od"         : EMD 2020 IRIS demand spread over all node pairs (used in the paper)
# "historical_unit" : optional, the newer IRIS -> facility OD (one unit per pair, not trips)
DEFAULT_DEMAND = "full_od"

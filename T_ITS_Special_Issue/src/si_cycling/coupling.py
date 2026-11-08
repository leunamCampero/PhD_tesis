"""
Interaction between new bike lanes and car capacity (Section II-E, Eq. 17 of the paper).

    theta_a = theta0_a * (1 - omega * z_a / n_a)

z_a = 1 if a bike arc running along car road a is upgraded from DoS <= 4 (on the road) to
DoS >= 5 (physically separated) AND the lanes of the road are not wider than 4.5 m (French
standard, CEREMA 2020; wider lanes can host the bike lane without removing car space);
n_a = car lanes per direction.  The reduction applies to both directions of the road.

omega (CHANGE, "moderated impact"): the paper uses omega = 1, i.e. one full lane is removed.
On Grenoble's roads with ONE lane per direction (88 % of the length) this gives theta = 0, i.e. the road
is closed to cars (the 2024 code then silently reset such capacities to 600 veh/h).  We use
omega = 0.5 by default: a separated bike lane takes about half a lane (the ITSC 2025 paper's
1/2 factor), so a single-lane road keeps half its capacity and a two-lane road 75 %.  The
paper's rule is available with ``omega=1`` (capacities floored at MIN_CAPACITY_SHARE).
"""
import numpy as np

from . import config as C


class Coupling:
    """Maps a bike DoS vector to the car capacities."""

    def __init__(self, bike, car, omega=C.OMEGA):
        self.bike, self.car, self.omega = bike, car, omega
        row = np.full(bike.E, -1)
        ok = bike.car_edge >= 0
        row[ok] = car.row_of_edge_id[bike.car_edge[ok]]
        narrow = car.lane_width[: car.E] <= C.LANE_WIDTH_THRESHOLD_M
        cand = (row >= 0) & (bike.dos0 < C.SEPARATED_DOS)
        cand[cand] &= narrow[row[cand]]
        self.cand = np.flatnonzero(cand)
        self.cand_road = row[self.cand]
        self.roads = np.unique(self.cand_road)                 # roads that may lose capacity
        n = car.lanes[: car.E].astype(float)
        self.road_factor = np.maximum(1.0 - omega / n, C.MIN_CAPACITY_SHARE)   # per road

    def reduced_roads(self, dos):
        up = np.asarray(dos)[self.cand] >= C.SEPARATED_DOS
        return np.unique(self.cand_road[up])

    def capacity(self, dos):
        """Directed car-arc capacities (2 E) for the bike configuration ``dos``."""
        cap = self.car.capacity0.copy()
        r = self.reduced_roads(dos)
        cap[r] *= self.road_factor[r]
        cap[r + self.car.E] *= self.road_factor[r]
        return cap

    def capacity_if_all(self, cap0):
        """Capacities if every candidate road were reduced (used for greedy penalties)."""
        cap = cap0.copy()
        r = self.roads
        cap[r] *= self.road_factor[r]
        cap[r + self.car.E] *= self.road_factor[r]
        return cap

    def road_of_bike_edge(self):
        out = np.full(self.bike.E, -1)
        out[self.cand] = self.cand_road
        return out

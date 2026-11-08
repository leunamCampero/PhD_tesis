"""
Interaction between new bike lanes and car capacity (Section II-E, Eq. 9 of the paper).

A bike arc that is upgraded from DoS <= 4 (on the road) to DoS >= 5 (physically separated)
needs road space.  If the arc runs along a car road (geometric matching, tools/build_data.py)
the capacity of that road is reduced in BOTH directions:

    theta_new = CAPACITY_FACTOR * theta        (paper: 1/2, half of a lane)

Bike arcs that are already DoS 5/6 are already part of today's road layout, and bike arcs
that do not run along a car road (parks, pedestrian streets, ...) never change car capacity.
"""
import numpy as np

from . import config as C


class Coupling:
    """Maps a bike DoS vector to the car capacities."""

    def __init__(self, bike, car, factor=C.CAPACITY_FACTOR):
        self.bike, self.car, self.factor = bike, car, factor
        row = np.full(bike.E, -1)
        ok = bike.car_edge >= 0
        row[ok] = car.row_of_edge_id[bike.car_edge[ok]]
        # candidates: on-road bike arcs (DoS <= 4) along a car road
        self.cand = np.flatnonzero((row >= 0) & (bike.dos0 < C.SEPARATED_DOS))
        self.cand_road = row[self.cand]                   # car road (row) of each candidate

    def reduced_roads(self, dos):
        """Rows of the car roads whose capacity is reduced by ``dos``."""
        up = np.asarray(dos)[self.cand] >= C.SEPARATED_DOS
        return np.unique(self.cand_road[up])

    def capacity(self, dos):
        """Directed car-arc capacities (2 E) for the bike configuration ``dos``."""
        cap = self.car.capacity0.copy()
        r = self.reduced_roads(dos)
        cap[r] *= self.factor
        cap[r + self.car.E] *= self.factor
        return cap

    def road_of_bike_edge(self):
        """Car road row of every candidate bike arc (array over bike edges, -1 if none)."""
        out = np.full(self.bike.E, -1)
        out[self.cand] = self.cand_road
        return out

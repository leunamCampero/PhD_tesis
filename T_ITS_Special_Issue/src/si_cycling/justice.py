"""
Sufficientarian measures (Section II-H of the paper).

Exposure-based path sufficiency (Eq. 30-35)
    S_p   = 1 / ( sum_{a in p} f_a d_a / sum_{a in p} d_a  + 1 )        in (0, 1)
    S_G   = mean of S_p over the OD pairs of TODAY's network
    S_thr = lambda * S_G,  lambda in {10, ..., 50} %
    I_p   = max(0, (S_thr - S_p) / S_thr)
    I_r   = sum_{u in r} q_uv I_p / sum_{u in r} q_uv        (trips leaving region r)
    J_SI  = sum_r n_r I_r / sum_r n_r                        (n_r = cyclists of region r)

Trip (travel-time) insufficiency (Eq. 36-39)
    bike:  t_p = b_p / v_b (cyclist distance / 15 km/h),  threshold T_thr = 20 min (paper)
    car:   t_p = UE travel time of the OD,  threshold = (1 + delta) x free-flow time of the OD
           (CHANGE: with an absolute 20 min no car trip of the city is insufficient - the
           longest peak-hour car trip takes 14 min - so the driver term would vanish; a
           relative "acceptable delay" threshold, delta = 25 %, measures exactly the harm that
           capacity reallocation does to drivers.)
    I = max(0, (t - T_thr) / T_thr), aggregated like J_SI -> J_TI^b and J_TI^c.

Regions = the 70 IRIS.  The thresholds are fixed on today's network, so that every solution
is judged against the same "sufficient" level.
"""
import numpy as np

from . import config as C
from .bike import traffic_factor
from .trees import path_sums


class Regions:
    """Demand-weighted regional aggregation (I_r and the population-weighted J)."""

    def __init__(self, Q, row_zone):
        self.Q = Q
        self.zones = np.unique(row_zone)
        self.rows = [row_zone == z for z in self.zones]
        self.n_r = np.array([Q[r].sum() for r in self.rows])

    def regional(self, I):
        """I_r for every region (trips leaving r, weighted by their demand)."""
        IQ = I * self.Q
        return np.array([IQ[r].sum() / n if n > 0 else 0.0 for r, n in zip(self.rows, self.n_r)])

    def network(self, I):
        """J = sum_r n_r I_r / sum_r n_r  (= demand-weighted mean of I)."""
        return float(np.sum(self.n_r * self.regional(I)) / self.n_r.sum())

    def share_below(self, I):
        """Share of trips that are insufficient (I > 0)."""
        return float(self.Q[(I > 0) & (self.Q > 0)].sum() / self.Q.sum())


class Sufficiency:
    """Thresholds of today's network + all sufficientarian indicators."""

    def __init__(self, bike, car, ev0, car_free_flow_od):
        self.bike, self.car = bike, car
        self.mask = bike.Q > 0
        self.bike_regions = Regions(bike.Q, bike.source_zone)
        self.car_regions = Regions(car.Q_base, car.source_zone)
        S0 = self.path_safety(ev0)
        self.S_G = float(S0[self.mask].mean())
        self.S_thr = {lam: lam * self.S_G for lam in C.LAMBDAS}
        self.T_thr_bike = C.T_THR_BIKE_MIN
        self.T_thr_car = (1 + C.CAR_DELAY_TOLERANCE) * car_free_flow_od   # (S_c x S_c), min

    # ---- exposure-based path sufficiency ------------------------------------------------
    def path_safety(self, ev):
        """S_p of every OD path (S x S)."""
        b = self.bike
        L = np.tile(b.length, 2)
        fd = path_sums(ev["pred"], b.arc_index, np.tile(traffic_factor(ev["dos"]), 2) * L)[:, b.sources]
        d = path_sums(ev["pred"], b.arc_index, L)[:, b.sources]
        with np.errstate(invalid="ignore", divide="ignore"):
            fbar = np.where(d > 0, fd / d, 0.0)
        return 1.0 / (fbar + 1.0)

    def path_insufficiency(self, S, lam):
        thr = self.S_thr[lam]
        return np.maximum(0.0, (thr - S) / thr) * self.mask

    def J_SI(self, S, lam):
        return self.bike_regions.network(self.path_insufficiency(S, lam))

    # ---- trip (time) insufficiency ------------------------------------------------------
    def bike_time(self, ev):
        """Bicycle OD travel times (min): cyclist distance / v_b."""
        return ev["od_distance"] / (C.V_BIKE_KMH / 3.6) / 60.0

    def bike_time_insufficiency(self, t_b):
        return np.maximum(0.0, (t_b - self.T_thr_bike) / self.T_thr_bike) * self.mask

    def car_time_insufficiency(self, t_c):
        thr = self.T_thr_car
        ok = (self.car.Q_base > 0) & (thr > 0)              # diagonal (thr = 0) has no trips
        return np.where(ok, np.maximum(0.0, t_c - thr) / np.where(ok, thr, 1.0), 0.0)

    def J_TI_bike(self, t_b):
        return self.bike_regions.network(self.bike_time_insufficiency(t_b))

    def J_TI_car(self, t_c):
        return self.car_regions.network(self.car_time_insufficiency(t_c))

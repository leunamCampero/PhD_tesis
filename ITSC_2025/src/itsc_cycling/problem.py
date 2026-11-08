"""
Problem 2 of the paper (Trade-off network upgrade) and its normalisation.

    min_{w in Omega}  eps * E_hat(w) + (1 - eps) * T_hat(w)                         Eq. (11)
    s.t.  sum_a d_a pi(w0_a, w_a) <= B                                             Eq. (12)
          theta_a(w) = 1/2 theta_a  if a bike arc along road a is upgraded to DoS >= 5  (13)
          bike flows = AON on the cyclist distance (lower level)                    (14)

    E(w)  = J_E = sum_a d_hat f_hat phi_hat           cyclist exposure (bike.py)
    T(w)  = sum_a phi*_a t_a(phi*_a; theta_a(w))      car TTT with the PRE-upgrade UE flows
                                                       phi* of Problem 1 (drivers do not
                                                       re-route during the optimisation)

Normalisation (change w.r.t. the paper, see README).  E and T have different units and very
different ranges (E drops by ~80 %, T rises by a few %).  Dividing by the original values
makes (1 - eps) T_hat almost constant, so eps would only matter close to 0.  We therefore
use the ideal / nadir normalisation of multi-objective optimisation:

    E_hat = (E - E_ideal) / (E_nadir - E_ideal),   T_hat = (T - T_ideal) / (T_nadir - T_ideal)

where the ideal and nadir values come from the two extreme runs eps = 1 (min E) and eps = 0
(min T) of the same budget.  Both terms are then in ~[0, 1] and eps spans the whole front.
The extreme runs use a lexicographic tie-break (1e-6 x the other objective) so that, e.g.,
eps = 0 returns the car-neutral solution with the lowest exposure.
"""
from functools import lru_cache

import numpy as np

from . import config as C
from . import costs, utils
from .bike import BikeNetwork
from .cars import CarNetwork
from .costs import COST_MATRIX
from .coupling import Coupling

TIE = 1e-6


class TradeOff:
    """Everything needed to evaluate a bike configuration for Problem 2."""

    def __init__(self, ue_file=None):
        ue = utils.load_ue(ue_file)                        # results of 02_car_user_equilibrium
        self.bike = BikeNetwork()
        self.car = CarNetwork(demand_multiplier=ue["multiplier"])
        self.cpl = Coupling(self.bike, self.car)
        self.phi = ue["x"]                                  # pre-upgrade UE flows phi*
        self.ev0 = self.bike.evaluate(self.bike.dos0)
        self.E0 = self.ev0["J_E"]
        self.T0 = self.car.total_travel_time(self.phi)
        self.B_max = costs.max_budget(self.bike)

    # ------------------------------------------------------------------------------------
    def car_ttt(self, dos):
        """T(w): car TTT with fixed pre-upgrade flows and the reduced capacities (veh.h)."""
        return self.car.total_travel_time(self.phi, self.cpl.capacity(dos))

    def measures(self, dos):
        dos = np.asarray(dos, int)
        roads = self.cpl.reduced_roads(dos)
        return dict(J_E=self.bike.evaluate(dos)["J_E"], TTT=self.car_ttt(dos),
                    reduced_roads=int(len(roads)),
                    reduced_road_km=float(self.car.length[roads].sum() / 1000),
                    budget_used=costs.upgrade_cost(self.bike, dos) / self.B_max)

    def objective(self, eps, norm=None):
        """Function dos -> objective of Problem 2 for weight eps (lower is better)."""
        if norm is None:                                    # extreme runs (phase 1)
            if eps == 1:
                return lambda dos: self.bike.evaluate(dos)["J_E"] / self.E0 + TIE * self.car_ttt(dos) / self.T0
            if eps == 0:
                return lambda dos: self.car_ttt(dos) / self.T0 + TIE * self.bike.evaluate(dos)["J_E"] / self.E0
            raise ValueError("intermediate eps needs the ideal / nadir normalisation")
        dE = max(norm["E_nadir"] - norm["E_ideal"], 1e-12)
        dT = max(norm["T_nadir"] - norm["T_ideal"], 1e-12)

        def f(dos):
            E = self.bike.evaluate(dos)["J_E"]
            T = self.car_ttt(dos)
            return eps * (E - norm["E_ideal"]) / dE + (1 - eps) * (T - norm["T_ideal"]) / dT
        return f

    # ------------------------------------------------------------------------------------
    # greedy scores used to seed the GA
    # ------------------------------------------------------------------------------------
    def road_penalty(self):
        """Increase of T if the road along each bike arc were the only one halved (veh.h)."""
        car, phi = self.car, self.phi
        t_full = car.time(phi, car.capacity0)
        t_half = car.time(phi, car.capacity0 * C.CAPACITY_FACTOR)
        dT_road = car.two_way(phi * (t_half - t_full)) / 60.0        # per road
        out = np.zeros(self.bike.E)
        road = self.cpl.road_of_bike_edge()
        ok = road >= 0
        out[ok] = dT_road[road[ok]]
        return out

    def seeds(self, eps, norm=None):
        """Greedy scores: Algorithm 1, exposure saved per euro, and the car-aware blend."""
        bike = self.bike
        cost6 = bike.length * COST_MATRIX[bike.dos0, 6]
        gain = self.ev0["x_hat"] * bike.exposure_arc(bike.dos0)       # ~ exposure removed
        util = np.where(cost6 > 0, gain / np.maximum(cost6, 1e-9), -1.0)
        pen = self.road_penalty()
        car_free = np.where(pen > 0, -1.0, util)
        dE = (norm["E_nadir"] - norm["E_ideal"]) if norm else self.E0
        dT = (norm["T_nadir"] - norm["T_ideal"]) if norm else 0.05 * self.T0
        blend = np.where(cost6 > 0, (eps * gain / dE - (1 - eps) * pen / dT) / np.maximum(cost6, 1e-9), -1.0)
        from .ga import algorithm1_priority
        out = [algorithm1_priority(bike), util, car_free, blend]
        if eps == 1:
            out = [algorithm1_priority(bike), util]
        return out


@lru_cache(maxsize=1)
def context():
    """One TradeOff per process (for parallel runs)."""
    return TradeOff()

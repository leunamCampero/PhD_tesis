"""
The two bi-level problems of the paper and the evaluation of a bike configuration.

``MultiModal.evaluate(dos)`` chains the whole model for one DoS vector w:
    1. bicycle AON assignment (lower level)        -> flows, cyclist distances, J_E
    2. mode choice                                 -> P_b(w), bike / car demand scales
    3. car capacities (coupling, Eq. 17)           -> theta(w)
    4. car flows: surrogate + projection + SURROGATE_UE_ITERS Frank-Wolfe correction
       iterations (default) or exact UE (``exact=True``)
    5. T_c (car TTT), car OD times, T_b (bike), T_tot = T_b + T_c         (Eq. 16, 28)
    6. sufficientarian measures J_SI(lambda), J_TI^b, J_TI^c              (Eq. 30-39)

Problem 2 (utilitarian)       min  gamma * J_E_hat + (1 - gamma) * T_tot_hat
Problem 3 (sufficientarian)   min  gamma * J_SI_hat(lambda) + (1 - gamma) * eta * J_TI_hat
                                   with J_TI = J_TI^b + J_TI^c
subject to the budget (0.15 B_max) and the capacity rule.

Normalisation (as in ITSC_2025, see README): every term is scaled with the ideal / nadir
values of the two extreme runs gamma = 1 and gamma = 0 of the same problem (and lambda), so
both terms lie in ~[0, 1] and gamma spans the whole Pareto front.

All car quantities are for the morning peak hour; bike demand is converted to the peak
hour with the same peak-hour factor, so T_tot is in traveller-hours per peak hour.
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import config as C
from . import costs, utils
from .bike import BikeNetwork
from .cars import CarNetwork
from .costs import COST_MATRIX
from .coupling import Coupling
from .justice import Sufficiency
from .modeshift import ModeChoice

TIE = 1e-6


class MultiModal:
    def __init__(self, surrogate=True):
        ue = utils.load_ue()
        self.bike = BikeNetwork()
        self.car = CarNetwork(demand_multiplier=ue["multiplier"])
        self.cpl = Coupling(self.bike, self.car)
        self.phi0 = ue["x"]
        self.ev0 = self.bike.evaluate(self.bike.dos0)
        self.mode = ModeChoice(self.bike, self.ev0, ue["multiplier"])
        A = csr_matrix((self.car.t0, (self.car.tail, self.car.head)), shape=(self.car.N,) * 2)
        self.car_ff_od = dijkstra(A, indices=self.car.sources)[:, self.car.sources]
        self.suf = Sufficiency(self.bike, self.car, self.ev0, self.car_ff_od)
        self.B_max = costs.max_budget(self.bike)
        self.surrogate = self.projector = None
        self.ue_iters = C.SURROGATE_UE_ITERS
        if surrogate:
            from .surrogate import CarSurrogate, FlowProjector
            if C.SURROGATE_MODEL == "ridge":                 # no PyTorch needed
                self.surrogate = CarSurrogate.load(C.RESULTS / "03_surrogate.pkl")
            else:                                            # gnn / ensemble / auto (step 3b)
                from .gnn import load_surrogate
                self.surrogate, self.ue_iters = load_surrogate()
            self.projector = FlowProjector(self.car)
        self.base = self.evaluate(self.bike.dos0, exact=True, x_exact=self.phi0)

    # ------------------------------------------------------------------------------------
    def car_od_times(self, t):
        A = csr_matrix((t, (self.car.tail, self.car.head)), shape=(self.car.N,) * 2)
        return dijkstra(A, indices=self.car.sources)[:, self.car.sources]

    def car_flows(self, cap, scale, exact=False, x_exact=None):
        """Car UE flows for capacities ``cap`` and demand scale ``scale``."""
        if x_exact is not None:
            return x_exact
        if exact:
            return self.car.solve_ue(cap=cap, scale=scale, x0=self.phi0 * scale)["x"]
        red = 1.0 - (cap[: self.car.E] / self.car.capacity0[: self.car.E])[self.surrogate.cand_roads]
        phi_hat = self.surrogate.predict_raw(red, scale)[0]
        phi = self.projector.project(phi_hat, self.projector.balance(scale))
        if self.ue_iters > 0:                  # warm-started equilibrium correction (config / step 3b)
            phi = self.car.solve_ue(cap=cap, scale=scale, x0=phi, rel_gap=1e-9,
                                    max_iter=self.ue_iters)["x"]
        return phi

    def evaluate(self, dos, exact=False, x_exact=None, lams=None):
        dos = np.asarray(dos, int)
        bike, car, suf = self.bike, self.car, self.suf
        ev = bike.evaluate(dos)
        p_b, s_b, s_c = self.mode.scales(ev)
        cap = self.cpl.capacity(dos)
        x = self.car_flows(cap, s_c, exact, x_exact)
        t = car.time(x, cap)
        od_c = self.car_od_times(t)
        t_b = suf.bike_time(ev)
        T_c = float(np.sum(x * t)) / 60.0                                        # veh.h
        T_b = float(np.sum(bike.Q * s_b * C.PEAK_HOUR_FACTOR * t_b)) / 60.0       # bike.h
        S = suf.path_safety(ev)
        out = dict(dos=dos, J_E=ev["J_E"], P_b=p_b, D_bar=self.mode.mean_distance(ev),
                   bike_scale=s_b, car_scale=s_c, T_c=T_c, T_b=T_b, T_tot=T_c + T_b,
                   J_TI_b=suf.J_TI_bike(t_b), J_TI_c=suf.J_TI_car(od_c),
                   reduced_roads=int(len(self.cpl.reduced_roads(dos))),
                   budget_used=costs.upgrade_cost(bike, dos) / self.B_max)
        out["J_TI"] = out["J_TI_b"] + out["J_TI_c"]
        for lam in (C.LAMBDAS if lams is None else lams):
            out[f"J_SI_{lam:g}"] = suf.J_SI(S, lam)
        out["_arrays"] = dict(S=S, t_b=t_b, t_c=od_c, x_car=x, x_bike=ev["flows"])
        return out

    # ------------------------------------------------------------------------------------
    # objectives
    # ------------------------------------------------------------------------------------
    @staticmethod
    def terms(kind, lam=None):
        if kind == "util":
            return "J_E", "T_tot"
        return f"J_SI_{lam:g}", "J_TI"

    def objective(self, kind, gamma, lam=None, norm=None):
        """Function dos -> objective (lower is better) of Problem 2 ('util') or 3 ('suff')."""
        a, b = self.terms(kind, lam)
        lams = None if lam is None else (lam,)
        if norm is None:                       # extreme runs: lexicographic tie-break
            if gamma not in (0, 1):
                raise ValueError("intermediate gamma needs the ideal / nadir normalisation")
            first, second = (a, b) if gamma == 1 else (b, a)
            ref = {k: max(abs(self.base[k]), 1e-9) for k in (a, b)}

            def f(dos):
                r = self.evaluate(dos, lams=lams)
                return r[first] / ref[first] + TIE * r[second] / ref[second]
            return f
        da = max(norm[a + "_nadir"] - norm[a + "_ideal"], 1e-12)
        db = max(norm[b + "_nadir"] - norm[b + "_ideal"], 1e-12)
        eta = C.ETA if kind == "suff" else 1.0

        def f(dos):
            r = self.evaluate(dos, lams=lams)
            return gamma * (r[a] - norm[a + "_ideal"]) / da + (1 - gamma) * eta * (r[b] - norm[b + "_ideal"]) / db
        return f

    # ------------------------------------------------------------------------------------
    # greedy scores used to seed the GA
    # ------------------------------------------------------------------------------------
    def road_penalty(self):
        """Car-time increase if the road along each bike arc were the only one reduced."""
        car, phi = self.car, self.phi0
        cap1 = self.cpl.capacity_if_all(car.capacity0)
        dT = car.two_way(phi * (car.time(phi, cap1) - car.time(phi, car.capacity0))) / 60.0
        out = np.zeros(self.bike.E)
        road = self.cpl.road_of_bike_edge()
        ok = road >= 0
        out[ok] = dT[road[ok]]
        return out

    def seeds(self, kind, gamma, lam=None):
        """Greedy scores: Algorithm 1, MLCC warm start, exposure / insufficiency per euro,
        car-free and car-aware variants."""
        from .ga import algorithm1_priority
        from .mlcc import mlcc_priority
        bike = self.bike
        cost6 = bike.length * COST_MATRIX[bike.dos0, 6]
        e = bike.exposure_arc(bike.dos0)
        util_gain = self.ev0["x_hat"] * e
        if kind == "suff":
            # flow of the trips that are insufficient today (path safety or bike time), I_p-weighted
            S0 = self.base["_arrays"]["S"]
            I = self.suf.path_insufficiency(S0, lam) + self.suf.bike_time_insufficiency(self.base["_arrays"]["t_b"])
            x_ins, _, _ = bike.assign(bike.dos0, Q=bike.Q * I)
            gain = x_ins / max(x_ins.max(), 1e-12) * e + 0.05 * util_gain
        else:
            gain = util_gain
        pen = self.road_penalty()
        per_eur = lambda g: np.where(cost6 > 0, g / np.maximum(cost6, 1e-9), -1.0)  # noqa: E731
        score = per_eur(gain)
        car_free = np.where(pen > 0, -1.0, score)
        rel_pen = pen / max(self.base["T_c"], 1e-9)
        blend = per_eur(gamma * gain / max(gain.max(), 1e-12) - (1 - gamma) * 50 * rel_pen)
        out = [algorithm1_priority(bike), score, car_free, blend, mlcc_priority(self, C.BUDGET_FRACTION)]
        if kind == "suff":
            out.append(per_eur(util_gain))
        return out


@lru_cache(maxsize=1)
def context():
    """One MultiModal problem (with surrogate) per process."""
    return MultiModal(surrogate=True)

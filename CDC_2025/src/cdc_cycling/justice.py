"""
Sufficientarian measures (Section II-E of the paper).

    path safety          s_p   = 1 / (E_p + epsilon)                            (<= 100)
    average safety       s_G   = mean of s_p over the OD paths of the ORIGINAL network
    threshold            s_thr = lambda * s_G,   lambda in {10, 20, 30, 40, 50} %
    path insufficiency   I_p   = max(0, (s_thr - s_p) / s_thr)
    regional             I_r   = sum_{u in r, v} q_uv I_p / sum_{u in r, v} q_uv
    network              J_I   = sum_r n_r I_r / sum_r n_r,  n_r = cyclists leaving IRIS r
    GA-S objective             J_I + eta * J_E

The thresholds are fixed once, on the original network, so that every solution is judged
against the same "sufficient" safety level.
"""
import numpy as np

from . import config as C


def path_safety(path_exposure):
    return 1.0 / (path_exposure + C.EPSILON)


class Sufficiency:
    """Thresholds of the original network + all sufficientarian indicators."""

    def __init__(self, net, original_eval):
        self.net = net
        self.Q = net.Q
        self.mask = self.Q > 0                                   # OD pairs with trips
        s0 = path_safety(original_eval["path_exposure"])
        self.s_G = float(s0[self.mask].mean())                   # average path safety
        self.thresholds = {lam: lam * self.s_G for lam in C.LAMBDAS}
        # cyclists of each IRIS (trips leaving it), for J_I
        self.zone_of_row = net.source_zone
        self.zones = np.unique(self.zone_of_row)
        self.n_r = np.array([self.Q[self.zone_of_row == z].sum() for z in self.zones])

    # ------------------------------------------------------------------------------------
    def path_insufficiency(self, ev, lam):
        thr = self.thresholds[lam]
        return np.maximum(0.0, (thr - path_safety(ev["path_exposure"])) / thr)

    def regional(self, ev, lam):
        """I_r for every IRIS (array aligned with self.zones)."""
        I = self.path_insufficiency(ev, lam) * self.Q
        out = np.zeros(len(self.zones))
        for i, z in enumerate(self.zones):
            rows = self.zone_of_row == z
            out[i] = I[rows].sum() / self.Q[rows].sum() if self.Q[rows].sum() > 0 else 0.0
        return out

    def J_I(self, ev, lam):
        I_r = self.regional(ev, lam)
        return float(np.sum(self.n_r * I_r) / np.sum(self.n_r))

    def objective(self, ev, lam, eta=C.ETA):
        """Upper level of Problem 2: J_I + eta * J_E."""
        return self.J_I(ev, lam) + eta * ev["J_E"]

    def share_below(self, ev, lam):
        """Share of cyclists (trips) whose path safety is below the threshold."""
        s = path_safety(ev["path_exposure"])
        return float(self.Q[(s < self.thresholds[lam]) & self.mask].sum() / self.Q.sum())

    def safety_histogram_data(self, ev):
        """(path safety, trips) of every OD pair, for the 'people vs path safety' histogram."""
        s = path_safety(ev["path_exposure"])
        return s[self.mask], self.Q[self.mask]

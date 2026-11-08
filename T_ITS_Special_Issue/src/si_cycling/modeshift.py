"""
Mode-choice (modal-shift) model of the paper (Section II-F, Eqs. 18-25).

    D_bar(w)  = mean cyclist distance over the OD pairs (km)
    V_b(w)    = alpha + beta * D_bar(w)                    (car utility V_c = 0)
    P_b(w)    = 1 / (1 + exp(-V_b(w))),    P_c = 1 - P_b
    q_b(w)    = P_b(w) * Q,    q_c(w) = P_c(w) * Q          Q = bike + car trips of each OD

beta (sensitivity to the mean cyclist distance) is the paper's estimate from the 2015-2025
Grenoble series, beta = -0.886 per km.  alpha is calibrated so that today's network gives the
2025 bike share P_b0 = 16.49 % (paper, Fig. 7a).  NOTE: the paper prints alpha = 0.128,
which with D_bar = 4.41 km gives P_b = 2.2 %, not 16.49 %; the calibration rule of the text
("alpha is calibrated so that the 2025 network reproduces the observed share") is used here.

The bicycle demand of today is the one of CDC_2025 (5 % of all trips), so Q = q_b / P_b0.
The car demand of the network also contains the exchange / through traffic (multiplier m of
step 2), which does not shift.  For a configuration w the demands therefore scale as

    bike:  q_b(w) = q_b0 * P_b(w) / P_b0
    car:   q_c(w) = q_c0 * [ (m - 1) + (1 - P_b(w)) / (1 - P_b0) ] / m
"""
import numpy as np

from . import config as C


def logit(p):
    return np.log(p / (1 - p))


class ModeChoice:
    """Binary logit bike / car on the mean cyclist distance."""

    def __init__(self, bike, ev0, multiplier, p0=C.P_BIKE_2025, beta=C.BETA_PER_KM):
        self.mask = bike.Q > 0                      # OD pairs (between bike trip ends) with trips
        self.p0, self.beta, self.m = p0, beta, float(multiplier)
        self.D0 = self.mean_distance(ev0)
        self.alpha = float(logit(p0) - beta * self.D0)

    def mean_distance(self, ev):
        """D_bar: unweighted mean cyclist distance (km) over the OD pairs with trips."""
        return float(ev["od_distance"][self.mask].mean() / 1000.0)

    def share(self, ev):
        """P_b(w)."""
        return float(1.0 / (1.0 + np.exp(-(self.alpha + self.beta * self.mean_distance(ev)))))

    def bike_scale(self, p):
        return p / self.p0

    def car_scale(self, p):
        return ((self.m - 1) + (1 - p) / (1 - self.p0)) / self.m

    def scales(self, ev):
        """(P_b, bike demand scale, car demand scale) for an evaluated configuration."""
        p = self.share(ev)
        return p, self.bike_scale(p), self.car_scale(p)

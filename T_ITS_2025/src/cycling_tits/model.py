"""Prediction models, upgrade costs and the two model variants (presets).

PRESET "dos" (default)
    Integer DoS weights 1..6, linear models in s_G and l_G (ITSC-2024 calibration)
    and the exponential cost per metre exp(b-4) - exp(a-4).

PRESET "cscs"
    CycleRAP (CSCS) weights, the NNLS model y = theta0 + theta1 * l_G of Eq. (10)
    with theta = [0.41, 0.58, 0.45, 0.57] and Cerema costs in EUR/m.

The former names "published" and "paper_text" are still accepted as aliases.

With a model that only depends on l_G ("cscs"), the percentage improvement
(y - y0)/(y_FS - y0) does not depend on theta, so safety and comfort curves are
identical.
"""
import numpy as np

from .metrics import largest_connected_component, strength

# Cost per metre to upgrade an edge from DoS a to DoS b (a <= b)
COSTS = {
    "exp": {a: {b: float(np.exp(b - 4) - np.exp(a - 4)) for b in range(a, 7)} for a in range(1, 7)},
    # Cerema (2024) - EUR per metre, used for the budgets B_max in EUR
    "cerema": {1: {1: 0, 2: 150, 3: 250, 4: 300, 5: 500, 6: 750},
               2: {2: 0, 3: 100, 4: 150, 5: 350, 6: 600},
               3: {3: 0, 4: 50, 5: 250, 6: 500},
               4: {4: 0, 5: 200, 6: 450},
               5: {5: 0, 6: 250},
               6: {6: 0}},
}

PRESETS = {
    "dos": {
        "scheme": "dos", "cost": "exp",
        "safety": {"const": 3.002424857775961, "s_G": 0.21250708, "l_G": 2.86754057},
        "comfort": {"const": 3.336445200761906, "l_G": 2.80839469},
    },
    "cscs": {
        "scheme": "cscs", "cost": "cerema",
        "safety": {"const": 0.41, "l_G": 0.58},
        "comfort": {"const": 0.45, "l_G": 0.57},
    },
}
ALIASES = {"published": "dos", "paper_text": "cscs"}   # names used by earlier versions


def canonical(preset):
    """Preset name after resolving the old aliases."""
    return ALIASES.get(preset, preset)


def alphas(n=10, start=0.01, stop=1.0):
    """Logarithmically spaced budget fractions alpha_i, i = 0..n-1 (Eq. 13)."""
    a, b = np.log10(start), np.log10(stop)
    return [float(10 ** (a + i * (b - a) / (n - 1))) for i in range(n)]


class Model:
    """Linear prediction of perceived safety or comfort from graph metrics."""

    def __init__(self, preset="dos", target="safety"):
        preset = canonical(preset)
        p = PRESETS[preset]
        self.preset, self.target = preset, target
        self.scheme, self.cost = p["scheme"], COSTS[p["cost"]]
        self.coef = p[target]

    def predict(self, G):
        y = self.coef["const"]
        if "l_G" in self.coef:
            y += self.coef["l_G"] * largest_connected_component(G, self.scheme)
        if "s_G" in self.coef:
            y += self.coef["s_G"] * strength(G, self.scheme)
        return y

    @property
    def y_fully_safe(self):
        """Prediction for the fully safe network (all metrics equal to 1)."""
        return sum(self.coef.values())

    def percent(self, y, y0):
        """% of the achievable improvement: (y - y0) / (y_FS - y0) * 100."""
        return 100.0 * (y - y0) / (self.y_fully_safe - y0)

    # ---- budget -----------------------------------------------------------
    def upgrade_cost(self, G, levels):
        """Total cost of moving every edge from its current DoS to ``levels``."""
        return float(sum(d["distance"] * self.cost[d["dos"]][int(l)]
                         for (_, _, d), l in zip(G.edges(data=True), levels)))

    def max_budget(self, G):
        """B_max: cost of upgrading the whole network to DoS 6."""
        return self.upgrade_cost(G, [6] * G.number_of_edges())


def max_budget(G, cost="cerema"):
    return float(sum(d["distance"] * COSTS[cost][d["dos"]][6] for _, _, d in G.edges(data=True)))

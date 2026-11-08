"""Upgrade costs (Cerema, EUR/m), budget and per-DoS statistics."""
import numpy as np
import pandas as pd

from . import config as C

# COST_MATRIX[a, b] = cost per metre of upgrading from DoS a to DoS b (b >= a), else +inf
COST_MATRIX = np.full((7, 7), np.inf)
for a, row in C.COST.items():
    for b, c in row.items():
        COST_MATRIX[a, b] = c


def upgrade_cost(net, dos, dos_from=None):
    """Total cost (EUR) of going from ``dos_from`` (default: original network) to ``dos``."""
    dos_from = net.dos0 if dos_from is None else dos_from
    return float(np.sum(net.length * COST_MATRIX[dos_from, dos]))


def max_budget(net):
    """B_max: cost of upgrading every arc of the original network to DoS 6."""
    return upgrade_cost(net, np.full(net.E, 6))


def km_per_dos(net, dos):
    """Kilometres of network in each DoS class (Table III of the paper)."""
    dos = np.asarray(dos)
    return {k: float(net.length[dos == k].sum() / 1000) for k in (6, 5, 4, 3, 2, 1)}


def km_table(net, solutions):
    """DataFrame with one row per solution {name: dos} and one column per DoS (km)."""
    return pd.DataFrame({name: km_per_dos(net, d) for name, d in solutions.items()}).T.round(1)

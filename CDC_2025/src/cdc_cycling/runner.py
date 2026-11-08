"""
One optimisation run = one function call, so that runs can be spread over several CPU cores
(concurrent.futures.ProcessPoolExecutor).  Every worker process builds its own Network.
"""
from functools import lru_cache

from . import config as C
from . import costs, ga
from .justice import Sufficiency
from .mlcc import run_mlcc
from .model import Network


@lru_cache(maxsize=1)
def context():
    """Network, original evaluation and thresholds (built once per process)."""
    net = Network()
    ev0 = net.evaluate(net.dos0)
    return net, ev0, Sufficiency(net, ev0)


def summarize(dos):
    """All indicators of one DoS configuration (J_E, and J_I / objective / % below per lambda)."""
    net, _, suf = context()
    ev = net.evaluate(dos)
    out = dict(J_E=ev["J_E"], budget_used=costs.upgrade_cost(net, dos) / costs.max_budget(net),
               km_per_dos=costs.km_per_dos(net, dos))
    for lam in C.LAMBDAS:
        out[f"J_I_{lam:g}"] = suf.J_I(ev, lam)
        out[f"objective_{lam:g}"] = suf.objective(ev, lam)
        out[f"below_{lam:g}"] = suf.share_below(ev, lam)
    return out


def run(algo, beta, lam=None, pop=16, gens=10, seed=0):
    """Run 'MLCC', 'GA-U' or 'GA-S' (needs ``lam``) with budget beta * B_max.

    Returns a dict with the DoS vector and the run information.
    """
    net, ev0, suf = context()
    B = beta * costs.max_budget(net)
    info = {}
    if algo == "MLCC":
        dos, _ = run_mlcc(net, B)
    elif algo == "GA-U":
        seeds = [ga.algorithm1_priority(net), ga.utilitarian_priority(net, ev0)]
        dos, info = ga.run_ga(net, B, lambda ev: ev["J_E"], seeds, pop, gens, seed=seed)
    elif algo == "GA-S":
        # GA-S also gets the utilitarian ordering, so it can never end worse than GA-U on Problem 2
        seeds = [ga.algorithm1_priority(net), ga.sufficientarian_priority(net, ev0, suf, lam, C.ETA),
                 ga.utilitarian_priority(net, ev0)]
        dos, info = ga.run_ga(net, B, lambda ev: suf.objective(ev, lam), seeds, pop, gens, seed=seed)
    else:
        raise ValueError(algo)
    return dict(algo=algo, beta=beta, lam=lam, dos=dos, info=info, summary=summarize(dos))

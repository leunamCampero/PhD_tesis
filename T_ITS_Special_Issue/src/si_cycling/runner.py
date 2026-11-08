"""
One optimisation run = one function call, so that runs can be spread over several CPU cores
(concurrent.futures.ProcessPoolExecutor).  Every worker builds its own MultiModal problem
(with the surrogate of step 3).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one core per run

from . import config as C  # noqa: E402
from . import costs, ga  # noqa: E402
from .problems import context  # noqa: E402


def summary(P, r):
    """JSON-friendly indicators of an evaluation (drop the arrays)."""
    return {k: v for k, v in r.items() if not k.startswith("_") and k != "dos"}


def run(kind, gamma, lam=None, norm=None, pop=C.GA_POP, gens=C.GA_GENERATIONS, seed=0):
    """GA for Problem 2 (kind='util') or 3 (kind='suff', needs ``lam``) with B = 0.15 B_max."""
    P = context()
    B = C.BUDGET_FRACTION * P.B_max
    dos, info = ga.run_ga(P.bike, B, P.objective(kind, gamma, lam, norm), P.seeds(kind, gamma, lam),
                          pop, gens, seed=seed)
    s = summary(P, P.evaluate(dos))
    s["km_per_dos"] = costs.km_per_dos(P.bike, dos)
    return dict(kind=kind, gamma=gamma, lam=lam, dos=dos, info=info, summary=s)

"""
One optimisation run = one function call, so that runs can be spread over several CPU cores
(concurrent.futures.ProcessPoolExecutor).  Every worker process builds its own TradeOff.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one core per run

from . import config as C  # noqa: E402
from . import costs, ga  # noqa: E402
from .problem import context  # noqa: E402


def run(alpha, eps, norm=None, pop=C.GA_POP, gens=C.GA_GENERATIONS, seed=0):
    """GA for Problem 2 with budget alpha * B_max and weight eps.

    ``norm`` (ideal / nadir values of this budget) is needed for 0 < eps < 1.
    Returns a dict with the DoS vector, run information and indicators.
    """
    P = context()
    B = alpha * P.B_max
    dos, info = ga.run_ga(P.bike, B, P.objective(eps, norm), P.seeds(eps, norm), pop, gens, seed=seed)
    m = P.measures(dos)
    m["km_per_dos"] = costs.km_per_dos(P.bike, dos)
    return dict(alpha=alpha, eps=eps, dos=dos, info=info, summary=m)

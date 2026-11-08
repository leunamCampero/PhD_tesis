"""
Genetic algorithm with efficient budget spending (Section III, Algorithm 1 of the paper).

Encoding (same as CDC_2025, see README "Suggested changes")
-----------------------------------------------------------
Chromosome = one priority y_a in [0, 1] per bike arc.  Decoding = Algorithm 1, but the arcs
are visited by decreasing PRIORITY instead of increasing DoS: an arc is raised to DoS 6 if
the remaining budget allows it.  Every chromosome is therefore feasible, and the paper's
Algorithm 1 is the special case y_a ~ 7 - DoS_a.

New here: arcs with y_a < SKIP are never upgraded.  With a car term in the objective some
upgrades are harmful (an on-road arc along a busy road), so the GA must be able to leave
budget unspent rather than being forced to buy them ("<= B" in Eq. 12 allows it).

Seeding: the population starts from noisy copies of a few greedy orderings (``seeds``):
Algorithm 1, exposure saved per euro, and the same score penalised by the car-time increase
(see problem.py).  pygmo's simple genetic algorithm (sga) then evolves the priorities.
"""
import time

import numpy as np
import pygmo as pg

from .costs import COST_MATRIX

SKIP = 0.05
_STATE = {}          # large objects kept outside the problem (pygmo copies it a lot)


def decode(net, priority, budget):
    """Algorithm 1 with a priority order: raise arcs to DoS 6 while the budget allows."""
    priority = np.asarray(priority)
    cost6 = net.length * COST_MATRIX[net.dos0, 6]
    x = net.dos0.copy()
    left = float(budget)
    for a in np.argsort(-priority, kind="stable"):
        if priority[a] < SKIP:
            break
        c = cost6[a]
        if 0 < c <= left:
            x[a] = 6
            left -= c
    return x


def algorithm1_priority(net):
    """Paper's Algorithm 1: lowest DoS first (ties: edge order)."""
    return (7 - net.dos0) + np.linspace(0.5, 0, net.E)


def to_unit(score):
    """Map a greedy score to genes: positive scores -> ranks in [SKIP, 1] (best = 1),
    non-positive scores -> [0, SKIP) (never upgraded)."""
    score = np.asarray(score, float)
    y = np.zeros(len(score))
    pos = score > 0
    for mask, lo, hi in ((pos, SKIP, 1.0), (~pos, 0.0, SKIP * 0.999)):
        idx = np.flatnonzero(mask)
        if idx.size:
            r = np.argsort(np.argsort(score[idx], kind="stable"), kind="stable")
            y[idx] = lo + (hi - lo) * (r / max(idx.size - 1, 1))
    return y


class UpgradeProblem:
    """pygmo problem: genes = arc priorities; ``objective(dos)`` -> float (lower is better)."""

    def __init__(self, net, budget, objective):
        self.key = id(self)
        _STATE[self.key] = dict(net=net, objective=objective, history=[], cache={})
        self.budget = float(budget)

    def _state(self):
        return _STATE[self.key]

    def fitness(self, y):
        st = self._state()
        dos = decode(st["net"], y, self.budget)
        k = dos.tobytes()
        if k not in st["cache"]:                       # many chromosomes decode the same way
            st["cache"][k] = float(st["objective"](dos))
        J = st["cache"][k]
        best = min(J, st["history"][-1][1]) if st["history"] else J
        st["history"].append((len(st["history"]) + 1, best))
        return [J]

    def get_bounds(self):
        n = self._state()["net"].E
        return np.zeros(n), np.ones(n)


def run_ga(net, budget, objective, seeds, pop_size=16, generations=10, noise=0.05, seed=0,
           verbose=False):
    """Run the GA.  ``seeds``: list of greedy scores used to build the population.

    Returns (best DoS vector, info dict).
    """
    rng = np.random.default_rng(seed)
    udp = UpgradeProblem(net, budget, objective)
    pop = pg.population(pg.problem(udp))
    base = [to_unit(s) for s in seeds]
    for i in range(pop_size):
        y = base[i % len(base)].copy()
        if i >= len(base):                                   # noisy copies of the seeds
            y = np.clip(y + rng.normal(0, noise, net.E), 0, 1)
        pop.push_back(y)
    algo = pg.algorithm(pg.sga(gen=1, cr=0.9, crossover="exponential", m=0.02,
                               mutation="gaussian", param_m=0.1,
                               selection="tournament", param_s=2, seed=seed))
    t0 = time.time()
    for g in range(generations):
        pop = algo.evolve(pop)
        if verbose:
            print(f"    generation {g + 1}/{generations}: best = {pop.champion_f[0]:.5f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    best = decode(net, pop.champion_x, budget)
    st = _STATE.pop(udp.key)
    info = dict(objective=float(pop.champion_f[0]), evaluations=len(st["history"]),
                distinct_evaluations=len(st["cache"]), history=list(st["history"]),
                seconds=time.time() - t0,
                settings=dict(pop_size=pop_size, generations=generations, seed=seed, noise=noise))
    return best, info

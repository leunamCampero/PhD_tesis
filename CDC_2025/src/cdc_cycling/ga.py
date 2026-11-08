"""
Genetic algorithm with efficient budget spending (Section II-F, Algorithm 1).

Two variants that differ ONLY in the objective (and in the arc scores used to seed them):
    GA-U  (utilitarian)      minimise J_E                         (Problem 1)
    GA-S  (sufficientarian)  minimise J_I(lambda) + eta * J_E      (Problem 2)

Encoding (change w.r.t. the 2024 code, see README "Suggested changes")
----------------------------------------------------------------------
Chromosome = one priority in [0, 1] per arc.  It is decoded with Algorithm 1, but the arcs
are visited by decreasing PRIORITY instead of by increasing DoS: each arc is raised to DoS 6
if the remaining budget allows it.  Hence every chromosome is feasible and spends the whole
budget ("efficient budget spending"), and the paper's Algorithm 1 is the special case
priority = 7 - DoS.  With the original encoding (DoS per arc + Algorithm 1 fill) the fill
spends almost all of a small budget on the same arcs whatever the objective, so GA-U and
GA-S return the same network.

Seeding: the population starts from noisy versions of
    * Algorithm 1 (lowest DoS first),
    * the objective's greedy score: exposure-weighted flow per euro on each arc, where GA-S
      uses only the flow of the trips that are below the safety threshold,
    * (GA-S only) the utilitarian score too, so GA-S never ends worse than GA-U on Problem 2.
"""
import time

import numpy as np
import pygmo as pg

from .costs import COST_MATRIX

_STATE = {}          # large objects kept outside the problem (pygmo copies it a lot)


def decode(net, priority, budget):
    """Algorithm 1 with a priority order: raise arcs to DoS 6 while the budget allows."""
    cost6 = net.length * COST_MATRIX[net.dos0, 6]
    x = net.dos0.copy()
    left = float(budget)
    for a in np.argsort(-np.asarray(priority), kind="stable"):
        c = cost6[a]
        if 0 < c <= left:
            x[a] = 6
            left -= c
    return x


def algorithm1_priority(net):
    """Paper's Algorithm 1: lowest DoS first (ties: edge order)."""
    return (7 - net.dos0) + np.linspace(0.5, 0, net.E)


def utilitarian_priority(net, ev0):
    """Exposure-weighted flow saved per euro if the arc becomes DoS 6."""
    cost6 = net.length * COST_MATRIX[net.dos0, 6]
    gain = ev0["x_hat"] * net.exposure_arc(net.dos0)
    return np.where(cost6 > 0, gain / np.maximum(cost6, 1e-9), -1.0)


def sufficientarian_priority(net, ev0, suf, lam, eta):
    """Same, but with the flow of the trips below the threshold (weighted by I_p)."""
    cost6 = net.length * COST_MATRIX[net.dos0, 6]
    I = suf.path_insufficiency(ev0, lam)
    x_ins, _ = net.assign(net.dos0, Q=net.Q * I)
    e = net.exposure_arc(net.dos0)
    gain = x_ins / max(x_ins.max(), 1e-12) * e + eta * ev0["x_hat"] * e
    return np.where(cost6 > 0, gain / np.maximum(cost6, 1e-9), -1.0)


def _to_unit(p):
    """Rank-transform priorities to [0, 1] (pygmo bounds)."""
    r = np.argsort(np.argsort(p, kind="stable"), kind="stable")
    return r / (len(p) - 1)


class UpgradeProblem:
    """pygmo problem: genes = arc priorities; ``objective(ev)`` -> float (lower is better)."""

    def __init__(self, net, budget, objective):
        self.key = id(self)
        _STATE[self.key] = dict(net=net, objective=objective, history=[])
        self.budget = float(budget)

    @property
    def net(self):
        return _STATE[self.key]["net"]

    @property
    def history(self):
        return _STATE[self.key]["history"]

    def fitness(self, y):
        J = _STATE[self.key]["objective"](self.net.evaluate(decode(self.net, y, self.budget)))
        best = min(J, self.history[-1][1]) if self.history else J
        self.history.append((len(self.history) + 1, best))
        return [J]

    def get_bounds(self):
        return np.zeros(self.net.E), np.ones(self.net.E)


def run_ga(net, budget, objective, seeds, pop_size=16, generations=10, noise=0.05, seed=0,
           verbose=False):
    """Run GA-U / GA-S.  ``seeds``: list of priority vectors used to build the population.

    Returns (best DoS vector, info dict).
    """
    rng = np.random.default_rng(seed)
    udp = UpgradeProblem(net, budget, objective)
    pop = pg.population(pg.problem(udp))
    base = [_to_unit(s) for s in seeds]
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
    info = dict(objective=float(pop.champion_f[0]), evaluations=len(udp.history),
                history=list(udp.history), seconds=time.time() - t0,
                settings=dict(pop_size=pop_size, generations=generations, seed=seed, noise=noise))
    _STATE.pop(udp.key, None)
    return best, info

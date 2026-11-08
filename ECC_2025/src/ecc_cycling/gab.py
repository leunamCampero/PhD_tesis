"""
GAB - Genetic Algorithm with efficient Budget spending (Section IV-C, Algorithm 1).

Chromosome  : the DoS of every arc, an integer vector x with  DoS_0(a) <= x_a <= 6.
Fitness     : 1. cost(x) > B                -> discarded (penalty 1e10)
              2. otherwise spend what is left of B with Algorithm 1 (``fill_budget``):
                 visit the arcs by increasing DoS and raise each to DoS 6 while the
                 remaining budget allows it
              3. fitness = J_E(filled x)     (lower is better; one traffic assignment)
Search      : pygmo's standard genetic algorithm (sga), as stated in the paper.
Start       : the original network where every arc is raised by one level with prob. 1/2,
              so the initial population is feasible.
"""
import time

import numpy as np
import pygmo as pg

from .costs import COST_MATRIX, upgrade_cost

PENALTY = 1e10

# pygmo deep-copies the problem object many times; keeping the (large) Network and the
# convergence history outside the object makes every copy cheap.
_NETWORKS = {}
_HISTORY = {}


def fill_budget(net, x, left):
    """Algorithm 1: lowest DoS first, raise to DoS 6 while the remaining budget allows it."""
    x = x.copy()
    for a in np.lexsort((np.arange(net.E), x)):          # sort by DoS, then by edge id
        if left <= 0:
            break
        delta = net.length[a] * (COST_MATRIX[net.dos0[a], 6] - COST_MATRIX[net.dos0[a], x[a]])
        if 0 < delta <= left:
            x[a] = 6
            left -= delta
    return x


class GABProblem:
    """pygmo user-defined problem: minimise J_E over integer DoS vectors under budget B."""

    def __init__(self, net, budget):
        self.key = id(net)
        _NETWORKS[self.key] = net
        _HISTORY[self.key] = []                     # (evaluation number, best J so far)
        self.budget = float(budget)

    @property
    def net(self):
        return _NETWORKS[self.key]

    @property
    def history(self):
        return _HISTORY[self.key]

    @property
    def n_evals(self):
        return len(self.history)

    def repair(self, x):
        """Integer, within bounds, budget filled.  Returns None if over budget."""
        x = np.clip(np.rint(x).astype(int), self.net.dos0, 6)
        cost = upgrade_cost(self.net, x)
        if cost > self.budget:
            return None
        return fill_budget(self.net, x, self.budget - cost)

    def fitness(self, x):
        y = self.repair(x)
        J = PENALTY if y is None else self.net.exposure(y)
        best = min(J, self.history[-1][1]) if self.history else J
        self.history.append((self.n_evals + 1, best))
        return [J]

    def get_bounds(self):
        # DoS_0(a) <= x_a <= 6  (arcs already at DoS 6 have lb = ub = 6)
        return self.net.dos0.astype(float), np.full(self.net.E, 6.0)

    def get_nix(self):
        return self.net.E                           # every variable is an integer

    def get_name(self):
        return "Grenoble DoS upgrade (min J_E s.t. budget)"


def initial_population(prob, size, rng):
    """Original DoS + 1 with probability 1/2 on every arc."""
    pop = pg.population(pg.problem(prob))
    x0 = prob.net.dos0
    for _ in range(size):
        pop.push_back(np.minimum(x0 + rng.integers(0, 2, size=x0.size), 6).astype(float))
    return pop


def run_gab(net, budget, pop_size=20, generations=20, seed=0, verbose=True):
    """Run GAB. Returns (best DoS vector, info dict with history and settings)."""
    rng = np.random.default_rng(seed)
    udp = GABProblem(net, budget)
    pop = initial_population(udp, pop_size, rng)
    algo = pg.algorithm(pg.sga(
        gen=1,                      # we loop over generations ourselves to print progress
        cr=0.9,                     # crossover probability
        crossover="exponential",
        m=2.0 / net.E,              # ~2 arcs mutate per child (larger values break the budget)
        mutation="uniform",         # integer genes: uniform draw within the bounds
        selection="tournament", param_s=2,
        seed=seed,
    ))
    t0 = time.time()
    for g in range(generations):
        pop = algo.evolve(pop)
        if verbose:
            print(f"  generation {g + 1:3d}/{generations}: best J_E = {pop.champion_f[0]:.5f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    # the fitness of a chromosome is the J_E of its *repaired* version -> rebuild it
    best = udp.repair(pop.champion_x)
    info = dict(J_E=float(pop.champion_f[0]), evaluations=udp.n_evals,
                history=list(udp.history), seconds=time.time() - t0,
                settings=dict(pop_size=pop_size, generations=generations, seed=seed,
                              crossover="exponential", cr=0.9, mutation="uniform",
                              m=2.0 / net.E, selection="tournament"))
    return best, info

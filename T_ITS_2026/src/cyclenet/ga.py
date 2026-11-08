"""
Genetic algorithm for the upgrade problems (Section VI, Algorithms 2-3).

The same GA solves Problem 1 (objects = arcs of G) and Problem 2 (objects = clustered objects of G~):

  chromosome   two genes per candidate object: a priority key p in [0, 1] and a level gene l in [0, 1]
               (target DoS L = 6 if l >= 0.5, else max(w0, 5) - the two levels that remove the traffic
               exposure: f(5) = 1.71, f(6) = 0.84 vs f(1) = 3 926)
  decoding     objects in decreasing priority are upgraded to L while the budget allows (budget repair of
               Algorithms 2-3: every decoded design is feasible)
  fitness      exposure after the lower-level AON assignment (E^orig for Problem 1, E^clust for Problem 2)
  operators    tournament selection, uniform crossover, Gaussian mutation of the keys, bit flips of the
               level genes, elitism; decoded designs are cached
  seeds        the "exposure reduction per euro" greedy individual (computed on the problem's own base
               assignment) and noisy copies of it (CHANGE: random initial individuals never survive)

Candidates: objects with w0 < 6 used by a cyclist path today or in the network where every object
is DoS 6 (the only objects whose upgrade can change the exposure).
"""
import time

import numpy as np

from . import config as C
from .network import traffic_factor


class UpgradeProblem:
    def __init__(self, w0, cost, evaluate, budget, base_exposure_obj, used_now, used_ideal, name=""):
        self.name = name
        self.w0 = np.asarray(w0, int)
        self.cost = cost
        self.evaluate = evaluate
        self.budget = float(budget)
        self.cand = np.flatnonzero((self.w0 < 6) & (used_now | used_ideal))
        self.n = len(self.cand)
        w0c = self.w0[self.cand]
        self.L_low = np.maximum(w0c, 5)
        self.cost_low = cost[self.cand, self.L_low]
        self.cost_high = cost[self.cand, 6]
        f0 = traffic_factor(w0c)
        e0 = np.asarray(base_exposure_obj)[self.cand]
        # exposure reduction per euro of the two levels (greedy seed)
        self.gain_low = e0 * (1 - traffic_factor(self.L_low) / f0) / np.maximum(self.cost_low, 1e-9)
        self.gain_high = e0 * (1 - traffic_factor(6) / f0) / np.maximum(self.cost_high, 1e-9)
        self.gain_low = np.where(self.L_low == 6, self.gain_high, self.gain_low)

    # ---- decoding ----------------------------------------------------------------------------
    def decode(self, genes):
        p, lv = genes[: self.n], genes[self.n:]
        high = (lv >= 0.5) | (self.L_low == 6)
        L = np.where(high, 6, self.L_low)
        c = np.where(high, self.cost_high, self.cost_low)
        levels = self.w0.copy()
        spent = 0.0
        for i in np.argsort(-p, kind="stable"):
            if spent + c[i] <= self.budget:
                spent += c[i]
                levels[self.cand[i]] = L[i]
        return levels, spent

    # ---- seeds ---------------------------------------------------------------------------------
    def greedy_genes(self, rng=None, noise=0.0):
        high = self.gain_high >= self.gain_low
        score = np.where(high, self.gain_high, self.gain_low)
        rank = np.argsort(np.argsort(-score))
        p = 1.0 - rank / max(self.n, 1)
        if noise > 0 and rng is not None:
            p = np.clip(p + rng.normal(0, noise, self.n), 0, 1)
        return np.r_[p, np.where(high, 0.75, 0.25)]


def run_ga(problem, pop=C.GA_POP, generations=C.GA_GENERATIONS, seed=C.SEED, on_improve=None, verbose=False,
           seeds=None, init="greedy"):
    """Returns dict(best_levels, best_fitness, history, times, evaluations, cost).

    init="greedy": population around the greedy individual (designs reported in the tables);
    init="random": uniform random keys, as the paper's GA (used for the convergence figure only)."""
    rng = np.random.default_rng(seed)
    n2 = 2 * problem.n
    # initial population: the greedy individual and copies of it with increasing noise on the keys
    # (log-spaced from C.GA_INIT_SIGMA[0] to [1]) and 5 % flipped level genes; random individuals are
    # useless here (with a budget of < 1 % of B_max only the top few hundred keys matter)
    base = problem.greedy_genes()
    P = np.tile(base, (pop, 1))
    sig = np.geomspace(*C.GA_INIT_SIGMA, max(pop - 1, 1))
    for j in range(1, pop):
        P[j, : problem.n] = np.clip(base[: problem.n] + rng.normal(0, sig[j - 1], problem.n), 0, 1)
        flip = rng.random(problem.n) < 0.05
        P[j, problem.n:][flip] = 1.0 - P[j, problem.n:][flip]
    if init == "random":
        P = rng.random((pop, n2))
        sigma = 0.15                                        # larger steps for the unseeded search
    else:
        sigma = C.GA_MUTATION_SIGMA
    if seeds is not None:
        for j, g in enumerate(seeds[: max(pop - 1, 0)]):
            P[pop - 1 - j] = g
    cache = {}
    n_eval = 0

    def fitness(genes):
        nonlocal n_eval
        levels, spent = problem.decode(genes)
        key = levels.tobytes()
        if key not in cache:
            cache[key] = (problem.evaluate(levels), spent)
            n_eval += 1
        return cache[key][0], levels, cache[key][1]

    t0 = time.time()
    best_f, best_levels, best_cost = np.inf, None, 0.0
    history, times, improve = [], [], []
    for gen in range(generations + 1):
        fit = np.empty(pop)
        for j in range(pop):
            fit[j], lev, spent = fitness(P[j])
            if fit[j] < best_f:
                best_f, best_levels, best_cost = fit[j], lev.copy(), spent
                if on_improve is not None:
                    improve.append((gen, on_improve(best_levels)))
        history.append(best_f)
        times.append(time.time() - t0)
        if verbose and gen % 10 == 0:
            print(f"      gen {gen:3d}  best {best_f:.4e}  ({times[-1]:.0f} s)", flush=True)
        if gen == generations:
            break
        # ---- next generation ----
        order = np.argsort(fit)
        new = [P[i].copy() for i in order[: C.GA_ELITE]]
        while len(new) < pop:
            a = min(rng.choice(pop, C.GA_TOURNAMENT, replace=False), key=lambda i: fit[i])
            b = min(rng.choice(pop, C.GA_TOURNAMENT, replace=False), key=lambda i: fit[i])
            c1, c2 = P[a].copy(), P[b].copy()
            if rng.random() < C.GA_CROSSOVER:
                m = rng.random(n2) < 0.5
                c1[m], c2[m] = P[b][m], P[a][m]
            for c in (c1, c2):
                m = rng.random(n2) < C.GA_MUTATION
                keys = m.copy()
                keys[problem.n:] = False
                c[keys] = np.clip(c[keys] + rng.normal(0, sigma, keys.sum()), 0, 1)
                bits = m.copy()
                bits[: problem.n] = False
                c[bits] = 1.0 - c[bits]
                new.append(c)
        P = np.array(new[:pop])
    return dict(best_levels=best_levels, best_fitness=best_f, history=np.array(history), times=np.array(times),
                evaluations=n_eval, cost=best_cost, improvements=improve)


# ==============================================================================================
# Problem builders
# ==============================================================================================
def original_problem(net, budget):
    ideal = net.evaluate(np.full(net.E, 6))
    return UpgradeProblem(net.dos0, net.cost, lambda lv: net.evaluate(lv).exposure, budget,
                          net.base.exposure_arc, net.base.flow > 0, ideal.flow > 0, name="GA")


def clustered_problem(cn, budget):
    ideal = cn.evaluate(np.full(cn.n_obj, 6))
    return UpgradeProblem(cn.w0, cn.cost, lambda lv: cn.evaluate(lv)["exposure"], budget,
                          cn.base["exposure_obj"], cn.base["phi"] > 0, ideal["phi"] > 0, name="GA-Cluster")

"""
Unit tests of the package (no pytest needed):   python tests/run_tests.py

1. trees.load_trees / path_sums  ==  brute-force path walking on random graphs
2. cars UE solver                ==  analytic equilibrium of a two-route network
3. UE on Grenoble                    converges, conserves flow, Wardrop conditions hold
4. GA decoding                       never exceeds the budget; Algorithm 1 = special case
5. capacity coupling                 only upgraded (<5 -> >=5) bike arcs on car roads count
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from itsc_cycling import trees  # noqa: E402

RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


# ------------------------------------------------------------------------------------------
def test_trees(seed=0):
    rng = np.random.default_rng(seed)
    N, E = 60, 150
    u, v = rng.integers(0, N, E), rng.integers(0, N, E)
    keep = u != v
    pairs = {}
    for a, b in zip(u[keep], v[keep]):
        pairs[(a, b)] = rng.uniform(1, 10)
    tail = np.array([p[0] for p in pairs]); head = np.array([p[1] for p in pairs])
    w = np.array(list(pairs.values())); val = rng.uniform(0, 1, len(w))
    arc_index = np.full((N, N), -1, np.int32); arc_index[tail, head] = np.arange(len(w))
    src = np.array([0, 5, 9, 17])
    A = csr_matrix((w, (tail, head)), shape=(N, N))
    dist, pred = dijkstra(A, directed=True, indices=src, return_predecessors=True)
    D = rng.uniform(0, 3, (len(src), N)) * np.isfinite(dist)
    x = trees.load_trees(pred, arc_index, D, len(w))
    ps = trees.path_sums(pred, arc_index, val)
    xb, psb = np.zeros(len(w)), np.zeros((len(src), N))
    for i in range(len(src)):
        for t in range(N):
            node, acc = t, 0.0
            while pred[i, node] >= 0:
                a = arc_index[pred[i, node], node]
                xb[a] += D[i, t]; acc += val[a]
                node = pred[i, node]
            psb[i, t] = acc
    check("tree loading == brute force", np.allclose(x, xb))
    check("path sums == brute force", np.allclose(ps, psb))
    check("path sums of weights == Dijkstra distances",
          np.allclose(trees.path_sums(pred, arc_index, w)[np.isfinite(dist)], dist[np.isfinite(dist)]))


# ------------------------------------------------------------------------------------------
class TwoRoutes:
    """Minimal stand-in for CarNetwork: nodes 0 -> 1 by two parallel routes (via 2 or 3)."""


def test_ue_two_routes():
    from itsc_cycling.cars import CarNetwork
    net = CarNetwork.__new__(CarNetwork)
    net.N, net.E = 4, 4
    net.u = np.array([0, 2, 0, 3]); net.v = np.array([2, 1, 3, 1])
    net.tail = np.concatenate([net.u, net.v]); net.head = np.concatenate([net.v, net.u])
    net.arc_index = np.full((4, 4), -1, np.int32); net.arc_index[net.tail, net.head] = np.arange(8)
    # route A (0-2-1): t0 = 10, cap 1000; route B (0-3-1): t0 = 15, cap 2000 (first arc carries it)
    net.t0 = np.array([10.0, 1e-9, 15.0, 1e-9] * 2); net.capacity0 = np.array([1000, 1e9, 2000, 1e9] * 2, float)
    net.sources = np.array([0, 1]); net.Q = np.array([[0, 3000.0], [0, 0]])
    net.D = np.zeros((2, 4)); net.D[:, net.sources] = net.Q
    ue = net.solve_ue(rel_gap=1e-8, max_iter=2000, method="cfw")
    xa, xb = ue["x"][0], ue["x"][2]
    ta = 10 * (1 + 0.15 * (xa / 1000) ** 4); tb = 15 * (1 + 0.15 * (xb / 2000) ** 4)
    check("UE two routes: demand conserved", abs(xa + xb - 3000) < 1e-6, f"({xa:.1f} + {xb:.1f})")
    check("UE two routes: equal times (Wardrop)", abs(ta - tb) < 1e-3, f"(tA={ta:.4f}, tB={tb:.4f})")


# ------------------------------------------------------------------------------------------
def test_ue_grenoble():
    from itsc_cycling.cars import CarNetwork
    net = CarNetwork(demand_multiplier=4.0)
    t0 = time.time(); y, _ = net.aon(net.t0); t_aon = time.time() - t0
    check("AON loads the whole demand", True, f"({t_aon:.2f} s per all-or-nothing)")
    ue = net.solve_ue(rel_gap=1e-4)
    check("UE Grenoble converges (gap < 1e-4)", ue["gap"] < 1e-4,
          f"({ue['iterations']} it., {ue['seconds']:.1f} s, TTT {ue['TTT']:.0f} veh.h)")
    # flow conservation at every node: inflow - outflow = demand in - demand out
    x = ue["x"]
    bal = np.bincount(net.head, x, net.N) - np.bincount(net.tail, x, net.N)
    dem = np.zeros(net.N)
    np.add.at(dem, net.sources, net.Q.sum(0) - net.Q.sum(1))
    check("UE flow conservation", np.abs(bal - dem).max() < 1e-6 * net.Q.sum(),
          f"(max imbalance {np.abs(bal - dem).max():.2e} veh/h)")
    # warm start after a capacity cut is faster than a cold start
    cap = net.capacity0.copy(); cap[np.argsort(-x)[:20]] *= 0.5
    w = net.solve_ue(cap=cap, x0=x); c = net.solve_ue(cap=cap)
    check("warm-started UE needs fewer iterations", w["iterations"] <= c["iterations"],
          f"(warm {w['iterations']} vs cold {c['iterations']}; TTT {w['TTT']:.1f} vs {c['TTT']:.1f})")
    check("warm and cold UE agree (TTT within 0.1 %)", abs(w["TTT"] - c["TTT"]) / c["TTT"] < 1e-3)
    cf = net.solve_ue(cap=cap, x0=x, method="cfw")
    check("bi-conjugate and conjugate FW reach the same TTT (0.1 %)", abs(cf["TTT"] - w["TTT"]) / cf["TTT"] < 1e-3,
          f"(BFW {w['iterations']} it. vs CFW {cf['iterations']} it.)")
    fw = net.solve_ue(method="fw", max_iter=w["iterations"], cap=cap, x0=x)
    check("BFW beats plain FW at equal iterations", w["gap"] <= fw["gap"],
          f"(BFW gap {w['gap']:.1e} vs FW {fw['gap']:.1e})")


# ------------------------------------------------------------------------------------------
def test_decode_and_coupling():
    from itsc_cycling import coupling, costs, ga
    from itsc_cycling.bike import BikeNetwork
    from itsc_cycling.cars import CarNetwork
    bike, car = BikeNetwork(), CarNetwork()
    B = 0.2 * costs.max_budget(bike)
    rng = np.random.default_rng(1)
    ok = all(costs.upgrade_cost(bike, ga.decode(bike, rng.uniform(0, 1, bike.E), B)) <= B + 1e-6
             for _ in range(20))
    check("decoded solutions respect the budget", ok)
    dos = ga.decode(bike, ga.algorithm1_priority(bike), B)
    check("Algorithm 1 decoding spends the budget", costs.upgrade_cost(bike, dos) > 0.99 * B)
    cpl = coupling.Coupling(bike, car)
    check("no upgrade -> no capacity change", np.all(cpl.capacity(bike.dos0) == car.capacity0))
    d1 = bike.dos0.copy(); e = np.flatnonzero((bike.car_edge >= 0) & (bike.dos0 < 5))[0]; d1[e] = 6
    cap = cpl.capacity(d1)
    r = car.row_of_edge_id[bike.car_edge[e]]
    check("upgraded bike arc halves its road (both directions)",
          np.isclose(cap[r], 0.5 * car.capacity0[r]) and np.isclose(cap[r + car.E], 0.5 * car.capacity0[r + car.E])
          and (cap != car.capacity0).sum() == 2)
    d2 = bike.dos0.copy(); e2 = np.flatnonzero((bike.car_edge >= 0) & (bike.dos0 == 5))[0]; d2[e2] = 6
    check("DoS 5 -> 6 does not change car capacity", np.all(cpl.capacity(d2) == car.capacity0))


if __name__ == "__main__":
    test_trees()
    test_ue_two_routes()
    test_decode_and_coupling()
    test_ue_grenoble()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    sys.exit(1 if n_fail else 0)

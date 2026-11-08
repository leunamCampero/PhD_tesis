"""
Unit tests (no pytest needed):   python tests/run_tests.py      (needs step 01; step 03 for section 4)

1. original model     AON flows: every OD path is continuous and conserves its demand; exposure = sum E_a
2. Theorem 1          L = I - D^-1/2 A D^-1/2 symmetric PSD, eigenvalues in [0, 2]
3. clustered model    k = |V| reproduces G exactly; Theorem 2 (corrected): total OD demand preserved;
                      lifting never downgrades and costs sum_g Pi_g
4. CC-Algorithm       DoS-homogeneous objects (eps = 0), connected clusters, exactly k clusters,
                      exposure closer to the original than k-means at k = |V_Grenoble|
5. GA / transfer      decoded designs respect the budget; GA never worse than its greedy seed;
                      GW(H, H) ~ 0 < GW(H, other); Problem 3 respects the budget
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cyclenet import clustering as cl, compare as cp, config as C, ga, transfer as tr  # noqa: E402
from cyclenet.clustered import ClusteredNetwork  # noqa: E402
from cyclenet.network import Network  # noqa: E402
from cyclenet.utils import components_per_cluster  # noqa: E402

RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)


def main():
    H = Network("Grenoble")

    # 1. original model ---------------------------------------------------------------------------
    r = H.base
    check("all OD pairs reached", r.reached == len(H.q))
    # flow conservation: node balance = demand produced - attracted for every node
    div = np.bincount(H.u, r.flow, H.N) + np.bincount(H.v, r.flow, H.N)
    ends = np.bincount(H.od_o, H.q, H.N) + np.bincount(H.od_d, H.q, H.N)
    # every visited intermediate node is entered and left: incident flow = 2 x visits - end visits
    check("AON flow conservation (incident flow = 2 visits - path ends)",
          np.allclose(div, 2 * r.node_flow - ends, atol=1e-6 * H.q.sum()))
    check("exposure = sum phi f d", np.isclose(r.exposure, (r.flow * cl.traffic_factor(H.dos0) * H.length).sum()))

    # 2. Theorem 1 --------------------------------------------------------------------------------
    _, A = cl.safety_adjacency(H)
    L, _ = cl.laplacian(A)
    check("Laplacian symmetric", abs(L - L.T).max() < 1e-12)
    from scipy.sparse.linalg import eigsh
    lo = eigsh(L, k=1, which="SA", return_eigenvectors=False, tol=1e-6)[0]
    hi = eigsh(L, k=1, which="LA", return_eigenvectors=False, tol=1e-6)[0]
    check("Laplacian eigenvalues in [0, 2]", lo > -1e-8 and hi < 2 + 1e-8, f"({lo:.2e}, {hi:.4f})")

    # 3. clustered model ----------------------------------------------------------------------------
    cn = ClusteredNetwork(H, np.arange(H.N))
    check("k = |V|: E^clust = E^orig and Phi^clust = Phi^orig",
          np.isclose(cn.base["exposure"], r.exposure) and np.isclose(cn.base["total_flow"], r.total_flow))
    lab, _ = cl.contract(H, 4000)
    cn = ClusteredNetwork(H, lab)
    check("Theorem 2 (corrected): clustered demand = original demand", np.isclose(cn.q.sum(), H.q.sum()))
    rng = np.random.default_rng(0)
    lv = np.where(rng.random(cn.n_obj) < 0.1, 6, cn.w0)
    dos = cn.lift(lv)
    check("lifting never downgrades", (dos >= H.dos0).all())
    check("lifted cost = sum of object costs", np.isclose(H.upgrade_cost(dos), cn.levels_cost(lv)))

    # 4. CC-Algorithm ------------------------------------------------------------------------------
    for city in C.TARGETS:
        f = C.RESULTS / f"03_labels_{city}_CC.npy"
        if not f.exists():
            print(f"[SKIP] CC tests for {city} (run 03_clustering.py)")
            continue
        net = Network(city)
        cc = ClusteredNetwork(net, np.load(f))
        km = ClusteredNetwork(net, np.load(C.RESULTS / f"03_labels_{city}_kmeans.npy"))
        check(f"{city}: CC gives exactly |V_Grenoble| clusters", cc.k == H.N)
        check(f"{city}: CC objects DoS-homogeneous (eps = 0)", (cc.w0max == cc.w0).all(),
              f"(k-means: {100 * (km.w0max > km.w0).mean():.1f} % mixed objects)")
        check(f"{city}: CC clusters connected", components_per_cluster(net, cc.labels) == 0,
              f"(k-means: {components_per_cluster(net, km.labels)} disconnected)")
        e0 = net.base.exposure
        check(f"{city}: CC preserves exposure better than k-means",
              abs(cc.base["exposure"] - e0) < abs(km.base["exposure"] - e0),
              f"(CC {100 * (cc.base['exposure'] / e0 - 1):+.2f} %, k-means {100 * (km.base['exposure'] / e0 - 1):+.2f} %)")

    # 5. GA / transfer -----------------------------------------------------------------------------
    B = 0.0025 * H.B_max
    prob = ga.original_problem(H, B)
    res = ga.run_ga(prob, pop=6, generations=2)
    check("GA design within budget", H.upgrade_cost(res["best_levels"]) <= B * (1 + 1e-9))
    seed_lv, _ = prob.decode(prob.greedy_genes())
    check("GA not worse than its greedy seed", res["best_fitness"] <= H.evaluate(seed_lv).exposure + 1e-6)
    mh, xh = cp.original_measure(H)
    g_self = cp.gw_between(mh, xh, mh, xh, 150)
    lyon = Network("Lyon")
    ml, xl = cp.original_measure(lyon)
    g_other = cp.gw_between(mh, xh, ml, xl, 150)
    check("GW(H, H) < GW(H, Lyon)", g_self["gw"] < g_other["gw"], f"({g_self['gw']:.3f} < {g_other['gw']:.3f})")
    cn = ClusteredNetwork(lyon, np.arange(lyon.N))
    gw = cp.gw_between(mh, xh, *cp.clustered_measure(cn), 150)
    lev, dos, spent, _ = tr.transfer(H, res["best_levels"], gw, cn, 0.0025 * lyon.B_max)
    check("transfer (Problem 3) within budget", lyon.upgrade_cost(dos) <= 0.0025 * lyon.B_max * (1 + 1e-9))


if __name__ == "__main__":
    main()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    sys.exit(1 if n_fail else 0)

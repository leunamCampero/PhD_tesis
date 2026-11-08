"""
Unit tests of the special-issue extensions (no pytest needed):   python tests/run_tests.py
(needs results/02_ue_original.npz of step 2; the ITSC_2025 tests cover trees / UE / GA)

1. mode choice      reproduces P_b0 today; shorter cyclist distances -> more cyclists, fewer cars
2. capacity rule    theta = theta0 (1 - omega z / n): only upgraded on-road arcs on narrow roads
3. projection       projected flows are >= 0 and conserve flow; a UE flow is left unchanged
4. justice          path safety in (0, 1]; DoS 6 everywhere -> no path insufficiency;
                    thresholds give zero insufficiency today for the car delay measure
5. GNN surrogate    shapes, pruned arcs, save / load, ensemble, feasibility after projection
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_cycling import config as C, utils  # noqa: E402

RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


def main():
    from si_cycling.problems import MultiModal
    from si_cycling.surrogate import FlowProjector
    from si_cycling.coupling import Coupling
    P = MultiModal(surrogate=False)
    bike, car = P.bike, P.car

    # 1. mode choice
    check("mode choice reproduces today's bike share", abs(P.mode.share(P.ev0) - C.P_BIKE_2025) < 1e-9)
    six = np.full(bike.E, 6)
    ev6 = bike.evaluate(six)
    p6, sb, sc = P.mode.scales(ev6)
    check("all DoS 6: shorter cyclist distance, more cyclists, fewer cars",
          P.mode.mean_distance(ev6) < P.mode.D0 and p6 > C.P_BIKE_2025 and sb > 1 and sc < 1,
          f"(D_bar {P.mode.D0:.3f} -> {P.mode.mean_distance(ev6):.3f} km, P_b {100 * p6:.2f} %, car x{sc:.4f})")

    # 2. capacity rule
    cap = P.cpl.capacity(bike.dos0)
    check("no upgrade -> original capacities", np.array_equal(cap, car.capacity0))
    e = P.cpl.cand[0]
    d = bike.dos0.copy(); d[e] = 6
    r = P.cpl.cand_road[0]; n = car.lanes[r]
    cap = P.cpl.capacity(d)
    check("upgrade on a candidate road: theta0 (1 - omega / n), both directions",
          np.isclose(cap[r], car.capacity0[r] * (1 - C.OMEGA / n)) and np.isclose(cap[r + car.E], car.capacity0[r + car.E] * (1 - C.OMEGA / n))
          and (cap != car.capacity0).sum() == 2, f"(n = {n})")
    cpl1 = Coupling(bike, car, omega=1.0)
    check("paper omega = 1 floors single-lane roads at MIN_CAPACITY_SHARE",
          np.isclose(cpl1.road_factor[car.lanes[: car.E] == 1], C.MIN_CAPACITY_SHARE).all())

    # 3. projection
    proj = FlowProjector(car)
    dvec = proj.balance(1.0)
    ok = np.abs(proj.B @ P.phi0 - dvec).max()
    check("UE flows satisfy B phi = d", ok < 1e-6 * car.Q.sum(), f"(max {ok:.1e})")
    check("projection leaves a feasible flow unchanged", np.allclose(proj.project(P.phi0, dvec), P.phi0, atol=1e-6))
    rng = np.random.default_rng(0)
    noisy = P.phi0 * rng.uniform(0.7, 1.3, P.phi0.size) + rng.normal(0, 20, P.phi0.size)
    pr = proj.project(noisy, dvec)
    viol = np.abs(proj.B @ pr - dvec).max()
    check("projection of a noisy flow: non-negative and conserving", pr.min() >= 0 and viol < 1.0,
          f"(max violation {viol:.2e} veh/h)")

    # 4. justice
    S0 = P.base["_arrays"]["S"][P.suf.mask]
    check("path safety in (0, 1]", (S0 > 0).all() and (S0 <= 1).all(), f"(S_G = {P.suf.S_G:.4f})")
    S6 = P.suf.path_safety(ev6)
    check("all DoS 6 -> no path insufficiency at any lambda",
          all(P.suf.J_SI(S6, lam) == 0 for lam in C.LAMBDAS))
    check("car delay insufficiency today is small", P.base["J_TI_c"] < 0.01, f"(J_TI^c = {P.base['J_TI_c']:.5f})")

    # 4b. representative designs (step 5b / 6, config.REPRESENTATIVE_RULE = "no_harm")
    f5b = C.RESULTS / "05b_all_designs.csv"
    if f5b.exists():
        import pandas as pd
        from si_cycling import selection
        df = pd.read_csv(f5b)
        reps = selection.representatives(df, rule="no_harm")
        gu, tol = reps["util"], C.NO_HARM_TOL_PP
        ok = all(reps[l][f"J_SI_{l:g}"] <= gu[f"J_SI_{l:g}"] for l in C.LAMBDAS)
        check("no_harm: GA-S(lambda) has lower path insufficiency than GA-U for every lambda", ok)
        ok = all(reps[l].bike_above_20 <= gu.bike_above_20 + tol and reps[l].car_above_thr <= gu.car_above_thr + tol
                 and reps[l].T_tot <= gu.T_tot * (1 + C.NO_HARM_MAX_TTOT) for l in C.LAMBDAS)
        check("no_harm: GA-S(lambda) respects the no-harm and travel-time conditions", ok)
        old = selection.representatives(df, rule="paper")
        check("paper rule reproduces the old comparison (GA-S(20 %) worse than GA-U)",
              old[0.2]["J_SI_0.2"] > old["util"]["J_SI_0.2"])
    else:
        print("[SKIP] selection tests (run 05b_exact_all_designs.py)")

    # 5. GNN surrogate (step 3b) - skipped if PyTorch is not installed
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[SKIP] GNN tests (PyTorch not installed)")
        return
    import tempfile
    from si_cycling.gnn import EnsembleSurrogate, GNNSurrogate
    z = np.load(C.RESULTS / "02_samples.npz")
    red, sc_, X = z["red"][:12].astype(float), z["scale"][:12], z["x"][:12].astype(float)
    g = GNNSurrogate(car, P.phi0, P.cpl.roads, hidden=16, layers=2)
    g.fit(red[:10], sc_[:10], X[:10], epochs=1, verbose=False)
    p = g.predict_raw(red[10:], sc_[10:])
    check("GNN prediction: one flow per arc, finite", p.shape == (2, 2 * car.E) and np.isfinite(p).all())
    check("GNN: pruned arcs keep s * phi_0", np.allclose(p[:, ~g.active], np.outer(sc_[10:], P.phi0)[:, ~g.active]),
          f"({int(g.active.sum())} active arcs)")
    with tempfile.TemporaryDirectory() as d:
        g.save(Path(d) / "g.pt")
        check("GNN save / load gives the same prediction",
              np.allclose(GNNSurrogate.load(Path(d) / "g.pt").predict_raw(red[10:], sc_[10:]), p))
    e = EnsembleSurrogate(g, g)
    check("ensemble of identical models = the model", np.allclose(e.predict_raw(red[10:], sc_[10:]), p))
    no_change = g.predict_raw(np.zeros((1, len(P.cpl.roads))), [1.0])
    proj_flow = proj.project(no_change[0], proj.balance(1.0))
    check("GNN + projection is a feasible flow", proj_flow.min() >= 0
          and np.abs(proj.B @ proj_flow - proj.balance(1.0)).max() < 1.0)


if __name__ == "__main__":
    main()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    sys.exit(1 if n_fail else 0)

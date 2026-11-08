"""
Step 2 - Car user equilibrium today and the exact UE samples for the surrogate.

1. Calibration of the car demand multiplier m (as ITSC_2025 step 2: modelled flows next to
   DoS 1-3 bike lanes = GTLVille AADT x 10 %) and the UE of today's network (gap 1e-4).
2. N_SAMPLES random bike configurations w, each with its exact post-upgrade UE:
       w        = Algorithm-1 decoding of a random priority vector with a random budget in
                  [2 %, 30 %] of B_max: 40 % uniformly random priorities, 60 % noisy versions
                  of the greedy orderings the GA starts from (so the samples cover the region
                  the optimiser explores);
       theta(w) = capacity rule,  s(w) = car demand scale of the mode-choice model;
       phi(w)   = UE(theta, s * q_c), conjugate Frank-Wolfe warm-started at s * phi_0.
   The paper used 912 samples and 5,928 CPU hours (6.5 CPU-h per sample).  With the fast
   tree loading and warm starts one sample takes ~20-40 CPU-seconds.

Options:  --n 1000 --workers N     (re-running continues from the samples already saved)
Outputs:  results/02_ue_original.npz/.json, results/02_samples.npz
Run time: ~3 min (calibration) + ~1-2 h for 1000 samples on 8 cores
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from si_cycling import config as C, utils  # noqa: E402

SAMPLES = C.RESULTS / "02_samples.npz"


def calibrate_and_solve():
    """Demand multiplier m and today's UE (identical procedure to ITSC_2025)."""
    from si_cycling.bike import BikeNetwork
    from si_cycling.cars import CarNetwork
    bike, car = BikeNetwork(), CarNetwork(1.0)
    target = {}
    for e in np.flatnonzero((bike.car_edge >= 0) & (bike.dos0 <= 3)):
        r = car.row_of_edge_id[bike.car_edge[e]]
        target[r] = max(target.get(r, 0), C.PEAK_HOUR_FACTOR * C.AADT[int(bike.dos0[e])])
    roads = np.array(sorted(target))
    tgt, L = np.array([target[r] for r in roads]), car.length[roads]
    m, x = 1.0, None
    for _ in range(8):
        car.set_demand_multiplier(m)
        ue = car.solve_ue(rel_gap=1e-3, x0=x)
        ratio = float(np.sum(L * car.two_way(ue["x"])[roads]) / np.sum(L * tgt))
        print(f"  m = {m:6.3f}: modelled / GTLVille = {ratio:5.3f}", flush=True)
        if abs(ratio - 1) < 0.01:
            break
        x, m = ue["x"] / ratio, float(np.clip(m / ratio, *C.DEMAND_MULTIPLIER_BOUNDS))
    car.set_demand_multiplier(m)
    ue = car.solve_ue(rel_gap=C.UE_REL_GAP, x0=ue["x"])
    utils.save_ue(utils.UE_FILE, ue, m)
    utils.save_json(dict(demand_multiplier=m, TTT_veh_h=ue["TTT"], gap=ue["gap"], iterations=ue["iterations"],
                         peak_hour_car_trips=float(car.Q.sum())), C.RESULTS / "02_ue_original.json")
    print(f"  m = {m:.3f}; UE TTT = {ue['TTT']:,.0f} veh.h, gap {ue['gap']:.1e} ({ue['iterations']} it.)")


@lru_cache(maxsize=1)
def ctx():
    """Light context for the samples (no surrogate needed)."""
    from si_cycling import costs, ga
    from si_cycling.bike import BikeNetwork
    from si_cycling.cars import CarNetwork
    from si_cycling.costs import COST_MATRIX
    from si_cycling.coupling import Coupling
    from si_cycling.modeshift import ModeChoice
    ue = utils.load_ue()
    bike, car = BikeNetwork(), CarNetwork(ue["multiplier"])
    cpl = Coupling(bike, car)
    ev0 = bike.evaluate(bike.dos0)
    cost6 = bike.length * COST_MATRIX[bike.dos0, 6]
    util = np.where(cost6 > 0, ev0["x_hat"] * bike.exposure_arc(bike.dos0) / np.maximum(cost6, 1e-9), -1.0)
    on_road = np.zeros(bike.E, bool)
    on_road[cpl.cand] = True
    seeds = [ga.algorithm1_priority(bike), util, np.where(on_road, -1.0, util), np.where(on_road, util * 3, util)]
    return dict(bike=bike, car=car, cpl=cpl, mode=ModeChoice(bike, ev0, ue["multiplier"]), phi0=ue["x"],
                B_max=costs.max_budget(bike), seeds=[ga.to_unit(s) for s in seeds], ga=ga)


def one_sample(k):
    c = ctx()
    rng = np.random.default_rng(10_000 + k)
    bike, car, cpl, ga = c["bike"], c["car"], c["cpl"], c["ga"]
    frac = rng.uniform(*C.SAMPLE_BUDGETS)
    if rng.random() < 0.4:
        y = rng.uniform(0, 1, bike.E)
    else:
        y = np.clip(c["seeds"][rng.integers(len(c["seeds"]))] + rng.normal(0, rng.uniform(0.02, 0.3), bike.E), 0, 1)
    dos = ga.decode(bike, y, frac * c["B_max"])
    ev = bike.evaluate(dos)
    p_b, _, s_c = c["mode"].scales(ev)
    cap = cpl.capacity(dos)
    ue = car.solve_ue(cap=cap, scale=s_c, x0=c["phi0"] * s_c, rel_gap=C.SAMPLE_UE_GAP)
    red = 1.0 - (cap[: car.E] / car.capacity0[: car.E])[cpl.roads]
    return dict(k=k, red=red.astype(np.float32), scale=s_c, x=ue["x"].astype(np.float32), TTT=ue["TTT"],
                gap=ue["gap"], it=ue["iterations"], sec=ue["seconds"], P_b=p_b, budget=frac,
                n_reduced=int((red > 0).sum()))


def save_samples(rows, cand_roads):
    rows = sorted(rows, key=lambda r: r["k"])
    np.savez_compressed(SAMPLES, k=np.array([r["k"] for r in rows]), red=np.stack([r["red"] for r in rows]),
                        scale=np.array([r["scale"] for r in rows]), x=np.stack([r["x"] for r in rows]),
                        TTT=np.array([r["TTT"] for r in rows]), gap=np.array([r["gap"] for r in rows]),
                        it=np.array([r["it"] for r in rows]), sec=np.array([r["sec"] for r in rows]),
                        P_b=np.array([r["P_b"] for r in rows]), budget=np.array([r["budget"] for r in rows]),
                        n_reduced=np.array([r["n_reduced"] for r in rows]), cand_roads=cand_roads)


def load_rows():
    if not SAMPLES.exists():
        return []
    z = np.load(SAMPLES)
    keys = ("k", "red", "scale", "x", "TTT", "gap", "it", "sec", "P_b", "budget", "n_reduced")
    return [{key: z[key][i] for key in keys} for i in range(len(z["k"]))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_SAMPLES)
    ap.add_argument("--workers", type=int, default=utils.n_workers())
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()
    utils.ensure_dirs()
    t0 = time.time()
    if not (args.skip_calibration and utils.UE_FILE.exists()):
        print("1. demand calibration and today's UE")
        calibrate_and_solve()
    cand_roads = ctx()["cpl"].roads
    rows = load_rows()
    done = {int(r["k"]) for r in rows}
    todo = [k for k in range(args.n) if k not in done]
    print(f"2. UE samples: {len(done)} already saved, {len(todo)} to compute with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one_sample, todo, chunksize=2), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(todo):
                save_samples(rows, cand_roads)
                el = time.time() - t0
                print(f"  {len(rows)}/{args.n} samples  ({el / 60:.1f} min, last: {r['it']} it., "
                      f"{r['sec']:.0f} s, {r['n_reduced']} roads reduced, P_b {100 * r['P_b']:.2f} %)", flush=True)
    z = np.load(SAMPLES)
    print(f"done: {len(z['k'])} samples; UE {np.mean(z['it']):.0f} it. / {np.mean(z['sec']):.0f} s on average "
          f"({np.sum(z['sec']) / 3600:.1f} CPU-hours in total); max gap {z['gap'].max():.1e}")

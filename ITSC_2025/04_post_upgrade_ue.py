"""
Step 4 - Impact on car routing and travel time (Section IV-B, Figs. 5-6 of the paper).

During the optimisation (step 3) the car flows are FIXED at the pre-upgrade equilibrium phi*
(drivers keep their routes).  Here every solution is re-assessed with a POST-upgrade user
equilibrium: drivers re-optimise their routes on the reduced capacities.

For every solution: TTT0 (before), TTT_fixed (no re-routing, = step 3), TTT_UE (re-routing),
mean OD travel time, share of road flows that change by more than 20 %.

Outputs: results/04_post_upgrade.csv, results/04_post_upgrade_flows.npz (UE flows of eps*)
Run time: ~5 min on 8 cores (warm-started conjugate Frank-Wolfe, ~20-40 s per UE)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from itsc_cycling import config as C, utils  # noqa: E402


@lru_cache(maxsize=1)
def ctx():
    from itsc_cycling.problem import TradeOff
    return TradeOff()


def post_upgrade(alpha, eps):
    P = ctx()
    car = P.car
    dos = utils.load_solution(C.RESULTS / "solutions" / f"{utils.solution_name(alpha, eps)}.csv")
    cap = P.cpl.capacity(dos)
    ue = car.solve_ue(cap=cap, x0=P.phi)
    ue0_od = utils.load_ue()["od_time"]
    f0, f1 = car.two_way(P.phi), car.two_way(ue["x"])
    used = f0 > 50                                           # roads with > 50 veh/h before
    change = np.abs(f1 - f0)[used] / f0[used]
    return dict(alpha=alpha, eps=eps, TTT0=P.T0, TTT_fixed=P.car_ttt(dos), TTT_UE=ue["TTT"],
                ue_gap=ue["gap"], ue_iterations=ue["iterations"], ue_seconds=ue["seconds"],
                mean_od_time0=float(np.sum(car.Q * ue0_od) / car.Q.sum()),
                mean_od_time=float(np.sum(car.Q * ue["od_time"]) / car.Q.sum()),
                share_roads_flow_change_gt20=float((change > 0.2).mean()),
                max_road_flow_change=float(change.max()),
                reduced_roads=int(len(P.cpl.reduced_roads(dos)))), ue["x"]


if __name__ == "__main__":
    t0 = time.time()
    pareto = pd.read_csv(C.RESULTS / "03_pareto.csv")
    star = utils.load_json(C.RESULTS / "03_pareto.json")["eps_star"]
    jobs = list(zip(pareto.alpha, pareto.eps))
    rows, flows = [], {}
    with ProcessPoolExecutor(utils.n_workers()) as ex:
        for (row, x) in ex.map(post_upgrade, *zip(*jobs)):
            rows.append(row)
            a, e = row["alpha"], row["eps"]
            if np.isclose(e, star[f"{a:.3f}"]["eps_star"]):
                flows[f"a{a:.3f}"] = x
            print(f"  alpha={a:.3f} eps={e:3.1f}: TTT fixed +{100 * (row['TTT_fixed'] / row['TTT0'] - 1):6.2f} %"
                  f"   after re-routing +{100 * (row['TTT_UE'] / row['TTT0'] - 1):6.2f} %"
                  f"   ({row['ue_iterations']} it., {row['ue_seconds']:.0f} s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["alpha", "eps"])
    df["TTT_fixed_pct"] = 100 * (df.TTT_fixed / df.TTT0 - 1)
    df["TTT_UE_pct"] = 100 * (df.TTT_UE / df.TTT0 - 1)
    df.to_csv(C.RESULTS / "04_post_upgrade.csv", index=False)
    np.savez_compressed(C.RESULTS / "04_post_upgrade_flows.npz", **flows)
    print(f"done in {(time.time() - t0) / 60:.1f} min")

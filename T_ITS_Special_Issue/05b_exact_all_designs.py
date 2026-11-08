"""
Step 5b - Exact-UE re-evaluation of EVERY design found by the optimiser (adds to step 5, which only
re-evaluates the utilitarian front, the lambda = 40 % front and the representative designs).

For the 11 utilitarian designs and the 5 x 11 sufficientarian designs:
    * all indicators with exact UE car flows (as step 5)
    * shares of users below the sufficiency thresholds (as Fig. 8), for every lambda:
        path_below_<lam>   % of cyclists whose path safety S_p < S_thr(lambda)
        bike_above_20      % of cyclists with a bike trip longer than 20 min
        car_above_thr      % of drivers above the delay threshold (1.25 x free-flow time)
    * the arrays (S, bike times, car OD times, flows) of every design, float32

Used by step 6 to (i) build the fronts from the pooled archive of all designs and (ii) pick the
representative designs (config.REPRESENTATIVE_RULE).  Step 5 files are not modified.

Outputs: results/05b_all_designs.csv, results/05b_arrays.npz
Run time: ~6 min on 8 cores
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

from si_cycling import config as C, utils  # noqa: E402


@lru_cache(maxsize=1)
def ctx():
    from si_cycling.problems import MultiModal
    return MultiModal(surrogate=False)            # exact UE only: no surrogate needed


def shares(P, arrays):
    suf = P.suf
    out = {f"path_below_{lam:g}": 100 * suf.bike_regions.share_below(suf.path_insufficiency(arrays["S"], lam))
           for lam in C.LAMBDAS}
    out["bike_above_20"] = 100 * suf.bike_regions.share_below(suf.bike_time_insufficiency(arrays["t_b"]))
    out["car_above_thr"] = 100 * suf.car_regions.share_below(suf.car_time_insufficiency(arrays["t_c"]))
    return out


def exact(job):
    name, kind, gamma, lam = job
    P = ctx()
    if kind == "original":
        ev = P.base
    else:
        dos = utils.load_solution(C.RESULTS / "solutions" / f"{name}.csv")
        ev = P.evaluate(dos, exact=True)
    row = {k: v for k, v in ev.items() if not k.startswith("_") and k != "dos"}
    row.update(name=name, kind=kind, gamma=gamma, lam=lam, **shares(P, ev["_arrays"]))
    return row, {k: np.asarray(v, np.float32) for k, v in ev["_arrays"].items()}


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("original", "original", np.nan, np.nan)]
    jobs += [(utils.solution_name("util", g), "util", g, np.nan) for g in C.GAMMAS]
    jobs += [(utils.solution_name("suff", g, lam), "suff", g, lam) for lam in C.LAMBDAS for g in C.GAMMAS]
    with ProcessPoolExecutor(utils.n_workers()) as ex:
        out = list(ex.map(exact, jobs))
    df = pd.DataFrame([r for r, _ in out])
    df.to_csv(C.RESULTS / "05b_all_designs.csv", index=False)
    np.savez_compressed(C.RESULTS / "05b_arrays.npz",
                        **{f"{r['name']}__{k}": v for r, a in out for k, v in a.items()})
    cols = ["name", "J_E", "J_SI_0.4", "J_TI_b", "J_TI_c", "T_tot", "path_below_0.4", "bike_above_20", "car_above_thr"]
    print(df[cols].round(4).to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")

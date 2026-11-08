"""
Step 3 - Solve Problem 2 (trade-off network upgrade) with the GA for every budget
alpha in {1/3, 2/5, 1/2} and every weight eps in {0, 0.1, ..., 1}  (33 runs, Fig. 3-4).

Phase 1: the extreme weights eps = 1 (min exposure) and eps = 0 (min car time) give the ideal
         and nadir values used to normalise the objective (problem.py).
Phase 2: the 9 intermediate weights.
Then, per budget, eps* = the solution closest to the utopia point (0, 0) of the min-max
normalised front (see README; the paper's Eq. 15 is also reported).

Options:  --pop 16 --generations 12 --workers N
Outputs:  results/03_pareto.json, results/03_pareto.csv, results/solutions/GA_a*_eps*.csv
Run time: ~30-40 min on 8 cores
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from itsc_cycling import config as C, utils  # noqa: E402
from itsc_cycling.runner import run  # noqa: E402


def run_batch(jobs, args):
    out = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, a, e, norm, args.pop, args.generations, 0): (a, e) for a, e, norm in jobs}
        for f in futs:
            a, e = futs[f]
            r = f.result()
            out[(a, e)] = r
            s = r["summary"]
            print(f"  alpha={a:.3f} eps={e:3.1f}: J_E={s['J_E']:.4f}  TTT={s['TTT']:,.1f} veh.h  "
                  f"roads cut={s['reduced_roads']:4d}  budget used={s['budget_used']:.3f}  "
                  f"({r['info']['seconds']:.0f} s, {r['info']['distinct_evaluations']} evaluations)",
                  flush=True)
    return out


def select_eps_star(df):
    """eps* per budget: closest point to (0, 0) after min-max normalising E and T over the
    11 solutions of that budget; also the paper's Eq. (15) on the weighted values."""
    rows = {}
    for a, g in df.groupby("alpha"):
        E = (g.J_E - g.J_E.min()) / (g.J_E.max() - g.J_E.min())
        T = (g.TTT - g.TTT.min()) / max(g.TTT.max() - g.TTT.min(), 1e-12)
        d = np.hypot(E, T)
        d_paper = np.hypot(g.eps * E, (1 - g.eps) * T)
        rows[a] = dict(eps_star=float(g.eps.iloc[int(np.argmin(d.values))]),
                       eps_star_paper_eq15=float(g.eps.iloc[int(np.argmin(d_paper.values))]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=C.GA_POP)
    ap.add_argument("--generations", type=int, default=C.GA_GENERATIONS)
    ap.add_argument("--workers", type=int, default=utils.n_workers())
    args = ap.parse_args()
    utils.ensure_dirs()
    t0 = time.time()

    print("phase 1: extreme weights (eps = 1 and eps = 0)")
    res = run_batch([(a, e, None) for a in C.ALPHAS for e in (1.0, 0.0)], args)
    norms = {}
    for a in C.ALPHAS:
        lo, hi = res[(a, 1.0)]["summary"], res[(a, 0.0)]["summary"]
        norms[a] = dict(E_ideal=lo["J_E"], E_nadir=hi["J_E"], T_ideal=hi["TTT"], T_nadir=lo["TTT"])

    print("phase 2: intermediate weights")
    mid = [e for e in C.EPSILONS if 0 < e < 1]
    res.update(run_batch([(a, e, norms[a]) for a in C.ALPHAS for e in mid], args))

    rows = []
    for (a, e), r in sorted(res.items()):
        utils.save_solution(r["dos"], C.RESULTS / "solutions" / f"{utils.solution_name(a, e)}.csv")
        s = r["summary"]
        rows.append(dict(alpha=a, eps=e, J_E=s["J_E"], TTT=s["TTT"], reduced_roads=s["reduced_roads"],
                         reduced_road_km=s["reduced_road_km"], budget_used=s["budget_used"],
                         **{f"km_dos{k}": v for k, v in s["km_per_dos"].items()},
                         objective=r["info"]["objective"], evaluations=r["info"]["evaluations"],
                         seconds=r["info"]["seconds"]))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "03_pareto.csv", index=False)
    star = select_eps_star(df)
    utils.save_json(dict(normalisation={f"{a:.3f}": v for a, v in norms.items()},
                         eps_star={f"{a:.3f}": v for a, v in star.items()},
                         history={f"{a:.3f}_{e:g}": r["info"]["history"] for (a, e), r in res.items()},
                         settings=vars(args), minutes=(time.time() - t0) / 60),
                    C.RESULTS / "03_pareto.json")
    print("\neps* (closest to utopia):", {f"{a:.3f}": v["eps_star"] for a, v in star.items()})
    print("eps* (paper Eq. 15)      :", {f"{a:.3f}": v["eps_star_paper_eq15"] for a, v in star.items()})
    print(f"done in {(time.time() - t0) / 60:.1f} min")

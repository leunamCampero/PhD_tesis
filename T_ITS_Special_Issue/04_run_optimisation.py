"""
Step 4 - Utilitarian (Problem 2) and sufficientarian (Problem 3) designs with B = 0.15 B_max.

    utilitarian      gamma in {0, 0.1, ..., 1}                         11 runs
    sufficientarian  lambda in {10, ..., 50} %  x  gamma in {0, ..., 1}  55 runs

Phase 1: extreme weights gamma = 1 / gamma = 0 (ideal and nadir values, normalisation).
Phase 2: intermediate weights.
Representative weight gamma* of every front: closest point to the utopia point of the
min-max normalised front (the paper picks illustrative values gamma*_lambda = lambda + 10 %;
both are reported).  The car flows come from the surrogate of step 3; step 5 re-evaluates
the selected designs with the exact UE.

Options:  --pop 16 --generations 12 --workers N
Outputs:  results/04_runs.csv, results/04_runs.json, results/solutions/*.csv
Run time: ~1.5-2 h on 8 cores
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from si_cycling import config as C, utils  # noqa: E402
from si_cycling.problems import MultiModal  # noqa: E402
from si_cycling.runner import run  # noqa: E402


def run_batch(jobs, args):
    out = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, k, g, lam, norm, args.pop, args.generations, 0): (k, g, lam)
                for k, g, lam, norm in jobs}
        for f in futs:
            key = futs[f]
            r = f.result()
            out[key] = r
            s = r["summary"]
            lab = "util" if key[2] is None else f"suff lambda={key[2]:g}"
            print(f"  {lab:18s} gamma={key[1]:3.1f}: J_E={s['J_E']:.3f} J_SI(0.4)={s['J_SI_0.4']:.4f} "
                  f"J_TI={s['J_TI']:.4f} (car {s['J_TI_c']:.4f}) T_tot={s['T_tot']:,.0f} P_b={100 * s['P_b']:.2f} % "
                  f"({r['info']['seconds'] / 60:.0f} min)", flush=True)
    return out


def norm_from(res, kind, lam=None):
    a, b = MultiModal.terms(kind, lam)
    s1, s0 = res[(kind, 1.0, lam)]["summary"], res[(kind, 0.0, lam)]["summary"]
    return {a + "_ideal": s1[a], a + "_nadir": s0[a], b + "_ideal": s0[b], b + "_nadir": s1[b]}


def gamma_star(df, a, b):
    """gamma closest to the utopia point of the min-max normalised front (a, b)."""
    A = (df[a] - df[a].min()) / max(df[a].max() - df[a].min(), 1e-12)
    B = (df[b] - df[b].min()) / max(df[b].max() - df[b].min(), 1e-12)
    return float(df.gamma.iloc[int(np.argmin(np.hypot(A, B).values))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=C.GA_POP)
    ap.add_argument("--generations", type=int, default=C.GA_GENERATIONS)
    ap.add_argument("--workers", type=int, default=utils.n_workers())
    args = ap.parse_args()
    utils.ensure_dirs()
    t0 = time.time()
    problems = [("util", None)] + [("suff", lam) for lam in C.LAMBDAS]

    print("phase 1: extreme weights")
    res = run_batch([(k, g, lam, None) for k, lam in problems for g in (1.0, 0.0)], args)
    norms = {(k, lam): norm_from(res, k, lam) for k, lam in problems}
    print("phase 2: intermediate weights")
    mid = [g for g in C.GAMMAS if 0 < g < 1]
    res.update(run_batch([(k, g, lam, norms[(k, lam)]) for k, lam in problems for g in mid], args))

    rows = []
    for (k, g, lam), r in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][2] or 0, kv[0][1])):
        utils.save_solution(r["dos"], C.RESULTS / "solutions" / f"{utils.solution_name(k, g, lam)}.csv")
        s = dict(r["summary"])
        km = s.pop("km_per_dos")
        rows.append(dict(kind=k, lam=lam if lam is not None else np.nan, gamma=g, **s,
                         **{f"km_dos{d}": v for d, v in km.items()}, objective=r["info"]["objective"],
                         seconds=r["info"]["seconds"]))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "04_runs.csv", index=False)
    stars = {}
    for k, lam in problems:
        g = df[(df.kind == k) & ((df.lam == lam) if lam is not None else df.lam.isna())]
        a, b = MultiModal.terms(k, lam)
        stars[f"{k}" if lam is None else f"{k}_{lam:g}"] = dict(
            gamma_star=gamma_star(g, a, b), gamma_paper=None if lam is None else round(lam + 0.1, 2))
    utils.save_json(dict(normalisation={f"{k}_{lam}": v for (k, lam), v in norms.items()}, gamma_star=stars,
                         history={f"{k}_{lam}_{g:g}": r["info"]["history"] for (k, g, lam), r in res.items()},
                         settings=vars(args), minutes=(time.time() - t0) / 60),
                    C.RESULTS / "04_runs.json")
    print("\ngamma*:", {k: v["gamma_star"] for k, v in stars.items()})
    print(f"done in {(time.time() - t0) / 60:.1f} min")

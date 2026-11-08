"""
Step 5 - optimisation (Sections III and VI, Figs. 6-8).

For every budget fraction beta (B = beta B_max) :
  Grenoble          GA on the original network (Problem 1; source strategy of the transfer)
  Lyon, Strasbourg  GA                     Problem 1 on the original network
                    GA-Cluster             Problem 2 on the CC-Algorithm network (k = |V_Grenoble|), lifted
                    GA-Cluster (k-means)   Problem 2 on the k-means network, lifted (baseline)
Every lifted design is evaluated on the ORIGINAL network (E^orig).  Runs in parallel (one per process).

Outputs  results/05_runs/<city>_<method>_<beta>.npz   design, convergence, run time
         results/05_optimisation.csv                  one line per run
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import config as C  # noqa: E402
from cyclenet.utils import log  # noqa: E402

OUT = C.RESULTS / "05_runs"
OUT_RANDOM = C.RESULTS / "05_runs_random_init"   # paper-like GA (random initial population): Fig. 7
METHOD_LABELS = {"GA": "GA", "CC": "GA-Cluster", "kmeans": "GA-Cluster (k-means)"}


def run_job(city, method, beta, pop, generations, init="greedy"):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from cyclenet import ga
    from cyclenet.clustered import ClusteredNetwork
    from cyclenet.network import Network

    net = Network(city)
    budget = beta * net.B_max
    t0 = time.time()
    if method == "GA":
        prob = ga.original_problem(net, budget)
        res = ga.run_ga(prob, pop=pop, generations=generations, init=init)
        dos = res["best_levels"]
        levels = dos
        lifted_hist = None
        build = 0.0
        E_model0 = net.base.exposure
    else:
        tb = time.time()
        cn = ClusteredNetwork(net, np.load(C.RESULTS / f"03_labels_{city}_{method}.npy"))
        build = time.time() - tb
        prob = ga.clustered_problem(cn, budget)
        res = ga.run_ga(prob, pop=pop, generations=generations, init=init,
                        on_improve=lambda lv: net.evaluate(cn.lift(lv)).exposure)
        levels = res["best_levels"]
        dos = cn.lift(levels)
        lifted_hist = np.array(res["improvements"])
        E_model0 = cn.base["exposure"]
    ev = net.evaluate(dos)
    row = dict(city=city, method=METHOD_LABELS[method], beta=beta, budget=budget, B_max=net.B_max,
               cost=net.upgrade_cost(dos), E_base=net.base.exposure, E_final=ev.exposure,
               E_norm=ev.exposure / net.base.exposure, E_model_base=E_model0, E_model_final=res["best_fitness"],
               E_model_norm=res["best_fitness"] / E_model0, seconds=time.time() - t0, build_seconds=build,
               ga_seconds=float(res["times"][-1]), evaluations=res["evaluations"], candidates=prob.n,
               upgraded_arcs=int((dos > net.dos0).sum()),
               upgraded_km=float(net.length[dos > net.dos0].sum() / 1000))
    np.savez_compressed((OUT if init == "greedy" else OUT_RANDOM) / f"{city}_{method}_{beta:g}.npz", dos=dos, levels=levels, history=res["history"],
                        times=res["times"], model_base=E_model0, lifted=lifted_hist if lifted_hist is not None
                        else np.zeros((0, 2)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=C.GA_POP)
    ap.add_argument("--generations", type=int, default=C.GA_GENERATIONS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--betas", type=float, nargs="*", default=list(C.BUDGET_FRACTIONS))
    ap.add_argument("--random-init", action="store_true",
                    help="paper-like GA from a random population (convergence figure, GA and GA-Cluster only)")
    args = ap.parse_args()
    init = "random" if args.random_init else "greedy"
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_RANDOM.mkdir(parents=True, exist_ok=True)
    if init == "greedy":
        jobs = [(C.REFERENCE, "GA", b) for b in args.betas]
        jobs += [(city, m, b) for city in C.TARGETS for m in ("GA", "CC", "kmeans") for b in args.betas]
    else:
        jobs = [(city, m, b) for city in C.TARGETS for m in ("GA", "CC") for b in args.betas]
    # longest jobs first
    jobs.sort(key=lambda j: (j[0] == C.REFERENCE, j[1] != "GA"))
    csv = C.RESULTS / ("05_optimisation.csv" if init == "greedy" else "05_convergence_random_init.csv")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_job, *j, args.pop, args.generations, init): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            log(f"{r['city']:10s} {r['method']:22s} beta {r['beta']:.2f}: E/E0 = {r['E_norm']:.3f} "
                f"(model {r['E_model_norm']:.3f})  {r['seconds'] / 60:.1f} min")
            pd.DataFrame(rows).sort_values(["city", "method", "beta"]).to_csv(csv, index=False)
    log("done")


if __name__ == "__main__":
    main()

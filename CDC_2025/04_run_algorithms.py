"""
STEP 4 - Solve Problems 1 and 2 at the budget B* of step 3 (Section III-C).

Runs (in parallel):  MLCC,  GA-U (Problem 1),  GA-S for every lambda (Problem 2).
All algorithms get the SAME budget, so the comparison is fair.

Writes results/solutions/<algo>_b<beta>[_lam<lambda>].csv and results/04_runs.json.
Runtime: ~15 min (pop 16, 10 generations, 7 runs in parallel).

Run:  python 04_run_algorithms.py [--beta 0.0567] [--pop 16] [--generations 10]
      (default: the paper's budget B_5 = 0.0567 B_max)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse                                                  # noqa: E402

from cdc_cycling import config as C                              # noqa: E402
from cdc_cycling import runner, utils                            # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beta", type=float, default=None, help="budget as a fraction of B_max")
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--generations", type=int, default=10)
    args = p.parse_args()
    utils.ensure_dirs()
    beta = args.beta if args.beta is not None else C.BUDGET_FRACTIONS[4]
    print(f"\nbudget B = {beta:.3f} B_max")

    tasks = [("MLCC", None), ("GA-U", None)] + [("GA-S", lam) for lam in C.LAMBDAS]
    out = {}
    with ProcessPoolExecutor(utils.n_workers()) as pool:
        futures = {pool.submit(runner.run, a, beta, lam, args.pop, args.generations): (a, lam)
                   for a, lam in tasks}
        for f, (a, lam) in futures.items():
            r = f.result()
            name = utils.solution_name(a, beta, lam)
            utils.save_solution(r["dos"], C.RESULTS / "solutions" / f"{name}.csv")
            out[name] = dict(algo=a, beta=beta, lam=lam, summary=r["summary"],
                             run={k: v for k, v in r["info"].items() if k != "history"},
                             history=r["info"].get("history"))
            s = r["summary"]
            print(f"   {name:22s} J_E = {s['J_E']:.4f}   J_I(40%) = {s['J_I_0.4']:.4f}   "
                  f"below(40%) = {s['below_0.4']:.1%}   budget used = {s['budget_used']:.3f}", flush=True)
    utils.save_json(dict(beta=beta, runs=out), C.RESULTS / "04_runs.json")
    print("saved: results/solutions/*.csv, results/04_runs.json")

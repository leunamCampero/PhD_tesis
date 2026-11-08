"""
STEP 3 - Budget sensitivity analysis (Section III-B, Fig. 2).

For every threshold lambda and every budget B_k (10 values, log-spaced from 1 % to 50 % of
B_max) Problem 2 is solved with GA-S.  The marginal improvement per euro

        (J(B_{k-1}) - J(B_k)) / (B_k - B_{k-1}),   J = J_I + eta J_E,  B_0 = 0 (today)

is normalised to [0, 1] for each lambda; B*_lambda is the budget where it is largest.

Note: with a well-converged GA the curve J(B) is concave (diminishing returns), so this
criterion picks the first budget B_1 = 1 % B_max.  Step 4 therefore uses the paper's budget
B_5 ~ 5.7 % B_max by default (see README, "Suggested changes").

50 GA runs, spread over the CPU cores.  Default: small GA (pop 8, 3 generations), ~15 min.

Run:  python 03_budget_sensitivity.py [--pop 8] [--generations 3]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse                                                  # noqa: E402
from collections import Counter                                  # noqa: E402

import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from cdc_cycling import config as C                              # noqa: E402
from cdc_cycling import plotting, runner, utils                  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--generations", type=int, default=3)
    args = p.parse_args()
    utils.ensure_dirs()

    tasks = [(beta, lam) for lam in C.LAMBDAS for beta in C.BUDGET_FRACTIONS]
    print(f"\n{len(tasks)} GA-S runs on {utils.n_workers()} processes ...", flush=True)
    rows = []
    with ProcessPoolExecutor(utils.n_workers()) as pool:
        futures = [pool.submit(runner.run, "GA-S", beta, lam, args.pop, args.generations) for beta, lam in tasks]
        for (beta, lam), f in zip(tasks, futures):
            s = f.result()["summary"]
            rows.append(dict(lam=lam, beta=beta, objective=s[f"objective_{lam:g}"],
                             J_I=s[f"J_I_{lam:g}"], J_E=s["J_E"], below=s[f"below_{lam:g}"]))
            print(f"   lambda {lam:.0%}  B = {beta:.3f} B_max:  J_I + eta J_E = {rows[-1]['objective']:.4f}", flush=True)
    df = pd.DataFrame(rows)

    # marginal improvement per budget step (B_0 = 0: today's network)
    orig = utils.load_json(C.RESULTS / "02_original.json")["per_lambda"]
    marg, best = {}, {}
    for lam in C.LAMBDAS:
        J = np.r_[orig[str(lam)]["objective"], df[df.lam == lam].sort_values("beta").objective.to_numpy()]
        B = np.r_[0.0, np.array(C.BUDGET_FRACTIONS)]
        m = np.maximum(0, (J[:-1] - J[1:]) / (B[1:] - B[:-1]))
        marg[lam] = m / m.max() if m.max() > 0 else m
        best[lam] = float(C.BUDGET_FRACTIONS[int(np.argmax(m))])
        df.loc[df.lam == lam, "marginal"] = marg[lam]
    beta_star = Counter(best.values()).most_common(1)[0][0]
    df.to_csv(C.RESULTS / "03_budget_sensitivity.csv", index=False, float_format="%.6g")
    utils.save_json(dict(B_star_per_lambda=best, beta_star=beta_star), C.RESULTS / "03_budget_star.json")
    print("\nB*_lambda / B_max: " + ", ".join(f"{lam:.0%}: {b:.3f}" for lam, b in best.items()))
    print(f"most frequent B*_lambda = {beta_star:.3f} B_max  (step 4 uses the paper's B_5 = "
          f"{C.BUDGET_FRACTIONS[4]:.3f} B_max unless --beta is given)")

    fig, ax = plt.subplots(figsize=(8, 3))
    plotting.marginal_lines(ax, C.BUDGET_FRACTIONS, marg)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig2_marginal_improvements.png", dpi=200)
    print("saved: figures/fig2_marginal_improvements.png, results/03_*")

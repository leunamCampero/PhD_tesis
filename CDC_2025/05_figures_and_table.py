"""
STEP 5 - Results of the case study (Section III-C, Figs. 3-7) from the solutions of step 4.

Fig. 3  DoS (top) and regional safety insufficiency I_r (bottom), lambda = 40 %
Fig. 4  cyclists vs path safety, with the threshold (lambda = 40 %)
Fig. 5  % of cyclists below / above the threshold, every lambda
Fig. 6  sufficientarian objective J_I + eta J_E, every lambda
Fig. 7  utilitarian exposure J_E
Table   results/05_summary.csv

GA-S is always the run made for the lambda being shown.

Run:  python 05_figures_and_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt                                  # noqa: E402
import pandas as pd                                              # noqa: E402

from cdc_cycling import config as C                              # noqa: E402
from cdc_cycling import plotting, utils                          # noqa: E402
from cdc_cycling.runner import context                           # noqa: E402

utils.ensure_dirs()
net, ev0, suf = context()
runs = utils.load_json(C.RESULTS / "04_runs.json")
beta = runs["beta"]


def solution(algo, lam=None):
    return utils.load_solution(C.RESULTS / "solutions" / f"{utils.solution_name(algo, beta, lam)}.csv")


# evaluate every solution once
evals = {"Original": ev0, "MLCC": net.evaluate(solution("MLCC")), "GA-U": net.evaluate(solution("GA-U"))}
for lam in C.LAMBDAS:
    evals[f"GA-S {lam:.0%}"] = net.evaluate(solution("GA-S", lam))


def scen(lam):
    """The four scenarios compared for one lambda."""
    return {"Original": evals["Original"], "MLCC": evals["MLCC"], "GA-U": evals["GA-U"],
            "GA-S": evals[f"GA-S {lam:.0%}"]}


# ---- table ------------------------------------------------------------------------------
rows = []
for name, ev in evals.items():
    r = dict(solution=name, J_E=ev["J_E"])
    for lam in C.LAMBDAS:
        r[f"J_I {lam:.0%}"] = suf.J_I(ev, lam)
        r[f"J_I+eta*J_E {lam:.0%}"] = suf.objective(ev, lam)
        r[f"% below {lam:.0%}"] = 100 * suf.share_below(ev, lam)
    rows.append(r)
table = pd.DataFrame(rows)
table.to_csv(C.RESULTS / "05_summary.csv", index=False, float_format="%.5g")
pd.set_option("display.width", 200)
cols = ["solution", "J_E"] + [f"J_I {l:.0%}" for l in C.LAMBDAS] + [f"% below {l:.0%}" for l in C.LAMBDAS]
print(f"\nRESULTS at B = {beta:.3f} B_max (eta = {C.ETA})\n")
print(table[cols].round(4).to_string(index=False))

# ---- Fig. 3: DoS + regional insufficiency, lambda = 40 % -----------------------------------
lam = 0.4
polys = plotting.iris_polygons()
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
for j, (name, ev) in enumerate(scen(lam).items()):
    plotting.dos_map(axes[0, j], net, ev["dos"], title=f"{name} DoS", legend=(j == 3), polys=polys)
    plotting.region_map(axes[1, j], net, suf.zones, suf.regional(ev, lam), polys=polys,
                        title=f"{name} insufficiency", colorbar=(j == 3), label="$\\mathcal{I}_r$")
fig.suptitle(f"$\\lambda = 40\\%$, budget {beta:.3f} $B_{{max}}$")
fig.tight_layout()
fig.savefig(C.FIGURES / "fig3_dos_and_insufficiency.png", dpi=200)
plt.close(fig)

# ---- Fig. 4: histograms ------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
for ax, (name, ev) in zip(axes.flat, scen(lam).items()):
    s, w = suf.safety_histogram_data(ev)
    plotting.safety_histogram(ax, s, w, suf.thresholds[lam], lam, title=name)
fig.tight_layout()
fig.savefig(C.FIGURES / "fig4_safety_histograms.png", dpi=200)
plt.close(fig)

# ---- Fig. 5: % below / above the threshold ------------------------------------------------------
fig, axes = plt.subplots(1, len(C.LAMBDAS), figsize=(18, 3.6), sharey=True)
for ax, l in zip(axes, C.LAMBDAS):
    plotting.threshold_bars(ax, {k: suf.share_below(ev, l) for k, ev in scen(l).items()},
                            title=f"$\\lambda = {l:.0%}$".replace("%", "\\%"))
axes[0].legend(frameon=False, fontsize=8, loc="center left")
fig.tight_layout()
fig.savefig(C.FIGURES / "fig5_threshold_distribution.png", dpi=200)
plt.close(fig)

# ---- Fig. 6: J_I + eta J_E ------------------------------------------------------------------------
obj = {a: {l: suf.objective(scen(l)[a], l) for l in C.LAMBDAS} for a in ("Original", "MLCC", "GA-U", "GA-S")}
fig, ax = plt.subplots(figsize=(9, 3.8))
plotting.grouped_bars(ax, obj, "$J_{\\mathcal{I}} + \\eta\\, J_{\\mathcal{E}}$")
fig.tight_layout()
fig.savefig(C.FIGURES / "fig6_sufficientarian_objective.png", dpi=200)
plt.close(fig)

# ---- Fig. 7: J_E ------------------------------------------------------------------------------------
names = ["Original", "MLCC", "GA-U"] + [f"GA-S {l:.0%}" for l in C.LAMBDAS]
colors = [plotting.ALGO_COLORS["Original"], plotting.ALGO_COLORS["MLCC"], plotting.ALGO_COLORS["GA-U"]] + \
         [plotting.LAMBDA_COLORS[l] for l in C.LAMBDAS]
fig, ax = plt.subplots(figsize=(10, 3.4))
plotting.exposure_bars(ax, {n: evals[n]["J_E"] for n in names}, colors)
fig.tight_layout()
fig.savefig(C.FIGURES / "fig7_exposure.png", dpi=200)
plt.close(fig)
print("\nsaved: results/05_summary.csv, figures/fig3 ... fig7")

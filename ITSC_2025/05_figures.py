"""
Step 5 - Figures and summary table (Figs. 3-6 of the paper).

fig3_pareto_front.png              car TTT increase vs cyclist exposure, 3 budgets, eps* marked
fig4_exposure_travel_time_bars.png normalised E and T for every eps (one panel per budget)
fig5_travel_time_comparison.png    TTT increase at eps*: fixed flows vs post-upgrade UE
fig6_upgrades_and_car_flows.png    top: upgrades at eps*; bottom: post-upgrade car-flow change
results/05_summary.csv             the numbers of the eps* solutions

Run time: ~1 min
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from itsc_cycling import config as C, plotting as P, utils  # noqa: E402
from itsc_cycling.problem import TradeOff  # noqa: E402

pr = pd.read_csv(C.RESULTS / "03_pareto.csv")
post = pd.read_csv(C.RESULTS / "04_post_upgrade.csv")
star = {float(k): v["eps_star"] for k, v in utils.load_json(C.RESULTS / "03_pareto.json")["eps_star"].items()}
T = TradeOff()
pr["E_rel"] = pr.J_E / T.E0
pr["T_pct"] = 100 * (pr.TTT / T.T0 - 1)


ALPHAS = sorted(pr.alpha.unique())


def key(a):
    return min(star, key=lambda k: abs(k - a))


# ---- Fig. 3: Pareto fronts -------------------------------------------------------------------
fig, axs = P.plt.subplots(1, 2, figsize=(10, 3.8))
for ax, zoom in zip(axs, (False, True)):
    for a, g in pr.groupby("alpha"):
        col, mk, lab = P.ALPHA_STYLE[min(P.ALPHA_STYLE, key=lambda k: abs(k - a))]
        g = g.sort_values("eps")
        ax.plot(g.T_pct, g.E_rel, ls=":", marker=mk, color=col, ms=5, label=lab)
        s = g[np.isclose(g.eps, star[key(a)])]
        ax.scatter(s.T_pct, s.E_rel, s=110, facecolor="none", edgecolor="#d62728", lw=1.6, zorder=5)
    ax.scatter([0], [1], marker="*", s=120, color="k", label="original network", zorder=6)
    ax.set_xlabel("car TTT increase, fixed flows (%)"), ax.set_ylabel(r"cyclist exposure $J_E / J_E^{0}$")
    if zoom:
        sel = pr[pr.eps.between(0.1, 0.9)]
        ax.set_xlim(-0.05 * max(sel.T_pct.max(), 1), sel.T_pct.max() * 1.1)
        ax.set_ylim(0, sel.E_rel.max() * 1.1)
        ax.set_title(r"zoom on $0.1 \leq \varepsilon \leq 0.9$ (red: $\varepsilon^\star$)")
    else:
        ax.set_title("Pareto fronts of Problem 2")
        ax.legend(frameon=False, fontsize=8)
print("figure:", P.save(fig, "fig3_pareto_front.png"))

# ---- Fig. 4: normalised E and T for every eps ----------------------------------------------------
fig, axs = P.plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
for ax, (a, g) in zip(axs, pr.groupby("alpha")):
    g = g.sort_values("eps")
    En = (g.J_E - g.J_E.min()) / (g.J_E.max() - g.J_E.min())
    Tn = (g.TTT - g.TTT.min()) / max(g.TTT.max() - g.TTT.min(), 1e-12)
    i = np.arange(len(g))
    ax.bar(i - 0.18, En, 0.36, color="#009988", label=r"cyclist exposure $\hat{E}$")
    ax.bar(i + 0.18, Tn, 0.36, color="#0077BB", label=r"car travel time $\hat{T}$")
    j = int(np.flatnonzero(np.isclose(g.eps.values, star[key(a)]))[0])
    ax.axvspan(j - 0.45, j + 0.45, color="#d62728", alpha=0.15, lw=0)
    ax.set_ylabel(r"min-max normalised"), ax.set_title(rf"$\alpha$ = {a:.3f}", fontsize=10)
    ax.set_xticks(i, [f"{e:g}" for e in g.eps])
axs[0].legend(frameon=False, ncol=2, fontsize=8)
axs[-1].set_xlabel(r"weight $\varepsilon$ (red: $\varepsilon^\star$)")
print("figure:", P.save(fig, "fig4_exposure_travel_time_bars.png"))

# ---- Fig. 5: travel time at eps*: fixed flows vs post-upgrade UE ------------------------------
rows = []
for a in ALPHAS:
    p = post[np.isclose(post.alpha, a) & np.isclose(post.eps, star[key(a)])].iloc[0]
    q = pr[np.isclose(pr.alpha, a) & np.isclose(pr.eps, star[key(a)])].iloc[0]
    rows.append(dict(alpha=a, eps_star=star[key(a)], J_E=q.J_E, exposure_reduction_pct=100 * (1 - q.E_rel),
                     TTT0=p.TTT0, TTT_fixed=p.TTT_fixed, TTT_UE=p.TTT_UE,
                     TTT_fixed_pct=p.TTT_fixed_pct, TTT_UE_pct=p.TTT_UE_pct,
                     mean_od_time0_min=p.mean_od_time0, mean_od_time_min=p.mean_od_time,
                     reduced_roads=int(q.reduced_roads), reduced_road_km=q.reduced_road_km,
                     share_roads_flow_change_gt20=p.share_roads_flow_change_gt20,
                     budget_used=q.budget_used))
summary = pd.DataFrame(rows)
summary.to_csv(C.RESULTS / "05_summary.csv", index=False)
fig, ax = P.plt.subplots(figsize=(6, 3.4))
i = np.arange(len(summary))
ax.bar(i - 0.2, summary.TTT_fixed_pct, 0.4, color="#bbbbbb", label="pre-upgrade routes (fixed flows)")
ax.bar(i + 0.2, summary.TTT_UE_pct, 0.4, color="#1a5fb4", label="post-upgrade user equilibrium")
for k, r in summary.iterrows():
    ax.text(k + 0.2, r.TTT_UE_pct, f"{r.TTT_UE_pct:.1f} %", ha="center", va="bottom", fontsize=8)
ax.set_xticks(i, [rf"$\alpha$={a:.3f}" + "\n" + rf"$\varepsilon^\star$={e:g}" for a, e in zip(summary.alpha, summary.eps_star)])
ax.set_ylabel("car TTT increase (%)"), ax.legend(frameon=False, fontsize=8)
print("figure:", P.save(fig, "fig5_travel_time_comparison.png"))

# ---- Fig. 6: upgrades and car-flow changes at eps* ------------------------------------------------
flows = np.load(C.RESULTS / "04_post_upgrade_flows.npz")
fig, axs = P.plt.subplots(2, 3, figsize=(13, 8.5))
f0 = T.car.two_way(T.phi)
for k, a in enumerate(ALPHAS):
    dos = utils.load_solution(C.RESULTS / "solutions" / f"{utils.solution_name(a, star[key(a)])}.csv")
    up = np.flatnonzero(dos != T.bike.dos0)
    P.bike_dos_map(axs[0, k], T.bike, dos, lw=0.6, highlight=up, legend=(k == 0),
                   title=rf"upgrades, $\alpha$={a:.3f}, $\varepsilon^\star$={star[key(a)]:g}")
    f1 = T.car.two_way(flows[f"a{a:.3f}"])
    rel = np.where(f0 > 50, 100 * (f1 - f0) / np.maximum(f0, 1), 0.0)
    P.car_map(axs[1, k], T.car, np.clip(rel, -60, 60), cmap="RdBu_r", lw=0.9,
              norm=TwoSlopeNorm(0, -60, 60), label="car flow change (%)",
              title="post-upgrade UE: change of car flow")
    for ax in axs[:, k]:
        P.iris_boundaries(ax, lw=0.15)
print("figure:", P.save(fig, "fig6_upgrades_and_car_flows.png"))

print("\n" + summary.round(3).to_string(index=False))

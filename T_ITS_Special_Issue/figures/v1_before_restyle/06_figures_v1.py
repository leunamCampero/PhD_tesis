"""
Step 6 - Figures and summary table of the special-issue paper (Figs. 9-17).

fig4_pareto_fronts.png             utilitarian (J_E vs T_tot) and sufficientarian (J_SI vs J_TI) fronts
fig5_tradeoff_bars.png             normalised indicators for every gamma (lambda = 40 %)
fig6_path_safety_histograms.png    cyclists vs path-sufficiency score (original / util / suff)
fig7_trip_time_histograms.png      cyclists and drivers vs trip time (original / util / suff)
fig8_insufficiency_shares.png      share of insufficient cyclists / drivers (lambda = 40 %)
fig9_maps.png                      DoS and regional path / bike-time / car-time insufficiency
fig10_path_insufficiency_lambda.png J_SI for every lambda (original / util / suff gamma*_lambda)
fig11_system_performance.png       exposure, total travel time and bike share
results/06_summary.csv             exact-UE indicators of original / utilitarian / sufficientarian

All values are the exact-UE re-evaluations of step 5.
Run time: ~1 min
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from si_cycling import config as C, plotting as P, utils  # noqa: E402
from si_cycling.problems import MultiModal  # noqa: E402

LAM = C.LAMBDA_FIGURES
COL = {"original": "#7f7f7f", "utilitarian": "#e69f00", "sufficientarian": "#1a5fb4"}
runs = pd.read_csv(C.RESULTS / "04_runs.csv")
ex = pd.read_csv(C.RESULTS / "05_exact.csv")
stars = utils.load_json(C.RESULTS / "04_runs.json")["gamma_star"]
arr = np.load(C.RESULTS / "05_arrays.npz")
g_u, g_s = utils.representative_gamma(stars, "util"), utils.representative_gamma(stars, "suff", LAM)
MM = MultiModal(surrogate=False)
suf = MM.suf
base = ex[ex.kind == "original"].iloc[0]
rep = {"original": base,
       "utilitarian": ex[ex.name == f"util_g{g_u:g}"].iloc[0],
       "sufficientarian": ex[ex.name == f"suff_lam{LAM:g}_g{g_s:g}"].iloc[0]}

# ---- summary table ---------------------------------------------------------------------------
cols = ["J_E", f"J_SI_{LAM:g}", "J_TI_b", "J_TI_c", "T_c", "T_b", "T_tot", "P_b", "D_bar", "reduced_roads", "budget_used"]
summary = pd.DataFrame({k: r[cols] for k, r in rep.items()}).T
summary.insert(0, "gamma", [np.nan, g_u, g_s])
summary.to_csv(C.RESULTS / "06_summary.csv")

# ---- Fig. 4: Pareto fronts ---------------------------------------------------------------------
fig, axs = P.plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
u = runs[runs.kind == "util"].sort_values("gamma")
axs[0].plot(100 * (u.T_tot / base.T_tot - 1), u.J_E / base.J_E, "o:", color=COL["utilitarian"])
for _, r in u.iterrows():
    axs[0].annotate(f"{r.gamma:g}", (100 * (r.T_tot / base.T_tot - 1), r.J_E / base.J_E), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
s = u[np.isclose(u.gamma, g_u)]
axs[0].scatter(100 * (s.T_tot / base.T_tot - 1), s.J_E / base.J_E, s=120, facecolor="none", edgecolor="r")
axs[0].scatter([0], [1], marker="*", s=120, color="k", label="original")
axs[0].set_xlabel(r"total travel time change $\mathcal{T}_{tot}$ (%)"), axs[0].set_ylabel(r"$J_E / J_E^0$")
axs[0].set_title(r"utilitarian (labels: $\gamma$)"), axs[0].legend(frameon=False, fontsize=8)
cmap = P.plt.get_cmap("viridis")
for i, lam in enumerate(C.LAMBDAS):
    g = runs[(runs.kind == "suff") & np.isclose(runs.lam, lam)].sort_values("gamma")
    c = cmap(i / (len(C.LAMBDAS) - 1))
    axs[1].plot(g.J_TI, g[f"J_SI_{lam:g}"], "o:", color=c, ms=4, label=rf"$\lambda$={int(100 * lam)} %")
    st = g[np.isclose(g.gamma, utils.representative_gamma(stars, "suff", lam))]
    axs[1].scatter(st.J_TI, st[f"J_SI_{lam:g}"], s=110, facecolor="none", edgecolor="r")
axs[1].set_xlabel(r"trip insufficiency $J_{TI}^b + J_{TI}^c$"), axs[1].set_ylabel(r"path insufficiency $J_{SI}$")
axs[1].set_title(r"sufficientarian (red: $\gamma^\star_\lambda$)"), axs[1].legend(frameon=False, fontsize=8)
# both formulations in the same space (exact UE): path insufficiency vs total travel time
for kind, lab in (("util", "utilitarian"), ("suff", "sufficientarian")):
    g = ex[ex.kind == kind]
    g = (g if kind == "util" else g[np.isclose(g.lam, LAM)]).sort_values("gamma")
    axs[2].plot(100 * (g.T_tot / base.T_tot - 1), g[f"J_SI_{LAM:g}"], "o:", color=COL[lab], label=lab)
    st = g[np.isclose(g.gamma, g_u if kind == "util" else g_s)]
    axs[2].scatter(100 * (st.T_tot / base.T_tot - 1), st[f"J_SI_{LAM:g}"], s=120, facecolor="none", edgecolor="r")
axs[2].scatter([0], [base[f"J_SI_{LAM:g}"]], marker="*", s=120, color="k", label="original")
axs[2].set_yscale("log")
axs[2].set_xlabel(r"total travel time change $\mathcal{T}_{tot}$ (%)")
axs[2].set_ylabel(rf"path insufficiency $J_{{SI}}$ ($\lambda$={int(100 * LAM)} %)")
axs[2].set_title("both fronts, exact UE (red: representative $\\gamma^\\star$)"), axs[2].legend(frameon=False, fontsize=8)
print("figure:", P.save(fig, "fig4_pareto_fronts.png"))

# ---- Fig. 5: normalised indicators per gamma (lambda = 40 %) ------------------------------------
fig, axs = P.plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
inds = [("J_E", "exposure $J_E$"), ("T_tot", r"$\mathcal{T}_{tot}$"), (f"J_SI_{LAM:g}", "path insuff. $J_{SI}$"),
        ("J_TI_b", "bike trip insuff."), ("J_TI_c", "driver trip insuff.")]
for ax, (kind, title) in zip(axs, (("util", "utilitarian"), ("suff", rf"sufficientarian, $\lambda$={int(100 * LAM)} %"))):
    g = ex[ex.kind == kind].sort_values("gamma")
    i = np.arange(len(g))
    w = 0.16
    for j, (c, lab) in enumerate(inds):
        v = g[c].values
        ref = base[c]
        ax.bar(i + (j - 2) * w, v / ref if ref > 0 else v, w, label=lab)
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_ylabel("value / original"), ax.set_title(title, fontsize=10)
    ax.set_xticks(i, [f"{x:g}" for x in g.gamma])
axs[0].legend(frameon=False, ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.25))
axs[-1].set_xlabel(r"weight $\gamma$")
print("figure:", P.save(fig, "fig5_tradeoff_bars.png"))

# ---- Fig. 6: path-sufficiency histograms ----------------------------------------------------------
mask = suf.mask
fig, axs = P.plt.subplots(3, 1, figsize=(7, 7), sharex=True)
bins = np.logspace(np.log10(2e-4), np.log10(0.6), 40)
for ax, k in zip(axs, rep):
    S = arr[f"{k}__S"][mask]
    q = MM.bike.Q[mask] * rep[k].bike_scale
    thr = suf.S_thr[LAM]
    ax.hist(S[S < thr], bins, weights=q[S < thr], color="#d62728", alpha=0.8, label="below threshold")
    ax.hist(S[S >= thr], bins, weights=q[S >= thr], color=COL[k], alpha=0.8, label="above threshold")
    ax.axvline(thr, color="r", ls="--", lw=1), ax.axvline(suf.S_G, color="b", ls="--", lw=1)
    ax.set_xscale("log"), ax.set_ylabel("cyclists / day")
    ax.set_title(f"{k}: {100 * q[S < thr].sum() / q.sum():.1f} % of cyclists below the threshold", fontsize=9)
axs[0].legend(frameon=False, fontsize=8)
axs[-1].set_xlabel(r"exposure-based path-sufficiency score $S_p$ (red: $S_{thr}$, blue: $S_G$)")
print("figure:", P.save(fig, "fig6_path_safety_histograms.png"))

# ---- Fig. 7: trip-time histograms ---------------------------------------------------------------
fig, axs = P.plt.subplots(3, 2, figsize=(11, 7), sharex="col")
cm = MM.car.Q_base > 0
for row, k in enumerate(rep):
    tb, tc = arr[f"{k}__t_b"][mask], arr[f"{k}__t_c"][cm]
    qb, qc = MM.bike.Q[mask] * rep[k].bike_scale, MM.car.Q[cm] * rep[k].car_scale
    thr_c = suf.T_thr_car[cm]
    a = axs[row, 0]
    a.hist(tb, 40, weights=qb, color=COL[k]), a.axvline(C.T_THR_BIKE_MIN, color="r", ls="--")
    a.set_title(f"{k} - cyclists: {100 * qb[tb > C.T_THR_BIKE_MIN].sum() / qb.sum():.1f} % above 20 min", fontsize=9)
    a = axs[row, 1]
    a.hist(tc / (thr_c / (1 + C.CAR_DELAY_TOLERANCE)), 40, weights=qc, color=COL[k])
    a.axvline(1 + C.CAR_DELAY_TOLERANCE, color="r", ls="--")
    a.set_title(f"{k} - drivers: {100 * qc[tc > thr_c].sum() / qc.sum():.2f} % above the threshold", fontsize=9)
axs[-1, 0].set_xlabel("bike trip time (min)"), axs[-1, 1].set_xlabel("car trip time / free-flow time")
print("figure:", P.save(fig, "fig7_trip_time_histograms.png"))

# ---- Fig. 8: shares of insufficient users ------------------------------------------------------------
shares = {}
for k in rep:
    S, tb, tc = arr[f"{k}__S"], arr[f"{k}__t_b"], arr[f"{k}__t_c"]
    shares[k] = [suf.bike_regions.share_below(suf.path_insufficiency(S, LAM)),
                 suf.bike_regions.share_below(suf.bike_time_insufficiency(tb)),
                 suf.car_regions.share_below(suf.car_time_insufficiency(tc))]
fig, axs = P.plt.subplots(1, 3, figsize=(11, 3.2))
for ax, j, title in zip(axs, range(3), ("cyclists below path-sufficiency threshold", "cyclists above 20 min",
                                         "drivers above the delay threshold")):
    v = [100 * shares[k][j] for k in rep]
    ax.bar(list(rep), v, color=[COL[k] for k in rep])
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.1f} %", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=9), ax.set_ylabel("%"), ax.tick_params(axis="x", labelsize=8)
print("figure:", P.save(fig, "fig8_insufficiency_shares.png"))

# ---- Fig. 9: maps ------------------------------------------------------------------------------------
import geopandas as gpd  # noqa: E402

iris = gpd.read_file(C.DATA / "regions" / "grenoble_iris.geojson")
fig, axs = P.plt.subplots(4, 3, figsize=(12, 15))
maps = []
for k in rep:
    S, tb, tc = arr[f"{k}__S"], arr[f"{k}__t_b"], arr[f"{k}__t_c"]
    maps.append([dict(zip(suf.bike_regions.zones, suf.bike_regions.regional(suf.path_insufficiency(S, LAM)))),
                 dict(zip(suf.bike_regions.zones, suf.bike_regions.regional(suf.bike_time_insufficiency(tb)))),
                 dict(zip(suf.car_regions.zones, suf.car_regions.regional(suf.car_time_insufficiency(tc))))])
vmax = [max(max(m[j].values()) for m in maps) for j in range(3)]
titles = ["path insufficiency $I_r$", "bike trip-time insufficiency", "car trip-time insufficiency"]
for c, k in enumerate(rep):
    dos = MM.bike.dos0 if k == "original" else utils.load_solution(
        C.RESULTS / "solutions" / (f"util_g{g_u:g}.csv" if k == "utilitarian" else f"suff_lam{LAM:g}_g{g_s:g}.csv"))
    P.bike_dos_map(axs[0, c], MM.bike, dos, lw=0.6, legend=(c == 0),
                   title=f"{k}: " + rf"$\bar D^b$={rep[k].D_bar:.2f} km, $P^b$={100 * rep[k].P_b:.2f} %")
    for j in range(3):
        ax = axs[j + 1, c]
        iris.assign(v=iris.zone_id.map(maps[c][j]).fillna(0)).plot(
            column="v", ax=ax, cmap="Reds", vmin=0, vmax=max(vmax[j], 1e-9), legend=(c == 2),
            edgecolor="0.6", lw=0.2)
        ax.set_axis_off(), ax.set_title(f"{k}: {titles[j]}", fontsize=9)
print("figure:", P.save(fig, "fig9_maps.png"))

# ---- Fig. 10: J_SI for every lambda -------------------------------------------------------------------
fig, ax = P.plt.subplots(figsize=(7, 3.4))
w = 0.27
for i, (k, color) in enumerate(COL.items()):
    vals = []
    for lam in C.LAMBDAS:
        if k == "original":
            vals.append(base[f"J_SI_{lam:g}"])
        elif k == "utilitarian":
            vals.append(rep["utilitarian"][f"J_SI_{lam:g}"])
        else:
            g = utils.representative_gamma(stars, "suff", lam)
            vals.append(ex[ex.name == f"suff_lam{lam:g}_g{g:g}"].iloc[0][f"J_SI_{lam:g}"])
    ax.bar(np.arange(5) + (i - 1) * w, vals, w, color=color, label=k)
ax.set_xticks(np.arange(5), [f"{int(100 * l)} %" for l in C.LAMBDAS]), ax.set_xlabel(r"threshold $\lambda$")
ax.set_ylabel(r"path insufficiency $J_{SI}$"), ax.legend(frameon=False, fontsize=8)
print("figure:", P.save(fig, "fig10_path_insufficiency_lambda.png"))

# ---- Fig. 11: system performance ------------------------------------------------------------------------
fig, axs = P.plt.subplots(1, 3, figsize=(11, 3.2))
names = list(rep)
axs[0].bar(names, [rep[k].J_E / base.J_E for k in names], color=[COL[k] for k in names])
axs[0].set_title(r"cyclist exposure $J_E / J_E^0$", fontsize=9)
axs[1].bar(names, [rep[k].T_c for k in names], color="#bbbbbb", label="cars")
axs[1].bar(names, [rep[k].T_b for k in names], bottom=[rep[k].T_c for k in names], color=[COL[k] for k in names],
           label="bikes")
for i, k in enumerate(names):
    axs[1].text(i, rep[k].T_tot, f"{rep[k].T_tot:,.0f}", ha="center", va="bottom", fontsize=8)
axs[1].set_title(r"total travel time $\mathcal{T}_{tot}$ (traveller-h, peak hour)", fontsize=9)
axs[1].legend(frameon=False, fontsize=7)
axs[2].bar(names, [100 * rep[k].P_b for k in names], color=[COL[k] for k in names])
axs[2].set_ylim(100 * base.P_b * 0.95, None), axs[2].set_title("bike share of bike + car trips (%)", fontsize=9)
for ax in axs:
    ax.tick_params(axis="x", labelsize=8)
print("figure:", P.save(fig, "fig11_system_performance.png"))
print("\n" + summary.round(4).to_string())

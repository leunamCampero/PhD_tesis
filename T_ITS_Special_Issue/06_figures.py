"""
Step 6 - Figures and summary table of the special-issue paper, in the style of paper/Images
(serif text, dotted grid, Okabe-Ito colours, labels Original / GA-U / GA-S).

fig4_pareto_fronts.png              utilitarian and sufficientarian fronts, both in the J_SI - T_tot plane
fig5_tradeoff_bars.png              min-max normalised indicators for every gamma (as paper Fig. 11)
fig6_path_safety_histograms.png     cyclists vs relative path safety, Original / GA-U / GA-S + change GA-S - GA-U
fig7_trip_time_histograms.png       cyclists and drivers vs trip time + change w.r.t. today
fig8_insufficiency_shares.png       % of cyclists / drivers below the sufficiency thresholds
fig9_maps.png                       DoS and regional insufficiency (shared colour scale per row) + GA-S - GA-U
fig10_path_insufficiency_lambda.png J_SI for every lambda: Original / GA-U / GA-S(lambda)
fig11_system_performance.png        exposure, total travel time and bike share
results/06_summary.csv              indicators of Original / GA-U / GA-S (lambda = 40 %)
results/06_representatives.csv      GA-S(lambda) vs GA-U for every lambda

Data: exact-UE evaluation of every design (step 5b).  The compared designs are chosen with
config.REPRESENTATIVE_RULE (src/si_cycling/selection.py).  The figures of the previous version are in
figures/v1_before_restyle/.
Run time: ~1 min
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import PowerNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from si_cycling import config as C, plotting as P, selection, utils  # noqa: E402
from si_cycling.problems import MultiModal  # noqa: E402

P.paper_style()
K = P.PAPER
LAM = C.LAMBDA_FIGURES
L = f"{LAM:g}"
CFG = K["config"]
NAMES = ("Original", "GA-U", "GA-S")

df = pd.read_csv(C.RESULTS / "05b_all_designs.csv")
arr = np.load(C.RESULTS / "05b_arrays.npz")
reps = selection.representatives(df)
base = df[df.kind == "original"].iloc[0]
rep = {"Original": base, "GA-U": reps["util"], "GA-S": reps[LAM]}
A = {k: {f: arr[f"{r['name']}__{f}"].astype(float) for f in ("S", "t_b", "t_c")} for k, r in rep.items()}
dos_of = {k: None if k == "Original" else utils.load_solution(C.RESULTS / "solutions" / f"{r['name']}.csv")
          for k, r in rep.items()}
MM = MultiModal(surrogate=False)
suf, bike, car = MM.suf, MM.bike, MM.car
dos_of["Original"] = bike.dos0
print("representatives:", {k: r["name"] for k, r in rep.items()})


def pct(x):
    return f"{100 * x:g} %"


# ---- tables ----------------------------------------------------------------------------------------
cols = ["J_E", f"J_SI_{L}", "J_TI_b", "J_TI_c", "T_c", "T_b", "T_tot", "P_b", "D_bar", f"path_below_{L}",
        "bike_above_20", "car_above_thr", "reduced_roads", "budget_used"]
summary = pd.DataFrame({k: r[cols] for k, r in rep.items()}).T
summary.insert(0, "design", [r["name"] for r in rep.values()])
summary.to_csv(C.RESULTS / "06_summary.csv")
tab = selection.table(df, reps)
tab.to_csv(C.RESULTS / "06_representatives.csv", index=False)

# ---- Fig. 4: Pareto fronts -------------------------------------------------------------------------------
U, S4 = selection.util_front(df), selection.suff_front(df, LAM)
fig, axs = P.plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
ax = axs[0]
ax.plot(100 * (U.T_tot / base.T_tot - 1), U.J_E / base.J_E, "o-", color=CFG["GA-U"], ms=5, label="GA-U front")
for _, r in U.iterrows():
    ax.annotate(f"{r.gamma:g}", (100 * (r.T_tot / base.T_tot - 1), r.J_E / base.J_E), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.scatter(100 * (rep["GA-U"].T_tot / base.T_tot - 1), rep["GA-U"].J_E / base.J_E, s=160, facecolor="none",
           edgecolor=K["thr"], lw=1.5, zorder=5, label="GA-U (compared)")
ax.scatter([0], [1], marker="*", s=160, color="k", label="Original", zorder=5)
ax.set_xlabel(r"total travel time change $\mathcal{T}_{\mathrm{tot}}$ (%)"), ax.set_ylabel(r"$J_\mathcal{E}/J_\mathcal{E}^0$")
ax.set_title(r"utilitarian front (labels: $\gamma$)"), ax.legend(fontsize=8)
ax = axs[1]
for lam in C.LAMBDAS:
    g = selection.suff_front(df, lam).sort_values("J_TI")
    y = g[f"J_SI_{lam:g}"].to_numpy()
    nd = y <= np.minimum.accumulate(y)                     # non-dominated designs (line); dominated: hollow
    ax.plot(g.J_TI[nd], y[nd], "o-", color=K["lam"][lam], ms=4, label=rf"$\lambda$ = {pct(lam)}")
    ax.scatter(g.J_TI[~nd], y[~nd], s=16, facecolor="none", edgecolor=K["lam"][lam], lw=0.8)
    ax.scatter(reps[lam].J_TI, reps[lam][f"J_SI_{lam:g}"], s=130, facecolor="none", edgecolor=K["thr"], lw=1.4)
ax.set_xlabel(r"trip insufficiency $J_{\mathrm{TI}}^\mathrm{b}+J_{\mathrm{TI}}^\mathrm{c}$")
ax.set_ylabel(r"path insufficiency $J_{\mathrm{SI}}$"), ax.set_yscale("log")
ax.set_title(r"sufficientarian fronts (red: GA-S($\lambda$), hollow: dominated)"), ax.legend(fontsize=8)
ax = axs[2]
for g, lab in ((U, "GA-U front"), (S4, rf"GA-S front, $\lambda$ = {pct(LAM)}")):
    c = CFG["GA-U" if g is U else "GA-S"]
    ax.plot(100 * (g.T_tot / base.T_tot - 1), g[f"J_SI_{L}"], "o-", color=c, ms=5, label=lab)
for k in ("GA-U", "GA-S"):
    ax.scatter(100 * (rep[k].T_tot / base.T_tot - 1), rep[k][f"J_SI_{L}"], s=160, facecolor="none",
               edgecolor=K["thr"], lw=1.5, zorder=5)
    ax.annotate(k, (100 * (rep[k].T_tot / base.T_tot - 1), rep[k][f"J_SI_{L}"]), xytext=(-55, -28) if k == "GA-U" else (14, 16),
                textcoords="offset points", fontsize=10, color=CFG[k], weight="bold",
                arrowprops=dict(arrowstyle="-", color=CFG[k], lw=0.8))
ax.scatter([0], [base[f"J_SI_{L}"]], marker="*", s=160, color="k", label="Original", zorder=5)
ax.set_yscale("log"), ax.set_xlabel(r"total travel time change $\mathcal{T}_{\mathrm{tot}}$ (%)")
ax.set_ylabel(rf"path insufficiency $J_{{\mathrm{{SI}}}}$ ($\lambda$ = {pct(LAM)})")
ax.set_title("both fronts (exact UE)"), ax.legend(fontsize=8)
print("figure:", P.save(fig, "fig4_pareto_fronts.png"))

# ---- Fig. 5: normalised indicators per gamma (paper Fig. 11) ---------------------------------------------
panels = [("util", U, [("J_E", r"Cyclist exposure ($J_\mathcal{E}$)", K["J_E"]),
                       ("T_b", r"Bike time ($\mathcal{T}^\mathrm{b}$)", K["J_TI_b"]),
                       ("T_c", r"Car time ($\mathcal{T}^\mathrm{c}$)", K["J_TI_c"])],
           "(a) utilitarian", rep["GA-U"].gamma),
          ("suff", S4, [(f"J_SI_{L}", r"Cyclist path insufficiency ($J_\mathrm{SI}$)", K["J_SI"]),
                        ("J_TI_b", r"Bike time insufficiency ($J_\mathrm{TI}^\mathrm{b}$)", K["J_TI_b"]),
                        ("J_TI_c", r"Car time insufficiency ($J_\mathrm{TI}^\mathrm{c}$)", K["J_TI_c"])],
           rf"(b) sufficientarian, $\lambda$ = {pct(LAM)}", rep["GA-S"].gamma)]
fig, axs = P.plt.subplots(2, 1, figsize=(11, 6.6), constrained_layout=True)
for ax, (kind, g, inds, title, gstar) in zip(axs, panels):
    rows = [base] + [r for _, r in g.iterrows()]
    x = np.arange(len(rows))
    w = 0.27
    for j, (c, lab, color) in enumerate(inds):
        v = np.array([r[c] for r in rows], float)
        v = (v - v.min()) / max(v.max() - v.min(), 1e-12)          # min-max over the 12 configurations
        ax.bar(x + (j - 1) * w, np.maximum(v, 1e-3), w, color=color, label=lab, edgecolor="white", lw=0.3)
    ax.set_yscale("log"), ax.set_ylim(1e-2, 1.6)
    labels = ["Original"] + [r"$\gamma$=" + (f"{int(round(100 * r.gamma))}%" if 0 < r.gamma < 1 else f"{r.gamma:g}")
                             for r in rows[1:]]
    ax.set_xticks(x, labels, fontsize=9)
    for t, r in zip(ax.get_xticklabels()[1:], rows[1:]):
        if np.isclose(r.gamma, gstar):
            t.set_color(K["thr"]), t.set_fontweight("bold")
    ax.set_ylabel("normalised value $\\in[0,1]$"), ax.set_title(title, fontsize=10, loc="left")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
axs[-1].set_xlabel(r"$\gamma$ - weight of the safety term   (red: design compared in Figs. 6-11)")
fig.text(0.01, -0.03, "Each indicator is min-max normalised over Original + the 11 designs (1 = worst, 0 = best). "
         r"At $\gamma\to 1$ only safety counts: the GA takes car lanes on busy roads and the car delay "
         "insufficiency reaches its maximum (7 % of drivers above the threshold vs 1.1 % today).",
         fontsize=8, wrap=True)
print("figure:", P.save(fig, "fig5_tradeoff_bars.png"))

# ---- Fig. 6: path-safety histograms -------------------------------------------------------------------------
mask = suf.mask
thr = suf.S_thr[LAM] / suf.S_G                          # = lambda on the relative scale
bins = np.logspace(-2.6, 1.0, 37)
q = {k: bike.Q[mask] * r.bike_scale for k, r in rep.items()}
rel = {k: A[k]["S"][mask] / suf.S_G for k in rep}
H = {k: np.histogram(rel[k], bins, weights=q[k])[0] for k in rep}
fig, axs = P.plt.subplots(4, 1, figsize=(7.5, 10), sharex=True, constrained_layout=True,
                          gridspec_kw=dict(height_ratios=[1, 1, 1, 0.8]))
for ax, k in zip(axs, NAMES):
    below = rel[k] < thr
    sh = 100 * q[k][below].sum() / q[k].sum()
    ax.hist(rel[k][below], bins, weights=q[k][below], color=K["below"], edgecolor="k", lw=0.4)
    ax.hist(rel[k][~below], bins, weights=q[k][~below], color=K["above"], edgecolor="k", lw=0.4)
    handles = [Patch(fc=K["below"], ec="k", lw=0.4, label=rf"below $S_\mathrm{{thr}}$: {sh:.1f} % of cyclists"),
               Patch(fc=K["above"], ec="k", lw=0.4, label=rf"above $S_\mathrm{{thr}}$: {100 - sh:.1f} %")]
    if k != "Original":
        ax.stairs(H["Original"], bins, color="0.25", lw=1.1, ls="--")
        handles.append(Line2D([], [], color="0.25", lw=1.1, ls="--", label="Original"))
    ax.axvline(thr, color=K["thr"], ls="--", lw=1.2), ax.axvline(1, color=K["avg"], ls="--", lw=1.2)
    handles += [Line2D([], [], color=K["thr"], ls="--", label=rf"$S_\mathrm{{thr}}$ ($\lambda$ = {pct(LAM)})"),
                Line2D([], [], color=K["avg"], ls="--", label=r"$S_\mathcal{G}$ today")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left")
    ax.set_xscale("log"), ax.set_ylabel("Cyclists / day")
    ax.set_title(k, fontsize=10, loc="left", color=CFG[k] if k != "Original" else "k", weight="bold")
for ax in axs[:3]:                                       # same y scale: the panels are directly comparable
    ax.set_ylim(0, 1.08 * max(h.max() for h in H.values()))
d = H["GA-S"] - H["GA-U"]
ctr = np.sqrt(bins[:-1] * bins[1:])
axs[3].bar(ctr, d, width=np.diff(bins), color=np.where(d >= 0, CFG["GA-S"], CFG["GA-U"]), edgecolor="k", lw=0.3)
axs[3].axvline(thr, color=K["thr"], ls="--", lw=1.2), axs[3].axhline(0, color="k", lw=0.6)
moved = (q["GA-S"][rel["GA-S"] >= thr].sum() - q["GA-U"][rel["GA-U"] >= thr].sum())
axs[3].set_title(f"GA-S minus GA-U: {moved:+,.0f} cyclists / day above the threshold "
                 "(blue: more with GA-S, orange: more with GA-U)", fontsize=9, loc="left")
axs[3].set_ylabel(r"$\Delta$ cyclists / day")
axs[3].set_xlabel(r"path safety relative to today's network average, $S_p / S_\mathcal{G}$")
print("figure:", P.save(fig, "fig6_path_safety_histograms.png"))

# ---- Fig. 7: trip-time histograms ---------------------------------------------------------------------------
cm = car.Q_base > 0
ff = suf.T_thr_car[cm] / (1 + C.CAR_DELAY_TOLERANCE)
data = {"bike": {k: (A[k]["t_b"][mask], bike.Q[mask] * rep[k].bike_scale) for k in rep},
        "car": {k: (A[k]["t_c"][cm] / ff, car.Q[cm] * rep[k].car_scale) for k in rep}}
spec = {"bike": (np.linspace(0, 40, 41), C.T_THR_BIKE_MIN, "bike trip time (min)", "Cyclists / day",
                 r"$T_\mathrm{thr}$ = 20 min", (16, 34)),
        "car": (np.linspace(1, 1.6, 49), 1 + C.CAR_DELAY_TOLERANCE, "car trip time / free-flow time",
                "Drivers / peak hour", r"$T^\mathrm{c}_\mathrm{thr}$ = 1.25 $\times$ free flow", (1.18, 1.45))}
fig, axs = P.plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
for col, mode in enumerate(("bike", "car")):
    b, t, xl, yl, tl, zoom = spec[mode]
    ax = axs[0, col]
    hh = {}
    for k in NAMES:
        x, w = data[mode][k]
        hh[k] = np.histogram(x, b, weights=w)[0]
        share = 100 * w[x > t].sum() / w.sum()
        if k == "Original":
            ax.hist(x, b, weights=w, color=K["hist"], edgecolor="k", lw=0.4, label=f"Original ({share:.2f} % above)")
        else:
            ax.stairs(hh[k], b, color=CFG[k], lw=1.8, label=f"{k} ({share:.2f} % above)")
    ax.axvline(t, color=K["thr"], ls="--", lw=1.2, label=tl)
    ax.set_ylabel(yl), ax.legend(fontsize=8, loc="upper right")
    ax.set_title(("(a) cyclists" if mode == "bike" else "(b) drivers"), loc="left", fontsize=10)
    ins = ax.inset_axes([0.55, 0.12, 0.42, 0.38])
    for k in NAMES:
        if k == "Original":
            ins.stairs(hh[k], b, fill=True, color=K["hist"], ec="k", lw=0.3)
        else:
            ins.stairs(hh[k], b, color=CFG[k], lw=1.5)
    ins.axvline(t, color=K["thr"], ls="--", lw=1)
    ins.set_xlim(*zoom), ins.set_yscale("log"), ins.tick_params(labelsize=7)
    ins.set_title("tail (log scale)", fontsize=7)
    ax = axs[1, col]
    ctr, wd = 0.5 * (b[:-1] + b[1:]), np.diff(b)
    for j, k in enumerate(("GA-U", "GA-S")):
        ax.bar(ctr + (j - 0.5) * wd * 0.42, hh[k] - hh["Original"], wd * 0.42, color=CFG[k], label=f"{k} - Original")
    ax.axhline(0, color="k", lw=0.6), ax.axvline(t, color=K["thr"], ls="--", lw=1.2)
    ax.set_xlabel(xl), ax.set_ylabel(r"$\Delta$ " + yl.lower()), ax.legend(fontsize=8)
    ax.set_title("change with respect to today (bars < 0: fewer users in that time band)", fontsize=9, loc="left")
print("figure:", P.save(fig, "fig7_trip_time_histograms.png"))

# ---- Fig. 8: shares of insufficient users ----------------------------------------------------------------------
fig, axs = P.plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
items = [(f"path_below_{L}", r"(a) path insufficiency: cyclists with $S_p<S_\mathrm{thr}$", r"$S_\mathrm{thr}$"),
         ("bike_above_20", r"(b) cyclists with a trip above 20 min", r"$T_\mathrm{thr}$"),
         ("car_above_thr", r"(c) drivers above the delay threshold", r"$T^\mathrm{c}_\mathrm{thr}$")]
for ax, (c, title, sym) in zip(axs, items):
    v = np.array([rep[k][c] for k in NAMES])
    ax.bar(NAMES, v, color=[CFG[k] for k in NAMES], edgecolor="k", lw=0.5, width=0.6)
    for i, x in enumerate(v):
        txt = f"{x:.2f} %" if c != f"path_below_{L}" else f"{x:.1f} %"
        if i:
            txt += f"\n({100 * (x / v[0] - 1):+.0f} %)"
        ax.text(i, x, txt, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, v.max() * 1.3), ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_ylabel("% of users not sufficient")
print("figure:", P.save(fig, "fig8_insufficiency_shares.png"))

# ---- Fig. 9: maps --------------------------------------------------------------------------------------------
import geopandas as gpd  # noqa: E402

iris = gpd.read_file(C.DATA / "regions" / "grenoble_iris.geojson")
reg = {}
for k in NAMES:
    reg[k] = [dict(zip(suf.bike_regions.zones, suf.bike_regions.regional(suf.path_insufficiency(A[k]["S"], LAM)))),
              dict(zip(suf.bike_regions.zones, suf.bike_regions.regional(suf.bike_time_insufficiency(A[k]["t_b"])))),
              dict(zip(suf.car_regions.zones, suf.car_regions.regional(suf.car_time_insufficiency(A[k]["t_c"]))))]
titles = [r"path insufficiency $\mathcal{I}_r$", "bike trip-time insufficiency", "car trip-time insufficiency"]
fig, axs = P.plt.subplots(4, 4, figsize=(16, 16.5), constrained_layout=True)
for c, k in enumerate(NAMES):
    P.bike_dos_map(axs[0, c], bike, dos_of[k], lw=0.6, legend=(c == 0),
                   title=f"{k}: " + rf"$\bar D^\mathrm{{b}}$ = {rep[k].D_bar:.2f} km, $P^\mathrm{{b}}$ = {100 * rep[k].P_b:.2f} %")
# DoS difference GA-S vs GA-U
ax = axs[0, 3]
seg = np.stack([bike.xy[bike.u], bike.xy[bike.v]], axis=1)
du, ds = dos_of["GA-U"], dos_of["GA-S"]
ax.add_collection(LineCollection(seg, colors="0.85", linewidths=0.4))
for sel, color in ((ds > du, CFG["GA-S"]), (du > ds, CFG["GA-U"])):
    ax.add_collection(LineCollection(seg[sel], colors=color, linewidths=1.6))
ax.autoscale(), ax.set_aspect(1 / np.cos(np.deg2rad(bike.xy[:, 1].mean()))), ax.set_axis_off()
ax.legend(handles=[Line2D([], [], color=CFG["GA-S"], lw=2, label=f"safer with GA-S ({int((ds > du).sum())} links)"),
                   Line2D([], [], color=CFG["GA-U"], lw=2, label=f"safer with GA-U ({int((du > ds).sum())} links)")],
          fontsize=8, loc="lower left")
ax.set_title("where the two designs differ", fontsize=10)
for j in range(3):
    vmax = max(max(reg[k][j].values()) for k in NAMES)
    norm = PowerNorm(0.5, vmin=0, vmax=max(vmax, 1e-9))      # same scale for the 3 maps, low values visible
    for c, k in enumerate(NAMES):
        ax = axs[j + 1, c]
        iris.assign(v=iris.zone_id.map(reg[k][j]).fillna(0)).plot(column="v", ax=ax, cmap="viridis", norm=norm,
                                                                  edgecolor="white", lw=0.3)
        mean = np.mean(list(reg[k][j].values()))
        ax.set_axis_off(), ax.set_title(f"{k}: {titles[j]} (mean {mean:.3g})", fontsize=9)
    sm = P.plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, ax=axs[j + 1, :3], shrink=0.75, label=titles[j])
    diff = {z: reg["GA-S"][j][z] - reg["GA-U"][j].get(z, 0) for z in reg["GA-S"][j]}
    m = max(max(abs(v) for v in diff.values()), 1e-9)
    ax = axs[j + 1, 3]
    iris.assign(v=iris.zone_id.map(diff).fillna(0)).plot(column="v", ax=ax, cmap="RdBu_r",
                                                         norm=TwoSlopeNorm(0, -m, m), edgecolor="0.6", lw=0.3)
    better = sum(v < 0 for v in diff.values())
    ax.set_axis_off(), ax.set_title(f"GA-S minus GA-U (blue = GA-S better: {better}/{len(diff)} zones)", fontsize=9)
    fig.colorbar(P.plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -m, m), cmap="RdBu_r"), ax=ax, shrink=0.75)
print("figure:", P.save(fig, "fig9_maps.png"))

# ---- Fig. 10: J_SI for every lambda -----------------------------------------------------------------------------
fig, ax = P.plt.subplots(figsize=(9, 3.8), constrained_layout=True)
w = 0.26
x = np.arange(len(C.LAMBDAS))
vals = {"Original": [base[f"J_SI_{l:g}"] for l in C.LAMBDAS],
        "GA-U": [reps["util"][f"J_SI_{l:g}"] for l in C.LAMBDAS],
        "GA-S": [reps[l][f"J_SI_{l:g}"] for l in C.LAMBDAS]}
for i, k in enumerate(NAMES):
    ax.bar(x + (i - 1) * w, vals[k], w, color=CFG[k], edgecolor="k", lw=0.4,
           label=k if k != "GA-S" else r"GA-S($\lambda$)")
    for xi, v in zip(x, vals[k]):
        ax.text(xi + (i - 1) * w, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
for xi, u_, s_ in zip(x, vals["GA-U"], vals["GA-S"]):
    ax.annotate(f"{100 * (s_ / u_ - 1):+.0f} %", (xi + 0.5 * w, max(u_, s_)), xytext=(0, 16), textcoords="offset points",
                ha="center", fontsize=9, color=CFG["GA-S"], weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=CFG["GA-S"], lw=0.6))
ax.set_xticks(x, [rf"$\lambda$ = {pct(l)}" for l in C.LAMBDAS]), ax.set_ylabel(r"$J_\mathrm{SI}$")
ax.set_title(r"path insufficiency for every threshold (blue %: GA-S($\lambda$) vs GA-U)", fontsize=10, loc="left")
ax.legend(fontsize=8)
print("figure:", P.save(fig, "fig10_path_insufficiency_lambda.png"))

# ---- Fig. 11: system performance ----------------------------------------------------------------------------------
fig, axs = P.plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
cols3 = [CFG[k] for k in NAMES]
v = [rep[k].J_E for k in NAMES]
axs[0].bar(NAMES, v, color=cols3, edgecolor="k", lw=0.4)
for i, y in enumerate(v):
    axs[0].text(i, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
axs[0].set_title(r"(a) cyclist exposure $J_\mathcal{E}$", fontsize=10, loc="left")
axs[1].bar(NAMES, [rep[k].T_c for k in NAMES], color="#bbbbbb", edgecolor="k", lw=0.4, label="cars")
axs[1].bar(NAMES, [rep[k].T_b for k in NAMES], bottom=[rep[k].T_c for k in NAMES], color=cols3, edgecolor="k",
           lw=0.4, label="bikes")
for i, k in enumerate(NAMES):
    axs[1].text(i, rep[k].T_tot, f"{rep[k].T_tot:,.0f}\n({100 * (rep[k].T_tot / base.T_tot - 1):+.2f} %)",
                ha="center", va="bottom", fontsize=8)
axs[1].set_ylim(0, base.T_tot * 1.25)
axs[1].set_title(r"(b) total travel time $\mathcal{T}_\mathrm{tot}$ (traveller-h, peak h)", fontsize=10, loc="left")
axs[1].legend(handles=[Patch(fc="#bbbbbb", ec="k", lw=0.4, label="cars"),
                       Patch(fc="white", ec="k", lw=0.4, label="bikes (colour of the design)")], fontsize=8, loc="lower right")
v = [100 * rep[k].P_b for k in NAMES]
axs[2].bar(NAMES, v, color=cols3, edgecolor="k", lw=0.4)
for i, y in enumerate(v):
    axs[2].text(i, y, f"{y:.2f} %", ha="center", va="bottom", fontsize=9)
axs[2].set_ylim(100 * base.P_b * 0.95, max(v) * 1.03)
axs[2].set_title(r"(c) bike share $P^\mathrm{b}$ (%)", fontsize=10, loc="left")
print("figure:", P.save(fig, "fig11_system_performance.png"))

print("\n" + summary.round(4).to_string())
print("\nGA-S(lambda) vs GA-U:\n" + tab.round(4).to_string(index=False))

"""
Step 7 - figures of the paper (style of paper/T_ITS_2026.pdf) and summary tables.

fig04_exposure_flow_sweep    E^clust and Phi^clust vs number of clusters, k-means vs CC  (T-ITS Fig. 4)
fig05_clustering_maps        original / k-means / CC networks of Lyon and Strasbourg      (T-ITS Fig. 5)
fig06_exposure_bars          normalised exposure of every strategy and budget            (T-ITS Fig. 6)
fig07_convergence            exposure and run time over the generations, paper-like GA from a random
                             population (05_optimisation.py --random-init)                (T-ITS Fig. 7)
fig08_<city>_strategies      the 8 maps of the optimisation / transfer at beta = BUDGET_FIGURE (Fig. 8)
fig09_gw_distances           GW and kernel distances to Grenoble (Section VII text)
itsc_fig06_top_kernels       top-100 exposure kernels                                     (ITSC Fig. 6)
itsc_fig07_node_strength     node strength of Grenoble and the clustered cities           (ITSC Fig. 7)
results/07_summary.csv       one line per city x method x beta
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import compare as cp, config as C  # noqa: E402
from cyclenet.clustered import ClusteredNetwork  # noqa: E402
from cyclenet.network import Network  # noqa: E402
from cyclenet.plotting import (BUDGET_COLORS, CLUSTER_COLORS, FLOW_COLOR, METHOD_COLORS, beta_label,  # noqa: E402
                               cluster_graph_map, dos_map, paper_style, plt, save)
from cyclenet.utils import log  # noqa: E402

R = C.RESULTS


# ==============================================================================================
def fig04(sweep, k_ref):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, city in zip(axs, C.TARGETS):
        d = sweep[sweep.city == city]
        km = d[d.method == "kmeans"].sort_values("k", ascending=False)
        cc = d[d.method == "CC"].sort_values("k", ascending=False)
        x = np.arange(len(km))
        E0 = km.E_orig.iloc[0]
        wdt = 0.4
        ax.bar(x - wdt / 2, km.E_clust / E0, wdt, color=CLUSTER_COLORS["kmeans"], edgecolor="0.3", lw=0.4,
               label=r"Exposure ($\mathcal{E}^{clust}$) $k$-means")
        ax.bar(x + wdt / 2, cc.E_clust / E0, wdt, color=CLUSTER_COLORS["CC"], edgecolor="0.3", lw=0.4,
               label=r"Exposure ($\mathcal{E}^{clust}$) CC Algorithm")
        for xi, (a, b) in enumerate(zip(km.dE_pct, cc.dE_pct)):
            ax.text(xi - wdt / 2, max(1, 1 + a / 100) + 0.005, f"{a:+.1f}", rotation=90, fontsize=6.5,
                    ha="center", va="bottom", color="#2f5f8a")
            ax.text(xi + wdt / 2, max(1, 1 + b / 100) + 0.005, f"{b:+.1f}", rotation=90, fontsize=6.5,
                    ha="center", va="bottom", color="#b36b00")
        ax.axhline(1.0, color="0.3", lw=0.8)
        lo = min(km.E_clust.min(), cc.E_clust.min()) / E0
        hi = max(km.E_clust.max(), cc.E_clust.max()) / E0
        ax.set_ylim(min(lo, 1) - 0.04, max(hi, 1) + 0.05)
        ax.set_xticks(x, km.k.astype(int), rotation=45, fontsize=8)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel(r"Clustered exposure $\mathcal{E}^{clust}/\mathcal{E}^{orig}$")
        ax.set_title(city)
        ref = int(np.flatnonzero(km.k.to_numpy() == k_ref)[0])
        ax.axvline(ref, color="blue", ls=":", lw=1.5)
        div = np.flatnonzero(np.abs(cc.dE_pct.to_numpy()) > 100 * C.CC_TAU_E0)
        if len(div):
            ax.axvline(div[0], color="red", ls=":", lw=1.5)
        ax2 = ax.twinx()
        ax2.plot(x, km.Phi_clust / km.Phi_orig.iloc[0], "--", marker="s", ms=3.5, color=CLUSTER_COLORS["kmeans"],
                 label=r"Clustered flow ($\Phi^{clust}$) $k$-means")
        ax2.plot(x, cc.Phi_clust / cc.Phi_orig.iloc[0], "--", marker="o", ms=3.5, color=FLOW_COLOR,
                 label=r"Clustered flow ($\Phi^{clust}$) CC")
        ax2.set_ylabel(r"Clustered flow $\Phi^{clust}/\Phi^{orig}$", color=FLOW_COLOR)
        ax2.tick_params(axis="y", colors=FLOW_COLOR)
        ax2.grid(False)
        ax2.set_ylim(0.85, 1.15)
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    extra = [Line2D([], [], color="blue", ls=":"), Line2D([], [], color="red", ls=":")]
    fig.legend(h1 + h2 + extra, l1 + l2 + ["Grenoble nodes", r"Divergence CC ($|\Delta\mathcal{E}|>1\%$)"],
               loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig04_exposure_flow_sweep.png")


def fig05(nets, cns):
    fig, axs = plt.subplots(2, 3, figsize=(12, 10.5))
    for row, city in enumerate(C.TARGETS):
        net = nets[city]
        dos_map(axs[row, 0], net, net.dos0, title=f"{city}: original ({net.N:,} nodes)" if row == 0 else
                f"{city} ({net.N:,} nodes)")
        for col, (m, ttl) in enumerate((("kmeans", "$K$-mean"), ("CC", "CC-Algorithm")), 1):
            cn = cns[(city, m)]
            mixed = 100 * (cn.w0max > cn.w0)[cn.arc_obj].mean()
            dos_map(axs[row, col], net, cn.object_dos_of_arcs(), legend_title="Min DoS",
                    title=f"{ttl}: $k$={cn.k:,}, mixed-DoS arcs {mixed:.1f} %")
    for ax, t in zip(axs[1], ["(a) Original", "(b) $K$-mean", "(c) CC-Algorithm"]):
        ax.set_xlabel(t, fontsize=10)
    fig.tight_layout()
    save(fig, "fig05_clustering_maps.png")


def fig06(summary):
    order = ["GA", "GA-Cluster", "GA-Cluster (k-means)", "Transfer", "Transfer (k-means)", "Transfer+"]
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.2))
    for ax, city in zip(axs, ("Strasbourg", "Lyon")):
        d = summary[summary.city == city]
        meths = [m for m in order if m in set(d.method)]
        x = np.arange(len(C.BUDGET_FRACTIONS))
        w = 0.8 / len(meths)
        for i, m in enumerate(meths):
            vals = [d[(d.method == m) & np.isclose(d.beta, b)].E_norm.mean() for b in C.BUDGET_FRACTIONS]
            ax.bar(x + (i - (len(meths) - 1) / 2) * w, vals, w, color=METHOD_COLORS[m], edgecolor="0.25", lw=0.5,
                   label=m, hatch="//" if "k-means" in m else None)
            for xi, v in zip(x, vals):
                ax.text(xi + (i - (len(meths) - 1) / 2) * w, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.6, rotation=90)
        ax.set_xticks(x, [beta_label(b) for b in C.BUDGET_FRACTIONS])
        ax.set_xlabel(r"Budget fraction $\beta$ ($B=\beta B_{max}$)")
        ax.set_ylabel(r"Normalized cyclist exposure $\mathcal{E}^{orig}$")
        ax.set_ylim(0, 1.08)
        ax.set_title(city, fontsize=10)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig06_exposure_bars.png")


def fig07(runs_dir):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
    for col, city in enumerate(("Strasbourg", "Lyon")):
        ax, axt = axs[0, col], axs[1, col]
        for b, colr in zip(C.BUDGET_FRACTIONS, BUDGET_COLORS):
            g = np.load(runs_dir / f"{city}_GA_{b:g}.npz")
            c = np.load(runs_dir / f"{city}_CC_{b:g}.npz")
            gens = np.arange(len(g["history"]))
            ax.plot(gens, g["history"] / g["model_base"], "-", marker="o", ms=2.2, color=colr, lw=1.2,
                    label=f"{beta_label(b)} GA")
            lifted = c["lifted"]
            E0 = g["model_base"]
            if len(lifted):
                steps = np.full(len(gens), np.nan)
                for gen, val in lifted:
                    steps[int(gen):] = val / E0
                ax.plot(gens, steps, "--", marker="*", ms=3, color=colr, lw=1.0, label=f"{beta_label(b)} GA-Cluster")
            axt.plot(gens, g["times"] / 60, "-", marker="o", ms=2, color=colr, lw=1.2, label=f"{beta_label(b)} GA")
            axt.plot(np.arange(len(c["times"])), c["times"] / 60, "--", marker="*", ms=3, color=colr, lw=1.0,
                     label=f"{beta_label(b)} GA-Cluster")
        ax.set_ylabel(r"Normalized cyclist traffic exposure $\mathcal{E}^{orig}$")
        ax.set_title(city, fontsize=10)
        axt.set_ylabel("Runtime (minutes)")
        for a in (ax, axt):
            a.set_xlabel("Generation")
            a.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig07_convergence.png")


def fig08(city, net, H, cns, beta):
    runs, tr = R / "05_runs", R / "06_runs"
    cn = cns[(city, "CC")]
    ga = np.load(runs / f"{city}_GA_{beta:g}.npz")
    gc = np.load(runs / f"{city}_CC_{beta:g}.npz")
    gh = np.load(runs / f"{C.REFERENCE}_GA_{beta:g}.npz")
    t = np.load(tr / f"{city}_Transfer_{beta:g}.npz")
    fig, axs = plt.subplots(2, 4, figsize=(17, 10))
    up = lambda dos: dos > net.dos0  # noqa: E731
    panels = [
        (axs[0, 0], lambda ax: dos_map(ax, net, net.dos0), f"(a) Original {city}"),
        (axs[0, 1], lambda ax: dos_map(ax, net, ga["dos"], highlight=up(ga["dos"])), "(b) GA on original network"),
        (axs[0, 2], lambda ax: dos_map(ax, net, gc["dos"], highlight=up(gc["dos"])), "(c) Clustered GA lifted to original"),
        (axs[0, 3], lambda ax: dos_map(ax, net, t["dos"], highlight=up(t["dos"])),
         f"(d) Transferred {C.REFERENCE} strategy lifted"),
        (axs[1, 0], lambda ax: cluster_graph_map(ax, cn), f"(e) Clustered {city}"),
        (axs[1, 1], lambda ax: cluster_graph_map(ax, cn, gc["levels"]), f"(f) GA on clustered {city}"),
        (axs[1, 2], lambda ax: dos_map(ax, H, gh["dos"], highlight=gh["dos"] > H.dos0), f"(g) GA on {C.REFERENCE}"),
        (axs[1, 3], lambda ax: cluster_graph_map(ax, cn, t["levels"]), f"(h) {C.REFERENCE} strategy transferred"),
    ]
    for ax, fn, ttl in panels:
        fn(ax)
        ax.set_xlabel(ttl, fontsize=10)
    fig.suptitle(f"{city}, {beta_label(beta)} (upgraded arcs drawn thick with a blue halo, unchanged arcs faded)", fontsize=11)
    fig.tight_layout()
    save(fig, f"fig08_{city.lower()}_strategies.png")


def fig09(comp):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    d = comp[comp.method.isin(["CC", "kmeans"]) & (comp.source == C.REFERENCE)]
    for ax, col, ttl in ((axs[0], "GW", r"GW($\mathcal{H}^{Grenoble}$, $\tilde{\mathcal{G}}$)"),
                         (axs[1], "kernel_distance", "kernel (graphon) distance")):
        x = np.arange(len(C.GW_SIZES)) if col == "GW" else np.arange(1)
        i = 0
        for city in C.TARGETS:
            for m, hatch in (("CC", None), ("kmeans", "//")):
                s = d[(d.target == city) & (d.method == m)]
                vals = s.GW.to_numpy() if col == "GW" else s.kernel_distance.to_numpy()[:1]
                colr = {"Lyon": "#D55E00", "Strasbourg": "#009E73"}[city]
                ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=colr, hatch=hatch, edgecolor="0.25", lw=0.5,
                       alpha=1 if m == "CC" else 0.55, label=f"{city} ({'CC' if m == 'CC' else 'k-means'})")
                i += 1
        ax.set_xticks(x, [f"M={m}" for m in C.GW_SIZES] if col == "GW" else [""])
        ax.set_title(ttl, fontsize=10)
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig09_gw_distances.png")


def itsc_figs(H, cns):
    nets = [("Grenoble", cp.kernel_original(H), H.lonlat),
            ("Lyon (Clustered)", cp.kernel_clustered(cns[("Lyon", "CC")]), _cl_lonlat(cns[("Lyon", "CC")])),
            ("Strasbourg (Clustered)", cp.kernel_clustered(cns[("Strasbourg", "CC")]),
             _cl_lonlat(cns[("Strasbourg", "CC")]))]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, K, ll) in zip(axs, nets):
        sub, _ = cp.top_kernel(K, ll)
        im = ax.imshow(sub, cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{name.split(' ')[0]} top {C.TOP_KERNEL_NODES} nodes")
        ax.set_xlabel("Nodes"), ax.set_ylabel("Nodes")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, "itsc_fig06_top_kernels.png")

    fig, axs = plt.subplots(3, 1, figsize=(12, 7.5))
    for ax, (name, K, ll) in zip(axs, nets):
        s = cp.strength(K)
        order = np.lexsort((ll[:, 1], ll[:, 0]))
        pos = s[s > 0]
        v = np.log1p(s[order] / np.median(pos))
        ax.bar(np.arange(len(v)), v / v.max(), width=1.0, color="#4a90c8", alpha=0.6, lw=0)
        ax.set_title(f"{name} node strength", fontsize=9)
        ax.set_ylabel("Strength")
        ax.set_xlabel("Node index (sorted by lon, lat)")
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, "itsc_fig07_node_strength.png")


def _cl_lonlat(cn):
    from cyclenet.plotting import cluster_lonlat
    return cluster_lonlat(cn)


# ==============================================================================================
def main():
    paper_style()
    H = Network(C.REFERENCE)
    nets = {c: Network(c) for c in C.TARGETS}
    cns = {(c, m): ClusteredNetwork(nets[c], np.load(R / f"03_labels_{c}_{m}.npy")) for c in C.TARGETS
           for m in ("CC", "kmeans")}
    sweep = pd.read_csv(R / "03_sweep.csv")
    fig04(sweep, H.N)
    fig05(nets, cns)
    log("figs 4-5")
    if (R / "04_comparison.csv").exists():
        fig09(pd.read_csv(R / "04_comparison.csv"))
    itsc_figs(H, cns)
    log("ITSC figs")
    if (R / "05_optimisation.csv").exists() and (R / "06_transfer.csv").exists():
        opt = pd.read_csv(R / "05_optimisation.csv")
        trf = pd.read_csv(R / "06_transfer.csv")
        cols = ["city", "method", "beta", "budget", "cost", "E_norm", "upgraded_km"]
        summary = pd.concat([opt[cols + ["seconds"]], trf[cols + ["seconds"]]], ignore_index=True)
        summary.to_csv(R / "07_summary.csv", index=False)
        fig06(summary)
        rnd = R / "05_runs_random_init"
        fig07(rnd if any(rnd.glob("*.npz")) else R / "05_runs")
        for city in C.TARGETS:
            fig08(city, nets[city], H, cns, C.BUDGET_FIGURE)
        log("figs 6-8")
    log("done")


if __name__ == "__main__":
    main()

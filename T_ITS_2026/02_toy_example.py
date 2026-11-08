"""
Step 2 - toy example of the clustering (T-ITS Fig. 2 / ITSC Fig. 2) and of the kernel comparison (ITSC Fig. 3).

A 9-node network G is clustered by the CC-Algorithm into k = 4, 3, 2 clusters; every arc / clustered object
is labelled {w, d (m), phi, E} (E in thousands).  A 4-node network H is compared with G~ (k = 4) through
the GW distance and the exposure kernels.

Outputs  figures/fig02_toy_clustering.png, figures/fig03_toy_kernels.png, results/02_toy.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import clustering as cl, compare as cp, config as C  # noqa: E402
from cyclenet.clustered import ClusteredNetwork  # noqa: E402
from cyclenet.network import Network  # noqa: E402
from cyclenet.plotting import DOS_COLORS, paper_style, plt, save  # noqa: E402


def toy(name, xy, arcs, od):
    xy = np.asarray(xy, float)
    nodes = pd.DataFrame({"node": range(len(xy)), "lon": xy[:, 0] * 1e-5, "lat": 45 + xy[:, 1] * 1e-5,
                          "x": xy[:, 0], "y": xy[:, 1], "lcc": True})
    u, v, w = map(np.array, zip(*arcs))
    d = np.hypot(*(xy[u] - xy[v]).T)
    edges = pd.DataFrame({"u": u, "v": v, "dos": w, "length_m": d, "slope_pct": 0.0})
    dem = pd.DataFrame(od, columns=["o", "d", "q"])
    return Network(name, nodes, edges, dem)


G_XY = [(0, 0), (70, 50), (320, 0), (380, 70), (140, 320), (220, 360), (500, 330), (570, 280), (560, 400)]
G_ARCS = [(0, 1, 6), (2, 3, 4), (4, 5, 5), (6, 7, 3), (7, 8, 3), (6, 8, 3),
          (1, 2, 2), (0, 4, 1), (5, 6, 4), (3, 7, 6)]
G_OD = [(0, 8, 10), (2, 5, 6), (4, 7, 8), (1, 3, 5), (8, 2, 4)]
H_XY = [(0, 0), (300, 20), (150, 300), (500, 330)]
H_ARCS = [(0, 1, 2), (0, 2, 1), (1, 3, 6), (2, 3, 4), (1, 2, 5)]
H_OD = [(0, 3, 10), (1, 2, 6), (2, 3, 8), (0, 1, 5), (3, 1, 4)]


def label(ax, x, y, w, d, phi, E, fs=6.5):
    """{w, d (m), phi, E (thousands)} as in the paper's caption (one coloured box per value)."""
    parts = ((f"{w}", "black"), (f"{d:.0f}m", "#1f3fbf"), (f"{phi:.0f}", "#c00000"), (f"{E / 1000:.1f}k", "#0b6623"))
    txt = ax.annotate("{", (x, y), fontsize=fs, ha="right", va="center", xytext=(-40, 0), textcoords="offset points",
                      bbox=dict(fc="white", ec="none", alpha=0.0, pad=0))
    off = -40
    for i, (t, col) in enumerate(parts):
        txt = ax.annotate(t + (", " if i < 3 else "}"), (x, y), fontsize=fs, color=col, ha="left", va="center",
                          xytext=(off, 0), textcoords="offset points",
                          bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.1))
        off += 4.2 * len(t) + 6


def draw_original(ax, net):
    r = net.base
    for a in range(net.E):
        p, q = net.xy[net.u[a]], net.xy[net.v[a]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=DOS_COLORS[net.dos0[a]], lw=2.2, zorder=1)
        m = 0.5 * (p + q)
        label(ax, m[0], m[1] + 18, net.dos0[a], net.length[a], r.flow[a], r.exposure_arc[a])
    ax.scatter(*net.xy.T, s=60, c="white", edgecolors="k", zorder=3)
    for i, (x, y) in enumerate(net.xy):
        ax.text(x, y, f"$v_{i + 1}$", fontsize=6, ha="center", va="center", zorder=4)
    ax.set_title(rf"$\Phi^{{orig}}$={r.total_flow:.0f}, $\mathcal{{E}}^{{orig}}$={r.exposure / 1000:.0f}k, $|V|$={net.N}",
                 fontsize=8)


def draw_clustered(ax, cn):
    r = cn.base
    net = cn.net
    for a in range(net.E):
        p, q = net.xy[net.u[a]], net.xy[net.v[a]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="0.85", lw=1, zorder=0)
    for g in range(cn.n_intra, cn.n_obj):
        rr, ss = cn.inter_r[g - cn.n_intra], cn.inter_s[g - cn.n_intra]
        p, q = cn.centroid[rr], cn.centroid[ss]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=DOS_COLORS[cn.w0[g]], lw=2.2, zorder=1)
        m = 0.5 * (p + q)
        label(ax, m[0], m[1] + 18, cn.w0[g], cn.d[g], r["phi"][g], r["exposure_obj"][g])
    for c in range(cn.k):
        mem = net.xy[cn.labels == c]
        rad = max(np.hypot(*(mem - cn.centroid[c]).T).max() + 45, 45)
        ax.add_patch(Circle(cn.centroid[c], rad, fill=False, ls="--", lw=1, ec="0.3"))
        ax.text(cn.centroid[c][0] - rad * 0.75, cn.centroid[c][1] + rad * 0.8, f"$C_{c + 1}$", fontsize=7)
        g = cn.intra_of_cluster[c]
        if g >= 0:
            label(ax, cn.centroid[c][0], cn.centroid[c][1] - rad - 14, cn.w0[g], cn.d[g], r["phi"][g],
                  r["exposure_obj"][g], fs=5.6)
    ax.scatter(*net.xy.T, s=12, c="k", zorder=3)
    ax.set_title(rf"$\Phi^{{clust}}$={r['total_flow']:.0f}, $\mathcal{{E}}^{{clust}}$={r['exposure'] / 1000:.0f}k, $k$={cn.k}",
                 fontsize=8)


def main():
    paper_style()
    G = toy("toy-G", G_XY, G_ARCS, G_OD)
    H = toy("toy-H", H_XY, H_ARCS, H_OD)
    rows = [dict(network="G", k=G.N, E=G.base.exposure, Phi=G.base.total_flow)]
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.2))
    draw_original(axs[0], G)
    clustered = {}
    for ax, k in zip(axs[1:], (4, 3, 2)):
        lab, info = cl.contract(G, k, delta=200.0)
        cn = ClusteredNetwork(G, lab)
        clustered[k] = cn
        draw_clustered(ax, cn)
        rows.append(dict(network="G~", k=k, E=cn.base["exposure"], Phi=cn.base["total_flow"], eps_used=info["eps"]))
    for ax, t in zip(axs, ["(a) Original", "(b) Cluster k=4", "(c) Cluster k=3", "(d) Cluster k=2"]):
        ax.set_xlabel(t, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        ax.margins(0.15)
    fig.text(0.5, -0.02, r"arc $a$ and clustered objects: {$w$, $d$ (m), $\phi$, $\mathcal{E}$ (thousands)}; colour = DoS",
             ha="center", fontsize=9)
    save(fig, "fig02_toy_clustering.png")

    # ---- kernel comparison H vs G~(k=4) (ITSC Fig. 3) -------------------------------------------
    cn = clustered[4]
    Kg = cp.kernel_clustered(cn).toarray()
    Kh = cp.kernel_original(H).toarray()
    mg, xg = cp.clustered_measure(cn)
    mh, xh = cp.original_measure(H)
    gw = cp.gw_between(mh, xh, mg, xg, m=4)
    dker = float(np.abs(Kg / Kg.sum() - Kh / Kh.sum()).mean())
    rows.append(dict(network="H", k=H.N, E=H.base.exposure, Phi=H.base.total_flow, GW_H_G4=gw["gw"],
                     kernel_distance=dker))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    draw_clustered(axs[0], cn)
    axs[0].set_aspect("equal"), axs[0].set_xticks([]), axs[0].set_yticks([]), axs[0].grid(False)
    vmax = max((Kg / Kg.sum()).max(), (Kh / Kh.sum()).max())
    for ax, K, lab, ttl in ((axs[1], Kg / Kg.sum(), "C", r"$\tilde{\mathcal{G}}$ (k=4)"), (axs[2], Kh / Kh.sum(), "v", r"$\mathcal{H}$")):
        im = ax.imshow(K, cmap="viridis", vmin=0, vmax=vmax, origin="lower")
        ax.set_xticks(range(4), [f"${lab}_{i + 1}$" for i in range(4)])
        ax.set_yticks(range(4), [f"${lab}_{i + 1}$" for i in range(4)])
        ax.set_title(ttl, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"GW = {gw['gw']:.3f},   kernel distance = {dker:.2e}", fontsize=9, y=0.02)
    save(fig, "fig03_toy_kernels.png")
    pd.DataFrame(rows).to_csv(C.RESULTS / "02_toy.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()

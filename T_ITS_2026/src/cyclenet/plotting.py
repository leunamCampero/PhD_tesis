"""Figures in the style of paper/T_ITS_2026.pdf (serif / Computer Modern, dotted grid, same colours)."""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import config as C  # noqa: E402

# DoS colours of the paper's maps (Figs. 1, 5, 8): 6 dark green ... 1 black
DOS_COLORS = {6: "#1b8a1b", 5: "#7ccc1a", 4: "#ffd21f", 3: "#ff8c00", 2: "#8b0000", 1: "#000000"}
DOS_WIDTH = {6: 0.9, 5: 0.9, 4: 0.7, 3: 0.8, 2: 0.9, 1: 0.9}
# methods (Fig. 6 of the paper: GA sky blue, GA-Cluster orange, Transfer green)
METHOD_COLORS = {"GA": "#56B4E9", "GA-Cluster": "#E69F00", "GA-Cluster (k-means)": "#999999",
                 "Transfer": "#009E73", "Transfer (k-means)": "#B3DFC9", "Transfer+": "#CC79A7"}
CLUSTER_COLORS = {"kmeans": "#5b8db8", "CC": "#fbb13c"}          # Fig. 4 bars
FLOW_COLOR = "#8b0000"
BUDGET_COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#D55E00"]       # Fig. 7, one per beta
CITY_COLORS = {"Grenoble": "#0072B2", "Lyon": "#D55E00", "Strasbourg": "#009E73"}


def paper_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": 10,
                         "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6, "axes.axisbelow": True,
                         "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
                         "savefig.dpi": 220, "savefig.bbox": "tight"})


def save(fig, name):
    C.FIGURES.mkdir(exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path)
    if path.suffix == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def beta_label(beta):
    return rf"$\beta={100 * beta:g}\%$"


# ==============================================================================================
# Maps
# ==============================================================================================
def boundary(ax, city, color="#1f3fbf", lw=1.3):
    try:
        gj = json.load(open(C.RAW / "boundaries.geojson", encoding="utf8"))
    except OSError:
        return
    for f in gj["features"]:
        if f["properties"]["nom"] == city:
            geom = f["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                ring = np.asarray(poly[0])
                ax.plot(ring[:, 0], ring[:, 1], color=color, lw=lw, zorder=0)


def dos_map(ax, net, dos, title=None, legend=True, legend_title=None, highlight=None, lw=1.0, loc="upper left"):
    """Arcs of ``net`` coloured by ``dos`` (arc array).  ``highlight``: arcs drawn wider."""
    seg = np.stack([net.lonlat[net.u], net.lonlat[net.v]], axis=1)
    dos = np.asarray(dos, int)
    width = np.array([DOS_WIDTH[d] for d in dos]) * lw
    order = np.argsort(dos)[::-1]                                  # unsafe arcs on top
    if highlight is None:
        ax.add_collection(LineCollection(seg[order], colors=[DOS_COLORS[d] for d in dos[order]],
                                         linewidths=width[order], capstyle="round"))
    else:                                                          # unchanged arcs faded, upgrades on top
        hl = np.asarray(highlight, bool)
        o = order[~hl[order]]
        ax.add_collection(LineCollection(seg[o], colors=[DOS_COLORS[d] for d in dos[o]], linewidths=width[o] * 0.8,
                                         alpha=0.28, capstyle="round"))
        o = order[hl[order]]
        ax.add_collection(LineCollection(seg[o], colors="#1f3fbf", linewidths=width[o] * 4.2, alpha=0.35,
                                         capstyle="round"))
        ax.add_collection(LineCollection(seg[o], colors=[DOS_COLORS[d] for d in dos[o]], linewidths=width[o] * 2.4,
                                         capstyle="round"))
    boundary(ax, net.city)
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.deg2rad(net.lonlat[:, 1].mean())))
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    if legend:
        h = [Line2D([], [], color=DOS_COLORS[d], lw=4) for d in range(6, 0, -1)]
        ax.legend(h, [f"DoS={d}" for d in range(6, 0, -1)], loc=loc, fontsize=6.5, title=legend_title,
                  title_fontsize=6.5, frameon=True, handlelength=1.4)
    if title:
        ax.set_title(title, fontsize=9)


def cluster_graph_map(ax, cn, levels=None, title=None, legend=True, lw=0.9):
    """Clustered network drawn on centroids: super-arcs coloured by their (min) DoS, intra objects as dots."""
    lv = cn.w0 if levels is None else np.asarray(levels, int)
    lonlat = cluster_lonlat(cn)
    inter = lv[cn.n_intra:]
    seg = np.stack([lonlat[cn.inter_r], lonlat[cn.inter_s]], axis=1)
    order = np.argsort(inter)[::-1]
    ax.add_collection(LineCollection(seg[order], colors=[DOS_COLORS[d] for d in inter[order]],
                                     linewidths=[DOS_WIDTH[d] * lw for d in inter[order]]))
    ci = cn.obj_cluster[: cn.n_intra]
    ax.scatter(lonlat[ci, 0], lonlat[ci, 1], s=1.2, c=[DOS_COLORS[d] for d in lv[: cn.n_intra]], lw=0, zorder=3)
    boundary(ax, cn.net.city)
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.deg2rad(lonlat[:, 1].mean())))
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    if legend:
        h = [Line2D([], [], color=DOS_COLORS[d], lw=4) for d in range(6, 0, -1)]
        ax.legend(h, [f"DoS={d}" for d in range(6, 0, -1)], loc="upper left", fontsize=6.5, title="Min DoS",
                  title_fontsize=6.5, frameon=True, handlelength=1.4)
    if title:
        ax.set_title(title, fontsize=9)


def cluster_lonlat(cn):
    lab = cn.labels
    cnt = np.bincount(lab)
    return np.column_stack([np.bincount(lab, cn.net.lonlat[:, 0]), np.bincount(lab, cn.net.lonlat[:, 1])]) / cnt[:, None]


__all__ = ["plt", "paper_style", "save", "dos_map", "cluster_graph_map", "DOS_COLORS", "METHOD_COLORS",
           "CLUSTER_COLORS", "FLOW_COLOR", "BUDGET_COLORS", "CITY_COLORS", "beta_label"]

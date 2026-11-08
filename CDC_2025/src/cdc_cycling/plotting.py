"""Figures of the CDC 2025 paper (matplotlib only)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Patch                    # noqa: E402

from . import config as C                                # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

DOS_COLORS = {6: "green", 5: "#66CD00", 4: "gold", 3: "darkorange", 2: "darkred", 1: "black"}
DRAW_ORDER = (4, 1, 2, 3, 5, 6)                 # DoS 4 below, DoS 6 on top (as in the paper)
# colours of the lambda values (paper's Okabe-Ito-like set), algorithms
LAMBDA_COLORS = {0.1: "#0072B2", 0.2: "#E69F00", 0.3: "#4EB265", 0.4: "#7BAFDE", 0.5: "#D1BBD7"}
ALGO_COLORS = {"Original": "#CCBB44", "MLCC": "#EE6677", "GA-U": "#AA3377", "GA-S": "#0072B2"}
INK = "#222222"


def _lam(lam):
    return f"$\\lambda={lam:.0%}$".replace("%", "\\%")


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------
def iris_polygons():
    """{zone_id: [rings (n x 2 arrays)]} from data/regions/grenoble_iris.geojson."""
    gj = json.loads((C.DATA / "regions" / "grenoble_iris.geojson").read_text(encoding="utf-8"))
    out = {}
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        out[int(f["properties"]["zone_id"])] = [np.asarray(p[0]) for p in polys]
    return out


def _frame(ax, net):
    pad = 0.003
    ax.set_xlim(net.xy[:, 0].min() - pad, net.xy[:, 0].max() + pad)
    ax.set_ylim(net.xy[:, 1].min() - pad, net.xy[:, 1].max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(net.xy[:, 1].mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _iris_outline(ax, polys, color="#BBBBBB", lw=0.4):
    for rings in polys.values():
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], color=color, lw=lw, zorder=0)


# ----------------------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------------------
def dos_map(ax, net, dos, title=None, legend=True, polys=None):
    """Network coloured by DoS, IRIS boundaries in grey."""
    _iris_outline(ax, polys or iris_polygons())
    dos = np.asarray(dos)
    for level in DRAW_ORDER:
        idx = np.flatnonzero(dos == level)
        seg = np.stack([net.xy[net.u[idx]], net.xy[net.v[idx]]], axis=1)
        ax.add_collection(LineCollection(seg, colors=DOS_COLORS[level], linewidths=1.0))
    _frame(ax, net)
    if legend:
        ax.legend(handles=[Patch(color=DOS_COLORS[k], label=f"DoS={k}") for k in (6, 5, 4, 3, 2, 1)],
                  loc="upper right", fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=11, color=INK)


def region_map(ax, net, zones, values, title=None, vmax=1.0, cmap="viridis", label=None,
               colorbar=True, polys=None):
    """Choropleth of one value per IRIS (e.g. regional insufficiency I_r)."""
    polys = polys or iris_polygons()
    verts, vals = [], []
    for z, v in zip(zones, values):
        for ring in polys[int(z)]:
            verts.append(ring)
            vals.append(v)
    pc = PolyCollection(verts, array=np.asarray(vals), cmap=cmap, edgecolors="#DDDDDD", linewidths=0.4)
    pc.set_clim(0, vmax)
    ax.add_collection(pc)
    _frame(ax, net)
    if colorbar:
        cb = plt.colorbar(pc, ax=ax, fraction=0.045, pad=0.01)
        cb.outline.set_visible(False)
        if label:
            cb.set_label(label)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    return pc


# ----------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------
def safety_histogram(ax, safety, trips, thr, lam, title=None):
    """Trips (people) per path-safety bin, with the threshold line."""
    ax.hist(safety, bins=np.linspace(0, 100, 21), weights=trips, color="bisque", edgecolor="black", lw=0.8)
    ax.axvline(thr, color="red", ls="--", lw=1.5, label=f"$s_{{thr}}={lam:.0%}\\,s_{{\\mathcal{{G}}}}$".replace("%", "\\%"))
    ax.set_xlabel("Path safety $s_p$")
    ax.set_ylabel("Cyclists (trips/day)")
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.grid(True, ls=":", color="#BBBBBB")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11)


def threshold_bars(ax, below, title=None):
    """Stacked bars: % of cyclists below / above the threshold for each algorithm."""
    names = list(below)
    b = np.array([100 * below[k] for k in names])
    x = np.arange(len(names))
    ax.bar(x, b, color="#FD8D3C", width=0.6, label="below $s_{thr}$", edgecolor="white")
    ax.bar(x, 100 - b, bottom=b, color="#C7E1A8", width=0.6, label="above $s_{thr}$", edgecolor="white")
    for i, v in enumerate(b):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cyclists (%)")
    if title:
        ax.set_title(title, fontsize=11)


def grouped_bars(ax, table, ylabel):
    """table = {algorithm: {lambda: value}} -> bars grouped by algorithm, coloured by lambda."""
    algos = list(table)
    lams = list(C.LAMBDAS)
    w = 0.8 / len(lams)
    for j, lam in enumerate(lams):
        xs = np.arange(len(algos)) + (j - (len(lams) - 1) / 2) * w
        vals = [table[a][lam] for a in algos]
        ax.bar(xs, vals, width=w, color=LAMBDA_COLORS[lam], label=_lam(lam), edgecolor="white", lw=0.5)
        for xx, v in zip(xs, vals):
            ax.text(xx, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90, color=INK)
    ax.set_xticks(np.arange(len(algos)), algos)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.yaxis.grid(True, ls=":", color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))


def exposure_bars(ax, values, colors):
    names = list(values)
    ax.bar(np.arange(len(names)), [values[k] for k in names], color=colors, edgecolor="white", width=0.7)
    for i, k in enumerate(names):
        ax.text(i, values[k], f"{values[k]:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(names)), names, fontsize=9)
    ax.set_ylabel("$J_{\\mathcal{E}}$", fontsize=14)
    ax.yaxis.grid(True, ls=":", color="#BBBBBB")
    ax.set_axisbelow(True)


def marginal_lines(ax, fractions, marg):
    """marg = {lambda: normalised marginal improvement per budget step}."""
    k = np.arange(1, len(fractions) + 1)
    for lam, m in marg.items():
        ax.plot(k, m, "o-", color=LAMBDA_COLORS[lam], label=_lam(lam), lw=1.5, ms=5)
    ax.set_xticks(k, [f"$B_{{{i}}}$" for i in k])
    ax.set_ylabel("normalised $\\Delta J / \\Delta B$")
    ax.grid(True, ls=":", color="#BBBBBB")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))

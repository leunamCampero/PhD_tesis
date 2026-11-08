"""Figures of the paper (Fig. 1 DoS map, Fig. 2 exposure bars, Fig. 3 flow / DoS maps).

Only matplotlib is needed: the network is drawn as a LineCollection and the city limits
come from data/network/grenoble_iris.geojson.
"""
import json

import matplotlib

matplotlib.use("Agg")                                   # write files, no window needed
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.collections import LineCollection       # noqa: E402
from matplotlib.patches import Patch                    # noqa: E402

from . import config as C                                # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

# DoS colours used in the paper's maps (6 = safest)
DOS_COLORS = {6: "green", 5: "#66CD00", 4: "gold", 3: "darkorange", 2: "darkred", 1: "black"}
DRAW_ORDER = (4, 1, 2, 3, 5, 6)          # as in the 2024 maps: DoS 4 first, DoS 6 on top
# bar colours of Fig. 2 (Okabe-Ito: colour-blind safe); bars are also labelled by name
BAR_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
INK, MUTED = "#222222", "#777777"


def _segments(net, idx):
    return np.stack([net.xy[net.u[idx]], net.xy[net.v[idx]]], axis=1)


def _boundary(ax):
    """Outline of the 70 IRIS of Grenoble."""
    gj = json.loads((C.DATA / "network" / "grenoble_iris.geojson").read_text())
    for f in gj["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0])
            ax.plot(ring[:, 0], ring[:, 1], color="#BBBBBB", lw=0.4, zorder=0)


def _frame(ax, net):
    _boundary(ax)
    pad = 0.003
    ax.set_xlim(net.xy[:, 0].min() - pad, net.xy[:, 0].max() + pad)
    ax.set_ylim(net.xy[:, 1].min() - pad, net.xy[:, 1].max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(net.xy[:, 1].mean())))   # metric aspect ratio
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def dos_map(ax, net, dos, title=None, legend=True):
    """Network coloured by DoS (Fig. 1 and the lower row of Fig. 3)."""
    dos = np.asarray(dos)
    for level in DRAW_ORDER:
        idx = np.flatnonzero(dos == level)
        ax.add_collection(LineCollection(_segments(net, idx), colors=DOS_COLORS[level], linewidths=1.2))
    _frame(ax, net)
    if legend:
        ax.legend(handles=[Patch(color=DOS_COLORS[k], label=f"DoS={k}") for k in (6, 5, 4, 3, 2, 1)],
                  loc="upper right", fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=11, color=INK)


def flow_map(ax, net, x_hat, title=None, colorbar=True):
    """Network coloured by normalised cyclist flow (upper row of Fig. 3)."""
    order = np.argsort(x_hat)                                  # strongest flows drawn on top
    lc = LineCollection(_segments(net, order), cmap="Blues", linewidths=1.0)
    lc.set_array(x_hat[order])
    lc.set_clim(0, 1)
    ax.add_collection(lc)
    _frame(ax, net)
    if colorbar:
        cb = plt.colorbar(lc, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label("normalised flow $\\hat{x}_a$", fontsize=9)
        cb.outline.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, color=INK)


def exposure_bars(values, path, title=None):
    """Fig. 2: J_E of each scenario. ``values`` = {label: J_E}."""
    fig, ax = plt.subplots(figsize=(1.3 * len(values) + 1.5, 3.2))
    labels = list(values)
    x = np.arange(len(labels))
    bars = ax.bar(x, [values[k] for k in labels], width=0.62,
                  color=BAR_COLORS[: len(labels)], edgecolor="white", linewidth=2)
    for b, k in zip(bars, labels):
        ax.annotate(f"{values[k]:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=10)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("$J_{\\mathcal{E}}$", fontsize=14)
    ax.set_ylim(0, max(values.values()) * 1.2 + 1e-9)
    ax.yaxis.grid(True, ls=":", color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig3_panel(net, scenarios, path):
    """Fig. 3: flows (top row) and DoS (bottom row) for each scenario {name: (dos, x_hat)}."""
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.4))
    axes = np.atleast_2d(axes).reshape(2, n)
    for j, (name, (dos, x_hat)) in enumerate(scenarios.items()):
        flow_map(axes[0, j], net, x_hat, title=f"{name} flows", colorbar=(j == n - 1))
        dos_map(axes[1, j], net, dos, title=f"{name} DoS", legend=(j == n - 1))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def convergence(history, path, title=None):
    """Best J_E found so far versus number of fitness evaluations (GAB)."""
    h = np.asarray([(n, j) for n, j in history if j < 1e9])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(h[:, 0], h[:, 1], color=BAR_COLORS[1], lw=2)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("best $J_{\\mathcal{E}}$")
    ax.yaxis.grid(True, ls=":", color="#CCCCCC")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

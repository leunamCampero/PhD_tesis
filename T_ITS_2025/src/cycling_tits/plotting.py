"""Figures in the style of the paper (LaTeX fonts, same colours/layout).

If no LaTeX installation is found, matplotlib's built-in Computer Modern
mathtext is used instead (figures look almost identical).
"""
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from .metrics import WEIGHTS
from .model import alphas

# Colours of the figures
METHOD_COLORS = {"Optimal": "wheat", "MLCC": "lime", "SGA": "royalblue",
                 "SGA-MLCC": "cyan", "Bayesian": "lightblue"}
CSCS_COLORS = {6: "forestgreen", 5: "lawngreen", 4: "gold", 3: "darkorange", 2: "#CC0000", 1: "black"}
GEN_COLORS = ["#000000", "darkviolet", "darkorange", "gold", "green",
              "lightseagreen", "dodgerblue", "grey", "red", "blue"]
BUDGET_COLORS = ["#000000", "grey", "red", "tomato", "darkorange", "orange",
                 "gold", "yellow", "lime", "limegreen"]


def set_style(fontsize=20):
    """LaTeX-like style used in all figures of the paper."""
    usetex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        "savefig.dpi": 300,
        "figure.dpi": 100,
    })
    return usetex


def _pct_label():
    return r"\%" if plt.rcParams["text.usetex"] else "%"


# --------------------------------------------------------------------- comparisons
def _alpha_axes(ax1, positions, fontsize=50):
    pts = np.array(alphas())
    ax1.scatter(positions, pts * 100, color="blue", s=400, zorder=5)
    ax1.plot(positions, pts * 100, color="blue", linestyle="-", zorder=4)
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 130)   # bars below 0.5 % are not visible
    ticks = [1, 2, 5, 10, 20, 30, 50, 100]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([str(t) for t in ticks])
    ax1.tick_params(axis="both", which="major", labelsize=fontsize)
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(pts * 100)
    ax2.set_yticklabels([f"{p:.2f}" for p in pts])
    ax2.minorticks_off()
    ax2.set_ylabel(r"$\alpha_{i}$", fontsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)
    ax2.grid(axis="y", linestyle="--", color="blue")
    return ax2


def comparison_bars(safety, comfort, outfile=None, methods=None):
    """Grouped log-scale bars: % of achievable safety (plain) and comfort (hatched)
    for every budget i = 0..9 and every method (Figs. 7a, 9a, 11a).

    safety, comfort: dict method -> list of 10 percentages.
    """
    set_style(50)
    methods = methods or [m for m in METHOD_COLORS if m in safety]
    k = len(methods)
    width = 0.9 / (2 * k)
    r = np.arange(10)
    fig, ax1 = plt.subplots(figsize=(45, 10))
    for j, m in enumerate(methods):
        floor = lambda v: [max(x, 1e-3) for x in v]  # log scale cannot show 0 %
        ax1.bar(r + j * width, floor(safety[m]), width=width, color=METHOD_COLORS[m], edgecolor="black")
        ax1.bar(r + (k + j) * width, floor(comfort[m]), width=width, color=METHOD_COLORS[m],
                edgecolor="black", hatch="\\\\")
    centre = r + (2 * k - 1) * width / 2
    ax1.set_xticks(centre)
    ax1.set_xticklabels([f"${i}$" for i in range(10)])
    ax1.set_xlabel(r"$i$ Values", fontsize=50)
    ax1.set_ylabel(_pct_label() + r" Achievable safety/comfort" + "\n" + r"perception", fontsize=50)
    _alpha_axes(ax1, centre)

    # legend as a small table: Safety | method | Comfort
    cell_text = [["Safety", "", "Comfort"]] + [["", m, ""] for m in methods]
    colors = [["white"] * 3] + [[METHOD_COLORS[m], "white", METHOD_COLORS[m]] for m in methods]
    table = plt.table(cellText=cell_text, cellColours=colors, cellLoc="center", loc="bottom",
                      bbox=[0, .67, 0.4, 0.37], colWidths=[0.06, 0.24 if k == 5 else 0.12, 0.06])
    for i in range(len(methods)):
        table[(i + 1, 2)].set_hatch("\\\\")
    for cell in table.get_celld().values():
        cell.set_linewidth(0)
        cell.set_fontsize(50)
    plt.subplots_adjust(top=1.2, bottom=0.4)
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------ GA evolution
def ga_evolution(sga, sga_mlcc, outfile=None, legend_outfile=None, target="safety"):
    """% of achievable improvement of the GA champion per generation (Figs. 7b, 9b, 11b).

    sga, sga_mlcc: dict i -> list of percentages (generation 0 = 0 %).
    """
    set_style(25)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in sorted(sga):
        ax.plot(range(len(sga[i])), sga[i], marker="o", linestyle="-", color=GEN_COLORS[i],
                label=f"SGA ($i=$ {i})")
    for i in sorted(sga_mlcc):
        ax.plot(range(len(sga_mlcc[i])), sga_mlcc[i], marker="+", linestyle="--", color=GEN_COLORS[i],
                label=f"SGA-MLCC ($i=$ {i})")
    ax.set_xlabel("Generation", fontsize=25)
    ax.set_ylabel(_pct_label() + f" Achievable {target} perception", fontsize=25)
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    if legend_outfile:
        fl = plt.figure(figsize=(12, 8))
        axl = fl.add_subplot(111)
        axl.axis("off")
        axl.add_artist(Legend(axl, handles, labels, fontsize=25, ncol=2, loc="center"))
        fl.savefig(legend_outfile, dpi=300, bbox_inches="tight")
        plt.close(fl)
    return fig


# --------------------------------------------------------------- fourteen cities
def fourteen_cities(percent, ylabel, outfile=None, show_cities=True, show_legend=True):
    """Bars per city and budget alpha_i (Fig. 12).  percent: dict city -> 10 values."""
    set_style(50)
    cities = list(percent)
    r = np.arange(len(cities))
    w = 0.08
    pts = np.array(alphas())
    fig, ax1 = plt.subplots(figsize=(45, 12 if show_cities else 9))
    for i in range(10):
        ax1.bar(r + i * w, [max(percent[c][i], 1e-3) for c in cities], color=BUDGET_COLORS[i],
                width=w, edgecolor="black")
    for i in range(10):
        ax1.scatter(r + i * w + w / 2, [pts[i] * 100] * len(cities), color="blue", s=100, zorder=5)
    ax1.set_ylabel(_pct_label() + f" Achievable {ylabel}" + "\n" + r"perception", fontsize=50)
    if show_cities:
        ax1.set_xlabel("Cities", fontsize=50)
        ax1.set_xticks(r + w * 10 / 2)
        ax1.set_xticklabels([c.replace("_", " ") for c in cities], rotation=15, fontsize=50)
    else:
        ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.set_yscale("log")
    ticks = [1, 2, 5, 10, 20, 30, 50, 100]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([str(t) for t in ticks], fontsize=50)
    ax1.set_ylim(0.5, 130)
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(pts * 100)
    ax2.set_yticklabels([f"{p:.2f}" for p in pts], fontsize=50)
    ax2.minorticks_off()
    ax2.set_ylabel(r"$\alpha_{i}$", fontsize=50)
    ax2.grid(axis="y", linestyle="--", color="blue")
    if show_legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=BUDGET_COLORS[i]) for i in range(10)]
        labels = [f"$\\alpha_{{{i}}}$ = {pts[i]:.2f}" for i in range(10)]
        ax1.legend(handles, labels, fontsize=35, loc="upper center", bbox_to_anchor=(0.5, 1.13),
                   ncol=10, frameon=True)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- predictions
def prediction_bars(cities, actual, nnls, automl, ylabel, outfile=None):
    """Actual vs NNLS vs Auto-ML predictions on the 4 test cities (Fig. 6)."""
    set_style(25)
    x = np.arange(len(cities))
    w = 0.3
    fig = plt.figure(figsize=(12, 6))
    plt.bar(x - w, actual, w, label="Actual Values", color="#FF7F50")
    plt.bar(x, nnls, w, label="NNLS Synthetic", color="#98F5FF")
    if automl is not None:
        plt.bar(x + w, automl, w, label="Auto-ML-Ensemble", color="black")
    plt.xticks(x, [c.replace("_", " ") for c in cities], fontsize=25)
    plt.yticks(np.arange(0, max(actual) + 0.05, 0.10), fontsize=25)
    plt.grid(axis="y", linestyle="--", color="red")
    plt.legend(fontsize=15)
    plt.xlabel("Municipalities", fontsize=25)
    plt.ylabel(f"{ylabel} Values", fontsize=25)
    plt.subplots_adjust(bottom=0.2)
    if outfile:
        plt.savefig(outfile, dpi=300)
    return fig


# --------------------------------------------------------------------- networks
def _legend_cscs(ax, **kw):
    w = WEIGHTS["cscs"]
    handles = [Patch(facecolor=CSCS_COLORS[k], edgecolor=CSCS_COLORS[k]) for k in range(6, 0, -1)]
    return ax.legend(handles, [f"{w[k]}" for k in range(6, 0, -1)], **kw)


def benchmark_network(G, levels=None, outfile=None, show_lengths=True, show_legend=True):
    """Benchmark example coloured by CSCS (Fig. 5)."""
    set_style(12)
    levels = levels or [d["dos"] for _, _, d in G.edges(data=True)]
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(figsize=(4.5 if show_legend else 3, 2))
    nx.draw_networkx_edges(G, pos, ax=ax, width=6, edge_color=[CSCS_COLORS[l] for l in levels])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color="black")
    if show_lengths:
        for (u, v), d in zip(G.edges(), (d for *_, d in G.edges(data=True))):
            (x1, y1), (x2, y2) = pos[u], pos[v]
            if x1 == x2:   # vertical edge: label on the outer side
                ax.text(x1 - 0.03, (y1 + y2) / 2, f"{d['distance']:.2f}m", ha="right", va="center", fontsize=11)
            else:
                ax.text((x1 + x2) / 2, y1 - 0.12, f"{d['distance']:.2f}m", ha="center", va="top", fontsize=11)
    ax.set_xlim(-0.25 if not show_lengths else -0.3, 1.12)
    ax.set_ylim(-0.25, 4.2)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if show_legend:
        _legend_cscs(ax, loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=11)
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig


def sioux_falls_network(G, outfile=None):
    """Sioux Falls network with random DoS, coloured by CSCS (Fig. 8)."""
    set_style(20)
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = [CSCS_COLORS[d["dos"]] for *_, d in G.edges(data=True)]
    nx.draw(G, pos, ax=ax, node_size=500, node_color="lightcyan", with_labels=True,
            font_size=18, font_weight="bold", edge_color=colors, width=3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['distance']:.1f}"
                                                      for u, v, d in G.edges(data=True)},
                                 ax=ax, font_size=15)
    _legend_cscs(ax, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20)
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig


def city_map(G, levels=None, outfile=None, title=None, highlight_changes=False):
    """Map of a cycling network coloured by CSCS (Figs. 3 and 10).

    With ``highlight_changes`` the upgraded edges are drawn thicker.
    """
    set_style(16)
    levels = levels or [d["dos"] for *_, d in G.edges(data=True)]
    orig = [d["dos"] for *_, d in G.edges(data=True)]
    fig, ax = plt.subplots(figsize=(10, 10))
    from matplotlib.collections import LineCollection
    for lvl in range(1, 7):  # draw best infrastructure on top
        segs = [[u, v] for (u, v), l in zip(G.edges(), levels) if l == lvl]
        widths = [3.0 if highlight_changes and l != o else (1.6 if lvl >= 5 else 0.8)
                  for (_, _), l, o in zip(G.edges(), levels, orig) if l == lvl]
        ax.add_collection(LineCollection(segs, colors=CSCS_COLORS[lvl], linewidths=widths))
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean([n[1] for n in G.nodes()]))))
    ax.axis("off")
    _legend_cscs(ax, loc="lower right", title="CSCS", fontsize=14)
    if title:
        ax.set_title(title)
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig

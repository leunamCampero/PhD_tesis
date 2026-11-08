"""Figures (matplotlib only, no LaTeX needed)."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402

from . import config as C  # noqa: E402

DOS_COLORS = {1: "#d73027", 2: "#fc8d59", 3: "#fee090", 4: "#bdbdbd", 5: "#91bfdb", 6: "#1a5fb4"}
METHOD_COLORS = {"Original": "#7f7f7f", "GA": "#1a5fb4"}
ALPHA_STYLE = {1 / 3: ("#0072b2", "o", r"$\alpha=1/3$"), 2 / 5: ("#009e73", "s", r"$\alpha=2/5$"),
               1 / 2: ("#e69f00", "^", r"$\alpha=1/2$")}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def _segments(xy, u, v):
    return np.stack([xy[u], xy[v]], axis=1)


def _frame(ax, xy):
    ax.set_aspect(1 / np.cos(np.deg2rad(xy[:, 1].mean())))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def iris_boundaries(ax, color="0.6", lw=0.4):
    try:
        import geopandas as gpd
        gpd.read_file(C.DATA / "regions" / "grenoble_iris.geojson").boundary.plot(ax=ax, color=color, lw=lw)
    except Exception:                                   # geopandas is optional for figures
        pass
    ax.set_xlabel(""), ax.set_ylabel("")


def bike_dos_map(ax, bike, dos, lw=1.0, title=None, legend=True, highlight=None):
    """Bike network coloured by DoS; ``highlight``: edges drawn thicker (e.g. upgrades)."""
    segs = _segments(bike.xy, bike.u, bike.v)
    order = np.argsort(dos, kind="stable")
    cols = [DOS_COLORS[d] for d in dos[order]]
    widths = np.full(bike.E, lw)
    if highlight is not None:
        widths[highlight] = 2.4 * lw
    ax.add_collection(LineCollection(segs[order], colors=cols, linewidths=widths[order]))
    ax.autoscale()
    _frame(ax, bike.xy)
    if legend:
        for d in range(1, 7):
            ax.plot([], [], color=DOS_COLORS[d], lw=2.5, label=f"DoS {d}")
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)


def car_map(ax, car, values=None, cmap="Reds", lw=1.0, title=None, vmax=None, label=None,
            shared=None, norm=None):
    """Car network; ``values`` per road (E) coloured with ``cmap`` (grey if None)."""
    segs = _segments(car.xy, car.u, car.v)
    if values is None:
        ax.add_collection(LineCollection(segs, colors="0.55", linewidths=lw * 0.8))
    else:
        order = np.argsort(np.abs(values))
        norm = norm or Normalize(0, vmax if vmax is not None else np.nanmax(values))
        lc = LineCollection(segs[order], array=values[order], cmap=cmap, norm=norm,
                            linewidths=lw * (0.6 + 1.6 * norm(np.abs(values[order])).clip(0, 1)))
        ax.add_collection(lc)
        cb = plt.colorbar(lc, ax=ax, fraction=0.035, pad=0.01)
        if label:
            cb.set_label(label)
    if shared is not None:
        ax.add_collection(LineCollection(segs[shared], colors="#1a5fb4", linewidths=lw * 1.6))
        ax.plot([], [], color="#1a5fb4", lw=2, label="roads with an on-road bike arc (DoS 1-4)")
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.autoscale()
    _frame(ax, car.xy)
    if title:
        ax.set_title(title, fontsize=10)


def save(fig, name):
    path = C.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = ["plt", "bike_dos_map", "car_map", "iris_boundaries", "save", "ListedColormap",
           "DOS_COLORS", "ALPHA_STYLE"]

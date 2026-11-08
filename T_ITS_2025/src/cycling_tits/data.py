"""Loading of the cycling networks and of the survey data.

Every network is stored as two CSV files in ``data/networks``:

* ``<name>_nodes.csv`` - node coordinates (and an integer id for Sioux Falls),
* ``<name>_edges.csv`` - one row per edge: end points, length in metres and the
  Degree of Separation (DoS, integer 1..6).

Graphs are rebuilt with the *same node and edge order* as the original 2024
pickles: the MLCC heuristic breaks ties using networkx iteration order, so the
order matters for deterministic results.

Edge attributes of the returned ``networkx.Graph``:
    ``dos``       integer degree of separation 1..6
    ``distance``  length in metres (name kept from the original code)
    ``wlen``      {"dos": .., "cscs": ..} weighted length sum(length * weight) as
                  stored in the original graphs.  For edges that were merged from
                  several OSM segments during simplification this is a mixture and
                  differs from ``distance * w(dos)``.  It is dropped as soon as the
                  DoS of the edge is changed by an optimiser.
"""
import math
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# Order used in every table/figure
CITIES = ["Angers", "Annecy", "Bordeaux", "Chambery", "Dunkerque", "Grenoble",
          "La_Rochelle", "Le_Havre", "Lyon", "Nantes", "Paris", "Rennes",
          "Strasbourg", "Tours"]


def pretty(city):
    """'La_Rochelle' -> 'La Rochelle'."""
    return city.replace("_", " ")


def load_network(name):
    """Return the cycling network ``name`` (a city, 'Benchmark' or 'Sioux_Falls')."""
    nodes = pd.read_csv(DATA / "networks" / f"{name}_nodes.csv")
    edges = pd.read_csv(DATA / "networks" / f"{name}_edges.csv")
    G = nx.Graph(name=name)
    labelled = "u" in edges.columns  # Sioux Falls uses integer node ids
    if labelled:
        for r in nodes.itertuples(index=False):
            G.add_node(int(r.node), pos=(r.x, r.y))
        for r in edges.itertuples(index=False):
            G.add_edge(int(r.u), int(r.v), distance=float(r.length_m), dos=int(r.dos), wlen=_wlen(r))
    else:
        # Benchmark nodes are integer grid coordinates, cities are (lon, lat)
        cast = int if name == "Benchmark" else float
        for r in nodes.itertuples(index=False):
            n = (cast(r.x), cast(r.y))
            G.add_node(n, pos=n)
        for r in edges.itertuples(index=False):
            u, v = (cast(r.u_x), cast(r.u_y)), (cast(r.v_x), cast(r.v_y))
            G.add_edge(u, v, distance=float(r.length_m), dos=int(r.dos), wlen=_wlen(r))
    return G


def _wlen(r):
    w = {}
    for k in ("dos", "cscs"):
        val = getattr(r, f"wlen_{k}")
        if not (isinstance(val, float) and math.isnan(val)):
            w[k] = float(val)
    return w


def edge_list(G):
    """Ordered list of edges (u, v); position i is decision variable i of the optimisers."""
    return [(u, v) for u, v in G.edges()]


def dos_vector(G):
    """Current DoS level of every edge, in ``edge_list`` order."""
    return [d["dos"] for _, _, d in G.edges(data=True)]


def with_levels(G, levels, reset="changed"):
    """Copy of ``G`` with the DoS of every edge replaced by ``levels`` (edge order).

    reset="changed": only upgraded edges lose their stored weighted length
                     (what MLCC / the 14-city study did);
    reset="all":     every edge is re-weighted as length * w(DoS)
                     (what the GA and Bayesian objective functions did).
    """
    H = G.copy()
    for (u, v), lvl in zip(edge_list(G), levels):
        d = H[u][v]
        if reset == "all" or int(lvl) != d["dos"]:
            d["wlen"] = {}
        d["dos"] = int(lvl)
    return H


def load_survey():
    """Perceived safety/comfort of the 14 cities (Baromètre Vélo, scaled to [0, 1])."""
    return pd.read_csv(DATA / "model" / "survey_barometre.csv", index_col="city")

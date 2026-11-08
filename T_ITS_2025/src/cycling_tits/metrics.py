"""Graph metrics (Section II-C of the paper): node strength s_G, closeness
centrality c_G and largest connected component l_G.

Edge weights are obtained from the DoS level of each edge through a *weighting
scheme*:

* ``"cscs"`` - CycleRAP Safety and Comfort Score (Table I of the paper):
  DoS 1..6 -> 1, 1.55, 2.18, 2.99, 4.64, 5.23
* ``"dos"``  - the raw integer DoS 1..6 (ITSC-2024 version), used by the default
  preset of the optimisation (see ``model.py``).

The implementations are ports of the original ``degree_library*``,
``closeness_library*`` and ``diameter_library*`` modules.
"""
import math

import numpy as np

import networkx as nx

WEIGHTS = {
    "cscs": {1: 1.0, 2: 1.55, 3: 2.18, 4: 2.99, 5: 4.64, 6: 5.23},
    "dos": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
}


def wlen(d, scheme):
    """Weighted length of an edge: stored value if available, else length * w(DoS)."""
    return d.get("wlen", {}).get(scheme, d["distance"] * WEIGHTS[scheme][d["dos"]])


def strength(G, scheme="cscs"):
    """Normalised global node strength s_G in [0, 1]."""
    w = WEIGHTS[scheme]
    total = 0.0
    for n in G.nodes():
        deg = G.degree(n)
        if deg:
            total += sum(w[d["dos"]] for _, _, d in G.edges(n, data=True)) / (w[6] * deg)
    return total / G.number_of_nodes()


def closeness(G, scheme="cscs"):
    """Normalised global closeness centrality c_G in [0, 1].

    Shortest paths use the *inverted* geographical weight d_e * w(7 - DoS_e)
    (good infrastructure = short), normalised by the fully-safe network where every
    edge has the best (smallest) inverted weight w(1).

    Node closeness is computed on G itself (as in the original code).
    """
    w = WEIGHTS[scheme]
    c1 = _closeness_all(G, lambda d: d["distance"] * w[7 - d["dos"]])
    c2 = _closeness_all(G, lambda d: d["distance"] * w[1])
    ok = c2 != 0
    return float(np.sum(c1[ok] / c2[ok]) / G.number_of_nodes())


def _closeness_all(G, weight, chunk=500):
    """Same values as ``nx.closeness_centrality(distance=..., wf_improved=True)``
    but computed with SciPy's compiled Dijkstra (minutes instead of hours on Paris)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    idx = {n: i for i, n in enumerate(G.nodes())}
    n = len(idx)
    rows, cols, vals = [], [], []
    for u, v, d in G.edges(data=True):
        rows += [idx[u], idx[v]]
        cols += [idx[v], idx[u]]
        vals += [weight(d)] * 2
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    out = np.zeros(n)
    for start in range(0, n, chunk):
        D = dijkstra(A, directed=False, indices=np.arange(start, min(start + chunk, n)))
        reach = np.isfinite(D)
        tot = np.where(reach, D, 0).sum(axis=1)
        k = reach.sum(axis=1) - 1          # reachable nodes other than the source
        c = np.where(tot > 0, k / np.where(tot > 0, tot, 1), 0.0)
        out[start:start + len(c)] = c * (k / (n - 1)) if n > 1 else 0.0
    return out


def _max_weighted_component(H):
    """Connected component of H with the largest total edge weight (first wins ties)."""
    best, best_w = None, 0
    for comp in nx.connected_components(H):
        w = sum(H.edges[u, v]["weight"] for u in comp for v in H.neighbors(u) if v in comp)
        if w > best_w:
            best, best_w = comp, w
    return best


def largest_connected_component(G, scheme="cscs"):
    """Largest-connected-component metric l_G in [0, 1].

    For every connected component C (sorted by number of nodes, largest first) and
    every threshold level k = 2..6, take the heaviest connected sub-network made
    of edges with DoS >= k and compute its share of the fully-safe length
    sum(d_e * w_e) / sum(d_e * w(6)).  Thresholds are weighted by exp(k) and
    components by exp(n - j).

    The threshold weight is exp(k) with k = 2..6 the DoS level.
    """
    w = WEIGHTS[scheme]
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    n = len(comps)
    total, norm_comp = 0.0, 0.0
    for j, comp in enumerate(comps):
        W1 = G.subgraph(comp)
        full = sum(d["distance"] * w[6] for _, _, d in W1.edges(data=True))
        S, norm_level = 0.0, 0.0
        for k in range(2, 7):
            H = nx.Graph()
            for u, v, d in W1.edges(data=True):
                if d["dos"] >= k:
                    H.add_edge(u, v, weight=wlen(d, scheme))
            if H.number_of_edges() > 0:
                W = W1.subgraph(_max_weighted_component(H))
                norm_level += math.exp(k)
                S += math.exp(k) * sum(wlen(d, scheme) for _, _, d in W.edges(data=True)) / full
            elif norm_level == 0:
                norm_level = 1
        total += math.exp(n - j) * (S / norm_level)
        if S != 0:
            norm_comp += math.exp(n - j)
    return total / norm_comp


def all_metrics(G, scheme="cscs", with_closeness=True):
    """Dictionary with s_G, c_G (optional, slow on big cities) and l_G."""
    out = {"s_G": strength(G, scheme), "l_G": largest_connected_component(G, scheme)}
    if with_closeness:
        out["c_G"] = closeness(G, scheme)
    return out

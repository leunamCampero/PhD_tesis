"""
Original (arc-level) cycling network: Section II of the paper.

    traffic factor     f_a = psi exp(AADT/1000) (DoS < 5), psi 2.03 (DoS 5), psi (DoS 6)    Eq. (2)
    cyclist distance   b_a = d_a s_a f_a^rho  (+ eta * angle, eta = 0: no angles in the graphs) Eq. (1)
    flows (AON)        phi_a = sum_uv q_uv 1[a in p_uv]                                     Eq. (4)
    exposure           E_a = phi_a f_a d_a,   E^orig = sum_a E_a  (|A| normalisation: scale only)

The graph is undirected (w, d, phi symmetric, as assumed in the paper's Theorem 1).  One all-or-
nothing assignment = one multi-source Dijkstra from the distinct origins + a vectorised walk of the
predecessor trees for every OD pair (``assign``).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from . import config as C

EARTH_R = 6_371_008.8


# ==============================================================================================
# Element-wise factors
# ==============================================================================================
_F = np.array([np.nan] + [C.PSI * np.exp(C.AADT[k] / 1000.0) for k in range(1, 5)]
              + [C.PSI * C.PEDESTRIAN_FACTOR, C.PSI])


def traffic_factor(dos):
    """f(w) of Eq. (2) for an int array of DoS levels (1..6)."""
    return _F[np.asarray(dos, dtype=int)]


def slope_factor(slope_pct):
    """Slope factor of the notebooks (|slope|, Table of Cooper 2017)."""
    s = np.abs(np.asarray(slope_pct, dtype=float))
    out = np.ones_like(s)
    lower = 0.0
    for upper, factor in C.SLOPE_TABLE:
        out = np.where((s >= lower) & (s < upper), factor, out)
        lower = upper
    return out


def cyclist_distance(d, s, dos):
    """b_a of Eq. (1)."""
    return np.maximum(d * s * traffic_factor(dos) ** C.RHO, 1e-6)


def cost_table(lengths, dos0):
    """(E x 7) cost of raising each arc from dos0 to L (column L, inf if L < dos0)."""
    tab = np.full((len(dos0), 7), np.inf)
    unit = np.full((7, 7), np.inf)
    for a, row in C.COST.items():
        for b, c in row.items():
            unit[a, b] = c
    tab[:, 1:] = lengths[:, None] * unit[np.asarray(dos0, int)][:, 1:]
    return tab


def planar(lon, lat, lon0=None, lat0=None):
    """Local equirectangular projection (metres)."""
    lon0 = np.mean(lon) if lon0 is None else lon0
    lat0 = np.mean(lat) if lat0 is None else lat0
    x = EARTH_R * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon - lon0)
    y = EARTH_R * np.deg2rad(lat - lat0)
    return np.column_stack([x, y])


# ==============================================================================================
# Undirected graph + OD demand: all-or-nothing assignment (shared by the original and the
# clustered network)
# ==============================================================================================
class UGraph:
    """Undirected graph with a fixed sparse pattern (only the arc weights change between calls)."""

    def __init__(self, n, u, v, od_o, od_d, q):
        self.N, self.E = n, len(u)
        self.u, self.v = np.asarray(u), np.asarray(v)
        tmp = csr_matrix((np.arange(1, 2 * self.E + 1, dtype=float), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
        self._perm = tmp.data.astype(np.int64) - 1            # csr.data = np.r_[w, w][perm]
        self._csr = tmp
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        keys = lo.astype(np.int64) * n + hi
        self._order = np.argsort(keys)
        self._keys = keys[self._order]
        self.origins, self.od_row = np.unique(od_o, return_inverse=True)
        self.od_d, self.q = np.asarray(od_d), np.asarray(q, float)

    def arc_of(self, a, b):
        """Arc id of the undirected pair a-b (arrays)."""
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return self._order[np.searchsorted(self._keys, lo.astype(np.int64) * self.N + hi)]

    def matrix(self, weight):
        A = self._csr.copy()
        A.data = np.r_[weight, weight][self._perm]
        return A

    def aon(self, weight):
        """One multi-source Dijkstra + vectorised walk of the trees for every OD pair.

        Returns arc flows, node visits (demand of the paths through each node, end nodes included),
        the cost of every OD pair and the number of OD pairs with a path."""
        if len(self.origins) == 0:
            return np.zeros(self.E), np.zeros(self.N), np.zeros(0), 0
        dist, pred = dijkstra(self.matrix(weight), directed=True, indices=self.origins, return_predecessors=True)
        row, cur, q = self.od_row, self.od_d, self.q
        od_cost = dist[row, cur]
        ok = np.isfinite(od_cost)
        row, cur, q = row[ok], cur[ok], q[ok]
        arcs, arc_q, nodes, node_q = [np.zeros(0, int)], [np.zeros(0)], [cur], [q]
        while len(cur):
            p = pred[row, cur]
            alive = p >= 0
            row, cur, p, q = row[alive], cur[alive], p[alive], q[alive]
            arcs.append(self.arc_of(p, cur))
            arc_q.append(q)
            nodes.append(p)
            node_q.append(q)
            cur = p
        flow = np.bincount(np.concatenate(arcs), np.concatenate(arc_q), minlength=self.E)
        node_flow = np.bincount(np.concatenate(nodes), np.concatenate(node_q), minlength=self.N)
        return flow, node_flow, od_cost, int(ok.sum())


# ==============================================================================================
# Assignment result
# ==============================================================================================
@dataclass
class Assignment:
    flow: np.ndarray          # phi_a (E)
    node_flow: np.ndarray     # demand of the paths visiting each node (N)
    exposure_arc: np.ndarray  # E_a = phi_a f_a d_a
    exposure: float           # E^orig (sum)
    total_flow: float         # Phi^orig = sum_a phi_a
    od_cost: np.ndarray       # cyclist distance of each OD pair
    reached: int              # OD pairs with a path


# ==============================================================================================
# Network
# ==============================================================================================
class Network:
    """Cycling network of one city: nodes, undirected arcs, demand.  Built by 01_prepare_networks."""

    def __init__(self, city, nodes=None, edges=None, demand=None):
        """Read data/networks/<city>_*.csv, or use the given DataFrames (toy example, tests)."""
        self.city = city
        nodes = pd.read_csv(C.NETWORKS / f"{city}_nodes.csv") if nodes is None else nodes
        edges = pd.read_csv(C.NETWORKS / f"{city}_edges.csv") if edges is None else edges
        dem = pd.read_csv(C.NETWORKS / f"{city}_demand_{C.DEMAND}.csv") if demand is None else demand
        self.N, self.E = len(nodes), len(edges)
        self.lonlat = nodes[["lon", "lat"]].to_numpy()
        self.xy = nodes[["x", "y"]].to_numpy()
        self.u = edges["u"].to_numpy()
        self.v = edges["v"].to_numpy()
        self.length = edges["length_m"].to_numpy()
        self.slope = edges["slope_pct"].to_numpy()
        self.dos0 = edges["dos"].to_numpy().astype(int)
        self.s = slope_factor(self.slope) if C.SLOPE_SOURCE == "file" else np.ones(self.E)
        self.cost = cost_table(self.length, self.dos0)
        self.B_max = float(self.cost[:, 6].sum())
        self.od_o = dem["o"].to_numpy()
        self.od_d = dem["d"].to_numpy()
        self.q = dem["q"].to_numpy()
        self._prepare_graph()
        self.base = self.evaluate(self.dos0)

    # ---- sparse structure re-used by every Dijkstra ----------------------------------------
    def _prepare_graph(self):
        self.ug = UGraph(self.N, self.u, self.v, self.od_o, self.od_d, self.q)

    # ---- all-or-nothing assignment ------------------------------------------------------------
    def assign(self, weight, dos):
        """AON with arc weight ``weight`` (cyclist distance); exposure uses f(dos)."""
        flow, node_flow, od_cost, reached = self.ug.aon(weight)
        e_arc = flow * traffic_factor(dos) * self.length
        return Assignment(flow, node_flow, e_arc, float(e_arc.sum()), float(flow.sum()), od_cost, reached)

    def evaluate(self, dos):
        dos = np.asarray(dos, dtype=int)
        return self.assign(cyclist_distance(self.length, self.s, dos), dos)

    def upgrade_cost(self, dos):
        dos = np.asarray(dos, int)
        return float(self.cost[np.arange(self.E), dos].sum())

    # ---- helpers -------------------------------------------------------------------------------
    def incident_sum(self, arc_values):
        """Sum of an arc quantity over the arcs incident to every node (each arc counted at u and v)."""
        return np.bincount(self.u, arc_values, self.N) + np.bincount(self.v, arc_values, self.N)


# ==============================================================================================
# Building the CSV networks from the raw pickles (step 01)
# ==============================================================================================
def build_city(city, rng=None):
    """Read data/raw/<city>_bike_network_with_slope.pkl, write nodes / edges / demand CSVs."""
    import pickle

    rng = np.random.default_rng(C.SEED) if rng is None else rng
    with open(C.RAW / f"{city}_bike_network_with_slope.pkl", "rb") as f:
        G = pickle.load(f)
    G = G.to_undirected() if G.is_directed() else G
    G.remove_edges_from([(a, b) for a, b in G.edges() if a == b])
    names = list(G.nodes())
    idx = {n: i for i, n in enumerate(names)}
    lon = np.array([n[0] for n in names], float)
    lat = np.array([n[1] for n in names], float)
    xy = planar(lon, lat)
    rows = []
    for a, b, d in G.edges(data=True):
        rows.append((idx[a], idx[b], int(d["key"]), float(d["distance"]), float(d.get("slope", 0.0))))
    e = pd.DataFrame(rows, columns=["u", "v", "dos", "length_m", "slope_pct"])
    e["length_m"] = e["length_m"].clip(lower=0.5)
    n = pd.DataFrame({"node": np.arange(len(names)), "lon": lon, "lat": lat, "x": xy[:, 0], "y": xy[:, 1]})

    # largest connected component (demand nodes are drawn there)
    A = csr_matrix((np.ones(len(e)), (e.u, e.v)), shape=(len(n), len(n)))
    _, comp = connected_components(A, directed=False)
    big = np.bincount(comp).argmax()
    n["lcc"] = comp == big

    C.NETWORKS.mkdir(parents=True, exist_ok=True)
    n.to_csv(C.NETWORKS / f"{city}_nodes.csv", index=False)
    e.to_csv(C.NETWORKS / f"{city}_edges.csv", index=False)

    # ---- demand "paper130": the notebooks' OD pairs -------------------------------------------
    with open(C.RAW / f"{city}_node_demand_dic.pkl", "rb") as f:
        dic = pickle.load(f)
    od = [(idx[o], idx[d], float(q)) for (o, d), q in dic.items() if o in idx and d in idx and o != d and q > 0]
    p = pd.DataFrame(od, columns=["o", "d", "q"])
    p["q"] *= C.TRIPS_PER_CITY / p["q"].sum()
    p.to_csv(C.NETWORKS / f"{city}_demand_paper130.csv", index=False)

    # ---- demand "gravity" -------------------------------------------------------------------------
    mid = 0.5 * (xy[e.u] + xy[e.v])
    tree = cKDTree(mid)
    dens = np.array([e.length_m.to_numpy()[ix].sum() for ix in tree.query_ball_point(xy, C.DENSITY_RADIUS_M)])
    cand = np.flatnonzero(n["lcc"].to_numpy())
    w = dens[cand] / dens[cand].sum()
    origins = rng.choice(cand, size=C.N_ORIGINS, replace=False, p=w)
    od = []
    for o in origins:
        dist = np.hypot(*(xy[cand] - xy[o]).T)
        g = dens[cand] * np.exp(-dist / C.GRAVITY_LENGTH_M)
        g[cand == o] = 0.0
        dests = rng.choice(cand, size=C.DEST_PER_ORIGIN, replace=False, p=g / g.sum())
        for d in dests:
            od.append((o, d, dens[o] * dens[d] * np.exp(-np.hypot(*(xy[d] - xy[o])) / C.GRAVITY_LENGTH_M)))
    g = pd.DataFrame(od, columns=["o", "d", "q"])
    g["q"] *= C.TRIPS_PER_CITY / g["q"].sum()
    g.to_csv(C.NETWORKS / f"{city}_demand_gravity.csv", index=False)
    return n, e, p, g

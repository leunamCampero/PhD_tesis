"""
Cyclist-flow model of the paper (Section IV-A) and the exposure objective (Section IV-B).

    traffic factor      f_a  = psi * exp(AADT_a / 1000)   if DoS_a < 5          Eq. (2)
                             = psi * 2.03                 if DoS_a = 5
                             = psi                        if DoS_a = 6
    cyclist distance    b    = sum_a d_a * s_a * f_a**rho  +  kappa * sum(turn angles)   Eq. (1)
    flows (AON)         x_a  = sum_{u,v} q_uv * [a on the b-shortest path u -> v]  Eqs. (5)-(6)
    exposure            J_E  = sum_a  d_hat_a * f_hat_a * x_hat_a                Eqs. (7)-(9)

(hats = min-max normalisation to [0, 1]).

The ``Network`` object holds the graph as numpy arrays (edge i joins nodes u[i] and v[i]),
so a DoS configuration is simply an integer vector ``dos`` of length E.
"""
from math import atan2, degrees

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import config as C


# ========================================================================================
# Element-wise factors
# ========================================================================================
def traffic_factor(dos):
    """f_a of Eq. (2) for an array of DoS levels."""
    dos = np.asarray(dos)
    aadt = np.vectorize(C.AADT.get)(dos) / 1000.0
    f = C.PSI * np.exp(aadt)                                 # DoS 1-4: exposed to traffic
    f = np.where(dos == 5, C.PSI * C.PEDESTRIAN_FACTOR, f)   # shared with pedestrians
    f = np.where(dos == 6, C.PSI, f)                         # fully separated: AADT has no effect
    return f


F_MIN = float(traffic_factor([6])[0])   # 0.84     -> f_hat = 0  (DoS 6)
F_MAX = float(traffic_factor([1])[0])   # 3919.21  -> f_hat = 1  (DoS 1, AADT 8448)


def slope_factor(slope_pct):
    """Uphill slope factor s_a of Table IV (slope in %). Slopes below 2 % -> 1."""
    s = np.asarray(slope_pct, dtype=float)
    out = np.ones_like(s)
    lower = -np.inf
    for upper, factor in C.SLOPE_TABLE:
        out = np.where((s >= lower) & (s < upper) & (s >= 2.0), factor, out)
        lower = upper
    return out


def directional_slope_factor(slope_pct):
    """Uphill: table value. Downhill: reciprocal of the uphill value of the same |slope|."""
    s = np.asarray(slope_pct, dtype=float)
    return np.where(s >= 0, slope_factor(s), 1.0 / slope_factor(-s))


def minmax(x):
    """Min-max normalisation to [0, 1] (all ones if x is constant)."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.ones_like(x) if hi == lo else (x - lo) / (hi - lo)


def turn_angle_deg(p, q, r):
    """Turning angle (0..180 deg) at q when going p -> q -> r; lon/lat projected locally."""
    k = np.cos(np.radians(q[1]))
    a1 = atan2(q[1] - p[1], (q[0] - p[0]) * k)
    a2 = atan2(r[1] - q[1], (r[0] - q[0]) * k)
    d = abs(degrees(a2 - a1)) % 360
    return min(d, 360 - d)


# ========================================================================================
# Network
# ========================================================================================
class Network:
    """Grenoble cycling network + cyclist demand."""

    def __init__(self, demand=C.DEFAULT_DEMAND):
        """demand: "full_od" (EMD 2020, as in the paper) or "historical_unit" (optional)."""
        self.demand_kind = demand
        nodes = pd.read_csv(C.DATA / "network" / "grenoble_nodes.csv")
        edges = pd.read_csv(C.DATA / "network" / "grenoble_edges.csv")
        self.nodes, self.edges = nodes, edges
        self.N, self.E = len(nodes), len(edges)
        self.xy = nodes[["lon", "lat"]].to_numpy()
        self.zone = nodes["zone_id"].to_numpy()
        self.zone_weight = nodes["zone_weight"].to_numpy()
        self.u = edges["u"].to_numpy()
        self.v = edges["v"].to_numpy()
        self.length = edges["length_m"].to_numpy()
        self.slope = edges["slope_pct"].to_numpy()          # % from u to v
        self.dos0 = edges["dos"].to_numpy().astype(int)

        # Directed arcs: arc k < E goes u->v, arc k + E goes v->u (same edge k).
        self.tail = np.concatenate([self.u, self.v])
        self.head = np.concatenate([self.v, self.u])
        self.arc_index = np.full((self.N, self.N), -1, dtype=np.int32)
        self.arc_index[self.tail, self.head] = np.arange(2 * self.E)

        self.d_hat = minmax(self.length)           # normalised arc length (fixed)
        self._load_demand()

    # ------------------------------------------------------------------------------------
    def graph(self, dos=None):
        """networkx view of the network (node ids = 0..N-1) with DoS and lengths."""
        dos = self.dos0 if dos is None else dos
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        for i in range(self.E):
            G.add_edge(int(self.u[i]), int(self.v[i]), eid=i, length=self.length[i], dos=int(dos[i]))
        return G

    # ------------------------------------------------------------------------------------
    # Arc costs b_a (the "cyclist distance" of one arc) - used as Dijkstra weights
    # ------------------------------------------------------------------------------------
    def arc_costs(self, dos):
        """Return (cost u->v, cost v->u) arrays of length E for DoS configuration ``dos``.

        b_a = d_a * s_a * f_a**rho, with the slope factor depending on the direction.
        """
        f_rho = traffic_factor(dos) ** C.RHO
        fwd = self.length * directional_slope_factor(self.slope) * f_rho
        bwd = self.length * directional_slope_factor(-self.slope) * f_rho
        return fwd, bwd

    def cyclist_distance(self, node_path, dos=None, kappa=C.KAPPA):
        """Cyclist distance b_(u,v) of Eq. (1) for an explicit node path (with turn costs).

        Returns (b, arc part, total turning in degrees).
        """
        dos = self.dos0 if dos is None else dos
        fwd, bwd = self.arc_costs(dos)
        arcs = [self.arc_index[a, b] for a, b in zip(node_path, node_path[1:])]
        cost = sum(fwd[k] if k < self.E else bwd[k - self.E] for k in arcs)
        pts = [self.xy[n] for n in node_path]
        turns = sum(turn_angle_deg(pts[j], pts[j + 1], pts[j + 2]) for j in range(len(pts) - 2))
        return cost + kappa * turns, cost, turns

    # ------------------------------------------------------------------------------------
    # Demand q_uv
    # ------------------------------------------------------------------------------------
    def _load_demand(self):
        dem = C.DATA / "demand"
        if self.demand_kind == "full_od":
            # q_uv = factor * OD[zone(u), zone(v)] * a_u * a_v   (a_u: share of u in its IRIS)
            od = pd.read_csv(dem / "od_iris_grenoble.csv")
            zones = np.unique(self.zone)
            zpos = {z: i for i, z in enumerate(zones)}
            M = np.zeros((len(zones), len(zones)))
            M[[zpos[z] for z in od.zone_o], [zpos[z] for z in od.zone_d]] = od.trips_matrix
            self._od_zone = M * C.DEMAND_FACTOR
            self._zi = np.array([zpos[z] for z in self.zone])
            self.sources = np.flatnonzero(self.zone_weight > 0)
        elif self.demand_kind == "historical_unit":
            od = pd.read_csv(dem / "historical_unit_od.csv")
            self._Q = csr_matrix((od.demand.to_numpy(float), (od.origin, od.destination)),
                                 shape=(self.N, self.N))
            self.sources = np.unique(od.origin)
        else:
            raise ValueError(f"unknown demand {self.demand_kind!r}")

    def demand_rows(self, rows):
        """Dense block q[rows, :] of the node-to-node demand."""
        if self.demand_kind == "full_od":
            zi = self._zi
            Q = self._od_zone[zi[rows]][:, zi] * self.zone_weight[rows, None] * self.zone_weight[None, :]
            Q[zi[rows][:, None] == zi[None, :]] = 0.0   # trips inside one IRIS are not routed
            return Q
        return self._Q[rows].toarray()

    def total_demand(self):
        if self.demand_kind == "full_od":
            return sum(self.demand_rows(c).sum() for c in np.array_split(self.sources, 8))
        return float(self._Q.sum())

    # ------------------------------------------------------------------------------------
    # All-or-nothing assignment (lower level of Problem 1)
    # ------------------------------------------------------------------------------------
    def flows(self, dos, chunk=4000):
        """Arc flows x_a (length E, both directions summed) for DoS configuration ``dos``.

        For every origin a shortest-path tree (weights b_a) is built with scipy's Dijkstra.
        * full OD matrix (11.7 M pairs): demand is pushed from the leaves to the root of each
          tree; visiting nodes by decreasing distance guarantees that a node is handled after
          all nodes below it in the tree.
        * sparse OD (historical OD): every path is walked back from the destination.
        """
        fwd, bwd = self.arc_costs(dos)
        A = csr_matrix((np.concatenate([fwd, bwd]), (self.tail, self.head)), shape=(self.N, self.N))
        if self.demand_kind != "full_od":
            return self._flows_by_paths(A)
        x = np.zeros(2 * self.E)
        for rows in np.array_split(self.sources, max(1, int(np.ceil(len(self.sources) / chunk)))):
            dist, pred = dijkstra(A, directed=True, indices=rows, return_predecessors=True)
            load = self.demand_rows(rows)                      # demand still to be routed
            order = np.argsort(-np.where(np.isinf(dist), -1.0, dist), axis=1, kind="stable")
            r = np.arange(len(rows))
            arcs, vals = [], []
            for j in range(self.N):
                t = order[:, j]
                p = pred[r, t]
                ok = p >= 0                                    # skip origin / unreachable
                if not ok.any():
                    continue
                rr, tt, pp = r[ok], t[ok], p[ok]
                val = load[rr, tt]
                load[rr, pp] += val                            # parent carries the subtree
                arcs.append(self.arc_index[pp, tt])
                vals.append(val)
            x += np.bincount(np.concatenate(arcs), np.concatenate(vals), minlength=2 * self.E)
        return x[: self.E] + x[self.E:]

    def _flows_by_paths(self, A):
        """AON flows for a sparse list of OD pairs (walk all shortest paths backwards at once)."""
        Q = self._Q.tocoo()
        dist, pred = dijkstra(A, directed=True, indices=self.sources, return_predecessors=True)
        row = np.searchsorted(self.sources, Q.row)            # row of each pair's origin tree
        keep = (Q.data != 0) & ~np.isinf(dist[row, Q.col])    # skip empty / unreachable pairs
        row, src, cur, q = row[keep], Q.row[keep], Q.col[keep].copy(), Q.data[keep]
        x = np.zeros(2 * self.E)
        active = cur != src
        while active.any():                                   # one arc per step for all paths
            r, c = row[active], cur[active]
            p = pred[r, c]
            x += np.bincount(self.arc_index[p, c], q[active], minlength=2 * self.E)
            cur[active] = p
            active = cur != src
        return x[: self.E] + x[self.E:]

    # ------------------------------------------------------------------------------------
    # Objective (upper level of Problem 1)
    # ------------------------------------------------------------------------------------
    def evaluate(self, dos):
        """Flows and exposure for one DoS configuration."""
        dos = np.asarray(dos, dtype=int)
        x = self.flows(dos)
        f = traffic_factor(dos)
        f_hat = (f - F_MIN) / (F_MAX - F_MIN)          # fixed bounds -> comparable scenarios
        x_hat = minmax(x)
        exposure_arc = self.d_hat * f_hat                # E_a of Eq. (9)
        return dict(
            J_E=float(np.sum(exposure_arc * x_hat)),     # Problem 1 objective
            flows=x, x_hat=x_hat, f=f, f_hat=f_hat, exposure_arc=exposure_arc,
        )

    def exposure(self, dos):
        return self.evaluate(dos)["J_E"]

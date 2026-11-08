"""
Lower level of the bi-level problem (Section II-C) and the two upper-level measures.

    traffic factor      f_a  = psi * exp(AADT_a / 1000)   if DoS_a < 5          Eq. (2)
                             = psi * 2.03                 if DoS_a = 5
                             = psi                        if DoS_a = 6
    cyclist distance    b    = sum_a d_a * s_a * f_a**rho  +  kappa * sum(turn angles)   Eq. (1)
    flows (AON)         x_a  = sum_{u,v} q_uv * [a on the b-shortest path u -> v]  Eqs. (3)-(4)
    path exposure       E_p  = sum_{a in p} d_hat_a * f_hat_a                     Eq. (5)
    total exposure      J_E  = sum_a d_hat_a * f_hat_a * x_hat_a                  Problem 1

``Network.evaluate(dos)`` runs ONE traffic assignment and returns both the arc flows (for
J_E) and the exposure of every OD path (for the sufficientarian measures in justice.py).
"""
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
    f = C.PSI * np.exp(np.vectorize(C.AADT.get)(dos) / 1000.0)  # DoS 1-4: exposed to traffic
    f = np.where(dos == 5, C.PSI * C.PEDESTRIAN_FACTOR, f)       # shared with pedestrians
    return np.where(dos == 6, C.PSI, f)                          # fully separated


F_MIN = float(traffic_factor([6])[0])   # f of DoS 6 -> f_hat = 0
F_MAX = float(traffic_factor([1])[0])   # f of DoS 1 -> f_hat = 1


def slope_factor(slope_pct):
    """Uphill slope factor s_a (Table III, slope in %). Below 2 % -> 1."""
    s = np.asarray(slope_pct, dtype=float)
    out = np.ones_like(s)
    lower = -np.inf
    for upper, factor in C.SLOPE_TABLE:
        out = np.where((s >= lower) & (s < upper) & (s >= 2.0), factor, out)
        lower = upper
    return out


def directional_slope_factor(slope_pct):
    """Uphill: table value; downhill: its reciprocal."""
    s = np.asarray(slope_pct, dtype=float)
    return np.where(s >= 0, slope_factor(s), 1.0 / slope_factor(-s))


def minmax(x):
    """Min-max normalisation to [0, 1] (all ones if x is constant)."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.ones_like(x) if hi == lo else (x - lo) / (hi - lo)


# ========================================================================================
# Network + demand
# ========================================================================================
class Network:
    """Grenoble cycling network, its 70 IRIS regions and the cyclist demand."""

    def __init__(self, nodes_per_iris=C.NODES_PER_IRIS):
        nodes = pd.read_csv(C.DATA / "network" / "grenoble_nodes.csv")
        edges = pd.read_csv(C.DATA / "network" / "grenoble_edges.csv")
        self.nodes, self.edges = nodes, edges
        self.N, self.E = len(nodes), len(edges)
        self.xy = nodes[["lon", "lat"]].to_numpy()
        self.zone = nodes["zone_id"].to_numpy()
        self.u = edges["u"].to_numpy()
        self.v = edges["v"].to_numpy()
        self.length = edges["length_m"].to_numpy()
        self.slope = edges["slope_pct"].to_numpy()          # % from u to v
        self.dos0 = edges["dos"].to_numpy().astype(int)

        # directed arcs: k < E is u->v, k + E is v->u (same edge k)
        self.tail = np.concatenate([self.u, self.v])
        self.head = np.concatenate([self.v, self.u])
        self.arc_index = np.full((self.N, self.N), -1, dtype=np.int32)
        self.arc_index[self.tail, self.head] = np.arange(2 * self.E)
        self.d_hat = minmax(self.length)

        self._build_demand(nodes["zone_weight"].to_numpy(float), nodes_per_iris)

    # ------------------------------------------------------------------------------------
    def _build_demand(self, weight, k):
        """Trip ends: the k heaviest nodes of every IRIS (weights re-normalised per IRIS).

        q_uv = 0.5 * 0.05 * OD[zone(u), zone(v)] * a_u * a_v   for u, v in different IRIS.
        """
        keep = np.zeros(self.N, bool)
        a = np.zeros(self.N)
        for z in np.unique(self.zone):
            idx = np.flatnonzero((self.zone == z) & (weight > 0))
            if k is not None:
                idx = idx[np.argsort(-weight[idx], kind="stable")[:k]]
            keep[idx] = True
            a[idx] = weight[idx] / weight[idx].sum()
        self.sources = np.flatnonzero(keep)                     # origins = destinations
        od = pd.read_csv(C.DATA / "demand" / "od_iris_grenoble.csv")
        self.zones = np.unique(self.zone)
        zpos = {z: i for i, z in enumerate(self.zones)}
        M = np.zeros((len(self.zones), len(self.zones)))
        M[[zpos[z] for z in od.zone_o], [zpos[z] for z in od.zone_d]] = od.trips_matrix
        zi = np.array([zpos[z] for z in self.zone[self.sources]])
        Q = M[zi][:, zi] * C.DEMAND_FACTOR * np.outer(a[self.sources], a[self.sources])
        Q[zi[:, None] == zi[None, :]] = 0.0                     # trips inside one IRIS: not routed
        self.Q = Q                                              # (S x S) bikes/day
        self.source_zone = self.zone[self.sources]

    # ------------------------------------------------------------------------------------
    def arc_costs(self, dos):
        """Dijkstra weights b_a = d_a * s_a * f_a**rho (u->v, v->u)."""
        f_rho = traffic_factor(dos) ** C.RHO
        fwd = self.length * directional_slope_factor(self.slope) * f_rho
        bwd = self.length * directional_slope_factor(-self.slope) * f_rho
        return fwd, bwd

    def exposure_arc(self, dos):
        """E_a = d_hat_a * f_hat_a (Eq. 6), in [0, 1]."""
        f = traffic_factor(dos)
        return self.d_hat * (f - F_MIN) / (F_MAX - F_MIN)

    # ------------------------------------------------------------------------------------
    def assign(self, dos, Q=None):
        """All-or-nothing assignment.  Returns (arc flows x, path exposure E_p (S x S)).

        ``Q`` (S x S) replaces the trip matrix (used to route only the insufficient trips).

        One Dijkstra tree per origin (scipy).  Visiting the nodes of a tree by increasing
        distance gives the path exposure (parent before child); by decreasing distance the
        demand is pushed from the leaves to the origin, which gives the arc flows.
        """
        fwd, bwd = self.arc_costs(dos)
        A = csr_matrix((np.concatenate([fwd, bwd]), (self.tail, self.head)), shape=(self.N, self.N))
        dist, pred = dijkstra(A, directed=True, indices=self.sources, return_predecessors=True)
        S = len(self.sources)
        r = np.arange(S)
        e_arc = np.tile(self.exposure_arc(dos), 2)            # same exposure in both directions
        order = np.argsort(np.where(np.isinf(dist), np.inf, dist), axis=1, kind="stable")

        # path exposure, parents first
        Ep = np.zeros((S, self.N))
        for j in range(1, self.N):
            t = order[:, j]
            p = pred[r, t]
            ok = p >= 0
            if ok.any():
                Ep[r[ok], t[ok]] = Ep[r[ok], p[ok]] + e_arc[self.arc_index[p[ok], t[ok]]]

        # flows, children first
        load = np.zeros((S, self.N))
        load[:, self.sources] = self.Q if Q is None else Q
        arcs, vals = [], []
        for j in range(self.N - 1, 0, -1):
            t = order[:, j]
            p = pred[r, t]
            ok = p >= 0
            if not ok.any():
                continue
            rr, tt, pp = r[ok], t[ok], p[ok]
            val = load[rr, tt]
            load[rr, pp] += val
            arcs.append(self.arc_index[pp, tt])
            vals.append(val)
        x = np.bincount(np.concatenate(arcs), np.concatenate(vals), minlength=2 * self.E)
        return x[: self.E] + x[self.E:], Ep[:, self.sources]

    def evaluate(self, dos):
        """Flows, J_E and path exposures for one DoS configuration."""
        dos = np.asarray(dos, dtype=int)
        x, Ep = self.assign(dos)
        x_hat = minmax(x)
        return dict(dos=dos, flows=x, x_hat=x_hat, J_E=float(np.sum(self.exposure_arc(dos) * x_hat)),
                    path_exposure=Ep)

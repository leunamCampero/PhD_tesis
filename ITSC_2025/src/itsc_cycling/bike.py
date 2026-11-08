"""
Bicycle layer (Section II-C of the paper) - same model as ECC_2025 / CDC_2025.

    traffic factor      f_a  = psi * exp(AADT_a / 1000)   if DoS_a < 5          Eq. (2)
                             = psi * 2.03                 if DoS_a = 5
                             = psi                        if DoS_a = 6
    cyclist distance    b_a  = d_a * s_a * f_a**rho       (Dijkstra weight)     Eq. (1)
    flows (AON)         phi_a = sum_{u,v} q_uv * [a on the b-shortest path u -> v]  Eq. (3)
    exposure            J_E  = sum_a d_hat_a * f_hat_a * phi_hat_a              Eq. (4)

(hats = min-max normalisation, as in the paper's Table II "all normalised between their
minimum and maximum values").  ``BikeNetwork.evaluate(dos)`` runs ONE all-or-nothing
assignment and returns the flows, J_E, and the cyclist distance of every OD pair.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import config as C
from .trees import TreeOrder, path_sums


# ========================================================================================
# Element-wise factors
# ========================================================================================
def traffic_factor(dos):
    """f_a of Eq. (2) for an array of DoS levels."""
    dos = np.asarray(dos)
    aadt = np.array([0] + [C.AADT[k] for k in range(1, 7)], float)[dos]
    f = C.PSI * np.exp(aadt / 1000.0)                      # DoS 1-4: exposed to traffic
    f = np.where(dos == 5, C.PSI * C.PEDESTRIAN_FACTOR, f)  # shared with pedestrians
    return np.where(dos == 6, C.PSI, f)                     # fully separated


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
    """Min-max normalisation to [0, 1] (zeros if x is constant)."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


# ========================================================================================
# Network + demand
# ========================================================================================
class BikeNetwork:
    """Grenoble cycling network (ECC/CDC graph), its IRIS and the cyclist demand."""

    def __init__(self, nodes_per_iris=C.BIKE_NODES_PER_IRIS):
        nodes = pd.read_csv(C.DATA / "network" / "bike_nodes.csv")
        edges = pd.read_csv(C.DATA / "network" / "bike_edges.csv")
        self.nodes, self.edges = nodes, edges
        self.N, self.E = len(nodes), len(edges)
        self.xy = nodes[["lon", "lat"]].to_numpy()
        self.zone = nodes["zone_id"].to_numpy()
        self.u = edges["u"].to_numpy()
        self.v = edges["v"].to_numpy()
        self.length = edges["length_m"].to_numpy()
        self.slope = edges["slope_pct"].to_numpy()          # % from u to v
        self.dos0 = edges["dos"].to_numpy().astype(int)
        self.car_edge = edges["car_edge"].to_numpy()        # car road it runs along (-1: none)

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

        q_uv = 0.5 * bike_share * OD[zone(u), zone(v)] * a_u * a_v   (u, v in different IRIS)
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
        self.zones, M = zone_matrix()
        zpos = {z: i for i, z in enumerate(self.zones)}
        zi = np.array([zpos[z] for z in self.zone[self.sources]])
        Q = M[zi][:, zi] * C.SYMMETRIC_MATRIX_FACTOR * C.BIKE_SHARE
        Q = Q * np.outer(a[self.sources], a[self.sources])
        Q[zi[:, None] == zi[None, :]] = 0.0                     # trips inside one IRIS: not routed
        self.Q = Q                                              # (S x S) bike trips/day
        self.source_zone = self.zone[self.sources]

    # ------------------------------------------------------------------------------------
    def arc_costs(self, dos):
        """Dijkstra weights b_a = d_a * s_a * f_a**rho (u->v, v->u)."""
        f_rho = traffic_factor(dos) ** C.RHO
        fwd = self.length * directional_slope_factor(self.slope) * f_rho
        bwd = self.length * directional_slope_factor(-self.slope) * f_rho
        return fwd, bwd

    def exposure_arc(self, dos):
        """E_a = d_hat_a * f_hat_a, in [0, 1]."""
        return self.d_hat * (traffic_factor(dos) - F_MIN) / (F_MAX - F_MIN)

    def trees(self, dos):
        """Shortest-path trees of the cyclist distance from every trip end."""
        fwd, bwd = self.arc_costs(dos)
        A = csr_matrix((np.concatenate([fwd, bwd]), (self.tail, self.head)), shape=(self.N, self.N))
        return dijkstra(A, directed=True, indices=self.sources, return_predecessors=True)

    def assign(self, dos, Q=None):
        """All-or-nothing assignment of Q (default: the cyclist demand).

        Returns (edge flows phi (E), cyclist distance b of every OD pair (S x S), trees).
        """
        dist, pred = self.trees(dos)
        load = np.zeros((len(self.sources), self.N))
        load[:, self.sources] = self.Q if Q is None else Q
        order = TreeOrder(pred, self.arc_index)
        x = order.load(load, 2 * self.E)
        return x[: self.E] + x[self.E:], dist[:, self.sources], (dist, pred)

    def path_exposure(self, dos, pred):
        """E_p = sum_{a in p} d_hat_a f_hat_a for every OD path (S x S)."""
        e = np.tile(self.exposure_arc(dos), 2)
        return path_sums(pred, self.arc_index, e)[:, self.sources]

    def evaluate(self, dos):
        """Flows, J_E and OD cyclist distances for one DoS configuration."""
        dos = np.asarray(dos, dtype=int)
        x, b, (dist, pred) = self.assign(dos)
        x_hat = minmax(x)
        return dict(dos=dos, flows=x, x_hat=x_hat, J_E=float(np.sum(self.exposure_arc(dos) * x_hat)),
                    od_distance=b, pred=pred)


def zone_matrix():
    """EMD 2020 IRIS x IRIS trip matrix (all modes, both directions stored)."""
    od = pd.read_csv(C.DATA / "demand" / "od_iris_grenoble.csv")
    zones = np.unique(np.r_[od.zone_o, od.zone_d])
    zpos = {z: i for i, z in enumerate(zones)}
    M = np.zeros((len(zones), len(zones)))
    M[[zpos[z] for z in od.zone_o], [zpos[z] for z in od.zone_d]] = od.trips_matrix
    return zones, M

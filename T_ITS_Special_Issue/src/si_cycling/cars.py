"""
Car layer (Section II-D of the paper): BPR travel times and the user-equilibrium (UE)
traffic assignment of Problem 1.

    BPR             t_a(phi) = t0_a * (1 + mu * (phi / theta_a) ** sigma)          Eq. (6)
    UE (Beckmann)   min sum_a int_0^phi_a t_a(x) dx   s.t. flow conservation      Problem 1
    total time      TTT = sum_a phi_a * t_a(phi_a)                                 (veh.h)

Network: OSM drive network of Grenoble (4,720 nodes, 6,175 roads, every road in both
directions -> 12,350 arcs).  Car trips start and end at 4 representative nodes per IRIS
(276 trip ends); the demand is the morning peak hour (veh/h), see config.py.

UE solver: conjugate Frank-Wolfe (CFW, Mitradjieva & Lindberg, Transp. Sci. 2013) with an
exact line search (bisection on the directional derivative of the Beckmann function).
Each iteration = one all-or-nothing assignment (trees.TreeOrder, ~0.1 s).  The relative
gap (TTT - sum_od q * shortest time) / TTT measures the distance to equilibrium.
Warm starts (``x0``) make the post-upgrade UE and the surrogate samples much faster,
because an upgrade changes only a few capacities.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import config as C
from .bike import zone_matrix
from .trees import TreeOrder


class CarNetwork:
    """Car layer: arcs, free-flow times, capacities, peak-hour demand, UE solver."""

    def __init__(self, demand_multiplier=1.0):
        nodes = pd.read_csv(C.DATA / "network" / "car_nodes.csv")
        edges = pd.read_csv(C.DATA / "network" / "car_edges.csv")
        edges = edges[edges.u != edges.v].reset_index(drop=True)   # no self-loops
        self.nodes, self.edges = nodes, edges
        self.N, self.E = len(nodes), len(edges)
        self.xy = nodes[["lon", "lat"]].to_numpy()
        self.edge_id = edges["edge_id"].to_numpy()                # id used in bike_edges.car_edge
        self.u, self.v = edges["u"].to_numpy(), edges["v"].to_numpy()
        # directed arcs: k < E is u->v, k + E is v->u (same road k)
        self.tail = np.concatenate([self.u, self.v])
        self.head = np.concatenate([self.v, self.u])
        self.arc_index = np.full((self.N, self.N), -1, dtype=np.int32)
        self.arc_index[self.tail, self.head] = np.arange(2 * self.E)
        length = edges["length_m"].to_numpy()
        speed = edges["speed_kmh"].to_numpy()
        self.length = np.tile(length, 2)
        self.t0 = np.tile(length / (speed / 3.6) / 60.0, 2)          # free-flow time (min)
        self.capacity0 = np.tile(edges["capacity_dir"].to_numpy(float), 2)   # veh/h
        self.lanes = np.tile(edges["lanes_dir"].to_numpy(), 2)
        self.lane_width = np.tile(edges["lane_width_m"].to_numpy(), 2)
        self.highway = np.tile(edges["highway"].to_numpy(), 2)
        # car road id (edge_id) -> row of edges (for the bike -> car matching)
        self.row_of_edge_id = np.full(int(self.edge_id.max()) + 1, -1)
        self.row_of_edge_id[self.edge_id] = np.arange(self.E)
        self._build_demand(demand_multiplier)

    # ------------------------------------------------------------------------------------
    def _build_demand(self, m):
        """Peak-hour car OD matrix between the trip ends (veh/h).

        q_uv = OD[zone(u), zone(v)] * 0.5 * car_share * peak_hour_factor * m / (k_u * k_v)
        where k_z = number of trip ends of IRIS z (trips spread evenly over them).
        """
        self.sources = np.flatnonzero(self.nodes["is_trip_end"].to_numpy())
        zone = self.nodes["zone_id"].to_numpy()[self.sources]
        zones, M = zone_matrix()
        zpos = {z: i for i, z in enumerate(zones)}
        zi = np.array([zpos[z] for z in zone])
        k = np.bincount(zi, minlength=len(zones))[zi].astype(float)
        Q = M[zi][:, zi] * C.SYMMETRIC_MATRIX_FACTOR * C.CAR_SHARE * C.PEAK_HOUR_FACTOR
        Q = Q / np.outer(k, k)
        Q[zi[:, None] == zi[None, :]] = 0.0                     # trips inside one IRIS: not routed
        self.Q_base = Q                                         # m = 1
        self.source_zone = zone
        self.set_demand_multiplier(m)

    def set_demand_multiplier(self, m):
        self.multiplier = float(m)
        self.Q = self.Q_base * self.multiplier
        self.D = np.zeros((len(self.sources), self.N))          # demand to every node
        self.D[:, self.sources] = self.Q

    # ------------------------------------------------------------------------------------
    # BPR and the Beckmann function
    # ------------------------------------------------------------------------------------
    def time(self, x, cap):
        """t_a(x) (min)."""
        return self.t0 * (1.0 + C.BPR_MU * (x / cap) ** C.BPR_SIGMA)

    def dtime(self, x, cap):
        """dt_a/dx."""
        return self.t0 * C.BPR_MU * C.BPR_SIGMA * x ** (C.BPR_SIGMA - 1) / cap ** C.BPR_SIGMA

    def beckmann(self, x, cap):
        s = C.BPR_SIGMA
        return float(np.sum(self.t0 * (x + C.BPR_MU * cap / (s + 1) * (x / cap) ** (s + 1))))

    def total_travel_time(self, x, cap=None):
        """TTT = sum_a x_a t_a(x_a), in vehicle-hours (flows fixed: no re-routing)."""
        cap = self.capacity0 if cap is None else cap
        return float(np.sum(x * self.time(x, cap))) / 60.0

    # ------------------------------------------------------------------------------------
    # All-or-nothing and user equilibrium
    # ------------------------------------------------------------------------------------
    def aon(self, t, scale=1.0):
        """All-or-nothing loading on times t.  Returns (arc flows, OD shortest times S x S)."""
        A = csr_matrix((t, (self.tail, self.head)), shape=(self.N, self.N))
        dist, pred = dijkstra(A, directed=True, indices=self.sources, return_predecessors=True)
        y = TreeOrder(pred, self.arc_index).load(self.D * scale, 2 * self.E)
        return y, dist[:, self.sources]

    def _line_search(self, x, d, cap, iters=30):
        """argmin_{tau in [0,1]} Beckmann(x + tau d): bisection on the derivative."""
        def slope(tau):
            return float(np.sum(self.time(x + tau * d, cap) * d))
        if slope(1.0) <= 0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if slope(mid) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def solve_ue(self, cap=None, scale=1.0, x0=None, rel_gap=C.UE_REL_GAP,
                 max_iter=C.UE_MAX_ITER, method=C.UE_METHOD, verbose=False):
        """User equilibrium (Problem 1) by Frank-Wolfe type algorithms.

        method: 'bfw' bi-conjugate FW (default), 'cfw' conjugate FW, 'fw' plain FW
                (Mitradjieva & Lindberg, Transp. Sci. 47(2), 2013).
        cap: arc capacities (default: original); scale: multiplies the car demand (mode
        shift); x0: warm start (flows of a similar network).  Returns a dict with the flows
        ``x``, times ``t``, ``TTT`` (veh.h), OD times ``od_time`` (min), gap and history.
        """
        cap = self.capacity0 if cap is None else cap
        t_start = time.time()
        if x0 is None:
            x, _ = self.aon(self.t0, scale)
        else:
            x = np.asarray(x0, float).copy()
        demand = float(self.Q.sum() * scale)
        s1 = s2 = None                      # previous two search points s_{k-1}, s_{k-2}
        tau_prev, hist, gap = None, [], np.inf
        for k in range(1, max_iter + 1):
            t = self.time(x, cap)
            y, od_t = self.aon(t, scale)
            tx = float(np.sum(t * x))
            sp = float(np.sum(self.Q * scale * od_t))          # = t . y
            gap = (tx - sp) / tx
            hist.append(gap)
            if verbose and (k % 10 == 0 or k == 1):
                print(f"      UE it {k:3d}: rel. gap {gap:.2e}  ({time.time() - t_start:.1f} s)")
            if gap < rel_gap:
                break
            s = y
            if method != "fw" and s1 is not None and tau_prev is not None and tau_prev < 1:
                h = self.dtime(x, cap)                          # Hessian diagonal
                if method == "bfw" and s2 is not None:
                    d1 = s1 - x                                 # \bar d_{k-1}
                    d2 = tau_prev * s1 - x + (1 - tau_prev) * s2  # \bar d_{k-2}
                    den_mu = float(np.sum(d2 * h * (s2 - s1)))
                    mu = -float(np.sum(d2 * h * (y - x))) / den_mu if den_mu != 0 else 0.0
                    den_nu = float(np.sum(d1 * h * d1))
                    nu = -float(np.sum(d1 * h * (y - x))) / den_nu if den_nu != 0 else 0.0
                    nu += mu * tau_prev / (1 - tau_prev)
                    mu, nu = max(mu, 0.0), max(nu, 0.0)
                    b0 = 1.0 / (1.0 + mu + nu)
                    s = b0 * y + nu * b0 * s1 + mu * b0 * s2
                else:                                           # conjugate FW
                    a, b = s1 - x, y - x
                    den = float(np.sum(a * h * (y - s1)))
                    alpha = float(np.sum(a * h * b)) / den if den != 0 else 0.0
                    alpha = min(max(alpha, 0.0), 0.99)
                    s = alpha * s1 + (1 - alpha) * y
            d = s - x
            tau = self._line_search(x, d, cap)
            if tau <= 0:                                        # no progress: restart with FW
                s, d = y, y - x
                tau = self._line_search(x, d, cap)
                s1 = s2 = None
            x = x + tau * d
            s2, s1, tau_prev = s1, s, tau
        t = self.time(x, cap)
        _, od_t = self.aon(t, scale)
        return dict(x=x, t=t, TTT=float(np.sum(x * t)) / 60.0, od_time=od_t, gap=gap,
                    iterations=k, history=hist, seconds=time.time() - t_start,
                    demand=demand)

    # ------------------------------------------------------------------------------------
    def two_way(self, arc_values):
        """Sum both directions of every road (E values)."""
        return arc_values[: self.E] + arc_values[self.E:]

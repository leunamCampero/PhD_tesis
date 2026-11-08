"""
Clustered network G~ of a partition of the nodes (Sections II-D/E and III-B of the paper).

Objects (G_agg = V~ u A~):
    intra object of cluster C_r       arcs A(C_r)        (only clusters with at least one inner arc)
    inter object (C_r, C_s), r < s    arcs A(C_r, C_s)
For every object g:  w0_g = min DoS of its arcs (conservative),  d~_g, s~_g = flow-weighted means
(<x>^phi of Eq. 9; plain mean if the object carries no flow),  f~_g = f(w~_g).

Clustered cyclist distance (Eqs. 10-14): a path C_r -> ... -> C_s costs the inter terms of its
super-arcs plus the intra term of every visited cluster.  It is computed on the cluster graph with
super-arc weight  b~_inter(r,s) + (b~_intra(r) + b~_intra(s)) / 2 : the end clusters get only half of
their intra term, a constant for each OD pair, so the shortest paths are exactly those of Eq. (14).

Clustered demand q~ (Eq. 15), inter / intra flows (Eqs. 16-17), clustered exposure (Eq. 18-19):
    E^clust = sum_inter phi~ f~ d~ + sum_intra phi~_intra f~ d~    (total; the paper divides by
    |A~| + k, which only rescales it: we compare totals with E^orig = sum_a E_a)
    Phi^clust = sum_inter phi~ + sum_intra phi~_intra   (intra flow only for clusters with an intra
    object, so that G~ = G when every cluster is a single node)
"""
import numpy as np

from . import config as C
from .network import UGraph, traffic_factor


def _wmean(values, weights, groups, n):
    """Weighted mean of ``values`` per group (plain mean where the weights sum to 0)."""
    sw = np.bincount(groups, weights, n)
    num = np.bincount(groups, values * weights, n)
    cnt = np.bincount(groups, minlength=n)
    plain = np.bincount(groups, values, n) / np.maximum(cnt, 1)
    return np.where(sw > 0, num / np.where(sw > 0, sw, 1), plain)


class ClusteredNetwork:
    def __init__(self, net, labels, base_flow=None):
        self.net = net
        _, labels = np.unique(np.asarray(labels), return_inverse=True)
        self.labels = labels
        self.k = int(labels.max()) + 1
        phi = net.base.flow if base_flow is None else base_flow
        lu, lv = labels[net.u], labels[net.v]
        inner = lu == lv

        # ---- objects ----------------------------------------------------------------------------
        intra_clusters = np.unique(lu[inner])
        self.n_intra = len(intra_clusters)
        self.intra_of_cluster = np.full(self.k, -1)
        self.intra_of_cluster[intra_clusters] = np.arange(self.n_intra)
        lo, hi = np.minimum(lu, lv), np.maximum(lu, lv)
        pair_key = lo.astype(np.int64) * self.k + hi
        pairs, pair_id = np.unique(pair_key[~inner], return_inverse=True)
        self.n_inter = len(pairs)
        self.inter_r, self.inter_s = pairs // self.k, pairs % self.k
        self.n_obj = self.n_intra + self.n_inter
        arc_obj = np.empty(net.E, dtype=int)
        arc_obj[inner] = self.intra_of_cluster[lu[inner]]
        arc_obj[~inner] = self.n_intra + pair_id
        self.arc_obj = arc_obj
        self.obj_is_intra = np.arange(self.n_obj) < self.n_intra
        self.obj_cluster = np.r_[intra_clusters, np.full(self.n_inter, -1)]

        # ---- object attributes (flow-weighted, Eq. 9) -------------------------------------------
        self.d = _wmean(net.length, phi, arc_obj, self.n_obj)
        self.s = _wmean(net.s, phi, arc_obj, self.n_obj)
        self.w0 = np.full(self.n_obj, 6)
        np.minimum.at(self.w0, arc_obj, net.dos0)
        self.w0max = np.zeros(self.n_obj, int)
        np.maximum.at(self.w0max, arc_obj, net.dos0)
        self.n_arcs = np.bincount(arc_obj, minlength=self.n_obj)

        # cost Pi_g(L) of raising every arc of g to at least L (Eq. 25)
        self.cost = np.full((self.n_obj, 7), np.inf)
        ar = np.arange(net.E)
        for L in range(1, 7):
            c = net.cost[ar, np.maximum(net.dos0, L)]
            self.cost[:, L] = np.bincount(arc_obj, c, self.n_obj)
        self.cost[np.arange(7)[None, :] < self.w0[:, None]] = np.inf
        self.cost[:, 0] = np.inf

        # ---- cluster graph and clustered demand (Eq. 15) ----------------------------------------
        self.centroid = np.column_stack([np.bincount(labels, net.xy[:, 0], self.k),
                                         np.bincount(labels, net.xy[:, 1], self.k)]) / np.bincount(labels)[:, None]
        q_o, q_d = labels[net.od_o], labels[net.od_d]
        key = q_o.astype(np.int64) * self.k + q_d
        uk, inv = np.unique(key, return_inverse=True)
        self.q_o, self.q_d = uk // self.k, uk % self.k
        self.q = np.bincount(inv, net.q)
        self.ug = UGraph(self.k, self.inter_r, self.inter_s, self.q_o, self.q_d, self.q)
        self.base = self.evaluate(self.w0)

    # ------------------------------------------------------------------------------------------
    def b(self, levels):
        """Clustered cyclist distance of every object (Eqs. 11-13)."""
        return self.d * self.s * traffic_factor(levels) ** C.RHO

    def evaluate(self, levels):
        """Clustered AON + exposure for the object DoS ``levels`` (n_obj)."""
        levels = np.asarray(levels, int)
        b = self.b(levels)
        b_intra = np.zeros(self.k)
        b_intra[self.obj_cluster[: self.n_intra]] = b[: self.n_intra]
        w = b[self.n_intra:] + 0.5 * (b_intra[self.inter_r] + b_intra[self.inter_s])
        flow_inter, visits, od_cost, reached = self.ug.aon(np.maximum(w, 1e-6))
        phi = np.empty(self.n_obj)
        phi[self.n_intra:] = flow_inter
        phi[: self.n_intra] = visits[self.obj_cluster[: self.n_intra]]
        e_obj = phi * traffic_factor(levels) * self.d
        return dict(phi=phi, exposure_obj=e_obj, exposure=float(e_obj.sum()), total_flow=float(phi.sum()),
                    exposure_inter=float(e_obj[self.n_intra:].sum()), exposure_intra=float(e_obj[: self.n_intra].sum()),
                    flow_inter=float(phi[self.n_intra:].sum()), flow_intra=float(phi[: self.n_intra].sum()),
                    reached=reached, od_cost=od_cost)

    # ------------------------------------------------------------------------------------------
    def lift(self, levels):
        """w_a^lift = max(w_a^0, w~_g) for every arc a of object g (Eq. 32)."""
        return np.maximum(self.net.dos0, np.asarray(levels, int)[self.arc_obj])

    def levels_cost(self, levels):
        return float(self.cost[np.arange(self.n_obj), np.asarray(levels, int)].sum())

    def object_dos_of_arcs(self, levels=None):
        """DoS of the object each original arc belongs to (for maps of the clustered network)."""
        lv = self.w0 if levels is None else np.asarray(levels, int)
        return lv[self.arc_obj]

    def node_mass(self, result=None):
        """Exposure mass of every cluster (Eq. 31): incident inter exposure + intra exposure."""
        r = self.base if result is None else result
        e = r["exposure_obj"]
        m = np.bincount(self.inter_r, e[self.n_intra:], self.k) + np.bincount(self.inter_s, e[self.n_intra:], self.k)
        m[self.obj_cluster[: self.n_intra]] += e[: self.n_intra]
        return m

    def diameters(self):
        """Bounding-box diagonal of every cluster (m)."""
        xy, lab = self.net.xy, self.labels
        out = np.zeros(self.k)
        mins = np.full((self.k, 2), np.inf)
        maxs = np.full((self.k, 2), -np.inf)
        np.minimum.at(mins, lab, xy)
        np.maximum.at(maxs, lab, xy)
        out[:] = np.hypot(*(maxs - mins).T)
        return out

"""
Fast utilities on shortest-path trees (one tree per origin), shared by the bicycle and the
car layer.

scipy's Dijkstra returns, for S origins, the distance and predecessor matrices (S x N).
Two operations are needed on these trees:

* ``load_trees``  push a demand matrix D (S x N, trips from origin s to node t) from the
                  leaves to the root: the load of the arc (pred[t] -> t) is the demand of
                  every node in the subtree of t.  This is the all-or-nothing assignment.
* ``path_sums``   sum an arc attribute along every tree path (path exposure, path length,
                  path travel time ...).

Both are fully vectorised over the S origins.  The nodes are processed by *hop depth*
(number of arcs from the root) instead of one node at a time, so the Python loop runs
~150 times (tree height) instead of N = 3,000 - 5,000 times; the depth itself and the path
sums are obtained by pointer jumping (Wyllie's list ranking) in log2(height) ~ 8 steps.
This makes one all-or-nothing assignment of the car network (276 origins, 4,720 nodes)
take ~0.1 s, so that a Frank-Wolfe user equilibrium takes seconds instead of minutes.
"""
import numpy as np


def _flat_parents(pred):
    """Flat index of the parent of every (origin, node) in an (S x (N+1)) array whose last
    column is a sentinel node (parent of the roots, its own parent, value 0)."""
    S, N = pred.shape
    itype = np.int32 if S * (N + 1) < 2 ** 31 else np.int64
    base = (np.arange(S, dtype=itype) * (N + 1))[:, None]
    par = np.empty((S, N + 1), dtype=itype)
    par[:, :N] = np.where(pred >= 0, pred, N)
    par[:, N] = N
    return (par + base).ravel()


def _jump(values, pred, dtype=float):
    """Pointer jumping: returns the sum of ``values`` over each node and all its ancestors.

    values: (S x N), 0 at the roots and unreachable nodes; pred: (S x N), -1 if none.
    Synchronous update acc <- acc + acc[parent], parent <- parent[parent] (Wyllie).
    """
    S, N = pred.shape
    par = _flat_parents(pred)
    acc = np.zeros((S, N + 1), dtype=dtype)
    acc[:, :N] = values
    acc = acc.ravel()
    sentinel = np.repeat(par[N::N + 1], N + 1)              # sentinel of each tree
    while True:
        acc = acc + np.take(acc, par)
        par = np.take(par, par)
        if np.array_equal(par, sentinel):                   # every node reached the root
            break
    return acc.reshape(S, N + 1)[:, :N]


def depth(pred):
    """Hop depth of every node in every tree (0 for roots / unreachable nodes)."""
    return _jump((pred >= 0), pred, dtype=np.int32)


def path_sums(pred, arc_index, arc_value):
    """Sum of ``arc_value`` along the tree path root -> t, for every (origin, node).

    arc_index: (N x N) int array, id of the arc u -> v (-1 if none); arc_value: per arc.
    """
    S, N = pred.shape
    cols = np.broadcast_to(np.arange(N), (S, N))
    ok = pred >= 0
    own = np.zeros((S, N))
    own[ok] = arc_value[arc_index[pred[ok], cols[ok]]]   # value of the arc entering t
    return _jump(own, pred)


class TreeOrder:
    """Pre-computed processing order of a set of trees (re-used for several loadings)."""

    def __init__(self, pred, arc_index):
        S, N = pred.shape
        self.pred = pred
        dep = depth(pred)
        s, t = np.nonzero(pred >= 0)
        d = dep[s, t].astype(np.int16)
        order = np.argsort(-d, kind="stable")             # deepest nodes first (radix sort)
        self.s, self.t, d = s[order], t[order], d[order]
        self.p = pred[self.s, self.t]
        self.arc = arc_index[self.p, self.t]
        # boundaries of the depth levels in the sorted arrays
        cuts = np.flatnonzero(np.diff(d)) + 1
        self.levels = np.split(np.arange(len(d)), cuts)
        self.shape = (S, N)

    def load(self, D, n_arcs):
        """All-or-nothing loading of the demand D (S x N).  Returns arc loads (n_arcs)."""
        L = np.array(D, dtype=float, copy=True)
        flat = L.reshape(-1)
        N = self.shape[1]
        for lev in self.levels:                            # deepest level first
            s, t, p = self.s[lev], self.t[lev], self.p[lev]
            np.add.at(flat, s * N + p, flat[s * N + t])     # child load -> parent
        # load of arc (pred[t] -> t) = final load of node t
        return np.bincount(self.arc, weights=flat[self.s * N + self.t], minlength=n_arcs)


def load_trees(pred, arc_index, D, n_arcs):
    """One-shot all-or-nothing loading (see TreeOrder.load)."""
    return TreeOrder(pred, arc_index).load(D, n_arcs)

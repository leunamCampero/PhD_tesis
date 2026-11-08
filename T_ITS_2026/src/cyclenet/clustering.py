"""
Clustering of a cycling network into k clusters (Section II-F, Algorithm 1).

* ``safety_adjacency``   A_ij = w_ij exp(-d_ij / gamma) (phi_ij / max phi)^alpha          Eq. (19)
* ``laplacian``          L = I - D^-1/2 A D^-1/2                                         Eq. (20)
* ``kmeans_baseline``    the "K-mean" of the paper / notebooks: k-means on the spectral embedding
                         of L concatenated with the (weighted) geographic coordinates.
* ``cc_algorithm``       CC-Algorithm (Algorithm 1).  CHANGE: the assignment step is a *constrained
                         greedy normalised-cut contraction* on the same Laplacian: starting from
                         singletons, the two adjacent clusters with the largest normalised
                         association  A_rs / sqrt(vol_r vol_s)  (the quantity spectral clustering
                         maximises) minus  lambda_1 x (exposure distortion of the merge)  are merged,
                         as long as the merged cluster satisfies the hard constraints of Algorithm 1:
                           - DoS homogeneity:  max DoS - min DoS <= eps_max inside every clustered
                             object it creates (the intra object and every inter object), and
                           - compactness: bounding-box diagonal <= delta_max,
                         until k clusters remain.  If the contraction gets stuck, eps_max and delta_max
                         are relaxed (x alpha_relax); the exposure / flow tolerances tau_E, tau_Phi are
                         checked on the clustered network of Section II-E and, if violated, lambda_1 is
                         increased and the tolerances relaxed (outer loop of Algorithm 1).  Fallback:
                         k-means.  Clusters are connected by construction (merges follow arcs), which
                         unconstrained k-means does not guarantee.
"""
import heapq
import time

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from . import config as C
from .clustered import ClusteredNetwork
from .network import traffic_factor


# ==============================================================================================
# Adjacency, Laplacian, spectral embedding
# ==============================================================================================
def safety_adjacency(net, flow=None, gamma=C.GAMMA_ADJ_M, alpha=C.ALPHA_FLOW):
    """Symmetric weight of every arc (Eq. 19) and the sparse matrix A."""
    phi = net.base.flow if flow is None else flow
    fl = (phi / phi.max()) ** alpha if (alpha > 0 and phi.max() > 0) else np.ones(net.E)
    w = net.dos0 * np.exp(-net.length / gamma) * fl
    A = csr_matrix((np.r_[w, w], (np.r_[net.u, net.v], np.r_[net.v, net.u])), shape=(net.N, net.N))
    return w, A


def laplacian(A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1)), 0.0)
    Dm = diags(inv)
    return diags(np.ones(A.shape[0])) - Dm @ A @ Dm, deg


def spectral_embedding(A, m=C.N_EIGEN, seed=C.SEED):
    """First m non-trivial eigenvectors of L (largest of D^-1/2 A D^-1/2, same eigenvectors)."""
    L, deg = laplacian(A)
    M = diags(np.ones(A.shape[0])) - L
    rng = np.random.default_rng(seed)
    vals, vecs = eigsh(M, k=m + 1, which="LA", v0=rng.standard_normal(A.shape[0]), tol=1e-6)
    order = np.argsort(-vals)
    return vecs[:, order[1: m + 1]], 1.0 - vals[order]


# ==============================================================================================
# Baseline: (spectral + geographic) k-means of the notebooks
# ==============================================================================================
def kmeans_features(net, embedding=None):
    if embedding is None:
        _, A = safety_adjacency(net)
        embedding, _ = spectral_embedding(A)
    z = lambda x: (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)  # noqa: E731
    return np.hstack([z(embedding), C.KMEANS_GEO_WEIGHT * z(net.xy)])


def kmeans_baseline(net, k, features=None, seed=C.SEED):
    from sklearn.cluster import KMeans
    X = kmeans_features(net) if features is None else features
    km = KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, max_iter=C.KMEANS_MAX_ITER, random_state=seed)
    return km.fit_predict(X)


# ==============================================================================================
# CC-Algorithm
# ==============================================================================================
class _Contraction:
    """Greedy constrained contraction (state kept between relaxations)."""

    def __init__(self, net, lam, flow=None, node_flow=None):
        self.net = net
        w, A = safety_adjacency(net)
        phi = net.base.flow if flow is None else flow
        T = net.base.node_flow if node_flow is None else node_flow
        deg = np.asarray(A.sum(axis=1)).ravel()
        n = net.N
        self.parent = np.arange(n)
        self.size = np.ones(n, int)
        self.vol = deg.astype(float)
        self.T = T.astype(float)
        self.bb = np.column_stack([net.xy, net.xy])            # minx, miny, maxx, maxy
        self.imin = np.full(n, 99)
        self.imax = np.full(n, -1)                             # no inner arc yet
        self.ver = np.zeros(n, int)
        self.alive = np.ones(n, bool)
        self.n_clusters = n
        self.lam = lam
        f = traffic_factor(net.dos0)
        # neighbours: nb[r][s] = [A_sum, wmin, wmax, phi_sum, len_sum, n_arcs, f*len*? ]
        self.nb = [dict() for _ in range(n)]
        for a in range(net.E):
            r, s = net.u[a], net.v[a]
            for x, y in ((r, s), (s, r)):
                e = self.nb[x].get(y)
                if e is None:
                    self.nb[x][y] = [w[a], net.dos0[a], net.dos0[a], phi[a], net.length[a], 1]
                else:
                    e[0] += w[a]; e[1] = min(e[1], net.dos0[a]); e[2] = max(e[2], net.dos0[a])  # noqa: E702
                    e[3] += phi[a]; e[4] += net.length[a]; e[5] += 1                             # noqa: E702
        # scales of the two terms of the merge score
        assoc = w / np.sqrt(deg[net.u] * deg[net.v])
        dist = np.maximum(T[net.u] + T[net.v] - 2 * phi, 0) * f * net.length
        self.s_assoc = np.median(assoc[assoc > 0]) if (assoc > 0).any() else 1.0
        self.s_dist = np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
        self.ftab = traffic_factor(np.arange(1, 7))
        self.heap = []

    # ---- union-find -------------------------------------------------------------------------
    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    # ---- merge evaluation -----------------------------------------------------------------------
    def score(self, r, s):
        e = self.nb[r][s]
        assoc = e[0] / np.sqrt(self.vol[r] * self.vol[s])
        dist = max(self.T[r] + self.T[s] - 2 * e[3], 0.0) * self.ftab[e[1] - 1] * e[4] / e[5]
        return assoc / self.s_assoc - self.lam * dist / self.s_dist

    def feasible(self, r, s, eps, delta):
        e = self.nb[r][s]
        lo = min(self.imin[r], self.imin[s], e[1])
        hi = max(self.imax[r], self.imax[s], e[2])
        if hi - lo > eps:
            return False
        bb = np.r_[np.minimum(self.bb[r, :2], self.bb[s, :2]), np.maximum(self.bb[r, 2:], self.bb[s, 2:])]
        if np.hypot(bb[2] - bb[0], bb[3] - bb[1]) > delta:
            return False
        small, big = (r, s) if len(self.nb[r]) < len(self.nb[s]) else (s, r)
        nb_big = self.nb[big]
        for t, et in self.nb[small].items():
            if t == big:
                continue
            ft = nb_big.get(t)
            if ft is not None and max(et[2], ft[2]) - min(et[1], ft[1]) > eps:
                return False
        return True

    def push(self, r, s):
        heapq.heappush(self.heap, (-self.score(r, s), self.ver[r], self.ver[s], r, s))

    def fill_heap(self):
        self.heap = []
        for r in np.flatnonzero(self.alive):
            for s in self.nb[r]:
                if r < s:
                    self.push(r, s)

    def merge(self, r, s):
        """Merge s into r."""
        e = self.nb[r].pop(s)
        self.nb[s].pop(r)
        self.parent[s] = r
        self.alive[s] = False
        self.size[r] += self.size[s]
        self.vol[r] += self.vol[s]
        self.T[r] += self.T[s] - e[3]      # through-demand of the merged cluster (shared arc counted once)
        self.bb[r, :2] = np.minimum(self.bb[r, :2], self.bb[s, :2])
        self.bb[r, 2:] = np.maximum(self.bb[r, 2:], self.bb[s, 2:])
        self.imin[r] = min(self.imin[r], self.imin[s], e[1])
        self.imax[r] = max(self.imax[r], self.imax[s], e[2])
        for t, et in self.nb[s].items():
            self.nb[t].pop(s)
            er = self.nb[r].get(t)
            if er is None:
                self.nb[r][t] = et
                self.nb[t][r] = list(et)
            else:
                er[0] += et[0]; er[1] = min(er[1], et[1]); er[2] = max(er[2], et[2])  # noqa: E702
                er[3] += et[3]; er[4] += et[4]; er[5] += et[5]                           # noqa: E702
                self.nb[t][r] = list(er)
        self.nb[s] = {}
        self.ver[r] += 1
        self.ver[s] += 1
        self.n_clusters -= 1
        for t in self.nb[r]:
            self.push(r, t)

    def run(self, k, eps, delta):
        """Merge until k clusters remain or no feasible merge exists.  Returns True if k reached."""
        self.fill_heap()
        while self.n_clusters > k and self.heap:
            _, vr, vs, r, s = heapq.heappop(self.heap)
            if not (self.alive[r] and self.alive[s]) or vr != self.ver[r] or vs != self.ver[s]:
                continue
            if s not in self.nb[r] or not self.feasible(r, s, eps, delta):
                continue
            self.merge(r, s)
        return self.n_clusters <= k

    def labels(self):
        roots = np.array([self.find(i) for i in range(self.net.N)])
        return np.unique(roots, return_inverse=True)[1]


def contract(net, k, eps=C.CC_EPS_MAX, delta=C.CC_DELTA_MAX_M, lam=C.CC_LAMBDA_E0, max_relax=12):
    """Constrained contraction to k clusters; relaxes eps / delta when stuck."""
    st = _Contraction(net, lam)
    relax = 0
    while not st.run(k, eps, delta):
        relax += 1
        if relax > max_relax:
            return None, dict(eps=eps, delta=delta, relaxations=relax)
        if eps < 5 and (relax % 2 == 1 or delta == np.inf):
            eps += 1
        else:
            delta = delta * C.CC_ALPHA_RELAX if delta < 1e5 else np.inf
    return st.labels(), dict(eps=eps, delta=delta, relaxations=relax)


def cc_algorithm(net, k, verbose=False):
    """CC-Algorithm (Algorithm 1).  Returns (labels, info).

    Outer loop t = 1..T_max: contraction with the penalty / compactness pair (lambda_1, delta_max) of
    CC_SCHEDULE[t]; the clustered network is built and its exposure and flow compared with the
    original ones; accepted as soon as |dE| <= tau_E and |dPhi| <= tau_Phi, otherwise the tolerances
    are relaxed (x alpha_relax) and the next pair is tried.  The best clustering (smallest
    max(|dE| / tau_E0, |dPhi| / tau_Phi0)) is returned if no pair meets the tolerances."""
    t0 = time.time()
    E0, P0 = net.base.exposure, net.base.total_flow
    tau_e, tau_p = C.CC_TAU_E0, C.CC_TAU_PHI0
    best = None
    for it, (lam, delta) in enumerate(C.CC_SCHEDULE[: C.CC_T_MAX], 1):
        labels, info = contract(net, k, lam=lam, delta=delta)
        if labels is None:
            continue
        cn = ClusteredNetwork(net, labels)
        dE = abs(cn.base["exposure"] - E0) / E0
        dP = abs(cn.base["total_flow"] - P0) / P0
        viol = max(dE / C.CC_TAU_E0, dP / C.CC_TAU_PHI0)
        cand = dict(labels=labels, dE=dE, dP=dP, lam=lam, delta_max=delta, tau_E=tau_e, tau_Phi=tau_p,
                    iteration=it, violation=viol, **info)
        if best is None or viol < best["violation"]:
            best = cand
        if verbose:
            print(f"    CC it {it}: lambda {lam:.3g} eps {info['eps']} delta {info['delta']:.0f} "
                  f"dE {100 * dE:.2f} % dPhi {100 * dP:.2f} %")
        if dE <= tau_e and dP <= tau_p:
            break
        tau_e *= C.CC_ALPHA_RELAX
        tau_p *= C.CC_ALPHA_RELAX
    if best is None:                                            # fallback of Algorithm 1
        labels = kmeans_baseline(net, k)
        best = dict(labels=labels, fallback=True)
    best.setdefault("fallback", False)
    best["seconds"] = time.time() - t0
    labels = best.pop("labels")
    return labels, best

"""
Network comparison (Section IV): exposure masses, Gromov-Wasserstein distance, exposure kernels.

    mu^H(v_i)  = sum of the exposure of the arcs incident to v_i                       Eq. (34)
    mu^G~(C_r) = incident inter-cluster exposure + intra-cluster exposure              Eq. (36)
    D_ij       = |mu_i - mu_j| + gamma ||x_i - x_j||                                     Eqs. (38-39)
    GW         = sqrt( min_T sum (D^H_ij - D^G_rs)^2 T_ir T_js )                         Eq. (41)

CHANGES (README): (i) the two terms of D are each scaled to [0, 1] (the raw |mu_i - mu_j| ~ 1e-4
is negligible against metres) and coordinates are standardised per city; (ii) the GW problem is
solved on the GW_TOP_M nodes / clusters with the largest exposure mass (as in the notebooks: a
6 773 x 6 773 GW problem needs 6 773^3 operations per iteration); (iii) entropic GW (Peyre et al.
2016) with log-domain Sinkhorn, no external OT library.

Kernel comparison (ITSC 2026): exposure-weighted adjacency kernel K (K_ij = exposure of the arc /
super-arc i-j, intra exposure on the diagonal).  The kernel distance is the mean absolute difference of
the degree-sorted empirical graphons (nodes sorted by strength, KERNEL_BLOCKS x KERNEL_BLOCKS block
averages of K / sum K) - label invariant and size independent.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from . import config as C


# ==============================================================================================
# Exposure measures and dissimilarities
# ==============================================================================================
def original_measure(net, result=None):
    r = net.base if result is None else result
    return net.incident_sum(r.exposure_arc), net.xy


def clustered_measure(cn, result=None):
    return cn.node_mass(result), cn.centroid


def top_subset(mass, coords, m):
    idx = np.argsort(-mass)[: min(m, len(mass))]
    idx = idx[mass[idx] > 0] if (mass[idx] > 0).sum() >= 2 else idx
    p = mass[idx] / mass[idx].sum()
    return idx, p, coords[idx]


def dissimilarity(p, coords, gamma=C.GW_GAMMA):
    z = (coords - coords.mean(0)) / coords.std(0).mean()
    dm = np.abs(p[:, None] - p[None, :])
    dx = np.hypot(z[:, None, 0] - z[None, :, 0], z[:, None, 1] - z[None, :, 1])
    return dm / max(dm.max(), 1e-300) + gamma * dx / max(dx.max(), 1e-300)


# ==============================================================================================
# Entropic Gromov-Wasserstein (square loss)
# ==============================================================================================
def _sinkhorn_log(p, q, M, eps, f=None, g=None, iters=60, tol=1e-7):
    lp, lq = np.log(p), np.log(q)
    f = np.zeros(len(p)) if f is None else f
    g = np.zeros(len(q)) if g is None else g
    for _ in range(iters):
        f_old = f
        f = eps * lp - eps * logsumexp((g[None, :] - M) / eps, axis=1)
        g = eps * lq - eps * logsumexp((f[:, None] - M) / eps, axis=0)
        if np.abs(f - f_old).max() < tol:
            break
    T = np.exp((f[:, None] + g[None, :] - M) / eps)
    return T, f, g


def gromov_wasserstein(D1, D2, p, q, eps=C.GW_EPSILON, outer=C.GW_OUTER_ITERS, tol=1e-5):
    """Returns (coupling T, GW distance = sqrt(sum (D1_ij - D2_rs)^2 T_ir T_js))."""
    constC = (D1 ** 2 @ p)[:, None] + (D2 ** 2 @ q)[None, :]
    T = np.outer(p, q)
    f = g = None
    for _ in range(outer):
        G = constC - 2.0 * D1 @ T @ D2.T
        T_new, f, g = _sinkhorn_log(p, q, G, eps * max(G.max() - G.min(), 1e-12), f, g)
        delta = np.abs(T_new - T).sum()
        T = T_new
        if delta < tol:
            break
    loss = float(((constC - 2.0 * D1 @ T @ D2.T) * T).sum())
    return T, np.sqrt(max(loss, 0.0))


def gw_between(mass_h, xy_h, mass_g, xy_g, m=C.GW_TOP_M):
    ih, ph, ch = top_subset(mass_h, xy_h, m)
    ig, pg, cg = top_subset(mass_g, xy_g, m)
    T, gw = gromov_wasserstein(dissimilarity(ph, ch), dissimilarity(pg, cg), ph, pg)
    return dict(T=T, gw=gw, idx_h=ih, idx_g=ig, p_h=ph, p_g=pg)


# ==============================================================================================
# Exposure kernels (ITSC 2026)
# ==============================================================================================
def kernel_original(net, result=None):
    r = net.base if result is None else result
    e = r.exposure_arc
    return csr_matrix((np.r_[e, e], (np.r_[net.u, net.v], np.r_[net.v, net.u])), shape=(net.N, net.N))


def kernel_clustered(cn, result=None):
    r = cn.base if result is None else result
    e = r["exposure_obj"]
    ei = e[cn.n_intra:]
    rows = np.r_[cn.inter_r, cn.inter_s, cn.obj_cluster[: cn.n_intra]]
    cols = np.r_[cn.inter_s, cn.inter_r, cn.obj_cluster[: cn.n_intra]]
    return csr_matrix((np.r_[ei, ei, e[: cn.n_intra]], (rows, cols)), shape=(cn.k, cn.k))


def strength(K):
    return np.asarray(K.sum(axis=1)).ravel()


def graphon(K, blocks=C.KERNEL_BLOCKS):
    """Degree-sorted empirical graphon: B x B block sums of K / sum(K), times B^2 (integrates to 1)."""
    s = strength(K)
    order = np.argsort(-s, kind="stable")
    n = K.shape[0]
    block_of = np.empty(n, int)
    block_of[order] = np.minimum((np.arange(n) * blocks) // n, blocks - 1)
    Kc = K.tocoo()
    W = np.zeros((blocks, blocks))
    np.add.at(W, (block_of[Kc.row], block_of[Kc.col]), Kc.data)
    return W / max(W.sum(), 1e-300) * blocks ** 2


def kernel_distance(K1, K2, blocks=C.KERNEL_BLOCKS):
    return float(np.abs(graphon(K1, blocks) - graphon(K2, blocks)).mean())


def top_kernel(K, lonlat, m=C.TOP_KERNEL_NODES):
    """Sub-kernel of the m strongest nodes, ordered by longitude (ITSC Fig. 6), log-scaled to [0, 1]."""
    s = strength(K)
    top = np.argsort(-s)[:m]
    top = top[np.lexsort((lonlat[top, 1], lonlat[top, 0]))]
    sub = K[top][:, top].toarray()
    pos = sub[sub > 0]
    if len(pos) == 0:
        return sub, top
    sub = np.log1p(sub / np.median(pos))
    return sub / sub.max(), top

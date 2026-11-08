"""
GW-based transfer of an optimised strategy from a source network H to a target network G
(Section V-B, Problem 3, Algorithm 4).

    u_i^H        = sum_{a ~ i} E_a^H (w*_a - w0_a) / (sum_{a ~ i} E_a^H + eps)            Eq. (42)
    u_{C_r}      = sum_i t*_ir u_i^H / (sum_i t*_ir + eps)                                 Eq. (43)
    S_g          = u_{C_r} (intra object of C_r),  (u_{C_r} + u_{C_s}) / 2  (inter object)  Eq. (44)
    Problem 3    max sum_g S_g w~_g   s.t.  sum_g Pi_g(w~_g) <= B^G                         Eqs. (45-46)
    lifting      w_a = max(w0_a, w~_g)                                                      Eq. (32)

The GW plan is computed on the top-M subsets (compare.gw_between); clusters outside the target subset
receive the inverse-distance-weighted score of their TRANSFER_NEIGHBOURS nearest subset clusters.
Problem 3 is a multiple-choice knapsack; it is solved greedily by the best ratio
S_g (L - w0_g) / Pi_g(L) over the levels L of each object (Algorithm 4, "rank and select").

``flow_aware=True`` (PROPOSAL, not in the paper): S_g is multiplied by the share of the target's own
cyclist flow on g (one AON on the target, no optimisation), so the budget is not spent on objects
that no cyclist uses.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C


def source_intensity(net_h, w_star):
    e0 = net_h.base.exposure_arc
    inc = np.clip(np.asarray(w_star) - net_h.dos0, 0, None)
    return net_h.incident_sum(e0 * inc) / (net_h.incident_sum(e0) + C.TRANSFER_EPS)


def cluster_scores(plan, u_src_sub, cn, idx_target_sub):
    """Eq. (43) on the GW subset, IDW propagation to the other clusters."""
    T = plan
    u_sub = (T.T @ u_src_sub) / (T.sum(axis=0) + C.TRANSFER_EPS)
    u = np.zeros(cn.k)
    u[idx_target_sub] = u_sub
    others = np.setdiff1d(np.arange(cn.k), idx_target_sub)
    if len(others):
        tree = cKDTree(cn.centroid[idx_target_sub])
        kk = min(C.TRANSFER_NEIGHBOURS, len(idx_target_sub))
        d, j = tree.query(cn.centroid[others], k=kk)
        d, j = d.reshape(len(others), -1), j.reshape(len(others), -1)
        w = 1.0 / np.maximum(d, 1.0)
        u[others] = (w * u_sub[j]).sum(1) / w.sum(1)
    return u


def object_scores(cn, u):
    S = np.empty(cn.n_obj)
    S[: cn.n_intra] = u[cn.obj_cluster[: cn.n_intra]]
    S[cn.n_intra:] = 0.5 * (u[cn.inter_r] + u[cn.inter_s])
    return S


def solve_problem3(cn, S, budget):
    """Greedy rank-and-select for max sum S_g w_g s.t. budget (Algorithm 4)."""
    levels = cn.w0.copy()
    best_ratio = np.zeros(cn.n_obj)
    best_L = cn.w0.copy()
    for L in range(2, 7):
        c = cn.cost[:, L]
        ok = (L > cn.w0) & np.isfinite(c) & (c > 0)
        ratio = np.where(ok, S * (L - cn.w0) / np.where(ok, c, 1), 0.0)
        better = ratio > best_ratio
        best_ratio[better], best_L[better] = ratio[better], L
    spent = 0.0
    for g in np.argsort(-best_ratio):
        if best_ratio[g] <= 0:
            break
        c = cn.cost[g, best_L[g]]
        if spent + c <= budget:
            spent += c
            levels[g] = best_L[g]
    return levels, spent


def transfer(net_h, w_star, gw, cn_target, budget, flow_aware=False):
    """Algorithm 4.  Returns (clustered levels, lifted arc DoS, spent, scores)."""
    u_h = source_intensity(net_h, w_star)
    u = cluster_scores(gw["T"], u_h[gw["idx_h"]], cn_target, gw["idx_g"])
    S = object_scores(cn_target, u)
    if flow_aware:
        phi = cn_target.base["phi"]
        S = S * phi / max(phi.max(), 1e-300)
    levels, spent = solve_problem3(cn_target, S, budget)
    return levels, cn_target.lift(levels), spent, S

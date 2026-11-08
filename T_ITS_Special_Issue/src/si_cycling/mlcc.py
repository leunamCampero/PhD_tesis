"""
MLCC - Maximize the Largest Connected Component (Algorithm 2 of the paper, from [ITSC 2024]).

Idea: grow ONE big, connected, fully separated (DoS 6) network.

    1. C  <- largest connected component (CC) made of DoS-6 arcs
             (if there is no DoS-6 arc: upgrade the largest CC of the best DoS present to 6)
    2. the other DoS-6 CCs, then the CCs of DoS 5, 4, 3, 2, 1 (largest first) are attached to C:
         - find the shortest "DoS-path" from C to the component, arc weight b_a * (7 - w_a)
           (b_a = cyclist distance of the arc, so safe arcs are cheap to use)
         - upgrade every arc on that path, then every arc of the component, to DoS 6
         - C grows; repeat
    3. stop when the budget cannot pay for any further upgrade or every arc is DoS 6.

When an arc cannot be raised to 6 with the remaining money it is raised to the best level
that is still affordable.  MLCC does not look at cyclist flows.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .costs import COST_MATRIX


class _Upgrader:
    """Keeps the current DoS vector and the remaining budget."""

    def __init__(self, net, budget):
        self.net = net
        self.dos = net.dos0.copy()
        self.left = float(budget)
        self.log = []                       # (edge, old DoS, new DoS, cost, budget left)

    def raise_edge(self, e):
        """Raise arc e to the highest affordable DoS (6 if possible). True if it changed."""
        now = self.dos[e]
        for target in range(6, now, -1):
            c = self.net.length[e] * (COST_MATRIX[self.net.dos0[e], target]
                                       - COST_MATRIX[self.net.dos0[e], now])
            if c <= self.left:
                self.dos[e] = target
                self.left -= c
                self.log.append((int(e), int(now), target, c, self.left))
                return True
        return False

    def raise_all(self, edges):
        return sum(self.raise_edge(e) for e in edges) > 0


def _components(net, mask):
    """Connected components of the sub-network made of the arcs in ``mask``.

    Returns a list of (node set, edge ids), largest (most nodes) first; ties by smallest
    edge id so that the algorithm is deterministic.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    A = csr_matrix((np.ones(idx.size), (net.u[idx], net.v[idx])), shape=(net.N, net.N))
    _, label = connected_components(A, directed=False)
    groups = {}
    for e in idx:
        groups.setdefault(label[net.u[e]], []).append(e)
    comps = [(set(net.u[es]) | set(net.v[es]), es) for es in groups.values()]
    comps.sort(key=lambda c: (-len(c[0]), min(c[1])))
    return comps


def _dos_path(net, dos, sources, targets):
    """Edges of the shortest DoS-path from any node of ``sources`` to any node of ``targets``.

    Arc weight = b_a * (7 - w_a): cyclist distance, penalised for low DoS.
    """
    fwd, bwd = net.arc_costs(dos)
    pen = 7 - dos
    W = csr_matrix((np.concatenate([fwd * pen, bwd * pen]), (net.tail, net.head)), shape=(net.N, net.N))
    src = np.fromiter(sources, dtype=int)
    dist, pred, _ = dijkstra(W, directed=True, indices=src, min_only=True, return_predecessors=True)
    tgt = np.fromiter(targets, dtype=int)
    t = tgt[np.argmin(dist[tgt])]
    if np.isinf(dist[t]):
        return []
    path = []
    while pred[t] >= 0:
        path.append(net.arc_index[pred[t], t] % net.E)
        t = pred[t]
    return path[::-1]


def run_mlcc(net, budget):
    """Run MLCC with ``budget`` (EUR). Returns (dos vector, upgrade log)."""
    up = _Upgrader(net, budget)

    # ---- step 1: the core C ------------------------------------------------------------
    six = _components(net, up.dos == 6)
    if not six:
        best = up.dos.max()
        up.raise_all(_components(net, up.dos == best)[0][1])
        six = _components(net, up.dos == 6)
    core_seed = next(iter(six[0][0]))          # any node of C identifies C later on

    def core_nodes():
        for nodes, _ in _components(net, up.dos == 6):
            if core_seed in nodes:
                return nodes
        return {core_seed}

    # ---- step 2: attach other DoS-6 CCs, then CCs of DoS 5, 4, ..., 1 ----------------
    for level in (6, 5, 4, 3, 2, 1):
        while True:
            core = core_nodes()
            cands = [c for c in _components(net, up.dos == level) if not (level == 6 and core_seed in c[0])]
            if not cands:
                break                                     # this level is done
            nodes, edges = cands[0]                       # largest component first
            changed = up.raise_all(_dos_path(net, up.dos, core, nodes))
            changed |= up.raise_all(edges)
            if not changed:                               # budget exhausted
                return up.dos, up.log
    return up.dos, up.log


# ========================================================================================
# Warm start of the GA (T-ITS special issue, Section III)
# ========================================================================================
def mlcc_priority(problem, budget_fraction):
    """Greedy score that reproduces the MLCC upgrade order inside the GA decoding.

    The arcs upgraded by MLCC with ``budget_fraction * B_max`` get the highest priorities, in
    the order MLCC upgraded them; the other arcs follow by exposure saved per euro.
    """
    net = problem.bike
    _, log = run_mlcc(net, budget_fraction * problem.B_max)
    cost6 = net.length * COST_MATRIX[net.dos0, 6]
    gain = problem.ev0["x_hat"] * net.exposure_arc(net.dos0)
    score = np.where(cost6 > 0, gain / np.maximum(cost6, 1e-9), -1.0)
    score = score / max(score.max(), 1e-12)                  # in (-inf, 1]
    first = {}
    for k, (e, *_rest) in enumerate(log):
        first.setdefault(e, k)
    for e, k in first.items():
        score[e] = 2.0 + (len(log) - k) / max(len(log), 1)  # above every other arc, MLCC order
    return score

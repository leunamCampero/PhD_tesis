"""MLCC - Maximize Largest Connected Component (Section IV-A of the paper).

De-duplicated port of ``upgrade_edges`` from the original cluster scripts
(super_compu_*.py / Linux_*.py, 2024), including its tie breaking and
early-stopping rules.

Implementation notes:
* shortest paths use ``weight = length * DoS``;
* every edge on a path or in a component is upgraded to the highest level it can
  afford (6, else 5, ... else 2); if not even level 2 is affordable for a DoS-1
  edge the algorithm stops.
"""
import networkx as nx

from .data import edge_list


def _try_upgrade(G, u, v, budget, cost, log):
    """Upgrade edge (u, v) to the best affordable DoS. Returns (budget, stop)."""
    e = G.edges[u, v]
    for target in range(6, 1, -1):
        if e["key"] < target:
            c = e["distance"] * cost[e["key"]][target]
            if c <= budget:
                log.append((u, v, e["key"], target, c, budget - c))
                e["key"] = target
                e["weight"] = e["distance"] * target
                budget -= c
            elif target == 2:
                return budget, True
    return budget, False


def _components_with_level(H, level, sort=True):
    sub = nx.Graph((u, v, d) for u, v, d in H.edges(data=True) if d["key"] == level)
    comps = list(nx.connected_components(sub))
    if sort:
        comps.sort(key=len, reverse=True)
    return comps


def _upgrade_all(G, H, budget, cost, log):
    for u, v in H.edges():
        budget, stop = _try_upgrade(G, u, v, budget, cost, log)
        if stop:
            return budget, True
    return budget, False


def _upgrade_path(G, path, budget, cost, log):
    for a, b in zip(path, path[1:]):
        budget, stop = _try_upgrade(G, a, b, budget, cost, log)
        if stop:
            return budget, True
    return budget, False


def _run(GG, budget, cost, log):
    comps = sorted(nx.connected_components(GG), key=len, reverse=True)
    for comp in comps:
        C = GG.subgraph(comp)  # view: upgrades on GG are visible here

        # Step 1.2 - connect the existing DoS-6 components, largest first
        ccs = _components_with_level(C, 6)
        prev = budget
        while len(ccs) > 1:
            path = nx.shortest_path(C, next(iter(ccs[0])), next(iter(ccs[1])), weight="weight")
            budget, stop = _upgrade_path(GG, path, budget, cost, log)
            if stop:
                return budget
            ccs = _components_with_level(C, 6)
            if budget == prev:
                return budget
            prev = budget

        keys = [d["key"] for _, _, d in C.edges(data=True)]
        top = max(keys)
        if all(k == top for k in keys):
            budget, stop = _upgrade_all(GG, C, budget, cost, log)
            if stop:
                return budget
            continue

        # Step 1.1 - upgrade (first) component of the best available level to 6
        seed = _components_with_level(C, top, sort=False)[0]
        prev = budget
        budget, stop = _upgrade_all(GG, GG.subgraph(seed), budget, cost, log)
        if stop:
            return budget
        if budget == prev and top != 6:
            return budget
        core = _components_with_level(C, 6, sort=False)
        if not core:
            return budget
        core = core[0]

        # Step 2 - attach the components of level 5, 4, ... 1 (largest first)
        for level in range(5, 0, -1):
            ccs = _components_with_level(C, level)
            prev = budget
            while ccs:
                path = nx.shortest_path(C, next(iter(core)), next(iter(ccs[0])), weight="weight")
                budget, stop = _upgrade_path(GG, path, budget, cost, log)
                if stop:
                    return budget
                budget, stop = _upgrade_all(GG, GG.subgraph(ccs[0]), budget, cost, log)
                if stop:
                    return budget
                core = _components_with_level(C, 6, sort=False)[0]
                ccs = _components_with_level(C, level)
                if budget == prev:
                    return budget
                prev = budget
    return budget


def mlcc(G, budget, cost):
    """Run MLCC on network ``G`` (edge attribute ``dos``) with ``budget``.

    Returns ``(levels, log)``: the new DoS of every edge (in ``edge_list(G)``
    order) and the list of upgrades ``(u, v, old, new, cost, remaining_budget)``.
    """
    GG = G.copy()
    for _, _, d in GG.edges(data=True):
        d["key"] = d["dos"]
        # stored weighted length (as in the original graphs), length*DoS once upgraded
        d["weight"] = d.get("wlen", {}).get("dos", d["distance"] * d["dos"])
    log = []
    _run(GG, budget, cost, log)
    levels = [GG[u][v]["key"] for u, v in edge_list(G)]
    return levels, log

"""Run (and cache) the algorithm comparisons of Sections V, VI and Appendix C.

Results are stored as JSON in ``results/<preset>/`` so that expensive runs
(GA / Bayesian on Grenoble take hours) are never repeated by accident: delete
the file (or pass ``force=True``) to recompute.
"""
import json
import time

from .data import RESULTS, load_network
from .model import Model, alphas, canonical
from .optimizers import brute_force, enumerate_all, run_bayesian, run_mlcc, run_sga

METHODS = ["Optimal", "MLCC", "SGA", "SGA-MLCC", "Bayesian"]


def _path(preset, name, target, tag=""):
    d = RESULTS / canonical(preset)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_{target}{tag}.json"


def compare_algorithms(name, preset="dos", target="safety", methods=METHODS,
                       generations=80, pop_size=50, bayes_calls=20, seed=0,
                       budgets=range(10), force=False, tag="", verbose=True):
    """Solve Problem 2 on network ``name`` for every budget alpha_i with every method.

    Returns ``{"alpha": [...], method: [result_i, ...]}`` where each result has
    ``y``, ``percent``, ``time_s``, ``levels`` (and ``history`` for the GAs).
    """
    path = _path(preset, name, target, tag)
    if path.exists() and not force:
        out = json.load(open(path))
        missing = [m for m in methods if m not in out]
        if not missing:
            if verbose:
                print(f"loaded cached results {path.relative_to(RESULTS.parent)}")
            return out
    else:
        out = {}
        missing = list(methods)

    G = load_network(name)
    model = Model(preset, target)
    out.update(alpha=[alphas()[i] for i in budgets], y0=model.predict(G),
               y_fully_safe=model.y_fully_safe, B_max=model.max_budget(G),
               settings=dict(generations=generations, pop_size=pop_size,
                             bayes_calls=bayes_calls, seed=seed))
    table = None
    for m in missing:
        out[m] = []
        for i in budgets:
            a = alphas()[i]
            t = time.time()
            if m == "Optimal":
                if table is None:
                    table = enumerate_all(model, G)
                r = brute_force(model, G, a, table)
            elif m == "MLCC":
                r = run_mlcc(model, G, a)
            elif m == "SGA":
                r = run_sga(model, G, a, init="random", generations=generations, pop_size=pop_size, seed=seed)
            elif m == "SGA-MLCC":
                mlcc_lv = out["MLCC"][list(budgets).index(i)]["levels"] if "MLCC" in out else None
                r = run_sga(model, G, a, init="mlcc", generations=generations, pop_size=pop_size,
                            seed=seed, mlcc_levels=mlcc_lv)
            elif m == "Bayesian":
                r = run_bayesian(model, G, a, n_calls=bayes_calls, seed=seed)
            out[m].append(r)
            if verbose:
                print(f"{name:12s} {target:7s} {m:9s} i={i}  {r['percent']:6.2f} %  ({time.time() - t:7.1f} s)",
                      flush=True)
        json.dump(out, open(path, "w"))  # save after every method
    return out


def percents(results, method):
    return [r["percent"] for r in results[method]]


def ga_histories(results, method):
    return {i: r["history"]["percent"] for i, r in enumerate(results[method])}


def format_time(seconds):
    """Human-readable time (s, m, hr)."""
    if seconds < 60:
        return f"{seconds:.3g} s" if seconds < 1 else f"{seconds:.2f} s"
    if seconds < 1800:
        return f"{seconds / 60:.2f} m"
    return f"{seconds / 3600:.2f} hr"


def time_table(results, methods):
    import pandas as pd
    rows = {m: [format_time(r["time_s"]) for r in results[m]] for m in methods if m in results}
    return pd.DataFrame(rows, index=[f"i = {i}" for i in range(len(next(iter(rows.values()))))]).T

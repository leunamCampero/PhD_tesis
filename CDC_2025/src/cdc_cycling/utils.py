"""Saving / loading results."""
import json
import os

import numpy as np
import pandas as pd

from . import config as C


def ensure_dirs():
    for d in (C.RESULTS, C.RESULTS / "solutions", C.FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))


def load_json(path):
    return json.loads(path.read_text())


def save_solution(dos, path):
    pd.DataFrame({"edge_id": np.arange(len(dos)), "dos": dos}).to_csv(path, index=False)


def load_solution(path):
    return pd.read_csv(path).dos.to_numpy().astype(int)


def solution_name(algo, beta, lam=None):
    """e.g. 'GA-S_b0.056_lam0.4'."""
    return f"{algo}_b{beta:g}" + ("" if lam is None else f"_lam{lam:g}")


def n_workers():
    """Parallel processes: all cores but one (each run is single-threaded)."""
    return max(1, (os.cpu_count() or 2) - 1)

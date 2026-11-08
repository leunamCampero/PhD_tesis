"""Saving / loading results."""
import json
import os

import numpy as np
import pandas as pd

from . import config as C

UE_FILE = C.RESULTS / "02_ue_original.npz"


def ensure_dirs():
    for d in (C.RESULTS, C.RESULTS / "solutions", C.FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def _conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=_conv))


def load_json(path):
    return json.loads(path.read_text())


def save_solution(dos, path):
    pd.DataFrame({"edge_id": np.arange(len(dos)), "dos": dos}).to_csv(path, index=False)


def load_solution(path):
    return pd.read_csv(path).dos.to_numpy().astype(int)


def save_ue(path, ue, multiplier):
    np.savez_compressed(path, x=ue["x"], t=ue["t"], od_time=ue["od_time"], multiplier=multiplier,
                        TTT=ue["TTT"], gap=ue["gap"])


def load_ue(path=None):
    path = UE_FILE if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 02_car_user_equilibrium.py first")
    z = np.load(path)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def solution_name(alpha, eps):
    """e.g. 'GA_a0.333_eps0.3'."""
    return f"GA_a{alpha:.3f}_eps{eps:g}"


def n_workers():
    """Parallel processes: all cores but one (each run is single-threaded)."""
    return max(1, (os.cpu_count() or 2) - 1)

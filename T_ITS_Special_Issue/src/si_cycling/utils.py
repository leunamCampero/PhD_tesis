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



def n_workers():
    """Parallel processes: all cores but one (each run is single-threaded)."""
    return max(1, (os.cpu_count() or 2) - 1)

def representative_gamma(stars, kind, lam=None, rule=None):
    """gamma* of a front (``stars`` = results/04_runs.json['gamma_star']), see config.GAMMA_STAR_RULE."""
    rule = rule or C.GAMMA_STAR_RULE
    if rule == "utopia":
        return stars["util" if kind == "util" else f"suff_{lam:g}"]["gamma_star"]
    lam = C.LAMBDA_FIGURES if kind == "util" else lam
    return round(lam + 0.1, 2)


def solution_name(kind, gamma, lam=None):
    """e.g. 'util_g0.3' or 'suff_lam0.4_g0.5'."""
    return f"{kind}_g{gamma:g}" if lam is None else f"{kind}_lam{lam:g}_g{gamma:g}"

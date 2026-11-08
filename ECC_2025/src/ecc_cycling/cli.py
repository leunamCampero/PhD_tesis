"""Small helpers shared by the numbered scripts (command-line options, saving results)."""
import argparse
import json

import numpy as np
import pandas as pd

from . import config as C


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--beta", type=float, default=C.BETA,
                   help="budget as a fraction of B_max (paper: 0.5)")
    p.add_argument("--demand", choices=["full_od", "historical_unit"], default=C.DEFAULT_DEMAND,
                   help="full_od = EMD 2020 demand (paper); historical_unit = newer IRIS->facility OD")
    return p


def tag(args):
    """Suffix used in every output file name, e.g. 'b0.5'."""
    t = f"b{args.beta:g}"
    return t if args.demand == C.DEFAULT_DEMAND else f"{t}_{args.demand}"


def ensure_dirs():
    for d in (C.RESULTS, C.RESULTS / "solutions", C.FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))


def save_solution(dos, path):
    pd.DataFrame({"edge_id": np.arange(len(dos)), "dos": dos}).to_csv(path, index=False)


def load_solution(path):
    return pd.read_csv(path).dos.to_numpy().astype(int)

"""One-off script that converted the original (2024) pickles into the clean,
portable CSV files stored in ``data/``.  It is kept for provenance only: you do
NOT need to run it (the CSV files are already in ``data/``).

Sources (legacy working folder of the PhD, not shipped with this project):
    LEGACY = .../CODE/2024/2024/First_project_2024

Usage:
    python tools/build_data_from_legacy.py <path-to-First_project_2024>
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CITIES = ["Angers", "Annecy", "Bordeaux", "Chambery", "Dunkerque", "Grenoble",
          "La_Rochelle", "Le_Havre", "Lyon", "Nantes", "Paris", "Rennes",
          "Strasbourg", "Tours"]


def export_graph(G, name, node_label=False, cscs_weights=None):
    """Write <name>_nodes.csv and <name>_edges.csv keeping networkx iteration order
    (the MLCC heuristic breaks ties using that order)."""
    nodes = []
    for n, d in G.nodes(data=True):
        if node_label:  # Sioux Falls: integer ids with a 'pos' attribute
            x, y = d["pos"]
            nodes.append({"node": n, "x": x, "y": y})
        else:           # node id is the (x, y) tuple itself
            nodes.append({"node": "", "x": n[0], "y": n[1]})
    edges = []
    for u, v, d in G.edges(data=True):
        # weighted lengths as stored in the original graphs.  For edges merged during
        # simplification they are the sum over the original OSM segments of
        # length * weight, i.e. NOT exactly length * DoS.
        row = {"length_m": d["distance"], "dos": int(d["key"]), "wlen_dos": d["weight"],
               "wlen_cscs": cscs_weights[(u, v)] if cscs_weights else None}
        if node_label:
            row.update(u=u, v=v)
        else:
            row.update(u_x=u[0], u_y=u[1], v_x=v[0], v_y=v[1])
        edges.append(row)
    out = DATA / "networks"
    pd.DataFrame(nodes).to_csv(out / f"{name}_nodes.csv", index=False)
    cols = (["u", "v"] if node_label else ["u_x", "u_y", "v_x", "v_y"]) + ["length_m", "dos", "wlen_dos", "wlen_cscs"]
    pd.DataFrame(edges)[cols].to_csv(out / f"{name}_edges.csv", index=False)


def main(legacy):
    legacy = Path(legacy)
    (DATA / "networks").mkdir(parents=True, exist_ok=True)

    # 1) 14 French cycling networks (integer DoS graphs, same edge order as the
    #    CycleRAP edge lists).
    for c in CITIES:
        G = pickle.load(open(legacy / f"{c}_Graph.pickle", "rb"))
        el = pickle.load(open(legacy / f"edge_list_{c}_CyRap.pickle", "rb"))
        export_graph(G, c, cscs_weights={(u, v): d["weight"] for u, v, d in el})
    # 2) Benchmark example (10 nodes, 11 edges) and Sioux Falls (24 nodes, 38 edges)
    el = pickle.load(open(legacy / "edge_list_Academic_example_CyRap.pickle", "rb"))
    export_graph(pickle.load(open(legacy / "academic_example_os.pickle", "rb")), "Benchmark",
                 cscs_weights={(u, v): d["weight"] for u, v, d in el})
    export_graph(pickle.load(open(legacy / "Sioux_Falls.pickle", "rb")), "Sioux_Falls", node_label=True)

    # 3) Synthetic Data Vault samples of the 2024 run (ten_four split)
    tf = legacy / "data_and_sythetic_data" / "ten_four"
    for k in ["safety", "comfort"]:
        pd.read_csv(tf / f"synthetic_{k}_10.csv").to_csv(DATA / "model" / f"sdv_synthetic_{k}.csv", index=False)

    # 4) Saved genetic-algorithm logs of the 2024 cluster runs (pygmo SGA / SGA-MLCC).
    #    Not used by the notebooks; archived outside the package.
    out = {}
    for net, prefix in [("Benchmark", "AE"), ("Sioux_Falls", "Sioux_Falls"), ("Grenoble", "Grenoble")]:
        out[net] = {}
        for target in ["safety", "comfort"]:
            out[net][target] = {}
            for i in range(10):
                r = pickle.load(open(legacy / "T_ITS_PLOTS" / f"{prefix}_{10*i}_results_{target}.pkl", "rb"))
                out[net][target][i] = {
                    "sga_value": [float(v) for v in r[0]], "sga_percent": [float(v) for v in r[1]],
                    "sga_cumtime_s": [float(v) for v in r[2]],
                    "sga_mlcc_value": [float(v) for v in r[3]], "sga_mlcc_percent": [float(v) for v in r[4]],
                    "sga_mlcc_cumtime_s": [float(v) for v in r[5]],
                }
    arch = ROOT / "_archive_old_files" / "comparison_with_published_paper" / "published_results"
    arch.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(arch / "ga_logs.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         r"C:\Users\Administrator\Documents\CODE\2024\2024\First_project_2024")

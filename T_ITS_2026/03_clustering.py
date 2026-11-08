"""
Step 3 - clustering of Lyon and Strasbourg (Section II-F, Algorithm 1, Figs. 4-5).

For every number of clusters k of the sweep (|V| ... |V_Grenoble| ... 5 000) both methods are run:
  * k-means   : spectral (Eq. 19-20) + geographic k-means of the notebooks (paper's "K-mean")
  * CC        : CC-Algorithm (constrained, exposure / flow checked)
and the clustered network of Section II-E is built and evaluated (clustered AON, E^clust, Phi^clust).

Outputs  results/03_sweep.csv                       one line per city x method x k
         results/03_labels_<city>_<method>.npy      partition at k = |V_Grenoble| (used by steps 4-6)
         results/03_clusters_summary.csv            diagnostics of these partitions
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import clustering as cl, config as C  # noqa: E402
from cyclenet.clustered import ClusteredNetwork  # noqa: E402
from cyclenet.network import Network  # noqa: E402
from cyclenet.utils import components_per_cluster, log  # noqa: E402


def sweep_values(n, k_ref):
    ks = np.unique(np.r_[np.linspace(n, C.SWEEP_MIN_K, C.SWEEP_POINTS).round().astype(int), k_ref])
    return ks[::-1]


def describe(net, cn, method, k, seconds, extra=None):
    E0, P0 = net.base.exposure, net.base.total_flow
    r = cn.base
    sizes = np.bincount(cn.labels)
    return dict(city=net.city, method=method, k=k, seconds=seconds,
                E_clust=r["exposure"], Phi_clust=r["total_flow"], E_orig=E0, Phi_orig=P0,
                dE_pct=100 * (r["exposure"] / E0 - 1), dPhi_pct=100 * (r["total_flow"] / P0 - 1),
                E_inter=r["exposure_inter"], E_intra=r["exposure_intra"],
                n_objects=cn.n_obj, n_intra=cn.n_intra, n_inter=cn.n_inter,
                mixed_objects_pct=100 * float((cn.w0max > cn.w0).mean()),
                mixed_arcs_pct=100 * float((cn.w0max > cn.w0)[cn.arc_obj].mean()),
                max_cluster_size=int(sizes.max()), max_diameter_m=float(cn.diameters().max()),
                disconnected_clusters=int(components_per_cluster(net, cn.labels)),
                demand_preserved=bool(np.isclose(cn.q.sum(), net.q.sum())),
                **(extra or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(C.TARGETS))
    args = ap.parse_args()
    C.RESULTS.mkdir(exist_ok=True)
    k_ref = Network(C.REFERENCE).N
    rows, summary = [], []
    for city in args.cities:
        net = Network(city)
        feats = cl.kmeans_features(net)
        for k in sweep_values(net.N, k_ref):
            t = time.time()
            lab_k = cl.kmeans_baseline(net, int(k), feats)
            tk = time.time() - t
            ck = ClusteredNetwork(net, lab_k)
            rows.append(describe(net, ck, "kmeans", int(k), tk))
            t = time.time()
            lab_c, info = cl.cc_algorithm(net, int(k))
            tc = time.time() - t
            cc = ClusteredNetwork(net, lab_c)
            rows.append(describe(net, cc, "CC", int(k), tc,
                                 {f"cc_{a}": b for a, b in info.items() if a not in ("seconds",)}))
            log(f"{city:10s} k={k:6d}  k-means dE {rows[-2]['dE_pct']:+6.2f} % dPhi {rows[-2]['dPhi_pct']:+6.2f} %"
                f" | CC dE {rows[-1]['dE_pct']:+6.2f} % dPhi {rows[-1]['dPhi_pct']:+6.2f} %  ({tk:.0f} s / {tc:.0f} s)")
            if k == k_ref:
                np.save(C.RESULTS / f"03_labels_{city}_kmeans.npy", lab_k)
                np.save(C.RESULTS / f"03_labels_{city}_CC.npy", lab_c)
                summary += [rows[-2], rows[-1]]
            pd.DataFrame(rows).to_csv(C.RESULTS / "03_sweep.csv", index=False)
    pd.DataFrame(summary).to_csv(C.RESULTS / "03_clusters_summary.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()

"""
Step 4 - comparison of the clustered Lyon / Strasbourg with Grenoble (Section IV; ITSC 2026 Sec. IV-VI).

For each target city and each clustering (CC-Algorithm, k-means) at k = |V_Grenoble|:
  * GW distance between the exposure measures (top-M subsets, M in GW_SIZES)
  * kernel (degree-sorted graphon) distance of the exposure kernels
Sanity references: GW(Grenoble, Grenoble) and GW(Lyon, Strasbourg) (both CC).

Outputs  results/04_comparison.csv
         results/04_gw_<city>_<method>.npz      GW plan at M = GW_TOP_M (used by the transfer, step 6)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import compare as cp, config as C  # noqa: E402
from cyclenet.clustered import ClusteredNetwork  # noqa: E402
from cyclenet.network import Network  # noqa: E402
from cyclenet.utils import log  # noqa: E402


def main():
    H = Network(C.REFERENCE)
    mh, xh = cp.original_measure(H)
    Kh = cp.kernel_original(H)
    rows = []
    clustered = {}
    for city in C.TARGETS:
        net = Network(city)
        for method in ("CC", "kmeans"):
            cn = ClusteredNetwork(net, np.load(C.RESULTS / f"03_labels_{city}_{method}.npy"))
            clustered[(city, method)] = cn
            mg, xg = cp.clustered_measure(cn)
            kd = cp.kernel_distance(Kh, cp.kernel_clustered(cn))
            for m in C.GW_SIZES:
                t = time.time()
                g = cp.gw_between(mh, xh, mg, xg, m)
                rows.append(dict(source=C.REFERENCE, target=city, method=method, M=m, GW=g["gw"],
                                 kernel_distance=kd, seconds=time.time() - t))
                log(f"GW({C.REFERENCE}, {city} {method}) M={m}: {g['gw']:.4f}  kernel {kd:.4f}  ({time.time() - t:.0f} s)")
                if m == C.GW_TOP_M:
                    np.savez_compressed(C.RESULTS / f"04_gw_{city}_{method}.npz", T=g["T"], idx_h=g["idx_h"],
                                        idx_g=g["idx_g"], gw=g["gw"])
            pd.DataFrame(rows).to_csv(C.RESULTS / "04_comparison.csv", index=False)
    # sanity references
    g = cp.gw_between(mh, xh, mh, xh, C.GW_TOP_M)
    rows.append(dict(source=C.REFERENCE, target=C.REFERENCE, method="original", M=C.GW_TOP_M, GW=g["gw"],
                     kernel_distance=cp.kernel_distance(Kh, Kh)))
    a, b = clustered[("Lyon", "CC")], clustered[("Strasbourg", "CC")]
    ma, xa = cp.clustered_measure(a)
    mb, xb = cp.clustered_measure(b)
    g = cp.gw_between(ma, xa, mb, xb, C.GW_TOP_M)
    rows.append(dict(source="Lyon", target="Strasbourg", method="CC", M=C.GW_TOP_M, GW=g["gw"],
                     kernel_distance=cp.kernel_distance(cp.kernel_clustered(a), cp.kernel_clustered(b))))
    pd.DataFrame(rows).to_csv(C.RESULTS / "04_comparison.csv", index=False)
    log(pd.DataFrame(rows).round(4).to_string())
    log("done")


if __name__ == "__main__":
    main()

"""
Step 6 - GW-based transfer of the Grenoble GA strategy to Lyon and Strasbourg (Section V-B, Problem 3,
Algorithm 4).

For every budget fraction beta, the Grenoble GA design of step 5 (same beta) is transferred to the
clustered target through the GW plan of step 4, Problem 3 is solved on the target's clustered objects with
the target budget beta B_max^G, and the design is lifted and evaluated on the original target network.
Variants: target clustered by CC ("Transfer") or k-means ("Transfer (k-means)"), and the flow-aware
proposal ("Transfer+", CC only).

Outputs  results/06_transfer.csv,  results/06_runs/<city>_<variant>_<beta>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import config as C, transfer as tr  # noqa: E402
from cyclenet.clustered import ClusteredNetwork  # noqa: E402
from cyclenet.network import Network  # noqa: E402
from cyclenet.utils import log  # noqa: E402

VARIANTS = (("Transfer", "CC", False), ("Transfer (k-means)", "kmeans", False), ("Transfer+", "CC", True))


def main():
    out = C.RESULTS / "06_runs"
    out.mkdir(parents=True, exist_ok=True)
    H = Network(C.REFERENCE)
    rows = []
    for city in C.TARGETS:
        net = Network(city)
        cns = {m: ClusteredNetwork(net, np.load(C.RESULTS / f"03_labels_{city}_{m}.npy")) for m in ("CC", "kmeans")}
        for beta in C.BUDGET_FRACTIONS:
            w_star = np.load(C.RESULTS / "05_runs" / f"{C.REFERENCE}_GA_{beta:g}.npz")["dos"]
            for name, method, flow_aware in VARIANTS:
                t = time.time()
                gw = dict(np.load(C.RESULTS / f"04_gw_{city}_{method}.npz"))
                cn = cns[method]
                levels, dos, spent, S = tr.transfer(H, w_star, gw, cn, beta * net.B_max, flow_aware=flow_aware)
                ev = net.evaluate(dos)
                rows.append(dict(city=city, method=name, beta=beta, budget=beta * net.B_max, cost=net.upgrade_cost(dos),
                                 E_base=net.base.exposure, E_final=ev.exposure, E_norm=ev.exposure / net.base.exposure,
                                 GW=float(gw["gw"]), seconds=time.time() - t,
                                 upgraded_arcs=int((dos > net.dos0).sum()),
                                 upgraded_km=float(net.length[dos > net.dos0].sum() / 1000)))
                np.savez_compressed(out / f"{city}_{name.replace(' ', '_').replace('(', '').replace(')', '')}_{beta:g}.npz",
                                    dos=dos, levels=levels, scores=S)
                log(f"{city:10s} {name:20s} beta {beta:.4f}: E/E0 = {rows[-1]['E_norm']:.3f}")
    pd.DataFrame(rows).to_csv(C.RESULTS / "06_transfer.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()

"""
Step 1 - networks, demand and today's flows / exposure of Grenoble, Lyon and Strasbourg (Section II).

Reads   data/raw/<city>_bike_network_with_slope.pkl, <city>_node_demand_dic.pkl, boundaries.geojson
Writes  data/networks/<city>_{nodes,edges,demand_gravity,demand_paper130}.csv
        results/01_networks.csv                  size, DoS mix, B_max, today's flow / exposure
        figures/fig01_grenoble_dos.png           Fig. 1
        figures/fig01b_original_networks.png     the three networks (Fig. 5a, 8a)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cyclenet import config as C  # noqa: E402
from cyclenet.network import Network, build_city  # noqa: E402
from cyclenet.plotting import dos_map, paper_style, plt, save  # noqa: E402
from cyclenet.utils import log  # noqa: E402


def main():
    C.RESULTS.mkdir(exist_ok=True)
    paper_style()
    rows = []
    nets = {}
    for city in C.CITIES:
        build_city(city)
        net = Network(city)
        nets[city] = net
        km = net.length.sum() / 1000
        share = {f"km_DoS{d}_pct": 100 * net.length[net.dos0 == d].sum() / 1000 / km for d in range(1, 7)}
        rows.append(dict(city=city, nodes=net.N, arcs=net.E, km=km, **share, B_max_MEUR=net.B_max / 1e6,
                         od_pairs=len(net.q), origins=len(net.ug.origins), demand=net.q.sum(),
                         arcs_used_pct=100 * (net.base.flow > 0).mean(), E_orig=net.base.exposure,
                         E_orig_per_arc=net.base.exposure / net.E, Phi_orig=net.base.total_flow,
                         mean_trip_km=float((net.base.od_cost * net.q).sum() / net.q.sum() / 1000),
                         demand_source=C.DEMAND, slope_source=C.SLOPE_SOURCE))
        log(f"{city:10s} {net.N:6d} nodes {net.E:6d} arcs {km:7.1f} km  B_max {net.B_max / 1e6:6.1f} M EUR  "
            f"E {net.base.exposure:.3e}  arcs used {rows[-1]['arcs_used_pct']:.1f} %")
    pd.DataFrame(rows).to_csv(C.RESULTS / "01_networks.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6.5))
    dos_map(ax, nets["Grenoble"], nets["Grenoble"].dos0, lw=1.2)
    save(fig, "fig01_grenoble_dos.png")

    fig, axs = plt.subplots(1, 3, figsize=(13, 5.2))
    for ax, city in zip(axs, C.CITIES):
        dos_map(ax, nets[city], nets[city].dos0, title=f"{city} ({nets[city].N:,} nodes)")
    save(fig, "fig01b_original_networks.png")
    log("done")


if __name__ == "__main__":
    main()

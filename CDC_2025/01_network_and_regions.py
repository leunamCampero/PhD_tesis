"""
STEP 1 - Network, IRIS regions, demand and budgets (Sections II-A, II-B, III-B).

* Fig. 1  : DoS of the Grenoble network with the 70 IRIS (iris.gpkg)
* map of the cyclists per IRIS (n_r, used to weight J_I)
* B_max and the 10 budgets B_k of the sensitivity analysis

Run:  python 01_network_and_regions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

from cdc_cycling import Network, costs, plotting, utils          # noqa: E402
from cdc_cycling import config as C                              # noqa: E402

utils.ensure_dirs()
net = Network()
Bmax = costs.max_budget(net)
km = costs.km_per_dos(net, net.dos0)

print(f"\nNETWORK: {net.N} nodes, {net.E} arcs, {net.length.sum() / 1000:.1f} km, 70 IRIS")
print("   km per DoS:  " + "  ".join(f"DoS{k} {v:.1f}" for k, v in km.items()))
print(f"\nDEMAND: {net.Q.sum():,.0f} bike trips/day between {len(net.sources)} trip-end nodes "
      f"({C.NODES_PER_IRIS} per IRIS)")
zones = np.unique(net.source_zone)
n_r = np.array([net.Q[net.source_zone == z].sum() for z in zones])
print(f"   cyclists per IRIS n_r: min {n_r.min():.0f}, median {np.median(n_r):.0f}, max {n_r.max():.0f}")
print(f"\nBUDGET: B_max = {Bmax / 1e6:.2f} M EUR")
print("   B_k / B_max = " + ", ".join(f"{b:.3f}" for b in C.BUDGET_FRACTIONS))

utils.save_json(dict(km_per_dos=km, B_max=Bmax, budget_fractions=C.BUDGET_FRACTIONS,
                     trips_per_day=net.Q.sum(), cyclists_per_iris=dict(zip(zones.tolist(), n_r))),
                C.RESULTS / "01_network_and_regions.json")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
plotting.dos_map(axes[0], net, net.dos0, title="Fig. 1 - DoS of the network")
plotting.region_map(axes[1], net, zones, n_r, vmax=n_r.max(), cmap="Blues",
                    label="cyclists (trips/day)", title="Cyclists leaving each IRIS ($n_r$)")
fig.tight_layout()
fig.savefig(C.FIGURES / "fig1_network_and_iris.png", dpi=200)
print("\nsaved: figures/fig1_network_and_iris.png, results/01_network_and_regions.json")

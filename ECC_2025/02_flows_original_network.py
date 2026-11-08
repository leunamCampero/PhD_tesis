"""
STEP 2 - Cyclist flows and exposure J_E of the ORIGINAL network (lower level of Problem 1).

* all-or-nothing assignment on the cyclist-distance shortest paths (Eqs. 5-6)
* exposure J_E = sum_a d_hat_a f_hat_a x_hat_a (Eqs. 7-9)
* Fig. 1 (DoS map) and the map of the original flows

Run:  python 02_flows_original_network.py [--demand full_od|historical_unit]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt                         # noqa: E402
import pandas as pd                                     # noqa: E402

from ecc_cycling import Network, cli, plotting          # noqa: E402
from ecc_cycling import config as C                     # noqa: E402

args = cli.parser(__doc__).parse_args()
cli.ensure_dirs()
net = Network(args.demand)

res = net.evaluate(net.dos0)
print(f"\nORIGINAL NETWORK, demand '{net.demand_kind}'")
print(f"   J_E = {res['J_E']:.4f}")
print(f"   arcs carrying flow: {(res['flows'] > 0).sum()} of {net.E};"
      f" max arc flow = {res['flows'].max():,.0f} bikes/day")

# per-arc results, useful for GIS
pd.DataFrame({"edge_id": range(net.E), "dos": net.dos0, "flow": res["flows"],
              "x_hat": res["x_hat"], "f_a": res["f"], "f_hat": res["f_hat"],
              "exposure_arc": res["exposure_arc"]}).to_csv(
    C.RESULTS / f"02_original_arcs_{cli.tag(args)}.csv", index=False, float_format="%.6g")
cli.save_json(dict(demand=net.demand_kind, J_E=res["J_E"]), C.RESULTS / f"02_original_{cli.tag(args)}.json")

# figures
fig, ax = plt.subplots(figsize=(6, 6))
plotting.dos_map(ax, net, net.dos0)
fig.tight_layout()
fig.savefig(C.FIGURES / "fig1_dos_original.png", dpi=200)
plt.close(fig)
fig, ax = plt.subplots(figsize=(6.5, 6))
plotting.flow_map(ax, net, res["x_hat"], title=f"Original flows (J_E = {res['J_E']:.3f})")
fig.tight_layout()
fig.savefig(C.FIGURES / f"flows_original_{cli.tag(args)}.png", dpi=200)
plt.close(fig)
print(f"saved: figures/fig1_dos_original.png, figures/flows_original_{cli.tag(args)}.png, results/02_*")

"""
STEP 2 - Today's network: flows, exposure and safety insufficiency (Sections II-C to II-E).

* J_E (utilitarian exposure), average path safety s_G and the thresholds s_thr = lambda s_G
* J_I and % of cyclists below the threshold for every lambda
* path-safety histogram and regional insufficiency map (lambda = 40 %)

Run:  python 02_original_network.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt                                  # noqa: E402

from cdc_cycling import config as C                              # noqa: E402
from cdc_cycling import plotting, utils                          # noqa: E402
from cdc_cycling.runner import context                           # noqa: E402

utils.ensure_dirs()
net, ev0, suf = context()

print(f"\nORIGINAL NETWORK")
print(f"   J_E = {ev0['J_E']:.4f}")
print(f"   average path safety s_G = {suf.s_G:.2f}   (s_p = 1/(E_p + {C.EPSILON}), max 100)")
rows = {}
for lam in C.LAMBDAS:
    rows[lam] = dict(threshold=suf.thresholds[lam], J_I=suf.J_I(ev0, lam),
                     objective=suf.objective(ev0, lam), below=suf.share_below(ev0, lam))
    print(f"   lambda {lam:.0%}: s_thr = {suf.thresholds[lam]:6.2f}   J_I = {rows[lam]['J_I']:.4f}   "
          f"J_I + eta J_E = {rows[lam]['objective']:.4f}   below threshold = {rows[lam]['below']:.1%}")
utils.save_json(dict(J_E=ev0["J_E"], s_G=suf.s_G, per_lambda=rows), C.RESULTS / "02_original.json")

lam = 0.4
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
s, w = suf.safety_histogram_data(ev0)
plotting.safety_histogram(axes[0], s, w, suf.thresholds[lam], lam, title="Original network")
plotting.region_map(axes[1], net, suf.zones, suf.regional(ev0, lam), label="$\\mathcal{I}_r$",
                    title="Regional safety insufficiency, $\\lambda=40\\%$")
fig.tight_layout()
fig.savefig(C.FIGURES / "original_safety.png", dpi=200)
print("saved: figures/original_safety.png, results/02_original.json")

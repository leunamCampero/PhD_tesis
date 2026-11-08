"""
STEP 5 - Compare the original network, GAB and MLCC (Fig. 2, Fig. 3, Table III).

All solutions are evaluated with the same model and demand.  Needs the solutions written by
03_run_mlcc.py and 04_run_gab.py.

Writes results/05_table3_<tag>.csv, figures/fig2_exposure_<tag>.png, figures/fig3_maps_<tag>.png.

Run:  python 05_compare_and_figures.py [--beta 0.5]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd                                     # noqa: E402

from ecc_cycling import Network, cli, costs, plotting   # noqa: E402
from ecc_cycling import config as C                     # noqa: E402

args = cli.parser(__doc__).parse_args()
cli.ensure_dirs()
net = Network(args.demand)
Bmax = costs.max_budget(net)
tag = cli.tag(args)

sols = {"Original": net.dos0}
for algo in ("gab", "mlcc"):
    f = C.RESULTS / "solutions" / f"{algo}_{tag}.csv"
    if not f.exists():
        sys.exit(f"missing {f.name}: run 0{3 if algo == 'mlcc' else 4}_run_{algo}.py first")
    sols[algo.upper()] = cli.load_solution(f)

rows, evals = [], {}
for name, dos in sols.items():
    ev = net.evaluate(dos)
    evals[name] = ev
    rows.append(dict(solution=name, J_E=ev["J_E"],
                     reduction=1 - ev["J_E"] / (evals["Original"]["J_E"]),
                     budget_used=costs.upgrade_cost(net, dos) / Bmax,
                     **{f"DoS{k} km": v for k, v in costs.km_per_dos(net, dos).items()}))
table = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(f"\nEXPOSURE AND km PER DoS  (demand '{net.demand_kind}', beta {args.beta:g})\n")
print(table.round(3).to_string(index=False))
table.to_csv(C.RESULTS / f"05_table3_{tag}.csv", index=False, float_format="%.5g")

plotting.exposure_bars({k: evals[k]["J_E"] for k in sols}, C.FIGURES / f"fig2_exposure_{tag}.png")
plotting.fig3_panel(net, {k: (sols[k], evals[k]["x_hat"]) for k in sols},
                    C.FIGURES / f"fig3_maps_{tag}.png")
print(f"\nsaved: results/05_table3_{tag}.csv, figures/fig2_exposure_{tag}.png, figures/fig3_maps_{tag}.png")

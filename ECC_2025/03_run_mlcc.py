"""
STEP 3 - MLCC: Maximize the Largest Connected Component (Algorithm 2), budget B = beta * B_max.

Writes results/solutions/mlcc_<tag>.csv (DoS of every arc) and results/03_mlcc_<tag>.json.
Runtime: ~20 s.

Run:  python 03_run_mlcc.py [--beta 0.5]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecc_cycling import Network, cli, costs             # noqa: E402
from ecc_cycling import config as C                     # noqa: E402
from ecc_cycling.mlcc import run_mlcc                   # noqa: E402

args = cli.parser(__doc__).parse_args()
cli.ensure_dirs()
net = Network(args.demand)
Bmax = costs.max_budget(net)
B = args.beta * Bmax

t0 = time.time()
dos, log = run_mlcc(net, B)
J = net.exposure(dos)
spent = costs.upgrade_cost(net, dos)
km = costs.km_per_dos(net, dos)

print(f"\nMLCC, B = {args.beta:g} x B_max = {B / 1e6:.2f} M EUR")
print(f"   {len(log)} arc upgrades, budget used {spent / Bmax:.3f} x B_max, {time.time() - t0:.1f} s")
print(f"   J_E = {J:.4f}")
print("   km per DoS:  " + "  ".join(f"DoS{k} {v:.1f}" for k, v in km.items()))

cli.save_solution(dos, C.RESULTS / "solutions" / f"mlcc_{cli.tag(args)}.csv")
cli.save_json(dict(J_E=J, budget_used=spent / Bmax, km_per_dos=km, upgrades=len(log)),
              C.RESULTS / f"03_mlcc_{cli.tag(args)}.json")
print(f"saved: results/solutions/mlcc_{cli.tag(args)}.csv")

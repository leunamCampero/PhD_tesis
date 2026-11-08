"""
STEP 4 - GAB: Genetic Algorithm with efficient Budget spending (pygmo SGA + Algorithm 1).

Writes results/solutions/gab_<tag>.csv, results/04_gab_<tag>.json (incl. convergence) and
figures/gab_convergence_<tag>.png.

Runtime: one fitness evaluation = one traffic assignment of 11.7 M OD pairs (~5 s), so the
default 20 individuals x 20 generations (~420 evaluations) takes ~40 min.  --quick: ~3 min.

Run:  python 04_run_gab.py [--beta 0.5] [--pop 20] [--generations 20] [--seed 0] [--quick]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecc_cycling import Network, cli, costs, plotting   # noqa: E402
from ecc_cycling import config as C                     # noqa: E402
from ecc_cycling.gab import run_gab                     # noqa: E402

p = cli.parser(__doc__)
p.add_argument("--pop", type=int, default=20, help="population size")
p.add_argument("--generations", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true", help="tiny run (pop 8, 3 generations) to test")
args = p.parse_args()
if args.quick:
    args.pop, args.generations = 8, 3
cli.ensure_dirs()
net = Network(args.demand)
Bmax = costs.max_budget(net)
B = args.beta * Bmax

print(f"\nGAB, B = {args.beta:g} x B_max = {B / 1e6:.2f} M EUR, "
      f"pop {args.pop}, {args.generations} generations")
dos, info = run_gab(net, B, pop_size=args.pop, generations=args.generations, seed=args.seed)
J = net.exposure(dos)
spent = costs.upgrade_cost(net, dos)
km = costs.km_per_dos(net, dos)
print(f"   J_E = {J:.4f}   budget used {spent / Bmax:.3f} x B_max   "
      f"{info['evaluations']} evaluations, {info['seconds'] / 60:.1f} min")
print("   km per DoS:  " + "  ".join(f"DoS{k} {v:.1f}" for k, v in km.items()))

name = cli.tag(args) + ("_quick" if args.quick else "")
cli.save_solution(dos, C.RESULTS / "solutions" / f"gab_{name}.csv")
cli.save_json({**info, "J_E": J, "budget_used": spent / Bmax, "km_per_dos": km},
              C.RESULTS / f"04_gab_{name}.json")
plotting.convergence(info["history"], C.FIGURES / f"gab_convergence_{name}.png", title="GAB convergence")
print(f"saved: results/solutions/gab_{name}.csv, figures/gab_convergence_{name}.png")

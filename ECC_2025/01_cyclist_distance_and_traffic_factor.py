"""
STEP 1 - The building blocks of the model (Sections II and IV-A of the paper).

Prints and saves:
  * Table I extended with the traffic factor f_a (Eq. 2), f_a^rho and its normalised value
  * the slope factors (Table IV)
  * the network: km per DoS, full-upgrade budget B_max and budget B
  * the cyclist demand (EMD 2020, x 0.5 x 0.05)
  * a worked example of the cyclist distance b_(u,v) (Eq. 1) for one OD pair

Run:  python 01_cyclist_distance_and_traffic_factor.py [--beta 0.5]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
from scipy.sparse import csr_matrix                                    # noqa: E402
from scipy.sparse.csgraph import dijkstra                              # noqa: E402

from ecc_cycling import Network, cli, costs                            # noqa: E402
from ecc_cycling import config as C                                    # noqa: E402
from ecc_cycling.model import F_MAX, F_MIN, slope_factor, traffic_factor  # noqa: E402

args = cli.parser(__doc__).parse_args()
cli.ensure_dirs()
net = Network(args.demand)
pd.set_option("display.width", 140)

# ---------------------------------------------------------------------------------------
# 1. Traffic factor per DoS class
# ---------------------------------------------------------------------------------------
levels = np.array([6, 5, 4, 3, 2, 1])
f = traffic_factor(levels)
tf = pd.DataFrame({
    "DoS": levels,
    "description": [C.DOS_NAMES[k] for k in levels],
    "AADT (veh/day)": [C.AADT[k] for k in levels],
    "f_a  (Eq. 2)": f.round(3),
    "f_a^rho (in b_a)": (f ** C.RHO).round(4),
    "f_hat (in J_E)": ((f - F_MIN) / (F_MAX - F_MIN)).round(5),
})
print("\nTRAFFIC FACTOR  (psi = %.2f, rho = %.2f)\n" % (C.PSI, C.RHO))
print(tf.to_string(index=False))
tf.to_csv(C.RESULTS / "01_traffic_factor.csv", index=False)

print("\nSLOPE FACTOR s_a (uphill; downhill = 1 / s_a)")
for s in (0, 1.9, 2, 3.9, 4, 5.9, 6, 10):
    print(f"   slope {s:5.1f} %  ->  s_a = {float(slope_factor(s)):.3f}")

# ---------------------------------------------------------------------------------------
# 2. Network and budget
# ---------------------------------------------------------------------------------------
km = costs.km_per_dos(net, net.dos0)
Bmax = costs.max_budget(net)
summary = dict(nodes=net.N, arcs=net.E, km_total=float(net.length.sum() / 1000),
               km_per_dos=km, B_max_eur=Bmax, beta=args.beta, B_eur=args.beta * Bmax)
print(f"\nNETWORK: {net.N} nodes, {net.E} arcs, {summary['km_total']:.1f} km")
print("   km per DoS:  " + "  ".join(f"DoS{k} {v:.1f}" for k, v in km.items()))
print(f"   B_max (everything to DoS 6, Cerema costs) = {Bmax / 1e6:.2f} M EUR;"
      f"  B = {args.beta:g} x B_max = {args.beta * Bmax / 1e6:.2f} M EUR")

# ---------------------------------------------------------------------------------------
# 3. Demand
# ---------------------------------------------------------------------------------------
od = pd.read_csv(C.DATA / "demand" / "od_iris_grenoble.csv")
M = od.pivot(index="zone_o", columns="zone_d", values="trips_matrix").to_numpy()
off = M.sum() - np.trace(M)
print("\nDEMAND (EMD 2020 IRIS x IRIS matrix, all modes, 70 IRIS of Grenoble)")
print(f"   matrix symmetric: {np.allclose(M, M.T)};  sum of off-diagonal entries = {off:,.0f}")
print(f"   x 0.5  (each trip is stored in both directions)   = {0.5 * off:,.0f} trips/day")
print(f"   x 0.05 (bike modal share)                          = {C.DEMAND_FACTOR * off:,.0f} bike trips/day")
print(f"   demand used: '{net.demand_kind}', total routed = {net.total_demand():,.0f}")
summary["demand"] = dict(off_diagonal_sum=off, trips_per_day=0.5 * off,
                         bike_trips_per_day=C.DEMAND_FACTOR * off, routed=net.total_demand())

# ---------------------------------------------------------------------------------------
# 4. Worked example of Eq. (1): the most used node of IRIS 49 -> the most used of IRIS 116
# ---------------------------------------------------------------------------------------
w = net.zone_weight
o = int(np.flatnonzero(net.zone == 49)[np.argmax(w[net.zone == 49])])
d = int(np.flatnonzero(net.zone == 116)[np.argmax(w[net.zone == 116])])
fwd, bwd = net.arc_costs(net.dos0)
A = csr_matrix((np.concatenate([fwd, bwd]), (net.tail, net.head)), shape=(net.N, net.N))
_, pred = dijkstra(A, indices=o, return_predecessors=True)
path = [d]
while path[-1] != o:
    path.append(int(pred[path[-1]]))
path = path[::-1]
b, b_arcs, turns = net.cyclist_distance(path)
arc_ids = [net.arc_index[a, c] % net.E for a, c in zip(path, path[1:])]
geo = net.length[arc_ids].sum()
print(f"\nCYCLIST DISTANCE, example path node {o} (IRIS 49) -> node {d} (IRIS 116), {len(arc_ids)} arcs")
print(f"   geographic length            {geo:8.1f} m")
print(f"   sum d_a s_a f_a^rho          {b_arcs:8.1f} m   (used as Dijkstra weight)")
print(f"   kappa * sum of turns         {C.KAPPA * turns:8.1f}     ({turns:.0f} deg in total)")
print(f"   b_(u,v)  (Eq. 1)             {b:8.1f}")
print("   DoS along the path:", "".join(str(k) for k in net.dos0[arc_ids]))
summary["example_path"] = dict(origin=o, destination=d, arcs=len(arc_ids), length_m=geo,
                               arc_term=b_arcs, turn_deg=turns, b=b)

cli.save_json(summary, C.RESULTS / f"01_network_and_demand_{cli.tag(args)}.json")
print(f"\nsaved: results/01_traffic_factor.csv, results/01_network_and_demand_{cli.tag(args)}.json")

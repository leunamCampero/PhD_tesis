"""
Step 1 - The multi-modal network of Grenoble (Section II-B, Fig. 1, Table I).

* bicycle layer: the ECC/CDC network (3452 nodes, 4993 arcs, DoS 1-6) and its demand
* car layer: OSM drive network (4720 nodes, 6175 roads), capacities, free-flow times
* which bike arcs run along a car road (the only ones whose upgrade can cost car capacity)
* budget B_max (cost of raising every arc to DoS 6) and the three budgets alpha * B_max

Output: results/01_networks.json, figures/fig1_multimodal_network.png
Run time: ~10 s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from itsc_cycling import config as C, costs, plotting as P, utils  # noqa: E402
from itsc_cycling.bike import BikeNetwork  # noqa: E402
from itsc_cycling.cars import CarNetwork  # noqa: E402
from itsc_cycling.coupling import Coupling  # noqa: E402

utils.ensure_dirs()
bike, car = BikeNetwork(), CarNetwork()
cpl = Coupling(bike, car)
B_max = costs.max_budget(bike)

shared_roads = np.unique(cpl.cand_road)
out = dict(
    bike=dict(nodes=bike.N, edges=bike.E, km=bike.length.sum() / 1000,
              km_per_dos=costs.km_per_dos(bike, bike.dos0), trip_ends=len(bike.sources),
              trips_per_day=float(bike.Q.sum())),
    car=dict(nodes=car.N, roads=car.E, km=car.length[: car.E].sum() / 1000, trip_ends=len(car.sources),
             peak_hour_trips_m1=float(car.Q_base.sum()),
             km_by_class={h: float(car.length[: car.E][car.highway[: car.E] == h].sum() / 1000)
                          for h in sorted(set(car.highway))}),
    coupling=dict(on_road_bike_arcs_along_car_roads=int(len(cpl.cand)),
                  their_km=float(bike.length[cpl.cand].sum() / 1000),
                  car_roads_concerned=int(len(shared_roads)),
                  car_road_km_concerned=float(car.length[shared_roads].sum() / 1000)),
    budget=dict(B_max=B_max, budgets={f"{a:.3f}": a * B_max for a in C.ALPHAS}),
)
utils.save_json(out, C.RESULTS / "01_networks.json")

print(f"bike layer : {bike.N} nodes, {bike.E} arcs, {out['bike']['km']:.0f} km, "
      f"{len(bike.sources)} trip ends, {bike.Q.sum():,.0f} bike trips/day")
print("             km per DoS:", {k: round(v, 1) for k, v in out["bike"]["km_per_dos"].items()})
print(f"car layer  : {car.N} nodes, {car.E} roads ({2 * car.E} arcs), {out['car']['km']:.0f} km, "
      f"{len(car.sources)} trip ends, {car.Q_base.sum():,.0f} veh/h (peak hour, before calibration)")
print(f"coupling   : {len(cpl.cand)} on-road bike arcs (DoS 1-4, {out['coupling']['their_km']:.0f} km) "
      f"run along {len(shared_roads)} car roads ({out['coupling']['car_road_km_concerned']:.0f} km)")
print(f"budget     : B_max = {B_max / 1e6:.2f} M EUR;  " +
      ", ".join(f"alpha={a:.3f}: {a * B_max / 1e6:.1f} M" for a in C.ALPHAS))

fig, axs = P.plt.subplots(1, 2, figsize=(10, 5.2))
P.bike_dos_map(axs[0], bike, bike.dos0, lw=0.8, title="Bicycle layer: degree of separation")
P.car_map(axs[1], car, lw=0.7, shared=shared_roads, title="Car layer (OSM drive network)")
for ax in axs:
    P.iris_boundaries(ax, lw=0.25)
print("figure:", P.save(fig, "fig1_multimodal_network.png"))

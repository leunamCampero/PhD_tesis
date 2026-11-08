"""
Step 1 - Multi-modal network, capacity rule and mode-choice model (Sections II-B to II-F).

* bicycle layer (ECC/CDC graph) and car layer (OSM drive network), as in ITSC_2025
* capacity rule theta = theta0 (1 - omega z / n): which roads can lose capacity and by how much
* mode choice: mean cyclist distance D_bar today, calibrated alpha, logit curve P_b(D_bar)
* budget B = 0.15 B_max

Outputs: results/01_networks_and_mode_choice.json, figures/fig1_multimodal_network.png,
         figures/fig2_mode_choice.png
Run time: ~15 s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from si_cycling import config as C, costs, plotting as P, utils  # noqa: E402
from si_cycling.bike import BikeNetwork  # noqa: E402
from si_cycling.cars import CarNetwork  # noqa: E402
from si_cycling.coupling import Coupling  # noqa: E402
from si_cycling.modeshift import ModeChoice  # noqa: E402

utils.ensure_dirs()
bike, car = BikeNetwork(), CarNetwork()
cpl = Coupling(bike, car)
ev0 = bike.evaluate(bike.dos0)
mode = ModeChoice(bike, ev0, multiplier=1.0)
B_max = costs.max_budget(bike)
lanes = car.lanes[: car.E]
L = car.length[: car.E]

out = dict(
    bike=dict(nodes=bike.N, edges=bike.E, km=bike.length.sum() / 1000, trips_per_day=float(bike.Q.sum()),
              km_per_dos=costs.km_per_dos(bike, bike.dos0)),
    car=dict(nodes=car.N, roads=car.E, km=float(L.sum() / 1000),
             km_by_lanes_per_direction={int(n): float(L[lanes == n].sum() / 1000) for n in np.unique(lanes)}),
    capacity_rule=dict(omega=C.OMEGA, roads_that_can_lose_capacity=int(len(cpl.roads)),
                       their_km=float(L[cpl.roads].sum() / 1000),
                       capacity_kept={int(n): float(max(1 - C.OMEGA / n, C.MIN_CAPACITY_SHARE)) for n in (1, 2, 3)},
                       capacity_kept_paper_omega1={int(n): float(max(1 - 1 / n, 0)) for n in (1, 2, 3)}),
    mode_choice=dict(P_b0=C.P_BIKE_2025, beta_per_km=C.BETA_PER_KM, D_bar0_km=mode.D0,
                     alpha_calibrated=mode.alpha, alpha_paper=0.128,
                     P_b_with_paper_alpha=float(1 / (1 + np.exp(-(0.128 + C.BETA_PER_KM * mode.D0))))),
    budget=dict(B_max=B_max, fraction=C.BUDGET_FRACTION, B=C.BUDGET_FRACTION * B_max),
)
utils.save_json(out, C.RESULTS / "01_networks_and_mode_choice.json")

one_lane = L[lanes == 1].sum() / L.sum()
print(f"bike: {bike.E} arcs, {out['bike']['km']:.0f} km, {bike.Q.sum():,.0f} trips/day;  "
      f"car: {car.E} roads, {L.sum() / 1000:.0f} km ({100 * one_lane:.0f} % of the length has one lane per direction)")
print(f"capacity rule (omega = {C.OMEGA}): {len(cpl.roads)} roads ({out['capacity_rule']['their_km']:.0f} km) "
      f"can lose capacity; kept share by lanes/direction: {out['capacity_rule']['capacity_kept']} "
      f"(paper omega = 1: {out['capacity_rule']['capacity_kept_paper_omega1']})")
print(f"mode choice: D_bar today = {mode.D0:.3f} km, beta = {C.BETA_PER_KM}, alpha calibrated = "
      f"{mode.alpha:.3f} -> P_b = {C.P_BIKE_2025:.4f}   (paper alpha 0.128 would give "
      f"{100 * out['mode_choice']['P_b_with_paper_alpha']:.1f} %)")
print(f"budget: B = {C.BUDGET_FRACTION} x {B_max / 1e6:.2f} M EUR = {C.BUDGET_FRACTION * B_max / 1e6:.1f} M EUR")

fig, axs = P.plt.subplots(1, 2, figsize=(10, 5.2))
P.bike_dos_map(axs[0], bike, bike.dos0, lw=0.8, title="Bicycle layer: degree of separation")
P.car_map(axs[1], car, lw=0.7, shared=cpl.roads, title="Car layer; roads that can lose capacity")
for ax in axs:
    P.iris_boundaries(ax, lw=0.25)
print("figure:", P.save(fig, "fig1_multimodal_network.png"))

fig, ax = P.plt.subplots(figsize=(5, 3.4))
D = np.linspace(mode.D0 - 1.0, mode.D0 + 0.5, 100)
ax.plot(D, 100 / (1 + np.exp(-(mode.alpha + C.BETA_PER_KM * D))), color="#1a5fb4")
ax.scatter([mode.D0], [100 * C.P_BIKE_2025], color="k", zorder=5, label="today (calibration point)")
ax.set_xlabel(r"mean cyclist distance $\bar{D}^b$ (km)"), ax.set_ylabel(r"bike share $P^b$ (%)")
ax.legend(frameon=False, fontsize=8), ax.set_title("Binary logit mode choice")
print("figure:", P.save(fig, "fig2_mode_choice.png"))

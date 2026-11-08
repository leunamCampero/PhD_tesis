"""
Step 2 - Pre-upgrade car user equilibrium (Problem 1, Fig. 2 of the paper).

1. Demand calibration.  The EMD matrix contains only the trips made inside the city; the
   roads also carry exchange and through traffic.  The peak-hour car OD matrix is multiplied
   by m so that the modelled flows on the roads next to DoS 1-3 bike lanes match, on average
   (length-weighted), the GTLVille traffic of Table I:  two-way peak flow = 10 % x AADT(DoS).
   m is found by a fixed-point iteration m <- m / ratio(m) (UE at gap 1e-3, warm started).
2. UE at gap 1e-4 with conjugate Frank-Wolfe -> phi* (kept fixed in Problem 2).

Outputs: results/02_ue_original.npz (flows phi*, times, OD times, m),
         results/02_ue_original.json, figures/fig2_original_car_flows.png,
         figures/fig2b_bpr_and_convergence.png
Run time: ~3-5 min
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from itsc_cycling import config as C, plotting as P, utils  # noqa: E402
from itsc_cycling.bike import BikeNetwork  # noqa: E402
from itsc_cycling.cars import CarNetwork  # noqa: E402
from itsc_cycling.coupling import Coupling  # noqa: E402

utils.ensure_dirs()
bike, car = BikeNetwork(), CarNetwork(1.0)
cpl = Coupling(bike, car)

# ---- roads next to DoS 1-3 bike lanes and their GTLVille peak flow ------------------------
target = {}
for e in np.flatnonzero((bike.car_edge >= 0) & (bike.dos0 <= 3)):
    r = car.row_of_edge_id[bike.car_edge[e]]
    target[r] = max(target.get(r, 0), C.PEAK_HOUR_FACTOR * C.AADT[int(bike.dos0[e])])
roads = np.array(sorted(target))
tgt = np.array([target[r] for r in roads])
L = car.length[roads]
print(f"calibration on {len(roads)} roads ({L.sum() / 1000:.0f} km) next to DoS 1-3 bike lanes, "
      f"target length-weighted two-way peak flow {np.sum(L * tgt) / L.sum():.0f} veh/h")

# ---- 1. demand multiplier m ---------------------------------------------------------------
m, x, hist = 1.0, None, []
t0 = time.time()
for it in range(8):
    car.set_demand_multiplier(m)
    ue = car.solve_ue(rel_gap=1e-3, x0=x)
    flow2 = car.two_way(ue["x"])[roads]
    ratio = float(np.sum(L * flow2) / np.sum(L * tgt))
    hist.append(dict(m=m, ratio=ratio, TTT=ue["TTT"], iterations=ue["iterations"]))
    print(f"  m = {m:6.3f}: modelled / GTLVille = {ratio:5.3f}  (UE {ue['iterations']} it., "
          f"{time.time() - t0:.0f} s)")
    if abs(ratio - 1) < 0.01:
        break
    x = ue["x"] / ratio                                   # warm start for the new demand
    m = float(np.clip(m / ratio, *C.DEMAND_MULTIPLIER_BOUNDS))

# ---- 2. accurate UE --------------------------------------------------------------------------
car.set_demand_multiplier(m)
ue = car.solve_ue(rel_gap=C.UE_REL_GAP, x0=ue["x"], verbose=True)
cold = car.solve_ue(rel_gap=C.UE_REL_GAP)                 # cold start, for the convergence plot
utils.save_ue(utils.UE_FILE, ue, m)

x, t = ue["x"], ue["t"]
vc = x / car.capacity0
free = float(np.sum(x * car.t0)) / 60
out = dict(
    demand_multiplier=m, calibration=hist, peak_hour_car_trips=float(car.Q.sum()),
    UE=dict(TTT_veh_h=ue["TTT"], free_flow_TTT_veh_h=free, congestion_delay_pct=100 * (ue["TTT"] / free - 1),
            rel_gap=ue["gap"], iterations=ue["iterations"], seconds=ue["seconds"],
            cold_start_iterations=cold["iterations"], cold_start_seconds=cold["seconds"]),
    vc=dict(mean_loaded=float(vc[x > 1].mean()), share_arcs_above_0_8=float((vc > 0.8).mean()),
            share_arcs_above_1=float((vc > 1).mean()), max=float(vc.max())),
    mean_od_time_min=float(np.sum(car.Q * ue["od_time"]) / car.Q.sum()),
    share_trips_above_20_min=float(car.Q[ue["od_time"] > 20].sum() / car.Q.sum()),
)
utils.save_json(out, C.RESULTS / "02_ue_original.json")
print(f"\ndemand multiplier m = {m:.3f}  ->  {car.Q.sum():,.0f} car trips in the peak hour")
print(f"UE: TTT = {ue['TTT']:,.0f} veh.h (free flow {free:,.0f}, delay +{out['UE']['congestion_delay_pct']:.1f} %), "
      f"gap {ue['gap']:.1e} in {ue['iterations']} it. ({ue['seconds']:.0f} s; cold start "
      f"{cold['iterations']} it., {cold['seconds']:.0f} s)")
print(f"v/c: mean {out['vc']['mean_loaded']:.2f}, > 0.8 on {100 * out['vc']['share_arcs_above_0_8']:.1f} % "
      f"of arcs, max {out['vc']['max']:.2f};  mean OD time {out['mean_od_time_min']:.1f} min")

# ---- figures ---------------------------------------------------------------------------------
fig, ax = P.plt.subplots(figsize=(6, 6))
f2 = car.two_way(x)
P.car_map(ax, car, f2 / f2.max(), cmap="Reds", lw=1.0, label="normalised car flow",
          title="Pre-upgrade user equilibrium (peak hour)")
P.iris_boundaries(ax, lw=0.2)
print("figure:", P.save(fig, "fig2_original_car_flows.png"))

fig, axs = P.plt.subplots(1, 2, figsize=(9, 3.2))
r = np.linspace(0, 1.6, 100)
axs[0].plot(r, 1 + C.BPR_MU * r ** C.BPR_SIGMA, color="k")
axs[0].plot(r, 1 + C.BPR_MU * (r / C.CAPACITY_FACTOR) ** C.BPR_SIGMA, color="#d73027",
            label=r"capacity $\times$ 1/2 (new separated bike lane)")
axs[0].set_xlabel(r"$\phi_a / \theta_a$ (original capacity)"), axs[0].set_ylabel(r"$t_a / t_a^0$")
axs[0].set_ylim(0.9, 5), axs[0].legend(frameon=False, fontsize=8), axs[0].set_title("BPR function")
axs[1].semilogy(np.arange(1, len(cold["history"]) + 1), cold["history"], color="#1a5fb4",
                label="conjugate Frank-Wolfe")
axs[1].axhline(C.UE_REL_GAP, color="0.5", ls=":")
axs[1].set_xlabel("iteration"), axs[1].set_ylabel("relative gap"), axs[1].set_title("UE convergence")
print("figure:", P.save(fig, "fig2b_bpr_and_convergence.png"))

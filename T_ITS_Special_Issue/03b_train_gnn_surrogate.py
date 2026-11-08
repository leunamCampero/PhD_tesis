"""
Step 3b - Graph neural network surrogate of the car UE, compared with the PCA + ridge model of
step 3 (which is kept: results/03_surrogate.pkl is not modified).

* same 1,000 UE samples (step 2) and the same 150 test samples as step 3 (seed 0)
* GNN on the car graph (src/si_cycling/gnn.py), trained on the other 850 samples
  (15 % of them for early stopping)
* compared on the test samples: ridge, GNN and the average of both ("ensemble"), each followed by
  the feasibility projection and by 0, 1 or 3 Frank-Wolfe correction iterations
* the best combination is written to results/03b_surrogate_choice.json and used by the optimiser
  when config.SURROGATE_MODEL = "auto": the fewest correction iterations whose mean |car TTT error|
  is below SURROGATE_TTT_TOLERANCE, then the smallest error

Outputs: results/03b_surrogate_gnn.pt, results/03b_gnn_metrics.json,
         results/03b_surrogate_choice.json, figures/fig3b_gnn_vs_ridge.png
Run time: ~2-3 h on CPU (training), ~20 min (comparison)
Options:  --skip-training  (re-use results/03b_surrogate_gnn.pt)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from si_cycling import config as C, plotting as P, utils  # noqa: E402
from si_cycling.gnn import GNNSurrogate  # noqa: E402
from si_cycling.surrogate import CarSurrogate, FlowProjector  # noqa: E402

GNN_FILE = C.RESULTS / "03b_surrogate_gnn.pt"
ITERS = (0, 1, 3)


def flow_metrics(pred, true, dev_true, min_flow=10.0):
    err = pred - true
    ok = true > min_flow
    rel = np.abs(err[ok]) / true[ok]
    return dict(MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                R2=float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)),
                R2_deviation=float(1 - np.sum(err ** 2) / np.sum((dev_true - dev_true.mean(0)) ** 2)),
                within_5pct=float((rel <= 0.05).mean()), within_10pct=float((rel <= 0.10).mean()),
                p95_abs_error=float(np.quantile(np.abs(err), 0.95)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="smoke test: 48 samples, 1 epoch, 10 tests, nothing saved")
    args = ap.parse_args()
    t0 = time.time()
    from si_cycling.problems import MultiModal
    z = np.load(C.RESULTS / "02_samples.npz")
    red, scale, X = z["red"].astype(float), z["scale"], z["x"].astype(float)
    n = len(scale)
    MM = MultiModal(surrogate=False)
    car, proj = MM.car, FlowProjector(MM.car)
    idx = np.random.default_rng(0).permutation(n)              # same split as step 3
    test, trval = idx[:int(0.15 * n)], idx[int(0.15 * n):]
    if args.quick:
        test, trval, args.epochs = test[:10], trval[:48], 1

    # ---- training -----------------------------------------------------------------------------
    if args.skip_training and GNN_FILE.exists() and not args.quick:
        gnn = GNNSurrogate.load(GNN_FILE)
        info = dict(val_R2_deviation=gnn.val_R2_deviation, epochs=len(gnn.history))
    else:
        print(f"training the GNN on {len(trval)} samples ({n - len(trval)} kept for the test)")
        gnn = GNNSurrogate(car, MM.phi0, MM.cpl.roads)
        info = gnn.fit(red[trval], scale[trval], X[trval], epochs=args.epochs)
        if not args.quick:
            gnn.save(GNN_FILE)
    ridge = CarSurrogate.load(C.RESULTS / "03_surrogate.pkl")
    print(f"GNN: {int(gnn.active.sum())} active arcs, validation R2 of the deviation {info['val_R2_deviation']:.3f}")

    # ---- comparison on the test samples ------------------------------------------------------
    raw = {"ridge": ridge.predict_raw(red[test], scale[test])}
    t1 = time.time()
    raw["gnn"] = gnn.predict_raw(red[test], scale[test])
    gnn_sec = (time.time() - t1) / len(test)
    raw["ensemble"] = 0.5 * (raw["ridge"] + raw["gnn"])
    dev_t = X[test] - np.outer(scale[test], MM.phi0)
    res, ttt_err = {}, {}
    for name, R in raw.items():
        proj_flows = np.array([proj.project(R[i], proj.balance(scale[r])) for i, r in enumerate(test)])
        res[name] = dict(flows=flow_metrics(proj_flows, X[test], dev_t))
        for it in ITERS:
            err, ti, sec = [], [], 0.0
            for i, r in enumerate(test):
                cap = car.capacity0.copy()
                cap[MM.cpl.roads] *= 1 - red[r]
                cap[MM.cpl.roads + car.E] *= 1 - red[r]
                x = proj_flows[i]
                if it:
                    ts = time.time()
                    x = car.solve_ue(cap=cap, scale=scale[r], x0=x, rel_gap=1e-9, max_iter=it)["x"]
                    sec += time.time() - ts
                Tt = car.total_travel_time(X[r], cap)
                err.append(car.total_travel_time(x, cap) / Tt - 1)
                ti.append(MM.suf.J_TI_car(MM.car_od_times(car.time(x, cap)))
                          - MM.suf.J_TI_car(MM.car_od_times(car.time(X[r], cap))))
            err = np.array(err)
            ttt_err[(name, it)] = err
            res[name][f"iters_{it}"] = dict(TTT_mean=float(err.mean()), TTT_mean_abs=float(np.abs(err).mean()),
                                            TTT_max_abs=float(np.abs(err).max()),
                                            J_TI_car_abs_error=float(np.mean(np.abs(ti))),
                                            correction_seconds=sec / len(test))
            print(f"  {name:8s} + {it} FW it.: car TTT error mean {100 * err.mean():+.2f} %, mean |.| "
                  f"{100 * np.abs(err).mean():.2f} %, max |.| {100 * np.abs(err).max():.2f} %", flush=True)
        f = res[name]["flows"]
        print(f"  {name:8s}: R2 {f['R2']:.4f}, R2 of the deviation {f['R2_deviation']:.3f}, "
              f"within 10 % {100 * f['within_10pct']:.1f} %")

    # ---- choice ----------------------------------------------------------------------------------
    tol = C.SURROGATE_TTT_TOLERANCE
    cands = sorted(((it, res[m][f"iters_{it}"]["TTT_mean_abs"], m) for m in res for it in ITERS))
    ok = [c for c in cands if c[1] <= tol]
    it, e, model = min(ok, key=lambda c: (c[0], c[1])) if ok else min(cands, key=lambda c: c[1])
    choice = dict(model=model, iterations=int(it), TTT_mean_abs_error=e, tolerance=tol)
    if args.quick:
        print(f"quick test OK (choice would be {choice}); nothing saved")
        sys.exit(0)
    utils.save_json(choice, C.RESULTS / "03b_surrogate_choice.json")
    utils.save_json(dict(test_samples=len(test), gnn_training=dict(
        val_R2_deviation=info["val_R2_deviation"], history=getattr(gnn, "history", []),
        hidden=gnn.hidden, layers=gnn.layers, active_arcs=int(gnn.active.sum())),
        gnn_prediction_seconds=gnn_sec, results=res, choice=choice, minutes=(time.time() - t0) / 60),
        C.RESULTS / "03b_gnn_metrics.json")
    print(f"\nchoice: {model} + {it} Frank-Wolfe iterations (mean |TTT error| {100 * e:.2f} %)")

    # ---- figure ----------------------------------------------------------------------------------
    col = {"ridge": "#bbbbbb", "gnn": "#1a5fb4", "ensemble": "#e69f00"}
    fig, axs = P.plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    h = gnn.history
    if h:
        axs[0].plot([r["epoch"] for r in h], [r["val_R2_deviation"] for r in h], "o-", color=col["gnn"], ms=3,
                    label="GNN (validation)")
        axs[0].axhline(res["ridge"]["flows"]["R2_deviation"], color="k", ls="--", lw=0.8, label="ridge (test)")
        axs[0].set_xlabel("epoch"), axs[0].set_ylabel(r"$R^2$ of $\varphi - s\,\varphi_0$")
        axs[0].set_title("GNN training", fontsize=10), axs[0].legend(frameon=False, fontsize=8)
    w = 0.25
    for k, m in enumerate(res):
        v = [100 * res[m][f"iters_{it}"]["TTT_mean_abs"] for it in ITERS]
        axs[1].bar(np.arange(len(ITERS)) + (k - 1) * w, v, w, color=col[m], label=m)
    axs[1].set_xticks(range(len(ITERS)), [f"{it} FW it." for it in ITERS])
    axs[1].set_ylabel("mean |car TTT error| (%)"), axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title("error on the car total travel time (test)", fontsize=10)
    bins = np.linspace(-3, 10, 40)
    for m in res:
        axs[2].hist(100 * ttt_err[(m, 0)], bins, color=col[m], alpha=0.6, label=m)
    axs[2].axvline(0, color="k", lw=0.8)
    axs[2].set_xlabel("car TTT error, surrogate + projection only (%)"), axs[2].set_ylabel("test samples")
    axs[2].legend(frameon=False, fontsize=8), axs[2].set_title("without correction iterations", fontsize=10)
    print("figure:", P.save(fig, "fig3b_gnn_vs_ridge.png"))
    print(f"done in {(time.time() - t0) / 60:.1f} min")

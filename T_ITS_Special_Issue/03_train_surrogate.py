"""
Step 3 - Train and validate the car-UE surrogate (Section II-G, Fig. 8 of the paper).

* samples of step 2 split 70 / 15 / 15 % into training, validation (model choice) and test
* PCA on the flow deviations + {ridge, MLP}; the better one on the validation set is kept
* test metrics on the projected flows (as in the paper): MAE, RMSE, R2, share of link flows
  within +-5 % / +-10 %, 95th / 99th percentile of the absolute error; and the error on the
  quantities the optimiser uses (car TTT, mean OD time, J_TI^c), for the surrogate alone and
  with the SURROGATE_UE_ITERS Frank-Wolfe correction iterations used by the optimiser
* learning curve: test R2 versus the number of training samples

Outputs: results/03_surrogate.pkl, results/03_surrogate_metrics.json,
         figures/fig3_surrogate.png
Run time: ~5-10 min
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402

from si_cycling import config as C, plotting as P, utils  # noqa: E402
from si_cycling.surrogate import CarSurrogate, FlowProjector  # noqa: E402


def metrics(pred, true, min_flow=10.0):
    err = pred - true
    ok = true > min_flow                         # relative errors on loaded links only
    rel = np.abs(err[ok]) / true[ok]
    return dict(MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                R2=float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)),
                within_5pct=float((rel <= 0.05).mean()), within_10pct=float((rel <= 0.10).mean()),
                p95_abs_error=float(np.quantile(np.abs(err), 0.95)),
                p99_abs_error=float(np.quantile(np.abs(err), 0.99)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-model", action="store_true", help="recompute metrics / figure, keep the saved model")
    args = ap.parse_args()
    t0 = time.time()
    from si_cycling.problems import MultiModal
    z = np.load(C.RESULTS / "02_samples.npz")
    red, scale, X = z["red"].astype(float), z["scale"], z["x"].astype(float)
    n = len(scale)
    MM = MultiModal(surrogate=False)
    car, proj = MM.car, FlowProjector(MM.car)
    assert np.array_equal(z["cand_roads"], MM.cpl.roads), "samples were made with another capacity rule"
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_test = int(0.15 * n)
    test, trval = idx[:n_test], idx[n_test:]
    print(f"{n} samples: {len(trval)} for training + validation, {n_test} for test")

    def evaluate(sur, rows):
        """Projected predictions and derived quantities for the samples ``rows``."""
        pred = sur.predict_raw(red[rows], scale[rows])
        out = []
        for i, r in enumerate(rows):
            phi = proj.project(pred[i], proj.balance(scale[r]))
            out.append(phi)
        return np.array(out)

    # R2 of the *deviation* from the scaled pre-upgrade UE (what the model has to learn);
    # the R2 on the flows themselves (> 0.99) is dominated by the differences between links
    dev_t = X[test] - np.outer(scale[test], MM.phi0)

    def r2_dev(pred):
        return float(1 - np.sum((pred - X[test]) ** 2) / np.sum((dev_t - dev_t.mean(0)) ** 2))

    # ---- learning curve -------------------------------------------------------------------
    curve = []
    for m in [s for s in (100, 200, 400, 700) if s < len(trval)] + [len(trval)]:
        sur = CarSurrogate(MM.phi0, MM.cpl.roads)
        sur.fit(red[trval[:m]], scale[trval[:m]], X[trval[:m]], verbose=False)
        p = evaluate(sur, test)
        curve.append(dict(n_train=int(m), R2=metrics(p, X[test])["R2"], R2_deviation=r2_dev(p), model=sur.kind))
        print(f"  {m:5d} training samples: test R2 = {curve[-1]['R2']:.4f}, R2 of the deviation = "
              f"{curve[-1]['R2_deviation']:.3f} ({sur.kind})", flush=True)
    print(f"final model on {len(trval)} samples:")
    sur = CarSurrogate(MM.phi0, MM.cpl.roads)
    info = sur.fit(red[trval], scale[trval], X[trval])
    if not args.keep_model:
        sur.save(C.RESULTS / "03_surrogate.pkl")

    # ---- test metrics -----------------------------------------------------------------------
    pred = evaluate(sur, test)
    raw = sur.predict_raw(red[test], scale[test])
    m_proj, m_raw = metrics(pred, X[test]), metrics(raw, X[test])
    bal = [np.abs(proj.B @ raw[i] - proj.balance(scale[r])).max() for i, r in enumerate(test)]
    R2_dev = r2_dev(pred)

    def derived(flows_of):
        """Relative error of car TTT and mean OD time, abs. error of J_TI^c, for the test samples."""
        TTT_err, od_err, ti_err = [], [], []
        for i, r in enumerate(test):
            cap = car.capacity0.copy()
            cap[MM.cpl.roads] *= 1 - red[r]
            cap[MM.cpl.roads + car.E] *= 1 - red[r]
            xp = flows_of(i, r, cap)
            Tp, Tt = car.total_travel_time(xp, cap), car.total_travel_time(X[r], cap)
            TTT_err.append((Tp - Tt) / Tt)
            odp, odt = MM.car_od_times(car.time(xp, cap)), MM.car_od_times(car.time(X[r], cap))
            od_err.append(np.sum(car.Q * (odp - odt)) / np.sum(car.Q * odt))
            ti_err.append(MM.suf.J_TI_car(odp) - MM.suf.J_TI_car(odt))
        TTT_err, od_err, ti_err = map(np.array, (TTT_err, od_err, ti_err))
        return dict(TTT_rel_error=dict(mean=float(TTT_err.mean()), mean_abs=float(np.abs(TTT_err).mean()),
                                       max_abs=float(np.abs(TTT_err).max())),
                    mean_od_time_rel_error_mean_abs=float(np.abs(od_err).mean()),
                    J_TI_car_abs_error_mean=float(np.abs(ti_err).mean())), TTT_err

    plain, e_plain = derived(lambda i, r, cap: pred[i])
    t1 = time.time()
    corr, e_corr = derived(lambda i, r, cap: car.solve_ue(cap=cap, scale=scale[r], x0=pred[i], rel_gap=1e-9,
                                                  max_iter=C.SURROGATE_UE_ITERS)["x"])
    corr_sec = (time.time() - t1) / len(test)
    res = dict(n_samples=n, n_train_val=len(trval), n_test=n_test, model=info["kind"],
               validation_R2=info["scores"], pca_components=sur.pca.n_components_, pca_explained=info["explained"],
               test_projected=m_proj, test_raw=m_raw, test_R2_of_deviation=R2_dev,
               raw_max_conservation_violation_veh_h=float(np.max(bal)),
               surrogate_only=plain, surrogate_plus_correction=corr,
               correction_iterations=C.SURROGATE_UE_ITERS, correction_seconds=corr_sec, learning_curve=curve,
               sample_stats=dict(mean_ue_iterations=float(z["it"].mean()), mean_ue_seconds=float(z["sec"].mean()),
                                 cpu_hours=float(z["sec"].sum() / 3600), max_gap=float(z["gap"].max())),
               minutes=(time.time() - t0) / 60)
    utils.save_json(res, C.RESULTS / "03_surrogate_metrics.json")
    print("\ntest set (projected flows):", {k: round(v, 4) for k, v in m_proj.items()})
    print("test set (before projection):", {k: round(v, 4) for k, v in m_raw.items()})
    print(f"flow conservation violated by up to {np.max(bal):.1f} veh/h before projection, 0 after")
    print(f"R2 of the deviation from s * phi_0: {R2_dev:.3f}")
    for lab, d in (("surrogate + projection", plain), (f"+ {C.SURROGATE_UE_ITERS} BFW iterations", corr)):
        e = d["TTT_rel_error"]
        print(f"{lab:24s}: car TTT error mean {100 * e['mean']:+.2f} %, mean |.| {100 * e['mean_abs']:.2f} %, "
              f"max |.| {100 * e['max_abs']:.2f} %; J_TI^c abs. error {d['J_TI_car_abs_error_mean']:.4f}")
    print(f"correction: {corr_sec:.1f} s per evaluation")

    # ---- figure -------------------------------------------------------------------------------
    fig, axs = P.plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axs[0].plot([c["n_train"] for c in curve], [c["R2_deviation"] for c in curve], "o-", color="#1a5fb4")
    axs[0].set_xlabel("training samples"), axs[0].set_ylabel(r"test $R^2$ of $\varphi - s\,\varphi_0$")
    axs[0].set_title(f"learning curve ({sur.kind}, {sur.pca.n_components_} PCA comp.)", fontsize=10)
    bins = np.linspace(min(e_plain.min(), e_corr.min(), 0) * 100, max(e_plain.max(), e_corr.max()) * 100, 30)
    axs[1].hist(100 * e_plain, bins, color="#bbbbbb", label="surrogate + projection")
    axs[1].hist(100 * e_corr, bins, color="#1a5fb4", alpha=0.8,
                label=f"+ {C.SURROGATE_UE_ITERS} Frank-Wolfe iterations")
    axs[1].axvline(0, color="k", lw=0.8)
    axs[1].set_xlabel("car TTT error vs exact UE (%)"), axs[1].set_ylabel("test samples")
    axs[1].set_title("error on the car total travel time", fontsize=10), axs[1].legend(frameon=False, fontsize=8)
    t, p = X[test].ravel(), pred.ravel()
    sel = np.random.default_rng(1).choice(len(t), min(60_000, len(t)), replace=False)
    axs[2].scatter(t[sel], p[sel], s=1, alpha=0.25, color="#1a5fb4")
    lim = [0, max(t.max(), p.max())]
    axs[2].plot(lim, lim, color="k", lw=0.8)
    axs[2].set_xlabel("UE flow (veh/h)"), axs[2].set_ylabel("surrogate + projection (veh/h)")
    axs[2].set_title(rf"test link flows, $R^2$ = {m_proj['R2']:.3f}", fontsize=10)
    print("figure:", P.save(fig, "fig3_surrogate.png"))

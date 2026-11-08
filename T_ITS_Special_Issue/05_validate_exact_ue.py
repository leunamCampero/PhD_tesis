"""
Step 5 - Re-evaluate the designs with the EXACT user equilibrium (no surrogate).

For today's network and every design of the utilitarian front and of the sufficientarian
front at lambda = 40 % (plus gamma* of the other lambdas):
    * all indicators with exact UE car flows   (what the paper reports)
    * surrogate vs exact: error on T_c, T_tot and J_TI^c at the optimum
Sensitivity of the car impact for the representative designs (gamma*, config.GAMMA_STAR_RULE):
    * "no mode shift": same design, car demand not reduced by the new cyclists
    * "paper omega = 1": a separated bike lane removes a full car lane (Eq. 17 literally)

Outputs: results/05_exact.csv, results/05_sensitivity.csv, results/05_arrays.npz
Run time: ~10 min on 8 cores
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from si_cycling import config as C, utils  # noqa: E402


@lru_cache(maxsize=1)
def ctx():
    from si_cycling.problems import MultiModal
    return MultiModal(surrogate=True)


def exact(job):
    name, kind, gamma, lam = job
    P = ctx()
    dos = P.bike.dos0 if kind == "original" else \
        utils.load_solution(C.RESULTS / "solutions" / f"{utils.solution_name(kind, gamma, lam)}.csv")
    ex = P.evaluate(dos, exact=True) if kind != "original" else P.base
    sg = P.evaluate(dos)                                       # surrogate
    row = {k: v for k, v in ex.items() if not k.startswith("_") and k != "dos"}
    row.update(name=name, kind=kind, gamma=gamma, lam=lam, T_c_surrogate=sg["T_c"], T_tot_surrogate=sg["T_tot"],
               J_TI_c_surrogate=sg["J_TI_c"])
    return row, {k: v for k, v in ex["_arrays"].items()}


def sensitivity(job):
    name, kind, gamma, lam = job
    from si_cycling.coupling import Coupling
    P = ctx()
    dos = utils.load_solution(C.RESULTS / "solutions" / f"{utils.solution_name(kind, gamma, lam)}.csv")
    ev = P.bike.evaluate(dos)
    _, _, s_c = P.mode.scales(ev)
    out = dict(name=name, T_c0=P.base["T_c"])
    for label, cap, scale in (("model", P.cpl.capacity(dos), s_c),
                              ("no_mode_shift", P.cpl.capacity(dos), 1.0),
                              ("paper_omega1", Coupling(P.bike, P.car, omega=1.0).capacity(dos), s_c)):
        ue = P.car.solve_ue(cap=cap, scale=scale, x0=P.phi0 * scale)
        out[f"T_c_{label}"] = ue["TTT"]
        out[f"J_TI_c_{label}"] = P.suf.J_TI_car(ue["od_time"])
    return out


if __name__ == "__main__":
    t0 = time.time()
    runs = pd.read_csv(C.RESULTS / "04_runs.csv")
    stars = utils.load_json(C.RESULTS / "04_runs.json")["gamma_star"]
    jobs = [("original", "original", np.nan, None)]
    for g in C.GAMMAS:
        jobs.append((f"util_g{g:g}", "util", g, None))
        jobs.append((f"suff_lam0.4_g{g:g}", "suff", g, 0.4))
    for lam in C.LAMBDAS:                      # gamma* of the other fronts (both rules)
        if lam != 0.4:
            for g in sorted({utils.representative_gamma(stars, "suff", lam, rule) for rule in ("paper", "utopia")}):
                jobs.append((f"suff_lam{lam:g}_g{g:g}", "suff", g, lam))
    reps = [("utilitarian", "util", utils.representative_gamma(stars, "util"), None),
            ("sufficientarian", "suff", utils.representative_gamma(stars, "suff", 0.4), 0.4)]

    with ProcessPoolExecutor(utils.n_workers()) as ex:
        out = list(ex.map(exact, jobs))
        sens = list(ex.map(sensitivity, reps))
    df = pd.DataFrame([r for r, _ in out])
    df["T_c_surrogate_error_pct"] = 100 * (df.T_c_surrogate / df.T_c - 1)
    df.to_csv(C.RESULTS / "05_exact.csv", index=False)
    sdf = pd.DataFrame(sens)
    sdf.to_csv(C.RESULTS / "05_sensitivity.csv", index=False)
    arrays = {}
    names = {"original": "original", reps[0][0]: f"util_g{reps[0][2]:g}", reps[1][0]: f"suff_lam0.4_g{reps[1][2]:g}"}
    for label, nm in names.items():
        i = [r["name"] for r, _ in out].index(nm)
        for k, v in out[i][1].items():
            arrays[f"{label}__{k}"] = v
    np.savez_compressed(C.RESULTS / "05_arrays.npz", **arrays)

    print(df[["name", "J_E", "J_SI_0.4", "J_TI_b", "J_TI_c", "T_c", "T_b", "T_tot", "P_b",
              "T_c_surrogate_error_pct"]].round(4).to_string(index=False))
    print("\nsurrogate error on T_c at the designs: mean |.| "
          f"{df.T_c_surrogate_error_pct.abs().mean():.2f} %, max |.| {df.T_c_surrogate_error_pct.abs().max():.2f} %")
    print("\ncar impact of the representative designs (exact UE, % vs today):")
    for _, r in sdf.iterrows():
        print(f"  {r['name']:16s} model {100 * (r.T_c_model / r.T_c0 - 1):+6.2f} %   without mode shift "
              f"{100 * (r.T_c_no_mode_shift / r.T_c0 - 1):+6.2f} %   paper omega=1 "
              f"{100 * (r.T_c_paper_omega1 / r.T_c0 - 1):+6.2f} %")
    print(f"done in {(time.time() - t0) / 60:.1f} min")

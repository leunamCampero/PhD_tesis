"""
Choice of the representative designs GA-U and GA-S(lambda) compared in the figures (step 6), from the
exact-UE evaluation of every design (step 5b, results/05b_all_designs.csv).

config.REPRESENTATIVE_RULE
  "paper"    GA-U: gamma = LAMBDA_FIGURES + 0.1; GA-S(lambda): gamma = lambda + 0.1 (the paper's illustrative
             choice).  Weakness: GA-S(20 %) uses gamma = 0.3 and GA-U gamma = 0.5, so GA-S puts LESS weight on
             safety than GA-U and can look worse on path insufficiency (old Fig. 10).
  "utopia"   each front's point closest to its own utopia point (min-max normalised).
  "no_harm"  (default, CHANGE) GA-U is the utopia point of the utilitarian front.  GA-S(lambda) is the design
             of the lambda-front with the LOWEST path insufficiency J_SI(lambda) among those that
               - do not leave more cyclists above 20 min or more drivers above the delay threshold than
                 GA-U (tolerance NO_HARM_TOL_PP percentage points: below the model resolution), and
               - cost at most NO_HARM_MAX_TTOT more total travel time than GA-U;
             i.e. "the most sufficient network that harms no traveller group more than the utilitarian one
             at almost the same system cost" (sufficientarian 'no-harm' principle).  Falls back to "utopia"
             if no design satisfies the conditions.
"""
import numpy as np
import pandas as pd

from . import config as C


def _utopia(front, a, b):
    fa = (front[a] - front[a].min()) / max(front[a].max() - front[a].min(), 1e-12)
    fb = (front[b] - front[b].min()) / max(front[b].max() - front[b].min(), 1e-12)
    return front.loc[np.hypot(fa, fb).idxmin()]


def util_front(df):
    return df[df.kind == "util"].sort_values("gamma")


def suff_front(df, lam):
    return df[(df.kind == "suff") & np.isclose(df.lam, lam)].sort_values("gamma")


def representatives(df, rule=None):
    """dict: 'util' -> row of GA-U, lam -> row of GA-S(lam) for every lambda (rows of 05b_all_designs)."""
    rule = rule or C.REPRESENTATIVE_RULE
    U = util_front(df)
    if rule == "paper":
        gu = U[np.isclose(U.gamma, round(C.LAMBDA_FIGURES + 0.1, 2))].iloc[0]
    else:
        gu = _utopia(U, "J_E", "T_tot")
    out = {"util": gu}
    for lam in C.LAMBDAS:
        S = suff_front(df, lam)
        a = f"J_SI_{lam:g}"
        if rule == "paper":
            out[lam] = S[np.isclose(S.gamma, round(lam + 0.1, 2))].iloc[0]
            continue
        if rule == "no_harm":
            tol = C.NO_HARM_TOL_PP
            ok = S[(S.bike_above_20 <= gu.bike_above_20 + tol) & (S.car_above_thr <= gu.car_above_thr + tol)
                   & (S.T_tot <= gu.T_tot * (1 + C.NO_HARM_MAX_TTOT))]
            if len(ok):
                out[lam] = ok.loc[ok[a].idxmin()]
                continue
        out[lam] = _utopia(S, a, "J_TI")
    return out


def table(df, reps):
    """One line per lambda: GA-S vs GA-U (for the log and the README)."""
    gu = reps["util"]
    rows = []
    for lam in C.LAMBDAS:
        gs = reps[lam]
        rows.append(dict(lam=lam, GA_S=gs["name"], GA_U=gu["name"],
                         J_SI_U=gu[f"J_SI_{lam:g}"], J_SI_S=gs[f"J_SI_{lam:g}"],
                         below_U=gu[f"path_below_{lam:g}"], below_S=gs[f"path_below_{lam:g}"],
                         bike20_U=gu.bike_above_20, bike20_S=gs.bike_above_20,
                         car_U=gu.car_above_thr, car_S=gs.car_above_thr,
                         J_E_U=gu.J_E, J_E_S=gs.J_E, T_tot_change_pct=100 * (gs.T_tot / gu.T_tot - 1)))
    return pd.DataFrame(rows)

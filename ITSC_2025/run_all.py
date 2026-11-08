"""Run every step of this project, in order.

    python run_all.py                 # all steps
    python run_all.py --list          # show the steps and their run times
    python run_all.py --from 3        # start at step 3 (outputs of the earlier steps must exist)
    python run_all.py --only 4        # run a single step
    python run_all.py --skip 5 tests  # skip some steps

Run it from the activated conda environment (cycling-ecc). It stops at the first step that fails.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (step id, command, what it does, approximate run time)
STEPS = [
    ('1', ['python', '01_networks_and_demand.py'],
     'bike and car layers, coupling, budgets (Fig. 1)', '~10 s'),
    ('2', ['python', '02_car_user_equilibrium.py'],
     'demand calibration and car UE (Fig. 2)', '~3 min'),
    ('3', ['python', '03_pareto_trade_off.py'],
     'GA Pareto trade-off (Figs. 3-4)', '~60 min'),
    ('4', ['python', '04_post_upgrade_ue.py'],
     'post-upgrade UE (Fig. 5)', '~10 min'),
    ('5', ['python', '05_figures.py'],
     'figures and summary table', '~1 min'),
    ('tests', ['python', 'tests/run_tests.py'],
     'unit tests', '~5 min'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="only list the steps")
    ap.add_argument("--from", dest="start", help="first step to run")
    ap.add_argument("--only", help="run only this step")
    ap.add_argument("--skip", nargs="*", default=[], help="steps to skip")
    a = ap.parse_args()
    ids = [s[0] for s in STEPS]
    for x in [a.start, a.only, *a.skip]:
        if x is not None and x not in ids:
            ap.error(f"unknown step {x!r}; steps are {ids}")
    todo = STEPS
    if a.start:
        todo = todo[ids.index(a.start):]
    if a.only:
        todo = [s for s in STEPS if s[0] == a.only]
    todo = [s for s in todo if s[0] not in a.skip]
    if a.list:
        for sid, cmd, what, rt in STEPS:
            print(f"{sid:>6}  {' '.join(cmd[-1:] if 'nbconvert' in cmd else cmd):45s} {what}  ({rt})")
        return
    t_all = time.time()
    for sid, cmd, what, rt in todo:
        print(f"\n=== step {sid}: {what}  (expected {rt})", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable if c == "python" else c for c in cmd], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"step {sid} failed (exit code {r.returncode}); fix it and resume with --from {sid}")
        print(f"=== step {sid} done in {time.time() - t0:.0f} s", flush=True)
    print(f"\nall steps done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()

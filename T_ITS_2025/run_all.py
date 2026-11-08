"""Run every step of this project, in order.

    python run_all.py                 # all steps
    python run_all.py --list          # show the steps and their run times
    python run_all.py --from 3        # start at step 3 (outputs of the earlier steps must exist)
    python run_all.py --only 4        # run a single step
    python run_all.py --skip 5 tests  # skip some steps

Run it from the activated conda environment (cycling-tits). It stops at the first step that fails.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (step id, command, what it does, approximate run time)
STEPS = [
    ('1', ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', 'notebooks/01_network_metrics.ipynb'],
     'graph metrics of the 14 cities', '~10 min'),
    ('2', ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', 'notebooks/02_model_regression_sdv.ipynb'],
     'OLS, SDV, NNLS prediction model', 'seconds'),
    ('3', ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', 'notebooks/03_benchmark_example.ipynb'],
     'all methods on the benchmark network', '~20 min'),
    ('4', ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', 'notebooks/04_sioux_falls.ipynb'],
     'all methods on Sioux Falls', '~15 min'),
    ('5', ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', 'notebooks/05_grenoble.ipynb'],
     'all methods on Grenoble', '1-2 days'),
    ('6', ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=-1', 'notebooks/06_fourteen_cities_mlcc.ipynb'],
     'MLCC on the 14 cities', '~1 h'),
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

"""Generate the numbered notebooks in ``notebooks/`` (source of truth for their content).

Regenerating a notebook removes its stored outputs; run it again afterwards.

    python tools/make_notebooks.py            # all notebooks
    python tools/make_notebooks.py 03 05      # only some of them
"""
import sys
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))   # the cycling_tits package

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from cycling_tits import CITIES, FIGURES, RESULTS, load_network, load_survey, pretty, Model, alphas
FIGURES.mkdir(exist_ok=True)"""

PRESET_MD = """**Choose the model variant** (see README, section *Two model variants*):

* `"dos"` (default) - integer DoS weights, linear models in $s_\\mathcal{G}$ and $l_\\mathcal{G}$ (ITSC-2024 calibration), exponential cost per metre.
* `"cscs"` - CycleRAP/CSCS weights, NNLS model $\\tilde\\theta=[0.41, 0.58, 0.45, 0.57]$ of Eq. (10), Cerema costs.
  With this model safety and comfort percentages are identical, because the prediction only depends on $l_\\mathcal{G}$."""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


ONLY = sys.argv[1:]   # optional: names of the notebooks to (re)generate


def write(name, cells):
    if ONLY and not any(name.startswith(o) for o in ONLY):
        return
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "cycling-tits", "display_name": "Python (cycling-tits)", "language": "python"}
    nbf.write(nb, NB / name)
    print("wrote", name)


# ----------------------------------------------------------------------------- 01
write("01_network_metrics.ipynb", [
    md("""# 01 - Cycling networks and graph metrics

Computes the graph metrics $s_\\mathcal{G}, c_\\mathcal{G}, l_\\mathcal{G}$ of the 14 cities (Section II-C, Table II of the paper), the size, length and budget $B_{\\max}$ of each network (Table VI) and draws the Grenoble network coloured by CSCS (Fig. 3).

The 14 networks (`data/networks/*.csv`) come from an OpenStreetMap extraction (2024) with the DoS classification of the ITSC-2024 paper, simplified. Re-extracting them from today's OSM would give different networks, so they are frozen here.

Closeness centrality takes a few minutes on the big cities (Paris), so the results are cached in `results/metrics_cities.csv`; set `RECOMPUTE = True` to recompute."""),
    code(SETUP),
    code("""from cycling_tits.metrics import strength, closeness, largest_connected_component
from cycling_tits.model import max_budget
from tqdm.auto import tqdm

RECOMPUTE = False
cache = RESULTS / "metrics_cities.csv"
if cache.exists() and not RECOMPUTE:
    metrics = pd.read_csv(cache, index_col="city")
else:
    rows = []
    for city in tqdm(CITIES):
        G = load_network(city)
        rows.append(dict(city=city, V=G.number_of_nodes(), E=G.number_of_edges(),
                         length_m=sum(d["distance"] for *_, d in G.edges(data=True)),
                         B_max_eur=max_budget(G, "cerema"),
                         s_G=strength(G, "cscs"), c_G=closeness(G, "cscs"),
                         l_G=largest_connected_component(G, "cscs")))
    metrics = pd.DataFrame(rows).set_index("city")
    RESULTS.mkdir(exist_ok=True)
    metrics.to_csv(cache)
metrics"""),
    md("## Table II - metrics and perceived safety / comfort (Baromètre Vélo, scaled to [0, 1])"),
    code("""survey = load_survey()
table2 = metrics[["s_G", "c_G", "l_G"]].round(4).join(survey[["y_safety", "y_comfort"]])
table2.index = [pretty(c) for c in table2.index]
table2"""),
    md("## Table VI - networks and budget $B_{\\max}$ (Cerema costs, EUR)"),
    code("""table6 = metrics[["V", "E", "length_m", "B_max_eur"]].copy()
table6["length_m"] = table6["length_m"].round(0).astype(int)
table6["B_max_eur"] = table6["B_max_eur"].astype(int).map("{:,}".format)
table6.index = [pretty(c) for c in table6.index]
table6"""),
    md("""## Fig. 3 - Grenoble cycling network coloured by CSCS"""),
    code("""from cycling_tits.plotting import city_map
G = load_network("Grenoble")
city_map(G, outfile=FIGURES / "Map_and_scale.png");"""),
    md("""## Sioux Falls test network (Fig. 8)
DoS values were drawn at random once (2024) and are frozen in `data/networks/Sioux_Falls_*.csv`."""),
    code("""from cycling_tits.plotting import sioux_falls_network
sioux_falls_network(load_network("Sioux_Falls"), outfile=FIGURES / "Sioux_Falls.png");"""),
])

# ----------------------------------------------------------------------------- 02
write("02_model_regression_sdv.ipynb", [
    md("""# 02 - Prediction model: OLS significance test, SDV enrichment and NNLS

Computes the OLS significance test (Table III: $P$- and $t$-values), the NNLS parameters $\\tilde\\theta$ of Eq. (10) and the predicted vs actual safety / comfort on the 4 test cities (Fig. 6).

Pipeline (Fig. 4 of the paper):
1. OLS on the 14 cities with $[1, s_\\mathcal{G}, c_\\mathcal{G}, l_\\mathcal{G}]$ -> only $l_\\mathcal{G}$ is significant.
2. Split: 10 training cities / 4 test cities (Annecy, Bordeaux, Chambery, Dunkerque).
3. Enrich the training set with 990 synthetic samples from the Synthetic Data Vault (SDV, Gaussian copula).
4. Non-negative least squares on the enriched training set -> $\\tilde\\theta$.
5. Compare with an AutoGluon (Auto-ML) ensemble on the test cities.

The synthetic samples of the 2024 run are stored in `data/model/sdv_synthetic_*.csv`. SDV sampling is random, so set `REGENERATE_SDV = True` to draw new ones (results will vary slightly - this is expected)."""),
    code(SETUP),
    code("""import statsmodels.api as sm
metrics = pd.read_csv(RESULTS / "metrics_cities.csv", index_col="city")   # from notebook 01
survey = load_survey()
df = metrics[["s_G", "c_G", "l_G"]].join(survey)   # full precision (as in the original regression)
df"""),
    md("## Table III - OLS regression on the 14 cities"),
    code("""rows = []
for target in ["safety", "comfort"]:
    res = sm.OLS(df[f"y_{target}"], sm.add_constant(df[["s_G", "c_G", "l_G"]])).fit()
    for name in ["const", "s_G", "c_G", "l_G"]:
        rows.append(dict(target=target, predictor=name.replace("const", "Intercept"),
                         coefficient=res.params[name], abs_t=abs(res.tvalues[name]), p_value=res.pvalues[name]))
table3 = pd.DataFrame(rows).set_index(["target", "predictor"]).round(4)
table3"""),
    code("""from scipy import stats
print("critical |t| for alpha = 0.05, df = 10:", round(stats.t.ppf(0.975, 10), 3))
print("significant predictors:", table3[table3.p_value < 0.05].index.tolist())"""),
    md("## Synthetic Data Vault enrichment of the training set"),
    code("""REGENERATE_SDV = False      # True: draw new synthetic samples (needs the `sdv` package)
N_SYNTHETIC = 1000           # 10 real + 1000 synthetic training rows (as in the 2024 run)

train = df[df.split == "train"]
test = df[df.split == "test"]
cols = ["s_G", "l_G", "c_G"]

synthetic = {}
for target in ["safety", "comfort"]:
    real = train[cols + [f"y_{target}"]]
    if REGENERATE_SDV:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
        meta = SingleTableMetadata(); meta.detect_from_dataframe(real)
        synth = GaussianCopulaSynthesizer(meta); synth.fit(real)
        synthetic[target] = synth.sample(N_SYNTHETIC)
    else:
        synthetic[target] = pd.read_csv(RESULTS.parent / "data" / "model" / f"sdv_synthetic_{target}.csv").iloc[:N_SYNTHETIC]
    print(target, "synthetic rows:", len(synthetic[target]))
synthetic["safety"].describe().round(3)"""),
    md("""## NNLS on the enriched training set -> $\\tilde\\theta$ (Eq. 10)
The original code fitted `LinearRegression(positive=True)` (= NNLS with a free intercept) on $[s_\\mathcal{G}, l_\\mathcal{G}, c_\\mathcal{G}]$; the positivity constraint drives the coefficients of $s_\\mathcal{G}$ and $c_\\mathcal{G}$ to 0, leaving $y = \\theta_0 + \\theta_1 l_\\mathcal{G}$."""),
    code("""from sklearn.linear_model import LinearRegression
theta, pred = {}, {}
for target in ["safety", "comfort"]:
    y = f"y_{target}"
    data = pd.concat([train[cols + [y]], synthetic[target]], ignore_index=True)
    reg = LinearRegression(positive=True).fit(data[cols], data[y])
    theta[target] = dict(intercept=reg.intercept_, **dict(zip(cols, reg.coef_)))
    pred[target] = reg.predict(test[cols])
pd.DataFrame(theta).T.round(4)"""),
    code("""th = [theta["safety"]["intercept"], theta["safety"]["l_G"], theta["comfort"]["intercept"], theta["comfort"]["l_G"]]
print("theta_tilde              :", np.round(th, 5))
print("truncated to 2 decimals  :", np.floor(np.array(th) * 100) / 100, " (values used by the preset 'cscs')")"""),
    md("""## Fig. 6 - predictions on the 4 test cities
The AutoGluon predictions were produced in a separate (Colab) session; their values are stored in `data/model/automl_predictions.csv`. An optional cell at the end re-runs AutoGluon if it is installed (it is *not* part of `environment.yml`: heavy and conflicts with pygmo)."""),
    code("""from cycling_tits.plotting import prediction_bars
automl = pd.read_csv(RESULTS.parent / "data" / "model" / "automl_predictions.csv", index_col="city").loc[test.index]
mape = lambda a, b: float(np.mean(100 * np.abs(np.asarray(a) - np.asarray(b)) / np.asarray(a)))
for target in ["safety", "comfort"]:
    actual = test[f"y_{target}"].values
    print(f"{target:8s} mean abs. % error  NNLS: {mape(actual, pred[target]):.2f} %   Auto-ML: {mape(actual, automl[target]):.2f} %")
    prediction_bars(list(test.index), actual, pred[target], automl[target].values,
                    target.capitalize(), outfile=FIGURES / f"{target}_prediction.png")"""),
    code("""# Optional: re-run the Auto-ML ensemble (requires `pip install autogluon.tabular` in a separate env)
RUN_AUTOGLUON = False
if RUN_AUTOGLUON:
    from autogluon.tabular import TabularPredictor
    for target in ["safety", "comfort"]:
        y = f"y_{target}"
        data = pd.concat([train[cols + [y]], synthetic[target]], ignore_index=True)
        p = TabularPredictor(label=y, problem_type="regression", verbosity=0).fit(data, presets="best_quality")
        print(target, p.predict(test[cols]).round(4).to_dict())"""),
])


# ----------------------------------------------------------------------- 03-05
def comparison_notebook(fname, name, title, figs, with_optimal, heavy):
    methods = ["Optimal", "MLCC", "SGA", "SGA-MLCC", "Bayesian"] if with_optimal else ["MLCC", "SGA", "SGA-MLCC", "Bayesian"]
    intro = {
        "Benchmark": """Small benchmark network (10 nodes, 11 edges) where the exact optimum can be computed by brute force (138,240 configurations). Draws the configurations found for $\\alpha_7$ (Fig. 5), the comparison of the algorithms and the GA evolution (Fig. 7) and the computation times (Table V).""",
        "Sioux_Falls": """Sioux Falls network (24 nodes, 38 edges) with random DoS. Comparison of the algorithms (Fig. 11, Appendix C) and computation times (Table VIII). Runtime: about 1-2 hours for everything (Bayesian is the slowest).""",
        "Grenoble": """Real Grenoble cycling network (3,539 nodes, 5,065 edges). Comparison of the algorithms (Fig. 9) and computation times (Table VII).

**Runtime warning:** with the paper settings (80 generations x 50 individuals, 20 Bayesian calls) the GA and Bayesian runs take **many hours** (up to ~1.5 h per budget and method). Results are cached in `results/<preset>/`, so this only has to be done once. For a quick check set `QUICK = True` (5 generations, 10 Bayesian calls) - the numbers will then be worse than with the full settings.""",
    }[name]
    cells = [
        md(f"# {title}\n\n{intro}\n\n{PRESET_MD}\n\nThe GA and the Bayesian optimiser are stochastic: their results vary from run to run (seed fixed below for repeatability). MLCC and the brute-force optimum are deterministic."),
        code(SETUP),
        code(f"""from cycling_tits.experiments import compare_algorithms, percents, ga_histories, time_table
from cycling_tits.plotting import comparison_bars, ga_evolution

PRESET = "dos"                # or "cscs"
TARGETS = ["safety", "comfort"]
METHODS = {methods}
QUICK = False                  # True: very short GA / Bayesian runs, only to test the notebook
GENERATIONS, POP_SIZE, BAYES_CALLS = (5, 10, 10) if QUICK else (80, 50, 20)
SEED = 0
TAG = "_quick" if QUICK else ""
OUT = FIGURES / PRESET; OUT.mkdir(exist_ok=True)"""),
        code(f"""results = {{t: compare_algorithms("{name}", PRESET, t, METHODS, generations=GENERATIONS, pop_size=POP_SIZE,
                                bayes_calls=BAYES_CALLS, seed=SEED, tag=TAG)
           for t in TARGETS}}"""),
        md("## % of achievable improvement  $100\\,|\\hat y_{\\omega_0}-\\hat y(\\omega^*)| / |\\hat y_{FS}-\\hat y_{\\omega_0}|$"),
        code("""tables = {t: pd.DataFrame({m: percents(results[t], m) for m in METHODS}, index=[f"i={i}" for i in range(10)]).T.round(2)
          for t in TARGETS}
tables["safety"]"""),
        code("""tables["comfort"]"""),
        md(f"## {figs['bars']} (a) - comparison of the algorithms (plain = safety, hatched = comfort)"),
        code(f"""comparison_bars({{m: percents(results["safety"], m) for m in METHODS}},
                {{m: percents(results["comfort"], m) for m in METHODS}},
                outfile=OUT / "{figs['bars_file']}", methods=METHODS);"""),
        md(f"## {figs['bars']} (b), (c) - evolution of SGA and SGA-MLCC over the generations (safety)"),
        code(f"""ga_evolution(ga_histories(results["safety"], "SGA"), ga_histories(results["safety"], "SGA-MLCC"),
             outfile=OUT / "{name}_comparation_GA_LCC_safety.png",
             legend_outfile=OUT / "{name}_comparation_GA_LCC_safety_legend.png");"""),
        md(f"## {figs['table']} - computation time"),
        code("""time_table(results["safety"], METHODS)"""),
    ]
    if name == "Benchmark":
        cells += [
            md("## Fig. 5 - benchmark network and the configurations found for safety with $\\alpha_7 \\approx 0.36$"),
            code("""from cycling_tits.plotting import benchmark_network
G = load_network("Benchmark")
benchmark_network(G, outfile=OUT / "Plot_T_ITS_academic_example_original.png")
for m in METHODS:
    benchmark_network(G, results["safety"][m][7]["levels"], show_lengths=False, show_legend=False,
                      outfile=OUT / f"Plot_T_ITS_academic_example_{m.lower().replace('-', '_')}.png")
    plt.title(m)
print("B_max (Cerema) =", round(__import__('cycling_tits').max_budget(G, 'cerema')), "EUR")"""),
        ]
    write(fname, cells)


comparison_notebook("03_benchmark_example.ipynb", "Benchmark", "03 - Benchmark example (Section V)",
                    dict(bars="Fig. 7", bars_file="maximize_safety_academic_example_LCC.png", table="Table V"), True, False)
comparison_notebook("04_sioux_falls.ipynb", "Sioux_Falls", "04 - Sioux Falls network (Appendix C)",
                    dict(bars="Fig. 11", bars_file="maximize_safety_academic_example_Sioux_Falls.png", table="Table VIII"), False, False)
comparison_notebook("05_grenoble.ipynb", "Grenoble", "05 - Grenoble cycling network (Section VI)",
                    dict(bars="Fig. 9", bars_file="maximize_safety_Grenoble_comparasion.png", table="Table VII"), False, True)

# ----------------------------------------------------------------------------- 06
write("06_fourteen_cities_mlcc.ipynb", [
    md(f"""# 06 - MLCC on the 14 French cities (Section VII)

Computes the % of achievable safety / comfort for the 10 budgets $\\alpha_i$ (Fig. 12) and draws Grenoble upgraded with MLCC for $\\alpha_7$ (Fig. 10). MLCC is deterministic. Runtime: about 1 hour the first time (Paris alone takes ~25 min), then cached.

{PRESET_MD}"""),
    code(SETUP),
    code("""from cycling_tits.optimizers import run_mlcc, evaluate
from cycling_tits.plotting import fourteen_cities, city_map
from tqdm.auto import tqdm

PRESET = "dos"           # or "cscs"
OUT = FIGURES / PRESET; OUT.mkdir(exist_ok=True)
cache = RESULTS / PRESET / "fourteen_cities_mlcc.json"
cache.parent.mkdir(parents=True, exist_ok=True)

# MLCC only depends on the costs, not on the safety/comfort model, so it is run once per
# city and budget and both models are evaluated on the same upgrade plan.
if cache.exists():
    res = json.load(open(cache))
else:
    models = {t: Model(PRESET, t) for t in ["safety", "comfort"]}
    res = {t: {} for t in models}
    for city in tqdm(CITIES):
        G = load_network(city)
        runs = [run_mlcc(models["safety"], G, a) for a in alphas()]
        for t, model in models.items():
            y0 = model.predict(G)
            ys = [evaluate(model, G, r["levels"]) if t != "safety" else r["y"] for r in runs]
            res[t][city] = {"y0": y0, "y": ys, "percent": [model.percent(y, y0) for y in ys],
                            "time_s": [r["time_s"] for r in runs], "levels_alpha7": runs[7]["levels"]}
    json.dump(res, open(cache, "w"))"""),
    code("""pct = {t: pd.DataFrame({pretty(c): res[t][c]["percent"] for c in CITIES}, index=[f"i={i}" for i in range(10)]).T.round(2)
       for t in ["safety", "comfort"]}
pct["safety"]"""),
    md("## Fig. 12 - safety (top) and comfort (bottom)"),
    code("""fourteen_cities({c: res["safety"][c]["percent"] for c in CITIES}, "safety",
                outfile=OUT / "maximize_safety_fourteen.png", show_cities=False)
fourteen_cities({c: res["comfort"][c]["percent"] for c in CITIES}, "comfort",
                outfile=OUT / "maximize_comfort_fourteen.png", show_legend=False);"""),
    md("## Fig. 10 - Grenoble upgraded by MLCC for safety with $\\alpha_7 \\approx 0.36$ (upgraded links drawn thicker)"),
    code("""G = load_network("Grenoble")
city_map(G, res["safety"]["Grenoble"]["levels_alpha7"], highlight_changes=True,
         outfile=OUT / "Map_and_scale_changes.png");"""),
])

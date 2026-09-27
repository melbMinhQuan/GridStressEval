# GridStressEval

Customer-load violation models, experiments, and interactive visualizations.

## Layout

```text
gridstress/
  config.py          Shared parameters, paths and composition enumeration
  results.py         Saved-run loading and probability summaries
  models/
    one_interval/
      all_on.py      Single-interval All-On model
      exact.py       Active exact binomial On-Off calculation
      validation.py  Exact-model input validation
      reference/
        monte_carlo.py  Historical independent-sample model
        interactive.py Historical cached-sample model
    multi_interval/
      all_on.py      Optimal/random 1--3 interval models and summaries
      allocation.py  Exact allocation dynamic programming
  plotting/          All-on and On-Off PNG generation
apps/                Local web viewer, HTML and Streamlit app
experiments/         Exact and multi-interval result/figure publishers
  reference/         Optional historical Monte Carlo experiment runner
data/                Published results and archived experiment runs
figures/             Generated all-on and On-Off plots
doc/                 LaTeX report, PDF, guides and reference papers
tests/               Model and integration regression checks
runtime/             Ignored caches, logs and preserved legacy bytecode
```

## Run

From the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 -m streamlit run apps/streamlit_app.py
```

Or use the original local viewer:

```sh
python3 -m apps.local_viewer --open
```

Both viewers use exact binomial probabilities, reusing `data/results.json` only
when its method is `exact_binomial`. No samples or seeds are needed.
Restart running viewers after changing saved results.
Each class has its own On-probability control (`q_S`, `q_M`, `q_L` in the report), initially
50%. Customers still switch independently. Batch defaults are configured with
`P_ON = {"S": 0.5, "M": 0.5, "L": 0.5}` in `gridstress/config.py`.
Old result files with a scalar `p_on` remain supported; new runs and interactive
JSON exports record an S/M/L probability mapping.

Customer **type** probabilities (`p_S`, `p_M`, `p_L`) are separate controls and
must sum to one. The overall risk is the trinomial-weighted sum of conditional
risks, not the pooled fraction of trials across uniformly enumerated compositions.
`TYPE_PROBABILITIES` in `gridstress/config.py` supplies an illustrative equal-thirds
starting mix, not an empirical population estimate. Both viewers allow custom
mixes and reuse the same conditional results when only these weights change.
The Streamlit plot's hover text shows both conditional risk and composition probability.

## Models, experiments and figures

```sh
python3 -m gridstress.models.one_interval.all_on
python3 -m experiments.exact
python3 -m experiments.reweight --type-probabilities 0.6 0.3 0.1
python3 -m gridstress.plotting.all_on
python3 -m gridstress.plotting.probabilities
```

The recommended `experiments.exact` command recalculates all compositions using
the published inputs, backs up previous data and generates exact PNGs.
Apps contain UI code; experiments orchestrate publishing; model calculations
live in the interval-specific packages above. App entrypoints are unchanged.

Monte Carlo is retained only for reference and is not imported by the active
exact or multi-interval models or viewers. To explicitly run the old code:

```sh
python3 -m gridstress.models.one_interval.reference.monte_carlo
python3 -m experiments.reference.monte_carlo
```

These optional commands overwrite
published data with estimates; run `experiments.exact` afterwards to restore
exact published figures. Viewers never use these estimates.
The Monte Carlo command publishes a single run using `gridstress/config.py`
defaults (10,000 samples, seed 42), replacing `data/results.json`.
The experiment command runs five seeds at 100,000 samples per composition,
archives individual runs and the previous results under `data/experiments/`,
then publishes the pooled results and regenerates the On-Off plots.
The `reweight` command is an optional example: it changes the published population
mix and overall risk without resimulating. It backs up the prior results first.
Historical experiment files retain their original, explicitly uniform summaries.

Compile the report from `doc/` with `tectonic load_violation_report.tex`.
Report numerical text is maintained manually when a new experiment is published.

The separate All-On multi-interval experiment runs with
`MPLBACKEND=Agg python3 -m experiments.multi_interval`. It writes
`data/multi_interval_results.json` and `figures/multi_interval/allocation_3d.png`
without changing the single-interval viewer data. Compile its report from `doc/`
with `tectonic multi_interval_evaluation_report.tex`. Report values use the
current shared configuration and must be updated if those parameters change.
Use `--intervals 3` to publish the three-interval results and figure, or
`--intervals 1` for the shared-interval case. Non-default results get an
`_3_intervals` or `_1_intervals` filename suffix.

Launch its separate interactive visualization with:

```sh
python3 -m streamlit run apps/multi_interval_app.py --server.port 8502
```

Open http://localhost:8502. Hover over compositions to inspect the chosen
allocation, interval current and composition probability. Select 2 or 3
intervals using the horizontal control. The Allocation comparison tab retains
the three-outcome scatter, total points reversed and composition bar chart.
Interval graphs shows the shared interval and the selected number of optimal
allocated intervals. Blue means within the limit and red means violating.
Axes always show original composition counts.

Optimal vs random compares binary violation outcomes (0 or 1). Random allocation
draws one assignment per composition, independently assigning each medium/large
customer uniformly among the selected intervals. Hover to see its actual assigned
counts and currents. Both methods report integer counts of points reversed.
The New random assignment button draws again; optimal results stay unchanged.
The initial random seed is 42 and exports record the seed and assignments.
The optimal model uses exact dynamic programming; random results are outcomes
of a single assignment, not a repeated-trial Monte Carlo probability estimate.

## Verify

```sh
python3 -B -m unittest discover -s tests
node tests/local_viewer_controls.cjs
```

## Streamlit Community Cloud

Use `apps/streamlit_app.py` as the entrypoint, root `requirements.txt` for
dependencies, and Python 3.11. If already deployed, update the old entrypoint
in the deployment settings. See [the guide](doc/STREAMLIT.md).

Input validation is kept at model and HTTP boundaries to reject invalid
probabilities, negative customer counts, zero samples and oversized requests.

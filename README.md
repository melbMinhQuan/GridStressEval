# GridStressEval

Customer-load violation models, experiments, and interactive visualizations.

## Layout

```text
gridstress/
  config.py          Shared parameters, paths and composition enumeration
  results.py         Saved-run loading and probability summaries
  models/            All-on, Monte Carlo and cached interactive engines
  plotting/          All-on and On-Off PNG generation
apps/                Local web viewer, HTML and Streamlit app
experiments/         Five-seed experiment runner
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

Both viewers initially display `data/results.json`. Changed settings use the
interactive engine. Restart running viewers after changing saved results.

## Models, experiments and figures

```sh
python3 -m gridstress.models.all_on
python3 -m gridstress.models.monte_carlo
python3 -m experiments.run
python3 -m gridstress.plotting.all_on
python3 -m gridstress.plotting.probabilities
```

The Monte Carlo command publishes a single run using `gridstress/config.py`
defaults (10,000 samples, seed 42), replacing `data/results.json`.
The experiment command runs five seeds at 100,000 samples per composition,
archives individual runs and the previous results under `data/experiments/`,
then publishes the pooled results and regenerates the On-Off plots.

Compile the report from `doc/` with `tectonic load_violation_report.tex`.
Report numerical text is maintained manually when a new experiment is published.

## Verify

```sh
python3 -B -m unittest discover -s tests
```

## Streamlit Community Cloud

Use `apps/streamlit_app.py` as the entrypoint, root `requirements.txt` for
dependencies, and Python 3.11. If already deployed, update the old entrypoint
in the deployment settings. See [the guide](doc/STREAMLIT.md).

Input validation is kept at model and HTTP boundaries to reject invalid
probabilities, negative customer counts, zero samples and oversized requests.

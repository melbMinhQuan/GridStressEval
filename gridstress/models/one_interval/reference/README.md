# Monte Carlo reference implementations

These implementations are preserved for future comparisons. They are not used
by the current viewers, which import `one_interval.exact` or
`multi_interval.all_on` instead.

- `monte_carlo.py`: independent binomial sampling for each composition.
- `interactive.py`: the former cached, shared-sample viewer engine.
- `experiments/reference/monte_carlo.py`: the historical five-seed runner.

Run from the repository root only when Monte Carlo is explicitly wanted:

```sh
python3 -m gridstress.models.one_interval.reference.monte_carlo
python3 -m experiments.reference.monte_carlo
```

Warning: these runners retain their old publishing behaviour and replace
`data/results.json` with estimates. The five-seed runner also regenerates plots.
Run `python3 -m experiments.exact` afterwards to republish exact results.
Archived data and regression tests have been preserved.

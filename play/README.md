# play

Scratch space for exploratory experiments. Nothing here is used by `gridstress/`,
`apps/` or `experiments/`. Outputs go to `play/out/` (gitignored).

## All-On allocation model

Math: `allocation_math.pdf` (source `allocation_math.tex`, build with `tectonic allocation_math.tex`).

| File | Contents |
| --- | --- |
| `config.py` | shared inputs: N, T, currents, V, customer mix, interval times, uptake |
| `solar.py` | VIC1 curtailment share per circuit, x = X / C |
| `circuit.py` | segments, allocations, current, overload, violation probability |
| `soak.py` | load profiles and solar soak = min(added load, curtailment) |
| `frontier.py` | Pareto front of soak vs circuits without overload |
| `plots.py` | figures |
| `run_allocation.py` | one circuit: allocation, hourly energy and soak, violation probability |
| `run_pareto.py` | Pareto fronts for 1, 2 and 3 intervals at several uptakes |

Run from the repo root:

```sh
python3 -m play.run_allocation     # edit the INPUTS block in run_allocation.py
python3 -m play.run_pareto         # edit UPTAKES in run_pareto.py
```

`python3 play/run_allocation.py` also works. Needs `numpy` and `matplotlib`.

## Experiments

Report: `experiments.pdf` (source `experiments.tex`). Code: `play/experiments/`, figures in
`play/out/experiments/`.

```sh
python3 -m play.experiments            # all (about 25 min; E6-E8 sweep every design)
python3 -m play.experiments e01 e03    # selected
```

| File | Experiment |
| --- | --- |
| `e01_onoff.py` | On-Off usage probability q |
| `e02_mix.py` | customer mix grid (p_M, p_L) |
| `e03_size.py` | circuit size N, limit 25 A x N |
| `e04_energy.py` | same daily energy: inflexible (E4) and flexible (E5) loads |
| `e06_dst.py` | daylight saving: local clock time |
| `e07_rooftop.py` | rooftop-solar timing instead of utility curtailment |
| `e08_evening.py` | evening-peak load as a third objective |
| `e09_ramp.py` | state-level ramps at interval edges |
| `e10_value.py` | upgrades avoided (cost per upgrade is a placeholder) |

Rooftop solar and demand data: `python script/pull_rooftop_demand.py` (2 API credits).

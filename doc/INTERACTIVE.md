# Interactive on/off model

From the project folder:

```sh
python3 -m apps.local_viewer --open
```

Open http://127.0.0.1:8765. Stop the server with Ctrl+C. If that port is in use,
choose another using `--port 8766`. Only local connections are accepted.

The S/M/L controls are **loads per active customer**, not counts. Every point
is one allocation with S + M + L = total customers. The camera is fixed at the
same angle as the exported PNG. Slider changes are applied after release;
each slider also has a numeric input. Download the current PNG or JSON in
the viewer. The existing batch models and their saved results are preserved.

Limits: 0–100 customers, loads 0–500, threshold 0–50,000, on probability
0–100%, and 1,000–100,000 trials per allocation. Off probability is 1 - p(on).
Defaults and initial results come from `data/results.json` when the server starts.
The saved five-seed run pools 500,000 trials per allocation. Changed settings
run a single-seed interactive simulation.

## Sampling and caching

Three independent customer groups use cumulative Bernoulli trials to build
active counts for every group size. A trial's load is
`small_load * active_S + medium_load * active_M + large_load * active_L`.
The estimate is `count(load > threshold) / samples`. Equality is allowed.

The bank is reused when changing customer loads or the threshold. Different
allocations share trials, so their estimates are correlated; each individual
allocation still follows the same independent-customer model as
`gridstress/models/monte_carlo.py`.
The random stream arrangement differs from the batch model, so estimates need
not match it bit for bit. Seed 42 makes viewer results reproducible.

Only the latest bank is persisted as `runtime/cache/latest_active_counts.npz` (arrays
and numeric metadata, loaded without pickle). A subsequent run can reuse it.
Three banks, 32 result sets and eight rendered figures are cached in memory.
There is no exhaustive parameter sweep or unbounded disk cache. PNGs and
per-combination JSON are downloaded only when requested in the viewer.

Monte Carlo uncertainty remains: at 10,000 trials, the worst-case approximate
95% margin of error is about one percentage point per combination. Zero observed
violations does not establish zero true probability. These are separate
per-combination estimates, not a simultaneous confidence statement for the plot.

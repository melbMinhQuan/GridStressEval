# Interactive on/off model

```sh
python3 -m apps.local_viewer --open
```

Open http://127.0.0.1:8765. Stop with Ctrl+C, or select another port with
--port 8766. Restart after publishing new data.

Current inputs are amperes per active customer. Each dot represents a composition
whose counts sum to the total customers. Its colour is the exact conditional
violation probability, calculated using independent binomial probabilities.
Equality with the threshold is allowed. Neither viewer uses samples or seeds.

Limits: 0–100 customers, currents 0–500 A, threshold 0–50,000 A, On probabilities
0–100%. Customer type percentages separately define trinomial weights and must
total 100%. Changing only these weights changes overall risk, not conditional risk.

The exact engine caches 32 result sets and the local viewer caches eight figures.
Download PNGs or JSON from the viewer. Historical Monte Carlo scripts and archives
remain available for comparison but are not used for displayed probabilities.

Republish exact results and PNGs using the saved inputs:

```sh
MPLBACKEND=Agg python3 -m experiments.exact
```

Previous results are backed up under data/experiments/ before publishing.

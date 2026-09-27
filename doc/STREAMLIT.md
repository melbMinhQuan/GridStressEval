# Streamlit viewer

Run from the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 -m streamlit run apps/streamlit_app.py
```

Enter values and click **Update visualization**. Both saved results and changed
settings use exact binomial probability sums, not Monte Carlo. No sample count
or seed is required. Repeated settings are cached.

Each dot's colour shows the violation probability conditional on its composition.
Overall probability weights these risks by trinomial composition probabilities.
Hover to inspect both values.

Customer type percentages must sum to 100 and differ from On probabilities.
Type inputs have one-decimal precision; the illustrative equal-thirds default
is rounded to 33.4%, 33.3%, 33.3%. Changing type percentages reuses conditional
results. Downloads record the actual inputs and method: exact_binomial.

## Cloud deployment

Push the changed code and data/results.json. Use apps/streamlit_app.py as the
entrypoint, Python 3.11, and root requirements.txt. Restart after publishing new
saved results. No local sample caches or logs are required.

Exact means evaluation under the model assumptions, subject to floating-point
rounding, not a guarantee that those assumptions describe a real network.
No cloud speed benchmark has been performed.

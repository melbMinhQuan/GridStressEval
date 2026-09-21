# Streamlit viewer

Run from the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 -m streamlit run apps/streamlit_app.py
```

The default settings load `data/results.json`, including the pooled five-seed
experiment. Changed settings use the existing interactive simulation engine
(seed 42). Rotation and hovering happen in the browser. Repeated settings use
cached results. Overall probability appears below the plot.

## Cloud deployment

Push these files and the current data/results.json to your GitHub repository.
In Streamlit Community Cloud, create an app using that repository and branch,
set the entrypoint to `apps/streamlit_app.py`, and select Python 3.11 in the
advanced settings. Dependencies are in the root `requirements.txt`.
Do not include local caches, logs or bytecode as deployment requirements.

## Measuring cloud speed

Expand **Run details and timing**. Measure the initial saved view, a fresh
parameter setting at 10,000 samples, another at 100,000 samples, and then a
repeat setting. Change on-probability to exercise sample-bank creation; changing
only the current or threshold reuses the bank and is usually cheaper.
Also measure the visible wait in the browser: server timings exclude transfer,
browser rendering and cold startup. Local timings are not cloud measurements.

Local AppTest measurements on Apple M1 / 8 GB, 22 September 2026, 40 customers:

| Case | App execution time |
| --- | ---: |
| Initial saved five-seed results | 0.290 s |
| New on-probability 51%, 10,000 samples | 0.229 s |
| New on-probability 52%, 100,000 samples | 1.963 s |
| Repeat cached 10,000-sample setting | 0.025 s |

These exercise the interactive engine, not the slower independent-composition
batch model. AppTest checks server execution; it does not measure browser load.
No cloud benchmark has been performed yet.

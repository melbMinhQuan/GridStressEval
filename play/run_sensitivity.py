"""Sensitivity of the 3-vs-2-interval soak gap to each parameter (one at a time).

For each parameter value, every interval design (as in run_design_pareto) is evaluated with every
per-circuit allocation; each K keeps its best design at the safest allocation. The gap is
    gap = soak_3 / soak_2 - 1.

Run from the repo root:  python3 -m play.run_sensitivity   (or python3 play/run_sensitivity.py)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_sensitivity.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock
from play.frontier import pareto_front

# ================================================================ INPUTS (this run)
BASE = {"uptake": config.UPTAKE, "latest": 15, "length": 3, "p_m": 0.30, "p_l": 0.01,
        "i_m": 40, "i_l": 100, "t": 1000}
SWEEPS = {                                        # one parameter varied at a time
    "uptake": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    "latest": [14, 15, 16, 17],                   # end of the allowed window (start fixed at EARLIEST)
    "length": [2, 2.5, 3],                        # hours per interval
    "p_m": [0.1, 0.2, 0.3, 0.4],
    "p_l": [0.005, 0.01, 0.02, 0.05],
    "i_m": [30, 40, 60],
    "i_l": [60, 100, 150],
    "t": [900, 1000, 1200],
}
LABELS = {"uptake": "uptake", "latest": "window end (h)", "length": "interval length (h)",
          "p_m": "p_M", "p_l": "p_L", "i_m": "I_M (A)", "i_l": "I_L (A)", "t": "limit T (A)"}
EARLIEST, STEP = 8, 0.5
PLOT = config.OUT_DIR / "sensitivity.png"
# ==================================================================================


def apply(p):
    """Set the model inputs in play.config for parameter set p (p_S = 1 - p_M - p_L)."""
    config.P_TYPE = {"S": 1 - p["p_m"] - p["p_l"], "M": p["p_m"], "L": p["p_l"]}
    config.CURRENT = {"S": 20, "M": p["i_m"], "L": p["i_l"]}
    config.T = p["t"]


def designs(k, p):
    starts = np.arange(EARLIEST, p["latest"] - p["length"] + 1e-9, STEP)
    return [[(float(s), float(s) + p["length"]) for s in c] for c in combinations(starts, k)]


def _safest(job):
    p, intervals = job
    apply(p)                                                  # each worker process sets its own config
    safe, soak, _ = pareto_front(intervals, p["uptake"])[-1]
    return 1 - safe, soak


def best(p, pool):
    """{k: (violation, soak, design)} for the best design of k intervals at the safest allocation."""
    out = {}
    for k in (2, 3):
        ds = designs(k, p)
        res = list(pool.map(_safest, [(p, d) for d in ds]))
        i = min(range(len(ds)), key=lambda j: (round(res[j][0], 10), -res[j][1]))   # safest, then most soak
        out[k] = (*res[i], ds[i])
    return out


def name(intervals):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in intervals)


def main():
    results = {}
    with ProcessPoolExecutor() as pool:
        for param, values in SWEEPS.items():
            print(f"\n{LABELS[param]}")
            print(f"  {'value':>8}  {'viol 2':>7} {'soak 2':>7}  {'viol 3':>7} {'soak 3':>7}  {'gap':>6}   best 3-interval design")
            for v in values:
                p = dict(BASE, **{param: v})
                r = best(p, pool)
                gap = r[3][1] / r[2][1] - 1
                results[(param, v)] = (r, gap)
                print(f"  {v:>8g}  {r[2][0]:>7.2%} {r[2][1]:>7.1f}  {r[3][0]:>7.2%} {r[3][1]:>7.1f}  {gap:>6.1%}   "
                      f"{name(r[3][2])}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, (param, values) in zip(axes.flat, SWEEPS.items()):
        gaps = [results[(param, v)][1] * 100 for v in values]
        ax.plot(values, gaps, "-o", color="#D9822B", lw=2)
        ax.axvline(BASE[param], color="#999", lw=1, ls=":")
        ax.set_title(LABELS[param], loc="left", fontweight="bold")
        if param == "uptake":
            ax.set_xscale("log")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Soak gain, 3 vs 2 intervals (%)")
    fig.suptitle("Sensitivity of the 3-interval soak gain (dotted line = default, others held at default)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    PLOT.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

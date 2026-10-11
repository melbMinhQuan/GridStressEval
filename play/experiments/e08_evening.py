"""E8 Evening peak: designs may run to 17:00; a third objective counts the added load that falls in
each day's top-6 demand hours (VIC1 operational demand, market time):
    peak load = sum_t E[D(t)] f(t) dt,  f(t) = share of days where slot t is in that day's top 6 hours."""
import csv
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.experiments.common import design_grid, name, safest_added_profile, save, style
from play.frontier import pareto_front

WINDOW, LENGTH, UPTAKE, TOP_HOURS = (8, 17), 3, 0.1, 6


def peak_share():
    """f(t) per 5-min slot."""
    with open(config.ROOFTOP_DEMAND_CSV, newline="") as f:
        demand = np.array([float(r["demand_mw"]) for r in csv.DictReader(f)]).reshape(-1, 24)
    top = np.argsort(-demand, axis=1)[:, :TOP_HOURS]
    hits = np.zeros(24)
    for row in top:
        hits[row] += 1
    return np.repeat(hits / len(demand), config.SLOTS // 24)


def _evaluate(job):
    intervals, f = job
    safe, soak, _ = pareto_front(intervals, UPTAKE)[-1]
    peak = (safest_added_profile(intervals, UPTAKE) * f).sum() * config.DT
    return 1 - safe, soak, peak


def run():
    f = peak_share()
    designs = {k: design_grid(k, *WINDOW, LENGTH) for k in (1, 2, 3)}
    with ProcessPoolExecutor() as pool:
        res = {k: list(pool.map(_evaluate, [(d, f) for d in ds], chunksize=1)) for k, ds in designs.items()}
    hourly = f[::config.SLOTS // 24]
    print(f"E8 evening peak: share of days each hour is in the day's top {TOP_HOURS} demand hours (AEST)")
    print("  " + " ".join(f"{h:02d}:{v:.0%}" for h, v in enumerate(hourly) if v > 0))
    print(f"E8 best designs ({LENGTH} h inside {WINDOW[0]}-{WINDOW[1]}, uptake {UPTAKE:.0%}, safest allocation)")
    for k in (1, 2, 3):
        pts = list(zip(designs[k], res[k]))
        front = [(d, r) for d, r in pts                              # nothing better on all three objectives
                 if not any(q[0] <= r[0] + 1e-12 and q[1] >= r[1] - 1e-9 and q[2] <= r[2] + 1e-9 and q != r
                            for _, q in pts)]
        print(f"  K={k}: {len(front)} Pareto-optimal designs (violation, soak, peak load), by peak load")
        for d, (v, s_, p) in sorted(front, key=lambda x: x[1][2]):
            print(f"    {name(d):<40} violation {v:7.3%}  soak {s_:6.1f}  peak load {p:5.1f} kWh/day")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    marks = {1: "s", 2: "o", 3: "D"}
    for k in (1, 2, 3):
        v, s, p = (np.array(x) for x in zip(*res[k]))
        sc = ax.scatter(p, s, c=np.log10(np.maximum(v * 100, 1e-3)), cmap="RdYlGn_r", vmin=-2, vmax=2,
                        marker=marks[k], s=28, edgecolor="none", label=f"{k} interval{'s' * (k > 1)}")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_ticks([-2, -1, 0, 1, 2], labels=["0.01", "0.1", "1", "10", "100"])
    cbar.set_label("P(violation) %")
    ax.set_xlabel(f"Added load in daily top-{TOP_HOURS} demand hours (kWh/day per circuit)  ← better")
    ax.set_ylabel("Solar soak per circuit (kWh/day)  ↑ better")
    ax.set_title(f"E8  Soak vs evening-peak load, all {LENGTH} h designs inside {WINDOW[0]}-{WINDOW[1]}",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style(ax)
    save(fig, "e08_evening.png")
    return res


if __name__ == "__main__":
    run()

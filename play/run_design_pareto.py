"""Pareto front over interval placement AND allocation, for 2 vs 3 intervals.

Every design of K intervals (fixed length, inside [EARLIEST, LATEST], overlaps allowed) is
evaluated with every per-circuit allocation (play.frontier.pareto_front). One design applies to
all circuits, so the combined front for K is the upper envelope of the design fronts.

Run from the repo root:  python3 -m play.run_design_pareto [--length 2] [--latest 15]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_design_pareto.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock
from play.frontier import pareto_front

# ================================================================ INPUTS (this run)
EARLIEST, LATEST = 8, 15                          # intervals must lie inside this window (hours, AEST)
LENGTH = 3                                        # hours per interval
STEP = 0.5                                        # start-time grid, hours
KS = (2, 3)
UPTAKES = [0.01, config.UPTAKE, 0.5]
PLOT = config.OUT_DIR / "design_pareto.png"
# ==================================================================================

COLORS = {2: "#2A7FB8", 3: "#D9822B"}


def designs(k):
    starts = np.arange(EARLIEST, LATEST - LENGTH + 1e-9, STEP)
    return [[(float(s), float(s) + LENGTH) for s in c] for c in combinations(starts, k)]


def name(intervals):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in intervals)


def _front(job):
    intervals, uptake = job
    return np.array([(round(safe, 10), soak) for safe, soak, _ in pareto_front(intervals, uptake)])  # rounding: equal
                                                                                         # safety compares equal


def envelope(fronts, grid):
    """Best soak at each safety level over all designs, and which design gives it."""
    best = np.full(len(grid), -np.inf)
    who = np.full(len(grid), -1)
    for i, f in enumerate(fronts):
        x, y = f[:, 0], f[:, 1]                                   # x increasing along each front
        if len(f) == 1:
            soak = np.where(np.isclose(grid, x[0]), y[0], -np.inf)
        else:
            soak = np.interp(grid, x, y, left=-np.inf, right=-np.inf)
        better = soak > best + 1e-12
        best[better], who[better] = soak[better], i
    # Pareto: drop points beaten by a safer point with at least as much soak
    safer_max = np.maximum.accumulate(best[::-1])[::-1]          # best soak at this or higher safety
    beaten = np.r_[safer_max[1:], -np.inf] >= best - 1e-9
    best[beaten] = -np.inf
    return best, who


def main():
    global LENGTH, LATEST, PLOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=LENGTH, help="hours per interval")
    parser.add_argument("--latest", type=float, default=LATEST, help="end of the allowed window")
    args = parser.parse_args()
    if (args.length, args.latest) != (LENGTH, LATEST):
        LENGTH, LATEST = args.length, args.latest
        PLOT = PLOT.with_name(f"design_pareto_{LENGTH:g}h_to_{LATEST:g}.png")
    all_designs = {k: designs(k) for k in KS}
    jobs = [(d, u) for k in KS for d in all_designs[k] for u in UPTAKES]
    print(f"{sum(len(v) for v in all_designs.values())} designs x {len(UPTAKES)} uptakes ...")
    with ProcessPoolExecutor() as pool:
        results = dict(zip(((tuple(d), u) for d, u in jobs), pool.map(_front, jobs)))

    fig, axes = plt.subplots(1, len(UPTAKES), figsize=(6.2 * len(UPTAKES), 5.4), squeeze=False)
    for ax, u in zip(axes[0], UPTAKES):
        print(f"\nuptake {u:.0%} ({config.circuits(u):,.0f} circuits)")
        for k in KS:
            fronts = [results[(tuple(d), u)] for d in all_designs[k]]
            for f in fronts:
                ax.plot(f[:, 0] * 100, f[:, 1], color=COLORS[k], lw=0.6, alpha=0.15)
            xs = np.concatenate([f[:, 0] for f in fronts])
            grid = np.unique(np.concatenate([xs, np.linspace(xs.min(), xs.max(), 400)]))
            best, who = envelope(fronts, grid)
            keep = np.isfinite(best)
            ax.plot(grid[keep] * 100, best[keep], "-", color=COLORS[k], lw=2.8)
            end = np.flatnonzero(keep)[-1]                        # safest point on the front
            ax.scatter(grid[end] * 100, best[end], s=60, color=COLORS[k], edgecolor="white", zorder=5)
            ax.annotate(f"{best[end]:.1f}", (grid[end] * 100, best[end]), xytext=(-6, 6),
                        textcoords="offset points", ha="right", fontsize=8, color=COLORS[k])
            print(f"  {k} intervals ({len(fronts)} designs), Pareto front:")
            for i in dict.fromkeys(who[keep]):                    # in order along the front
                on = keep & (who == i)
                print(f"    {name(all_designs[k][i]):<40} safe {grid[on].min():7.2%} - {grid[on].max():7.2%}  "
                      f"soak {best[on].max():6.1f} - {best[on].min():6.1f}")
        ax.set_xlabel("Circuits without overload (%)  →  better")
        ax.set_title(f"Uptake {u:.0%} ({config.circuits(u):,.0f} circuits)", loc="left", fontweight="bold")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("Expected solar soak per circuit (kWh/day)  ↑  better")
    handles = [plt.Line2D([], [], color=COLORS[k], lw=2.8, label=f"{k} intervals: best over all designs")
               for k in KS] + [plt.Line2D([], [], color="#888", lw=0.8, alpha=0.5, label="one design (faint)")]
    fig.legend(handles=handles, frameon=False, loc="upper right", ncol=3)
    fig.suptitle(f"Pareto front over interval placement and allocation ({LENGTH} h intervals inside "
                 f"{clock(EARLIEST)}-{clock(LATEST)}, overlaps allowed)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    PLOT.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

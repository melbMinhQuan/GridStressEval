"""Compare 1, 2 and 3 intervals with full trade-off curves.

For each K, every design (3 h intervals starting on the hour inside the window, overlaps allowed)
and every share rule alpha is evaluated; within a design circuits may mix rules (convex hull).
One design applies to the whole network, so the front for K is the upper envelope over designs.
    x = circuits without overload (trinomial weights)
    y = mean soak over compositions (unweighted)
Settings: soak metric (linear alignment or capped) x solar source x uptake.

Run from the repo root:  python3 -m play.run_compare_curves
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock, compositions, segments, weight

# ================================================================ INPUTS (this run)
SETTINGS = [                                      # (title, metric, solar source, uptake)
    ("Solar alignment (linear), rooftop", "linear", "rooftop", None),
    ("Solar alignment (linear), utility curtailment", "linear", "curtailment", None),
    ("Capped soak, curtailment, uptake 1%", "capped", "curtailment", 0.01),
    ("Capped soak, curtailment, uptake 10%", "capped", "curtailment", 0.10),
]
STARTS = range(7, 14)                             # interval start hours
LENGTH = 3
KS = (1, 2, 3)
PLOT = config.OUT_DIR / "compare_curves.png"
# ==================================================================================

COLORS = {1: "#6B6B6B", 2: "#2A7FB8", 3: "#D9822B"}


def designs(k):
    return [[(s, s + LENGTH) for s in c] for c in combinations(STARTS, k)]


def _hull(job):
    metric, source, uptake, design = job
    config.SOLAR_SOURCE = source
    from play.run_curve_search import linear_tables
    from play.run_strategy_pareto import evaluate_rule, mixed_front, rules
    from play.soak import soak_table
    from play.solar import excess_per_circuit
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    segs = segments(design)
    if metric == "linear":
        tables = linear_tables(segs)
    else:
        excess = excess_per_circuit(None, uptake)
        tables = [soak_table(a, b, excess) for a, b, _ in segs]
    pts = [(*evaluate_rule(a, segs, tables, comps, w), a) for a in rules(len(design))]
    return [((round(x, 10), y), a) for (x, y), a in mixed_front(pts)]


def envelope(hulls, grid):
    """Best soak at each safety level over designs (one design network-wide), Pareto-filtered."""
    best = np.full(len(grid), -np.inf)
    who = np.full(len(grid), -1)
    for i, h in enumerate(hulls):
        x = np.array([p[0][0] for p in h])
        y = np.array([p[0][1] for p in h])
        soak = (np.where(np.isclose(grid, x[0]), y[0], -np.inf) if len(h) == 1
                else np.interp(grid, x, y, left=-np.inf, right=-np.inf))
        better = soak > best + 1e-12
        best[better], who[better] = soak[better], i
    safer = np.r_[np.maximum.accumulate(best[::-1])[::-1][1:], -np.inf]
    best[safer >= best - 1e-9] = -np.inf
    return best, who


def name(design):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in design)


def main():
    all_designs = {k: designs(k) for k in KS}
    jobs = [(m, s, u, d) for _, m, s, u in SETTINGS for k in KS for d in all_designs[k]]
    print(f"{len(jobs)} design evaluations ({', '.join(f'{len(v)} x {k}' for k, v in all_designs.items())} per setting)")
    with ProcessPoolExecutor() as pool:
        hulls = dict(zip(((m, s, u, tuple(d)) for m, s, u, d in jobs), pool.map(_hull, jobs, chunksize=1)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (title, m, s, u) in zip(axes.flat, SETTINGS):
        print(f"\n{title}")
        for k in KS:
            hs = [hulls[(m, s, u, tuple(d))] for d in all_designs[k]]
            for h in hs:                                           # every design, faint
                ax.plot([p[0][0] * 100 for p in h], [p[0][1] for p in h], color=COLORS[k], lw=0.6, alpha=0.18)
            xs = np.concatenate([[p[0][0] for p in h] for h in hs])
            grid = np.unique(np.concatenate([xs, np.linspace(xs.min(), xs.max(), 600)]))
            best, who = envelope(hs, grid)
            keep = np.isfinite(best)
            ax.plot(grid[keep] * 100, best[keep], color=COLORS[k], lw=3 if k > 1 else 0,
                    marker="o" if k == 1 else None, ms=9, label=f"{k} interval{'s' * (k > 1)}")
            print(f"  {k} interval(s): front from {grid[keep].min():.1%} to {grid[keep].max():.1%} safe, "
                  f"soak {best[keep].max():.1f} -> {best[keep][-1]:.1f}")
            for i in dict.fromkeys(who[keep]):
                on = keep & (who == i)
                print(f"     {name(all_designs[k][i]):<34} safe {grid[on].min():6.1%} - {grid[on].max():6.1%}  "
                      f"soak {best[on].max():7.1f} - {best[on].min():7.1f}")
                mid = np.flatnonzero(on)[len(np.flatnonzero(on)) // 2]
                if k > 1 and on.sum() > 20:
                    ax.annotate(name(all_designs[k][i]), (grid[mid] * 100, best[mid]), xytext=(0, 8 if k == 3 else -14),
                                textcoords="offset points", fontsize=6.5, color=COLORS[k], ha="center")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Circuits without overload, trinomial (%)  →  better")
        ax.set_ylabel("Mean soak over compositions (kWh/day)  ↑  better")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"1 vs 2 vs 3 intervals: best front over interval placement ({LENGTH} h, starts "
                 f"{STARTS[0]}:00-{STARTS[-1]}:00) and share rules", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.955, "bold: best design at each safety level (labels: design)   faint: every single design",
             fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

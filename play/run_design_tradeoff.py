"""How interval placement creates (or removes) the soak-vs-safety trade-off.

2 intervals, every share rule alpha; front = best mix of rules (upper-right convex hull).
x = circuits without overload (trinomial), y = mean soak over compositions (unweighted).

Run from the repo root:  python3 -m play.run_design_tradeoff
"""
import sys
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
SOURCE, UPTAKE = "rooftop", 0.02                  # peaked solar, solar comparable to load
DESIGNS = {                                       # label: intervals
    "poor: 06-09 | 11-14": [(6, 9), (11, 14)],
    "medium: 07-10 | 11-14": [(7, 10), (11, 14)],
    "good: 09-12 | 12-15": [(9, 12), (12, 15)],
}
COLORS = ["#C0392B", "#D9822B", "#2E8B57"]
PLOT = config.OUT_DIR / "design_tradeoff.png"
# ==================================================================================

config.SOLAR_SOURCE = SOURCE
from play.run_strategy_pareto import evaluate_rule, mixed_front, rules  # noqa: E402  (after setting the source)
from play.soak import soak_table  # noqa: E402
from play.solar import excess_per_circuit  # noqa: E402


def main():
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    excess = excess_per_circuit(None, UPTAKE)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    print(f"{SOURCE} solar, uptake {UPTAKE:.0%}, 2 intervals, every share rule")
    for (label, iv), color in zip(DESIGNS.items(), COLORS):
        segs = segments(iv)
        tables = [soak_table(a, b, excess) for a, b, _ in segs]
        pts = [(*evaluate_rule(a, segs, tables, comps, w), a) for a in rules(2)]
        hull = mixed_front(pts)
        ax.scatter([(1 - v) * 100 for v, _, _ in pts], [s for _, s, _ in pts], s=16, color=color, alpha=0.25, lw=0)
        xs, ys = [x * 100 for (x, _), _ in hull], [y for (_, y), _ in hull]
        ax.plot(xs, ys, "-o", color=color, lw=2.6, ms=6, label=label)
        for (x, y), a in (hull[0], hull[-1]):
            ax.annotate(f"{a[0]:.0%} / {a[1]:.0%}", (x * 100, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=color)
        loss = 1 - ys[-1] / ys[0]
        print(f"  {label:<24} front {1 - hull[0][0][0]:.1%} -> {1 - hull[-1][0][0]:.1%} violation, "
              f"soak {ys[0]:.0f} -> {ys[-1]:.0f} kWh/day ({loss:.1%} soak given up)")
    ax.set_xlabel("Circuits without overload, trinomial (%)  →  better")
    ax.set_ylabel("Mean solar soak over compositions, unweighted (kWh/day)  ↑  better")
    ax.set_title(f"Interval placement sets the trade-off ({SOURCE} solar, uptake {UPTAKE:.0%}, 2 intervals)",
                 loc="left", fontweight="bold")
    ax.text(0.01, 0.01, "dots: every share rule   line: best front (mixing rules)   labels: share in first / second interval",
            transform=ax.transAxes, fontsize=8, color="#555")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

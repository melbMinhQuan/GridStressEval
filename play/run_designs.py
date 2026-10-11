"""Compare interval designs (times and overlaps) on violation probability and solar soak.

For each design and uptake, the safest end of the Pareto front is reported: every circuit
uses its best safe allocation. Violation does not depend on uptake; soak does.

Run from the repo root:  python3 -m play.run_designs   (or python3 play/run_designs.py)
"""
import sys
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_designs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from play import config
from play.circuit import clock
from play.frontier import pareto_front

# ================================================================ INPUTS (this run)
DESIGNS = [                                       # hours; 10.5 = 10:30
    [(11, 14)],
    [(10, 13)],
    [(12, 15)],
    [(8, 11), (11, 14)],                          # baseline 2: back to back
    [(9, 12), (12, 15)],                          # shifted later
    [(8, 11), (10, 13)],                          # 1 h overlap
    [(9, 12), (11, 14)],                          # 1 h overlap, later
    [(10, 13), (11, 14)],                         # 2 h overlap
    [(8, 11), (10.5, 13.5), (11, 14)],            # baseline 3
    [(9, 12), (10.5, 13.5), (11, 14)],            # first interval moved to 9-12
    [(8, 11), (10, 13), (12, 15)],                # chained 1 h overlaps
    [(9, 12), (11, 14), (13, 16)],                # chained 1 h overlaps, later
    [(8, 11), (11, 14), (14, 17)],                # no overlap
    [(7, 10), (10, 13), (13, 16)],                # no overlap, earlier
]
UPTAKES = [0.01, config.UPTAKE, 0.5]
PLOT = config.OUT_DIR / "designs.png"
# ==================================================================================

COLORS = {1: "#6B6B6B", 2: "#2A7FB8", 3: "#D9822B"}


def name(intervals):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in intervals)


def main():
    rows = []
    for intervals in DESIGNS:
        safest = {u: pareto_front(intervals, u)[-1] for u in UPTAKES}
        rows.append((intervals, 1 - safest[UPTAKES[0]][0], {u: safest[u][1] for u in UPTAKES}))

    head = "".join(f"{f'soak {u:.0%}':>11}" for u in UPTAKES)
    print(f"{'#':>2}  {'intervals':<36}{'violation':>10}{head}   (kWh/day per circuit, safest allocation)")
    for i, (intervals, violation, soak) in enumerate(rows, 1):
        print(f"{i:>2}  {name(intervals):<36}{violation:>10.2%}" + "".join(f"{soak[u]:>11.1f}" for u in UPTAKES))

    fig, axes = plt.subplots(1, len(UPTAKES), figsize=(6.2 * len(UPTAKES), 5.6), squeeze=False)
    for ax, u in zip(axes[0], UPTAKES):
        labels = {}                                                   # designs at the same point share a label
        for i, (intervals, violation, soak) in enumerate(rows, 1):
            k = len(intervals)
            ax.scatter(violation * 100, soak[u], s=70, color=COLORS[k], edgecolor="white", zorder=3,
                       marker={1: "s", 2: "o", 3: "D"}[k])
            labels.setdefault((round(violation, 6), round(soak[u], 1)), []).append(str(i))
        for (violation, soak), ids in labels.items():
            ax.annotate(",".join(ids), (violation * 100, soak), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(-0.005, 150)
        ax.invert_xaxis()                                             # less violation to the right
        ax.set_xticks([0, 0.1, 1, 10, 100])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_xlabel("P(violation), log scale  →  better")
        ax.set_title(f"Uptake {u:.0%} ({config.circuits(u):,.0f} circuits)", loc="left", fontweight="bold")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("Solar soak per circuit (kWh/day)  ↑  better")
    key = "\n".join(f"{i:>2}  {name(iv)}" for i, (iv, _, _) in enumerate(rows, 1))
    fig.text(1.0, 0.5, key, va="center", ha="left", family="monospace", fontsize=8)
    handles = [plt.Line2D([], [], ls="", marker=m, color=COLORS[k], label=f"{k} interval{'s' * (k > 1)}")
               for k, m in ((1, "s"), (2, "o"), (3, "D"))]
    fig.legend(handles=handles, frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Interval designs: violation vs solar soak (safest allocation, All-On)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    PLOT.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT, dpi=150, bbox_inches="tight")
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

"""Sweep 3-interval designs with overlaps: violation and soak at the safest allocation.

Run from the repo root:  python3 -m play.run_overlap_sweep   (or python3 play/run_overlap_sweep.py)
"""
import sys
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_overlap_sweep.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from play import config
from play.circuit import clock
from play.frontier import pareto_front

# ================================================================ INPUTS (this run)
LENGTH = 3                                        # hours per interval
STARTS = [7, 8, 9, 10, 11, 12, 13, 14]            # candidate start hours
UPTAKES = [0.01, config.UPTAKE, 0.5]
PLOT = config.OUT_DIR / "overlap_sweep.png"
REFERENCES = {                                    # drawn as horizontal lines
    "2 intervals 08-11 | 11-14": [(8, 11), (11, 14)],
    "3 no overlap 08-11 | 11-14 | 14-17": [(8, 11), (11, 14), (14, 17)],
}
# ==================================================================================


def overlap_hours(intervals):
    """Total hours where at least two intervals overlap."""
    edges = sorted({t for iv in intervals for t in iv})
    return sum(b - a for a, b in zip(edges, edges[1:])
               if sum(s <= a and b <= e for s, e in intervals) >= 2)


def disjoint_count(intervals):
    """Largest number of intervals that are pairwise non-overlapping (greedy by end time)."""
    count, end = 0, -1
    for s, e in sorted(intervals, key=lambda iv: iv[1]):
        if s >= end:
            count, end = count + 1, e
    return count


def name(intervals):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in intervals)


def main():
    designs = [[(s, s + LENGTH) for s in starts] for starts in combinations(STARTS, 3)]
    designs = [d for d in designs if overlap_hours(d) > 0]
    rows = []
    for d in designs:
        safest = {u: pareto_front(d, u)[-1] for u in UPTAKES}
        rows.append((d, 1 - safest[UPTAKES[0]][0], {u: safest[u][1] for u in UPTAKES}))
    rows.sort(key=lambda r: (round(r[1], 6), -r[2][config.UPTAKE]))

    head = "".join(f"{f'soak {u:.0%}':>10}" for u in UPTAKES)
    print(f"{len(rows)} overlapping 3-interval designs ({LENGTH} h each), sorted by violation then soak at "
          f"{config.UPTAKE:.0%} uptake")
    print(f"{'intervals':<30}{'overlap h':>10}{'disjoint':>9}{'violation':>11}{head}")
    for d, violation, soak in rows:
        print(f"{name(d):<30}{overlap_hours(d):>10g}{disjoint_count(d):>9}{violation:>11.2%}"
              + "".join(f"{soak[u]:>10.1f}" for u in UPTAKES))

    refs = {label: {u: pareto_front(d, u)[-1] for u in UPTAKES} for label, d in REFERENCES.items()}
    colors = {1: "#C0392B", 2: "#2A7FB8", 3: "#2E8B57"}
    fig, axes = plt.subplots(1, len(UPTAKES), figsize=(6 * len(UPTAKES), 5.2), squeeze=False)
    for ax, u in zip(axes[0], UPTAKES):
        for j, (d, violation, soak) in enumerate(rows):
            c = disjoint_count(d)
            ax.scatter(overlap_hours(d) + (j % 7 - 3) * 0.03, soak[u], s=55, color=colors[c],
                       edgecolor="white", zorder=3)
        for (label, ref), ls in zip(refs.items(), ("--", ":")):
            safe, soak, _ = ref[u]
            ax.axhline(soak, color="#444", lw=1.2, ls=ls, label=f"{label} ({1 - safe:.2%} violation)")
        best = max(rows, key=lambda r: r[2][u])
        ax.annotate(f"best overlap: {name(best[0])}", (overlap_hours(best[0]), best[2][u]), xytext=(8, 4),
                    textcoords="offset points", fontsize=8, ha="left")
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.7, 3.6)
        ax.set_xlabel("Hours with overlapping intervals")
        ax.set_title(f"Uptake {u:.0%} ({config.circuits(u):,.0f} circuits)", loc="left", fontweight="bold")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("Solar soak per circuit (kWh/day)  ↑  better")
    handles = [plt.Line2D([], [], ls="", marker="o", color=colors[c],
                          label=f"{c} non-overlapping: violation {v}")
               for c, v in ((1, "79.6%"), (2, "4.2%"))]
    handles += axes[0][0].get_legend_handles_labels()[0]
    fig.legend(handles=handles, frameon=False, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Overlapping 3-interval designs (safest allocation, All-On)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    PLOT.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

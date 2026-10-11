"""Baseline designs A2 and A3: the highest-soak and the lowest-violation allocation of each.

Each circuit picks its own allocation (play.frontier.pareto_front): front[0] maximises soak,
front[-1] minimises P(violation). Solar: 2025 VIC1 rooftop per circuit (all circuits); soak counts the
added load inside intervals; violation is trinomial-weighted, All-On.

Run from the repo root:  python3 -m play.run_baseline_points
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from play import config
from play.frontier import pareto_front

DESIGNS = {"A2: 08-11 | 11-14": [(8, 11), (11, 14)],
           "A3: 08-11 | 10:30-13:30 | 11-14": [(8, 11), (10.5, 13.5), (11, 14)]}
COLORS = ["#2A7FB8", "#D9822B"]
PLOT = config.OUT_DIR / "baseline_points.png"


def main():
    config.SOLAR_SOURCE = "rooftop_local"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (label, design), color, dx in zip(DESIGNS.items(), COLORS, (-16, 9)):
        front = pareto_front(design, config.UPTAKE)
        pts = {"highest soak": front[0], "lowest violation": front[-1]}
        for (name, (safe, soak, mix)), marker in zip(pts.items(), ("o", "s")):
            ax.scatter((1 - safe) * 100, soak, s=110, marker=marker, color=color, edgecolor="white", zorder=4,
                       label=f"{label}, {name}")
            ax.annotate(f"{1 - safe:.2%}, {soak:.2f} kWh/day", ((1 - safe) * 100, soak), xytext=(0, dx),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
            print(f"{label:<34} {name:<17} P(violation) {1 - safe:6.2%}  soak {soak:6.1f} kWh/day  "
                  f"M+L per interval {', '.join(f'{v:.0%}' for v in mix)}")
    ax.set_xlabel("P(violation) (%)  ←  better")
    ax.set_ylabel("Expected solar soak per circuit (kWh/day)  ↑  better")
    ax.invert_xaxis()
    ax.margins(x=0.25)
    ax.set_ylim(130, 136)                                             # +-2% around the points: no false slope
    ax.set_title("Baseline design A: highest-soak and lowest-violation allocation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

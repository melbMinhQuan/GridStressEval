"""Pareto fronts with domestic (rooftop) solar and total circuit demand.

    soak = mean_d sum_{t covered by an interval} min(D_total(t), x_d(t)) dt,
    D_total = N P_S + (P_M - P_S) M_g + (P_L - P_S) L_g,   x = VIC1 rooftop MW / C,  C = u N_VIC / N.
Violation: trinomial, total current. Points: share rules, fronts with mixing. Pairs: 2 vs 3 intervals.

Run from the repo root:  python3 -m play.run_rooftop_fronts
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

from play.circuit import compositions, segments, weight  # noqa: E402
from play.run_report import DESIGNS, SINGLE, evaluate, rules  # noqa: E402
from play.run_strategy_pareto import mixed_front  # noqa: E402
from play.soak import in_range  # noqa: E402
from play.solar import excess_per_circuit  # noqa: E402

UPTAKES = [0.01, 0.1]
PAIRS = [("A2", "A3"), ("B2", "B3"), ("C2", "C3")]
PLOT = config.OUT_DIR / "rooftop_fronts.png"


def total_table(a, b, x_all):
    """F[m, l] = mean_d sum_{t in [a,b)} min(s (N P_S + dP_M m + dP_L l), x_d(t)) dt,  s = DEMAND_SCALE."""
    load, n = config.load_kw(), config.N
    x = x_all[:, in_range(a, b)]
    table = np.zeros((n + 1, n + 1))
    for m in range(n + 1):
        d = config.DEMAND_SCALE * (n * load["S"] + (load["M"] - load["S"]) * m
                                   + (load["L"] - load["S"]) * np.arange(n + 1 - m))
        table[m, : n + 1 - m] = np.minimum(d[:, None, None], x[None]).sum(axis=2).mean(axis=1) * config.DT
    return table


def main():
    config.SOLAR_SOURCE = "rooftop_actual"                            # set here, not at import
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    labels = {**SINGLE, **DESIGNS[2], **DESIGNS[3]}
    full = {name.split(":")[0]: name for name in labels}
    res = {}
    for u in UPTAKES:
        x = excess_per_circuit(None, u)
        print(f"\nuptake {u:.0%}: rooftop per circuit, noon mean {x[:, 144:156].mean():.0f} kW")
        for label, design in labels.items():
            segs = segments(design)                                   # covered time only
            tables = [total_table(a, b, x) for a, b, _ in segs]
            pts = [(*evaluate(a, segs, tables, comps, w), a) for a in rules(len(design))]
            hull = mixed_front(pts)
            res[(u, label)] = (pts, hull)
            print(f"  {label:<34} " + "  ->  ".join(f"{np.round(a, 2).tolist()} safe {xx:.1%} soak {y:.1f}"
                                                 for (xx, y), a in hull))
    (single,) = SINGLE
    fig, axes = plt.subplots(len(UPTAKES), len(PAIRS), figsize=(18, 4.9 * len(UPTAKES)), squeeze=False)
    for row, u in zip(axes, UPTAKES):
        for ax, (two, three) in zip(row, PAIRS):
            (x1, y1), _ = res[(u, single)][1][0]
            ax.scatter(x1 * 100, y1, s=90, color="#6B6B6B", zorder=5, label=single)
            for key, color in ((two, "#2A7FB8"), (three, "#D9822B")):
                pts, hull = res[(u, full[key])]
                ax.scatter([(1 - v) * 100 for v, _, _ in pts], [s for _, s, _ in pts], s=10, color=color, alpha=0.25, lw=0)
                wide = key == two
                ax.plot([xx * 100 for (xx, _), _ in hull], [y for (_, y), _ in hull], "-o", color=color,
                        lw=6 if wide else 2.2, ms=8 if wide else 4, alpha=0.55 if wide else 1,
                        zorder=3 if wide else 4, label=full[key])
            ax.set_title(f"{two} vs {three}, uptake {u:.0%}", loc="left", fontweight="bold")
            ax.set_xlabel("Circuits without overload (%)  →  better")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.grid(color="#e6e6e6")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        row[0].set_ylabel("Expected rooftop solar soak (kWh/day)  ↑")
    fig.suptitle("Domestic solar: soak = min(total circuit demand, rooftop share), summed over interval hours",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

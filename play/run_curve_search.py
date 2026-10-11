"""Search for settings that give a real soak-vs-safety trade-off curve.

For every combination of solar source, uptake, soak metric and 2-interval design, all share rules
are evaluated and the best front (mixing rules) is built. A front is a "curve" when it gives up a
lot of soak to gain safety:
    soak span   = 1 - soak(safest end) / soak(most-soak end)
    safety span = safe share at the safest end - safe share at the most-soak end
Soak metrics:
    capped : sum_t min(D(t), x_d(t)) dt            (current model)
    linear : sum_t D(t) s_d(t) dt, s = X / max X   (alignment of added load with solar, no cap)

Run from the repo root:  python3 -m play.run_curve_search
"""
import sys
from itertools import product
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import compositions, segments, weight
from play.run_strategy_pareto import evaluate_rule, mixed_front, rules
from play.soak import in_range, soak_table
from play.solar import _profile_mw, excess_per_circuit

# ================================================================ INPUTS (this run)
SOURCES = ["curtailment", "rooftop"]
UPTAKES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]     # only used by the capped metric
DESIGNS = {"good 09-12|12-15": [(9, 12), (12, 15)],
           "medium 07-10|11-14": [(7, 10), (11, 14)],
           "poor 06-09|11-14": [(6, 9), (11, 14)]}
SHOW = 6                                          # fronts to plot (largest soak span)
PLOT = config.OUT_DIR / "curve_search.png"
# ==================================================================================


def linear_tables(segs):
    """table[m, l] = (dP_M m + dP_L l) * mean_d sum_{t in g} s_d(t) dt, s = X / max X."""
    mw, _ = _profile_mw(config.SOLAR_SOURCE)
    s = mw / mw.max()
    load = config.load_kw()
    m, l = np.meshgrid(np.arange(config.N + 1), np.arange(config.N + 1), indexing="ij")
    added = (load["M"] - load["S"]) * m + (load["L"] - load["S"]) * l
    return [added * s[:, in_range(a, b)].sum(axis=1).mean() * config.DT for a, b, _ in segs]


def run_case(design, tables_for, comps, w):
    segs = segments(design)
    tables = tables_for(segs)
    pts = [(*evaluate_rule(a, segs, tables, comps, w), a) for a in rules(2)]
    hull = mixed_front(pts)
    (x0, y0), a0 = hull[0]
    (x1, y1), a1 = hull[-1]
    return dict(hull=hull, pts=pts, soak_span=1 - y1 / y0 if y0 else 0.0, safety_span=x1 - x0,
                vertices=len(hull), ends=(a0, a1))


def main():
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    cases = []
    for source in SOURCES:
        config.SOLAR_SOURCE = source
        for (label, design), u in product(DESIGNS.items(), UPTAKES):
            excess = excess_per_circuit(None, u)
            r = run_case(design, lambda segs: [soak_table(a, b, excess) for a, b, _ in segs], comps, w)
            cases.append(dict(source=source, metric="capped", uptake=u, design=label, **r))
        for label, design in DESIGNS.items():
            r = run_case(design, linear_tables, comps, w)
            cases.append(dict(source=source, metric="linear", uptake=None, design=label, **r))

    print(f"{len(cases)} cases, 2 intervals, every share rule, best front with mixing")
    print(f"{'source':<12}{'metric':<8}{'uptake':>7}  {'design':<20}{'vertices':>9}{'safety span':>13}"
          f"{'soak span':>11}   rule: most soak -> safest")
    for c in sorted(cases, key=lambda c: -c["soak_span"]):
        u = f"{c['uptake']:.1%}" if c["uptake"] else "-"
        a0, a1 = c["ends"]
        print(f"{c['source']:<12}{c['metric']:<8}{u:>7}  {c['design']:<20}{c['vertices']:>9}"
              f"{c['safety_span']:>12.1%}{c['soak_span']:>11.1%}   {a0[0]:.0%}/{a0[1]:.0%} -> {a1[0]:.0%}/{a1[1]:.0%}")

    top = sorted(cases, key=lambda c: -c["soak_span"])[:SHOW]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, c in zip(axes.flat, top):
        ax.scatter([(1 - v) * 100 for v, _, _ in c["pts"]], [s for _, s, _ in c["pts"]], s=14, color="#999", alpha=0.4, lw=0)
        xs, ys = [x * 100 for (x, _), _ in c["hull"]], [y for (_, y), _ in c["hull"]]
        ax.plot(xs, ys, "-o", color="#C0392B", lw=2.4, ms=5)
        for (x, y), a in c["hull"]:
            ax.annotate(f"{a[0]:.0%}/{a[1]:.0%}", (x * 100, y), xytext=(4, 5), textcoords="offset points", fontsize=7)
        u = f", uptake {c['uptake']:.1%}" if c["uptake"] else ""
        ax.set_title(f"{c['design']}\n{c['source']}, {c['metric']}{u}: soak span {c['soak_span']:.0%}",
                     loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("Circuits without overload, trinomial (%)  →")
        ax.set_ylabel("Mean soak, unweighted  ↑")
        ax.grid(color="#e6e6e6")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Settings with the largest soak-vs-safety trade-off (2 intervals; labels: share first/second interval)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

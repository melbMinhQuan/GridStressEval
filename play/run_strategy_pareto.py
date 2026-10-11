"""Pareto front over allocation strategies: one share rule alpha applied to every circuit.

alpha_k = share of each circuit's medium and large customers placed in interval k (rounded to
whole customers). For each alpha:
    x = P(violation)  = sum_n w(n) r(n, alpha)            (trinomial weights)
    y = soak          = (1/|n|) sum_n soak(n, alpha)      (plain mean over compositions)

Circuits may mix rules, so the front is the upper-right convex hull of the rule points.

Run from the repo root:  python3 -m play.run_strategy_pareto
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
from play.circuit import clock, compositions, segments, weight
from play.soak import soak_table
from play.solar import excess_per_circuit

# ================================================================ INPUTS (this run)
DESIGNS = {1: [(11, 14)], 2: [(9, 12), (12, 15)], 3: [(8, 11), (9, 12), (12, 15)]}
UPTAKES = [0.01, 0.1, 0.5]
STEP = 0.05                                       # grid for alpha
PLOT = config.OUT_DIR / "strategy_pareto.png"
# ==================================================================================

COLORS = {1: "#6B6B6B", 2: "#2A7FB8", 3: "#D9822B"}


def rules(k):
    ticks = round(1 / STEP)
    for head in product(range(ticks + 1), repeat=k - 1):
        if sum(head) <= ticks:
            yield np.array(head + (ticks - sum(head),)) / ticks


def split(count, alpha):
    """Largest-remainder rounding of count * alpha."""
    raw = count * alpha
    parts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - parts))[: count - parts.sum()]:
        parts[i] += 1
    return parts


def evaluate_rule(alpha, segs, tables, comps, w):
    over = soak = 0.0
    n, amps = config.N, config.CURRENT
    for (_, n_m, n_l), wi in zip(comps, w):
        med, lar = split(n_m, alpha), split(n_l, alpha)
        peak, s = 0.0, 0.0
        for (_, _, cov), table in zip(segs, tables):
            m, l = med[cov].sum(), lar[cov].sum()
            peak = max(peak, amps["S"] * (n - m - l) + amps["M"] * m + amps["L"] * l)
            s += table[m, l]
        over += wi * (peak > config.T)
        soak += s
    return over, soak / len(comps)


def front(points):
    """Points not beaten on both (lower violation, higher soak), sorted by violation."""
    keep = [p for p in points if not any(q[0] <= p[0] and q[1] >= p[1] and (q[0], q[1]) != (p[0], p[1])
                                         for q in points)]
    return sorted(keep, key=lambda p: p[0])


def mixed_front(points):
    """Pareto front when circuits may use a mix of rules: the upper-right convex hull of
    (safe, soak) points, from the most-soak rule to the safest rule. Vertices are pure rules;
    points between two vertices mix those two rules across circuits."""
    pts = sorted({(1 - v, s): a for v, s, a in points}.items())            # by safe share
    hull = []
    for (x, y), a in pts:                                                  # upper hull, left to right
        while len(hull) >= 2:
            (x1, y1), _ = hull[-2]
            (x2, y2), _ = hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:            # middle point not above the chord
                hull.pop()
            else:
                break
        hull.append(((x, y), a))
    top = max(range(len(hull)), key=lambda i: (hull[i][0][1], hull[i][0][0]))
    return hull[top:]                                                      # most soak -> safest


def main():
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    fig, axes = plt.subplots(1, len(UPTAKES), figsize=(6 * len(UPTAKES), 5.4), sharex=True)
    zfig, zaxes = plt.subplots(1, len(UPTAKES), figsize=(6 * len(UPTAKES), 4.8))
    for ax, zax, u in zip(axes, zaxes, UPTAKES):
        excess = excess_per_circuit(None, u)
        print(f"\nuptake {u:.0%}: x = P(violation) trinomial, y = mean soak over {len(comps)} compositions")
        for k, iv in DESIGNS.items():
            segs = segments(iv)
            tables = [soak_table(a, b, excess) for a, b, _ in segs]
            pts = [(*evaluate_rule(a, segs, tables, comps, w), a) for a in rules(k)]
            hull = mixed_front(pts)
            ax.scatter([(1 - p[0]) * 100 for p in pts], [p[1] for p in pts], s=14, color=COLORS[k], alpha=0.3,
                       lw=0, zorder=2)
            xs, ys = [x * 100 for (x, _), _ in hull], [y for (_, y), _ in hull]
            ax.plot(xs, ys, "-", color=COLORS[k], lw=2.4, zorder=3)
            ax.scatter(xs, ys, s=45 if k > 1 else 90, color=COLORS[k], edgecolor="white", zorder=4,
                       label=f"{k} interval{'s' * (k > 1)}: " + " | ".join(f"{clock(a)}-{clock(b)}" for a, b in iv))
            if k == 1:                                                # off the zoomed range
                continue
            zys = getattr(zax, "_ys", []) + ys
            zax._ys = zys
            pad = max(0.5, 0.15 * (max(zys) - min(zys)))
            zax.set_ylim(min(zys) - pad, max(zys) + pad)
            zax.plot(xs, ys, "-o", color=COLORS[k], lw=2.4, ms=6, label=f"{k} interval{'s' * (k > 1)}")
            for i, ((x, y), a) in enumerate(hull):
                zax.annotate(f"{np.round(a, 2).tolist()}", (x * 100, y), xytext=(4, 6 if i % 2 else -11),
                             textcoords="offset points", fontsize=7, color=COLORS[k])
            (bx, by), ba = hull[-1]                                   # safest rule on the front
            ax.annotate(f"{np.round(ba, 2).tolist()}", (bx * 100, by), xytext=(-4, 6),
                        textcoords="offset points", ha="right", fontsize=7, color=COLORS[k])
            print(f"  {k} intervals, {len(pts)} rules, {len(hull)} hull vertices (most soak -> safest):")
            for (x, y), a in hull:
                print(f"    alpha {np.round(a, 2)}  P(violation) {1 - x:7.2%}  mean soak {y:6.1f} kWh/day")
        zax.set_xlim(80, 94)
        zax.set_xlabel("Circuits without overload, trinomial (%)  →  better")
        zax.set_title(f"Uptake {u:.0%} (zoom on the fronts)", loc="left", fontweight="bold")
        zax.grid(color="#e6e6e6")
        for side in ("top", "right"):
            zax.spines[side].set_visible(False)
        ax.set_xlim(15, 100)
        ax.set_xlabel("Circuits without overload, trinomial (%)  →  better")
        ax.set_title(f"Uptake {u:.0%}", loc="left", fontweight="bold")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Mean soak over compositions, unweighted (kWh/day)  ↑")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("Allocation strategies (one share rule for every circuit)", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.905, "dots: every rule   line: best front when circuits mix two rules (vertices are single rules)   label: safest rule",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(PLOT, dpi=150)
    zaxes[0].set_ylabel("Mean soak over compositions, unweighted (kWh/day)  ↑")
    zaxes[-1].legend(frameon=False, loc="lower left")
    zfig.suptitle("Best fronts with mixed rules (labels: share per interval at each vertex)",
                  x=0.01, ha="left", fontweight="bold")
    zfig.tight_layout()
    zoom = PLOT.with_name("strategy_pareto_zoom.png")
    zfig.savefig(zoom, dpi=150)
    print(f"\ngraph: {PLOT}\ngraph: {zoom}")


if __name__ == "__main__":
    main()

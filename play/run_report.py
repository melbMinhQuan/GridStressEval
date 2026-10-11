"""Figures and numbers for the technical report (doc/power_saver_intervals_report.tex).

Solar: 2025 VIC1 rooftop (domestic) solar shared over all circuits. Soak counts only the load the tariff adds, (P_j - P_S) for medium/large customers inside their
interval and 0 elsewhere (existing demand is already reflected in the curtailment data), over the
fixed window 08:00-15:00, weighted by w(n). Overload uses the total current. Each Pareto point is
one share rule alpha applied to every circuit, with mixing between rules (convex hull).

Run from the repo root:  python3 -m play.run_report
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
from play.circuit import clock, compositions, weight
from play.run_strategy_pareto import mixed_front
from play.soak import in_range
from play.solar import _profile_mw, excess_per_circuit

# ================================================================ INPUTS (this run)
WINDOW = (8, 15)                                  # soak is summed over this window for every design
DESIGNS = {
    2: {"A2: 08-11 | 11-14": [(8, 11), (11, 14)],
        "B2: 09-12 | 12-15": [(9, 12), (12, 15)],
        "C2: 08-11 | 12-15": [(8, 11), (12, 15)]},
    3: {"A3: 08-11 | 10:30-13:30 | 11-14": [(8, 11), (10.5, 13.5), (11, 14)],
        "B3: 08-11 | 09-12 | 12-15": [(8, 11), (9, 12), (12, 15)],
        "C3: 09-11 | 11-13 | 13-15": [(9, 11), (11, 13), (13, 15)]},
}
SINGLE = {"1: 11-14": [(11, 14)]}
SOLAR = "rooftop_local"                           # 2025 VIC1 rooftop solar / all 72,500 circuits
UPTAKES = [config.UPTAKE]                         # no effect on rooftop_local; one row
STEP = 0.05
SEED = 7
OUT = config.OUT_DIR / "report"
# ==================================================================================

SEASONS = {"Summer": (12, 1, 2), "Autumn": (3, 4, 5), "Winter": (6, 7, 8), "Spring": (9, 10, 11)}


def window_segments(intervals):
    """Segments of WINDOW cut at interval edges, including stretches with no interval."""
    a0, b0 = WINDOW
    edges = sorted({a0, b0} | {t for iv in intervals for t in iv if a0 < t < b0})
    return [(a, b, [k for k, (s, e) in enumerate(intervals) if s <= a and b <= e])
            for a, b in zip(edges, edges[1:])]


def added_table(a, b, excess):
    """F[m, l] = mean_d sum_{t in [a,b)} min(dP_M m + dP_L l, x_d(t)) dt  (added load only)."""
    load, n = config.load_kw(), config.N
    x = excess[:, in_range(a, b)]
    table = np.zeros((n + 1, n + 1))
    for m in range(n + 1):
        added = (load["M"] - load["S"]) * m + (load["L"] - load["S"]) * np.arange(n + 1 - m)
        table[m, : n + 1 - m] = np.minimum(added[:, None, None], x[None]).sum(axis=2).mean(axis=1) * config.DT
    return table


def rules(k):
    ticks = round(1 / STEP)
    for head in product(range(ticks + 1), repeat=k - 1):
        if sum(head) <= ticks:
            yield np.array(head + (ticks - sum(head),)) / ticks


def split(count, alpha):
    raw = count * alpha
    parts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - parts))[: count - parts.sum()]:
        parts[i] += 1
    return parts


def evaluate(alpha, segs, tables, comps, w):
    """(P(violation), expected soak) for share rule alpha."""
    amps, n = config.CURRENT, config.N
    over = soak = 0.0
    for (_, n_m, n_l), wi in zip(comps, w):
        med, lar = split(n_m, alpha), split(n_l, alpha)
        peak = s = 0.0
        for (_, _, cov), table in zip(segs, tables):
            m, l = med[cov].sum(), lar[cov].sum()
            peak = max(peak, amps["S"] * (n - m - l) + amps["M"] * m + amps["L"] * l)
            s += table[m, l]
        over += wi * (peak > config.T)
        soak += wi * s
    return over, soak


def curtailment_figure():
    mw, days = _profile_mw(SOLAR)
    mw = mw * 1000 / config.circuits(1.0)                            # kW per circuit
    months = np.array([int(d[5:7]) for d in days])
    rng = np.random.default_rng(SEED)
    hours = np.arange(config.SLOTS) * config.DT
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, (season, ms) in zip(axes, SEASONS.items()):
        for d in sorted(rng.choice(np.flatnonzero(np.isin(months, ms)), 4, replace=False)):
            ax.plot(hours, mw[d], lw=1.4, label=days[d])
        ax.axvspan(*WINDOW, color="#888", alpha=0.08, lw=0)
        ax.set_xlim(5, 20)
        ax.set_title(season, loc="left", fontweight="bold")
        ax.set_xlabel("Hour (AEST)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(color="#e6e6e6")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Rooftop solar per circuit (kW)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_solar.png", dpi=150)


PAIRS = [("A2", "A3"), ("B2", "B3"), ("C2", "C3")]


def main():
    config.SOLAR_SOURCE = SOLAR
    OUT.mkdir(parents=True, exist_ok=True)
    curtailment_figure()
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    results = {}
    for u in UPTAKES:
        excess = excess_per_circuit(None, u)
        for label, design in {**SINGLE, **DESIGNS[2], **DESIGNS[3]}.items():
            segs = window_segments(design)
            tables = [added_table(a, b, excess) for a, b, _ in segs]
            pts = [(*evaluate(a, segs, tables, comps, w), a) for a in rules(len(design))]
            hull = mixed_front(pts)
            results[(u, label)] = dict(pts=pts, hull=hull)
            print(f"uptake {u:.0%} {label:<34} " + "  ->  ".join(
                f"{np.round(a, 2).tolist()} safe {x:.1%} soak {y:.2f}" for (x, y), a in hull))

    labels = {**DESIGNS[2], **DESIGNS[3]}
    full = {name.split(":")[0]: name for name in labels}
    (single,) = SINGLE
    fig, axes = plt.subplots(len(UPTAKES), len(PAIRS), figsize=(6 * len(PAIRS), 5.2 * len(UPTAKES)), squeeze=False)
    for row, u in zip(axes, UPTAKES):
        for ax, (two, three) in zip(row, PAIRS):
            (x1, y1), _ = results[(u, single)]["hull"][0]
            ax.scatter(x1 * 100, y1, s=90, color="#6B6B6B", zorder=5, label=single)
            for key, color in ((two, "#2A7FB8"), (three, "#D9822B")):
                r = results[(u, full[key])]
                ax.scatter([(1 - v) * 100 for v, _, _ in r["pts"]], [s_ for _, s_, _ in r["pts"]], s=10,
                           color=color, alpha=0.25, lw=0)
                wide = key == two                                 # 2 intervals drawn wide underneath 3
                ax.plot([x * 100 for (x, _), _ in r["hull"]], [y for (_, y), _ in r["hull"]], "-o", color=color,
                        lw=6 if wide else 2.2, ms=8 if wide else 4, alpha=0.55 if wide else 1,
                        zorder=3 if wide else 4, label=full[key])
            ax.set_title(f"{two} vs {three}", loc="left", fontweight="bold")
            ax.set_xlabel("Circuits without overload (%)  →  better")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.grid(color="#e6e6e6")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        row[0].set_ylabel("Expected solar soak per circuit (kWh/day)  ↑")
    fig.suptitle("2 vs 3 intervals per design choice (dots: single rules, lines: fronts with mixing)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_pareto.png", dpi=150)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()

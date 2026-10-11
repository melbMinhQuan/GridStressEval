"""Figure 6 for several allocation strategies, one figure each: overload against soak (rooftop solar soaked,
excess not subtracted; --net for net soak) as rho grows, one panel per uptake, zoom on today's mix below.

Lowest overload and best soak come from the run_load_scale cache (net soak, converted with the fixed counted
energy E: soak = (net + E) / 2). The share rules are fixed splits for every circuit. One interval has only
one allocation and is drawn in every figure.

Run from the repo root:  python3 -m play.run_strategy_curves [--net]   (after python3 -m play.run_load_scale)
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from play import config
from play import run_participation as rp
from play.run_allocation_scatter import Setting
from play.run_load_scale import CACHE, K_REPORT

NET = "--net" in sys.argv
rp.NET_SOAK = NET
OUT = config.ROOT / "doc" / "vis"
LABEL = rp.SOAK_LABEL if NET else "Soak (% of rooftop solar, 08–16)"
RULES = {                                          # name: (two-interval split, three-interval split)
    "even split": ([1, 1], [1, 1, 1]),
    "middle-heavy": ([1, 1], [20, 80, 20]),
    "no middle": ([1, 1], [1, 0, 1]),
    "morning-heavy": ([70, 30], [60, 30, 10]),
}
TITLES = {"lowest overload": "Lowest overload", "best soak": "Best soak",
          "even split": "Even split: 50/50, 33/33/33", "middle-heavy": "Middle-heavy: 50/50, 17/67/17",
          "no middle": "No middle: 50/50, 50/0/50", "morning-heavy": "Morning-heavy: 70/30, 60/30/10"}


def energy(k, rho, u):
    """Counted energy, MWh/day: every customer's I_S over the covered hours plus the interval load."""
    kw, p = config.load_kw(), rp.mix(rho)
    hours = rp.covered(rp.choose(k, "Summer")).sum() * config.DT
    return u * config.VIC_CUSTOMERS / 1000 * (kw["S"] * hours + rp.LENGTH * (p["M"] * (kw["M"] - kw["S"])
                                                                            + p["L"] * (kw["L"] - kw["S"])))


def cached(z, f, k, which):
    """(overload %, soak) over RHO from the net-soak cache; which = 0 lowest overload, 1 best soak."""
    c = z[f"{f}_{k}"][which].copy()
    if not NET:
        c[:, 1] = (c[:, 1] + np.array([energy(k, rho, f) for rho in rp.RHO])) / 2
    return c


def rule_curves(f):
    """{(rule, K): (overload %, soak) over RHO} for the share rules at uptake f."""
    rp.UPTAKE = f
    out = {}
    designs = {k: rp.year_designs(k) for k in (2, 3)}
    for rho in rp.RHO:
        settings = {k: Setting(designs[k], rho) for k in (2, 3)}
        for rule, splits in RULES.items():
            for k, split in zip((2, 3), splits):
                s = settings[k]
                share = np.array(split, float) / sum(split)
                out.setdefault((rule, k), []).append(s(np.ones((len(s.keys), 1)) * share)[:2])
    return {key: np.array(v) for key, v in out.items()}


def figure(name, curves, path):
    """curves: {(f, K): (overload %, soak) over RHO}. Top: all designs; bottom: zoom on 2 and 3 at today's mix."""
    today = np.isclose(rp.RHO, rp.TODAY)
    fig, axes = plt.subplots(2, len(K_REPORT), figsize=(14, 7.6))
    for j, f in enumerate(K_REPORT):
        now = np.array([curves[(f, k)][today][0] for k in (2, 3)])
        x0, x1 = max(now[:, 0].min() - 5, -1), now[:, 0].max() + 5
        x1 = max(x1, x0 + 15)
        y = rp.pct(now[:, 1])
        y0, y1 = y.min() - 2, y.max() + 2
        for row, ks in ((0, (1, 2, 3)), (1, (2, 3))):
            ax = axes[row, j]
            for k in ks:
                c = curves[(f, k)]
                v = rp.pct(c[:, 1])
                ax.plot(c[:, 0], v, "-o", ms=3 + row, lw=2, color=rp.COLORS[k], label=f"{k} interval{'s' * (k > 1)}")
                ax.scatter(c[today, 0], v[today], s=90, facecolor="none", edgecolor=rp.COLORS[k], lw=2, zorder=5)
            ax.set_xlabel("Circuits that overload (%)")
            rp.style(ax)
        axes[0, j].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec="#8a8984"))
        axes[1, j].set_xlim(x0, x1)
        axes[1, j].set_ylim(y0, y1)
        axes[0, j].set_title(f"u = {100 * f:.0f}%", loc="left", fontweight="bold", fontsize=10)
        for k in (2, 3):
            c = curves[(f, k)][today][0]
            print(f"  {name:<16} u={f:.0%} K={k}: today overload {c[0]:5.1f}%  soak {rp.pct(c[1]):5.1f}%")
    fig.suptitle(TITLES[name], x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.supylabel(LABEL, fontsize=10)
    axes[1, 0].legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(CACHE)
    base = rp.UPTAKE
    rules = {f: rule_curves(f) for f in K_REPORT}
    rp.UPTAKE = base
    suffix = "_net" if NET else ""
    for i, name in enumerate(TITLES, 1):
        curves = {}
        for f in K_REPORT:
            curves[(f, 1)] = cached(z, f, 1, 0)
            for k in (2, 3):
                if name in RULES:
                    curves[(f, k)] = rules[f][(name, k)]
                else:
                    curves[(f, k)] = cached(z, f, k, 0 if name == "lowest overload" else 1)
        path = OUT / f"strategy{i}_{name.replace(' ', '_').replace('-', '_')}{suffix}.png"
        figure(name, curves, path)
        print(f"graph: {path}")


if __name__ == "__main__":
    main()

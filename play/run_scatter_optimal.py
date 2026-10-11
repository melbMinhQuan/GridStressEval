"""Scatter of every share rule for A2 and A3, the ideal point, and the rule closest to it.

Soak (state totals, no per-circuit split):
    D_tot(t) = DEMAND_SCALE * (N_VIC / N) * sum_n w(n) D_n(t)     total demand, MW
    soak     = mean_d sum_{t in interval hours} min(D_tot(t), R_d(t)) dt     R = total VIC1 rooftop, MWh/day
Violation: trinomial-weighted, All-On. Ideal point: no violation and the highest soak of any rule.
Distance: sqrt(violation^2 + (1 - soak / ideal soak)^2), both on a 0-1 scale.

Run from the repo root:  python3 -m play.run_scatter_optimal
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

DESIGNS = {"A2: 08-11 | 11-14": [(8, 11), (11, 14)],
           "A3: 08-11 | 10:30-13:30 | 11-14": [(8, 11), (10.5, 13.5), (11, 14)]}
COLORS = ["#2A7FB8", "#D9822B"]
PLOT = config.OUT_DIR / "scatter_optimal.png"


def main():
    from play.circuit import compositions, segments, weight
    from play.run_report import rules, split
    from play.soak import in_range
    from play.solar import _profile_mw
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    rooftop, _ = _profile_mw("rooftop_local")                         # (days, slots) MW, all of VIC
    circuits = config.VIC_CUSTOMERS / config.N
    load, amps, n = config.load_kw(), config.CURRENT, config.N
    pts = {}
    for label, d in DESIGNS.items():
        segs = segments(d)
        r = [rooftop[:, in_range(a, b)] for a, b, _ in segs]
        out = []
        for alpha in rules(len(d)):
            viol = 0.0
            demand = np.zeros(len(segs))                              # mean circuit demand per segment, kW
            for (_, n_m, n_l), wi in zip(comps, w):
                med, lar = split(n_m, alpha), split(n_l, alpha)
                peak = 0.0
                for g, (_, _, cov) in enumerate(segs):
                    m, l = med[cov].sum(), lar[cov].sum()
                    peak = max(peak, amps["S"] * (n - m - l) + amps["M"] * m + amps["L"] * l)
                    demand[g] += wi * (load["S"] * (n - m - l) + load["M"] * m + load["L"] * l)
                viol += wi * (peak > config.T)
            total_mw = config.DEMAND_SCALE * circuits * demand / 1000
            soak = sum(np.minimum(tm, rg).sum(axis=1).mean() * config.DT for tm, rg in zip(total_mw, r))
            out.append((viol, soak, alpha))
        pts[label] = out
    ideal = max(s for p in pts.values() for _, s, _ in p)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    config.SOLAR_SOURCE = "rooftop_local"
    from play.experiments.common import min_peak_table, safest_added_profile, violation, weights
    from play.soak import SLOT
    wt = weights(config.N, config.P_TYPE)
    base = config.N * config.load_kw()["S"]
    dist = lambda v, s_: np.hypot(v, 1 - s_ / ideal)                   # distance to the ideal point
    for (label, p), color, dy in zip(pts.items(), COLORS, (-14, 10)):
        v = np.array([q[0] for q in p])
        s_ = np.array([q[1] for q in p])
        ax.scatter(v * 100, s_, s=30, color=color, alpha=0.45, lw=0, label=f"{label} ({len(p)} rules)")
        # lowest violation among share rules (most soak at that level)
        j = np.flatnonzero(np.isclose(v, v.min()))[s_[np.isclose(v, v.min())].argmax()]
        ax.scatter(v[j] * 100, s_[j], s=90, marker="s", color=color, edgecolor="black", lw=0.8, zorder=6,
                   label=f"{label}: lowest violation, share rule")
        # lowest violation with each circuit's own best split
        d = DESIGNS[label]
        pv = violation(min_peak_table(d, wt.keys()), wt, config.T)
        total_mw = config.DEMAND_SCALE * circuits * (base + safest_added_profile(d, config.UPTAKE)) / 1000
        inside = np.zeros(len(SLOT), bool)
        for a, b in d:
            inside |= (SLOT >= a) & (SLOT < b)
        ps = np.minimum(total_mw[inside], rooftop[:, inside]).sum(axis=1).mean() * config.DT
        ax.scatter(pv * 100, ps, s=110, marker="D", color=color, edgecolor="black", lw=0.8, zorder=6,
                   label=f"{label}: lowest violation, own split per circuit")
        # closest to the ideal over ALL points of this design (share rules and own split)
        cands = [(dist(v[i], s_[i]), v[i], s_[i]) for i in range(len(v))] + [(dist(pv, ps), pv, ps)]
        dmin, cv, cs = min(cands)
        ax.scatter(cv * 100, cs, s=320, facecolor="none", edgecolor=color, lw=2.5, zorder=5,
                   label=f"{label}: closest to ideal (distance {dmin:.3f})")
        ax.annotate(f"{cv:.1%}, {cs:,.0f} MWh/day", (cv * 100, cs), xytext=(0, dy + (6 if dy > 0 else -6)),
                    textcoords="offset points", ha="center", fontsize=8, color=color)
        print(f"{label:<34} share-rule lowest {v[j]:.2%} / {s_[j]:.1f} (distance {dist(v[j], s_[j]):.3f});  "
              f"own split {pv:.2%} / {ps:.1f} (distance {dist(pv, ps):.3f});  closest: {cv:.2%} / {cs:.1f}")
    ax.scatter(0, ideal, s=160, color="#C0392B", marker="*", zorder=6, label=f"ideal: 0% violation, {ideal:,.0f} MWh/day")
    print(f"ideal point: 0% violation, soak {ideal:,.1f} MWh/day")
    ax.set_xlabel("P(violation) (%)  ←  better")
    ax.set_ylabel("Rooftop solar soaked, VIC total (MWh/day)  ↑")
    ax.set_xlim(85, -3)
    lows = [q[1] for p in pts.values() for q in p]
    ax.set_ylim(min(lows) - 0.05 * (ideal - min(lows)), ideal + 0.08 * (ideal - min(lows)))
    ax.set_title("A2 vs A3: every share rule, lowest-violation points, the ideal and the closest point", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

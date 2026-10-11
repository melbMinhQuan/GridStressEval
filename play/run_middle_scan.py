"""Soak against the share of medium and large customers in the middle interval, the rest split evenly over
the outer two, for the report's windows and the proposal's windows, at load scales k = 1.0 and 1.5. Shares are
applied exactly to the expected added load (no rounding to whole customers), which is what the state-level soak
needs. Right: summer load shapes against mean rooftop solar.

Run from the repo root:  python3 -m play.run_middle_scan
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
from play import run_participation as rp
from play.soak import SLOT, in_range

SHARES = np.round(np.arange(0, 1.001, 0.05), 2)
SCALES = {1.0: "#f0905c", 1.5: "#7a2e0b"}
SHAPES = {0.0: "#2a78d6", 0.15: "#eb6834", 0.4: "#8a8984"}          # middle shares drawn in the summer panel
PLOT = config.OUT_DIR / "report" / "fig_middle_scan.png"


def load(design, season, a, k):
    """Added load counted for soak (MW per slot): middle share a, outer windows (1 - a) / 2 each."""
    p, kw = rp.mix(rp.TODAY), config.load_kw()
    per_circuit = config.N * (p["M"] * (kw["M"] - kw["S"]) + p["L"] * (kw["L"] - kw["S"]))          # kW
    total = k * rp.UPTAKE * config.VIC_CUSTOMERS / config.N * per_circuit / 1000 * rp.season_weight(season)
    out = np.zeros(config.SLOTS)
    for (s, e), share in zip(design, ((1 - a) / 2, a, (1 - a) / 2)):
        out[in_range(s, e)] += share * total
    return out


def soak(designs, a, k):
    days = {s: len(rp.solar_season(s)) for s in rp.SEASONS}
    return sum(rp.net(load(designs[s], s, a, k), rp.solar_season(s)) * days[s]
               for s in rp.SEASONS) / sum(days.values())


def main():
    report = {s: rp.choose(3, s) for s in rp.SEASONS}
    proposal = {s: ((9.0, 12.0), (11.0, 14.0), (12.0, 15.0)) for s in rp.SEASONS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, (title, designs) in zip(axes, (("Report windows (middle 12-15 in summer)", report),
                                           ("Proposal windows: 9-12 | 11-14 | 12-15", proposal))):
        for k, color in SCALES.items():
            vals = np.array([soak(designs, a, k) for a in SHARES])
            i = vals.argmax()
            ax.plot(SHARES * 100, vals, lw=2, color=color, label=f"k = {k:.1f}")
            ax.scatter(SHARES[i] * 100, vals[i], s=60, color=color, zorder=5)
            print(f"{title}, k={k}: best middle share {SHARES[i]:.0%}, soak {vals[i]:,.0f} (0%: {vals[0]:,.0f}, "
                  f"+{100 * (vals[i] / vals[0] - 1):.1f}%)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("Customers in the middle interval (%)")
        rp.style(ax)
    axes[0].set_ylabel("Net soak (MWh/day)")
    axes[0].legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(SLOT, rp.solar_season("Summer").mean(axis=0), color="#333333", lw=1.8, label="Rooftop solar")
    for a, color in SHAPES.items():
        ax.plot(SLOT, load(report["Summer"], "Summer", a, 1.5), lw=2, color=color, label=f"{a:.0%} in the middle")
    ax.set_xlim(8, 17)
    ax.set_title("Summer, report windows, k = 1.5", loc="left", fontweight="bold", fontsize=10)
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    rp.style(ax)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    plt.close(fig)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

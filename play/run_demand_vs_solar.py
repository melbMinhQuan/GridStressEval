"""Current model: circuit demand vs rooftop solar per circuit over the day.

Demand: expected total circuit load for A2 and A3 at their safest allocation (w(n)-weighted),
including the baseline N P_S. Solar: 2025 VIC1 rooftop / all circuits (mean and 10-90% of days).

Run from the repo root:  python3 -m play.run_demand_vs_solar
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
PLOT = config.OUT_DIR / "demand_vs_solar.png"


def main():
    config.SOLAR_SOURCE = "rooftop_local"
    from play.experiments.common import safest_added_profile
    from play.soak import SLOT
    from play.solar import excess_per_circuit
    x = excess_per_circuit(None, config.UPTAKE)                       # (days, slots) kW per circuit
    scale = config.DEMAND_SCALE
    base = scale * config.N * config.load_kw()["S"]
    hours = SLOT + config.DT / 2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lo, hi = np.percentile(x, [10, 90], axis=0)
    ax.fill_between(hours, lo, hi, color="#F1C40F", alpha=0.25, lw=0, label="rooftop solar, 10-90% of days")
    ax.plot(hours, x.mean(axis=0), color="#B7950B", lw=2.5, label=f"rooftop solar, mean (peak {x.mean(0).max():.0f} kW)")
    ax.axhline(base, color="#555", ls="--", lw=1.5, label=f"baseline demand {scale:g} x {config.N} x {config.load_kw()['S']:.1f} kW = {base:.1f} kW")
    for (label, d), color in zip(DESIGNS.items(), COLORS):
        added = scale * safest_added_profile(d, config.UPTAKE)
        inside = added > 0
        ax.step(hours, np.where(inside, base + added, np.nan), where="mid", color=color, lw=2.2,
                label=f"total demand, {label} (safest allocation)")
        print(f"{label:<34} total demand in intervals {np.nanmin(np.where(inside, base + added, np.nan)):.0f}-"
              f"{(base + added).max():.0f} kW")
    print(f"baseline {base:.0f} kW; rooftop per circuit mean peak {x.mean(0).max():.1f} kW, max {x.max():.1f} kW")
    ax.set_xlim(5, 20)
    ax.set_xticks(range(5, 21))
    ax.set_ylim(0, None)
    ax.set_xlabel("Hour (AEST)")
    ax.set_ylabel("kW per circuit (40 customers)")
    ax.set_title(f"Circuit demand (x DEMAND_SCALE = {scale:g}) vs rooftop solar per circuit (2025 average)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

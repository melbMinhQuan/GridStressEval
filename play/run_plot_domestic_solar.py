"""Plot 2025 VIC1 domestic (rooftop) solar from Open Electricity (hourly, MW, market time UTC+10).

Run from the repo root:  python3 -m play.run_plot_domestic_solar
"""
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from play import config

PLOT = config.OUT_DIR / "domestic_solar.png"
SEASONS = {"Summer": (12, 1, 2), "Autumn": (3, 4, 5), "Winter": (6, 7, 8), "Spring": (9, 10, 11)}
COLORS = {"Summer": "#D9822B", "Autumn": "#8E44AD", "Winter": "#2A7FB8", "Spring": "#2E8B57"}
SEED = 7


def main():
    with open(config.ROOFTOP_DEMAND_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    mw = np.array([float(r["solar_rooftop_mw"]) for r in rows]).reshape(-1, 24)
    days = np.array([np.datetime64(r["interval_start"][:10]) for r in rows[::24]])
    months = np.array([int(str(d)[5:7]) for d in days])
    hours = np.arange(24) + 0.5

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])

    ax = fig.add_subplot(gs[0, :])                                         # whole year, hourly
    t = np.repeat(days, 24).astype("datetime64[h]") + np.tile(np.arange(24), len(days)).astype("timedelta64[h]")
    ax.plot(t, mw.ravel(), color="#D9822B", lw=0.3, alpha=0.6, label="hourly MW")
    ax.plot(days, mw.max(axis=1), color="#8B4513", lw=1.2, label="daily peak MW")
    ax.set_ylabel("VIC1 rooftop solar (MW)")
    ax.set_title(f"2025 VIC1 domestic (rooftop) solar: total {mw.sum() / 1e6:.2f} TWh, peak {mw.max():,.0f} MW",
                 loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(frameon=False, loc="upper center", ncol=2)

    rng = np.random.default_rng(SEED)
    for i, (season, ms) in enumerate(SEASONS.items()):                    # season profiles
        ax2 = fig.add_subplot(gs[1, i], sharey=None if i == 0 else fig.axes[1])
        sel = np.isin(months, ms)
        lo, hi = np.percentile(mw[sel], [10, 90], axis=0)
        ax2.fill_between(hours, lo, hi, color=COLORS[season], alpha=0.15, lw=0, label="10-90% of days")
        ax2.plot(hours, mw[sel].mean(axis=0), color=COLORS[season], lw=2.5, label="mean")
        for d in rng.choice(np.flatnonzero(sel), 3, replace=False):
            ax2.plot(hours, mw[d], color=COLORS[season], lw=0.8, ls="--", alpha=0.7)
        ax2.set_xlim(4, 20)
        ax2.set_xticks(range(4, 21, 4))
        ax2.set_title(f"{season}: mean daily {mw[sel].sum(axis=1).mean() / 1000:.1f} GWh", loc="left", fontweight="bold")
        ax2.set_xlabel("Hour (AEST)")
        if i == 0:
            ax2.set_ylabel("MW (dashed: 3 random days)")
            ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for a in fig.axes:
        a.grid(color="#e6e6e6")
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"total {mw.sum() / 1e6:.2f} TWh, peak {mw.max():,.0f} MW, per circuit peak "
          f"{mw.max() * 1000 / (config.VIC_CUSTOMERS / config.N):.0f} kW (all {config.VIC_CUSTOMERS / config.N:,.0f} circuits)")
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot 5-minute utility-solar excess (curtailment) for one day per season.

Reads the CSVs produced by pull_curtailment.py. Seasons are southern-hemisphere
(Summer = DJF, Autumn = MAM, Winter = JJA, Spring = SON). By default the day
shown for each season is the one with the most curtailed solar energy; pass
``--pick median`` for a typical day, or ``--dates`` to choose them yourself.

Usage:
    python script/plot_solar_excess.py
    python script/plot_solar_excess.py --pick median
    python script/plot_solar_excess.py --dates 2025-01-15 2025-04-15 2025-07-15 2025-10-15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT.parent / "figures" / "solar_excess"

SEASONS = ["Summer", "Autumn", "Winter", "Spring"]
SEASON_OF_MONTH = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn", 4: "Autumn", 5: "Autumn",
    6: "Winter", 7: "Winter", 8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}
COLOR = "#D9822B"  # single series -> one hue, no legend


def pick_days(daily: pd.DataFrame, how: str) -> dict[str, pd.Timestamp]:
    daily = daily.assign(season=daily.index.month.map(SEASON_OF_MONTH))
    days = {}
    for season in SEASONS:
        energy = daily.loc[daily.season == season, "curtailment_solar_utility_energy"]
        if how == "peak":
            days[season] = energy.idxmax()
        else:  # day whose energy is closest to the seasonal median
            days[season] = (energy - energy.median()).abs().idxmin()
    return days


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="VIC1")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--pick", choices=["peak", "median"], default="peak")
    parser.add_argument("--dates", nargs=4, metavar="YYYY-MM-DD",
                        help="explicit Summer Autumn Winter Spring dates")
    args = parser.parse_args()

    stem = f"NEM_{args.region}_curtailment_{args.year}"
    five = pd.read_csv(DATA / f"{stem}_5m.csv", parse_dates=["interval_start"])
    five = five.set_index("interval_start")
    daily = pd.read_csv(DATA / f"{stem}_1d.csv", parse_dates=["interval_start"])
    daily = daily.set_index("interval_start")

    if args.dates:
        tz = five.index.tz
        days = {s: pd.Timestamp(d, tz=tz) for s, d in zip(SEASONS, args.dates)}
    else:
        days = pick_days(daily, args.pick)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, season in zip(axes.flat, SEASONS):
        day = days[season].normalize()
        series = five.loc[day: day + pd.Timedelta(minutes=23 * 60 + 55),
                          "curtailment_solar_utility"]
        # Plot against a fixed reference date so all panels share a 00:00-24:00 axis.
        x = pd.Timestamp("2000-01-01") + (series.index - day)
        ax.fill_between(x, series.values, color=COLOR, alpha=0.18, linewidth=0)
        ax.plot(x, series.values, color=COLOR, linewidth=2)

        energy = series.sum() / 12  # MW over 5-min intervals -> MWh
        peak_t = series.idxmax()
        ax.set_title(f"{season} — {day:%a %d %b %Y}", loc="left", fontsize=11,
                     fontweight="bold", color="#222")
        ax.text(0.02, 0.93,
                f"{energy:,.0f} MWh curtailed · peak {series.max():,.0f} MW at {peak_t:%H:%M}",
                transform=ax.transAxes, fontsize=9, color="#555", va="top")

        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bbb")
        ax.tick_params(colors="#555", labelsize=9)

    ax = axes[0, 0]
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ymax = max(a.get_ylim()[1] for a in axes.flat)
    ax.set_ylim(0, ymax * 1.12)  # headroom so the annotation clears the peak
    for a in axes[:, 0]:
        a.set_ylabel("Solar excess (MW)", color="#555")
    for a in axes[1, :]:
        a.set_xlabel("Time of day (AEST, UTC+10)", color="#555")

    label = "highest-curtailment day" if not args.dates and args.pick == "peak" else (
        "median day" if not args.dates else "selected day")
    fig.suptitle(f"{args.region} utility-solar excess (curtailment), 5-minute, "
                 f"{label} per season, {args.year}",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "selected" if args.dates else args.pick
    path = OUT / f"{args.region}_{args.year}_solar_excess_by_season_{suffix}.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")
    for season in SEASONS:
        print(f"  {season}: {days[season]:%Y-%m-%d}")


if __name__ == "__main__":
    main()

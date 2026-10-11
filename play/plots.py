"""Figures: one circuit's load vs solar curtailment, and Pareto fronts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock
from play.soak import SLOT

COLORS = {1: "#6B6B6B", 2: "#2A7FB8", 3: "#D9822B"}


def _style(ax):
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path):
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_allocation(path, intervals, base, added, excess, soaked, soak_title):
    """Interval timeline, circuit load, and added load vs curtailment with the soak shaded."""
    fig, (strip, top, bottom) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                             gridspec_kw={"height_ratios": [0.35, 1, 1.3]})
    hours = SLOT + config.DT / 2
    inside = np.zeros(len(SLOT), bool)
    for a, b in intervals:
        inside |= (SLOT >= a) & (SLOT < b)
    shown = lambda y: np.where(inside, y, np.nan)                     # time outside intervals ignored

    for i, (a, b) in enumerate(intervals):
        strip.barh(i, b - a, left=a, height=0.6, color="#888")
        strip.text(a - 0.1, i, f"I{i + 1}", ha="right", va="center", fontsize=9)
        strip.text((a + b) / 2, i, f"{clock(a)}-{clock(b)}", ha="center", va="center", fontsize=8, color="white")
    strip.set_ylim(len(intervals) - 0.4, -0.6)
    strip.set_yticks([])
    strip.set_title("Free intervals", loc="left", fontweight="bold")
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)

    for ax in (top, bottom):
        ax.axvspan(min(a for a, _ in intervals), max(b for _, b in intervals), color="#888", alpha=0.07, lw=0)
        _style(ax)

    top.step(hours, shown(base + added), where="mid", color="#2A7FB8", lw=2, label="circuit load (all customers)")
    top.step(hours, shown(base), where="mid", color="#2A7FB8", lw=1, ls="--", label="small load only (baseline)")
    top.set_ylim(0, None)
    top.set_ylabel("kW per circuit")
    top.set_title(f"Customer load, one circuit ({config.N} customers)", loc="left", fontweight="bold")
    top.legend(frameon=False, loc="lower right")

    bottom.plot(hours, excess.mean(axis=0), color="#D9822B", lw=2, label="solar curtailment share")
    bottom.fill_between(hours, 0, shown(soaked), step="mid", color="#2E8B57", alpha=0.35, lw=0,
                        label="solar soak = min(added load, curtailment)")
    bottom.step(hours, shown(added), where="mid", color="#2A7FB8", lw=2, label="load added by free intervals")
    bottom.set_ylabel("kW per circuit")
    bottom.set_xlabel("Time of day")
    bottom.set_title(soak_title, loc="left", fontweight="bold")
    bottom.legend(frameon=False, loc="upper right")
    bottom.set_xlim(5, 20)
    bottom.set_xticks(range(5, 21))
    bottom.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: clock(v)))
    _save(fig, path)


def plot_fronts(path, fronts, uptakes, ks, title):
    """One panel per uptake with the Pareto fronts for the interval counts in ks."""
    fig, axes = plt.subplots(1, len(uptakes), figsize=(6 * len(uptakes), 5.2), squeeze=False)
    for ax, uptake in zip(axes[0], uptakes):
        for k in ks:
            front = fronts[uptake][k]
            ring = {1: dict(ms=10), 2: dict(mfc="none", mew=2, ms=11), 3: dict(ms=5)}[k]  # 2 as ring: may sit under 3
            ax.plot([p[0] * 100 for p in front], [p[1] for p in front], "-o", color=COLORS[k], lw=2,
                    label=f"{k} interval{'s' * (k > 1)}", zorder=3 if k == 2 else 4, **ring)
        ax.margins(0.15)
        xs = [p[0] * 100 for k in ks for p in fronts[uptake][k]]
        pad = max(0.15 * (max(xs) - min(xs)), 0.05)
        ax.set_xlim(min(xs) - pad, min(max(xs) + pad, 100.02))          # share cannot exceed 100%
        for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            digits = max(1, int(np.ceil(-np.log10(max(hi - lo, 1e-9)))) + 1)   # enough to tell ticks apart
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=digits: f"{v:.{d}f}"))
        ax.set_title(f"Uptake {uptake:.0%} ({config.circuits(uptake):,.0f} circuits)", loc="left", fontweight="bold")
        ax.set_xlabel(f"{'Compositions' if 'equal' in title else 'Circuits'} without overload (%)  →  better")
        _style(ax)
    axes[0][0].set_ylabel("Expected solar soak per circuit (kWh/day)  ↑  better")
    axes[0][-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_summary(path, fronts, uptakes, uniform=False):
    """Safest point of each front: circuits needing an upgrade, and soak, for 1, 2, 3 intervals."""
    ks = sorted(next(iter(fronts.values())))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})

    over = [(1 - fronts[uptakes[0]][k][-1][0]) * 100 for k in ks]      # same for every uptake
    bars = left.bar([f"{k} interval{'s' * (k > 1)}" for k in ks], over, color=[COLORS[k] for k in ks], width=0.6)
    left.bar_label(bars, labels=[f"{v:.1f}%" for v in over], padding=3)
    unit = "Compositions" if uniform else "Circuits"
    left.set_ylabel(f"{unit} that overload (%)  ↓  better")
    left.set_title(f"{unit} needing an upgrade", loc="left", fontweight="bold")
    left.set_ylim(0, max(over) * 1.15)

    width = 0.8 / len(ks)
    for i, k in enumerate(ks):
        x = np.arange(len(uptakes)) + (i - (len(ks) - 1) / 2) * width
        soak = [fronts[u][k][-1][1] for u in uptakes]
        bars = right.bar(x, soak, width * 0.92, color=COLORS[k], label=f"{k} interval{'s' * (k > 1)}")
        right.bar_label(bars, labels=[f"{v:.0f}" for v in soak], padding=2, fontsize=8)
    right.set_xticks(np.arange(len(uptakes)), [f"uptake {u:.0%}" for u in uptakes])
    right.set_ylabel("Solar soak per circuit (kWh/day)  ↑  better")
    right.set_title("Curtailed solar soaked (2025 average)", loc="left", fontweight="bold")
    right.legend(frameon=False)
    for ax in (left, right):
        _style(ax)
        ax.grid(axis="x", visible=False)
    mix = ("every composition weighted equally" if uniform else
           f"mix {config.P_TYPE['S']:.0%} S / {config.P_TYPE['M']:.0%} M / {config.P_TYPE['L']:.0%} L")
    fig.suptitle(f"Safest allocation per circuit (All-On, {mix})", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def plot_soak_profile(path, profiles, curtailment, uptakes):
    """Mean soak over the day for 2 and 3 intervals (safest allocation), one panel per uptake."""
    hours = SLOT + config.DT / 2
    fig, axes = plt.subplots(1, len(uptakes), figsize=(6 * len(uptakes), 4.8), squeeze=False)
    for ax, uptake in zip(axes[0], uptakes):
        ax.plot(hours, curtailment[uptake], color="#999", lw=1.2, ls="--", label="curtailment share (mean)")
        for k, style in ((2, dict(lw=3)), (3, dict(lw=1.8, ls=(0, (4, 2))))):
            p = profiles[uptake][k]
            ax.plot(hours, p, color=COLORS[k], label=f"{k} intervals: {p.sum() * config.DT:.1f} kWh/day", **style)
        for k, (a, b) in enumerate(config.INTERVAL_SETS[3]):
            ax.axvspan(a, b, color="#888", alpha=0.05, lw=0)
        ax.set_xlim(6, 16)
        ax.set_xticks(range(6, 17, 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: clock(v)))
        ax.set_ylim(0, None)
        ax.set_title(f"Uptake {uptake:.0%} ({config.circuits(uptake):,.0f} circuits)", loc="left", fontweight="bold")
        ax.set_xlabel("Time of day")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        _style(ax)
    axes[0][0].set_ylabel("Mean solar soak per circuit (kW)")
    fig.suptitle("Mean solar soak over the day, safest allocation (2025 average, all compositions)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)

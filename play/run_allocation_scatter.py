"""Allocation strategies for 1, 2 and 3 intervals: (overload, soak) of many strategies, the ideal point and
the strategy closest to it. Seasonal intervals and model of play/run_participation.py, All-On.

A strategy gives every circuit composition n a share of its medium and large customers per interval
(largest remainder, M and L separately):
    share rules   the same shares for every circuit, on a 0.05 grid
    random        each composition draws its own shares from a flat Dirichlet
    optimised     lowest overload and best soak, chosen per composition (play/run_participation.py)
Ideal point (fixed): 0% overload and all solar inside the interval window (08:00-16:00), the most any
design could soak. Distance: sqrt(overload^2 + (1 - soak/ideal)^2), both on a 0-1 scale.

Run from the repo root:  python3 -m play.run_allocation_scatter
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from play import config
from play import run_participation as rp
from play.circuit import segments
from play.experiments.common import weights
from play.soak import SLOT, in_range
from play.solar import _profile_mw

MIXES = {"Today's mix (ρ = 31%)": 0.31, "ρ = 50%": 0.50}
SCALES = [0.10, 0.13, 0.20]                       # uptakes, one panel each
ZOOM_X = (-1, 25)                                 # overload % of the zoomed row; its y range fits each panel
STEP = 0.05                                       # share-rule grid
N_RANDOM = 300                                    # random strategies per design
SEED = 7
PLOT = config.OUT_DIR / "report" / "fig2_allocation_scatter.png"
LOAD_PLOT = config.OUT_DIR / "report" / "Evidence_for_seasonal_weights.png"
GROSS = "--soak" in sys.argv                       # plain soak (no excess subtracted) instead of net soak
if GROSS:
    rp.NET_SOAK = False
    PLOT = config.ROOT / "doc" / "vis" / "fig2_allocation_scatter_soak.png"
FILL = "--window" in sys.argv                       # plain soak over 10-15 for every design, share of 10-15 solar
if FILL:
    rp.SOAK_HOURS, rp.SOAK_WINDOW, rp.REFERENCE, rp.NET_SOAK = "window", (10, 15), (10, 15), False
    PLOT = config.ROOT / "doc" / "vis" / "fig4_10to15.png"
LABEL = ("Soak (% of rooftop solar, 10–15)" if FILL else "Soak (% of rooftop solar, 08–16)" if GROSS
         else rp.SOAK_LABEL)


def split(counts, shares):
    """Largest-remainder split of counts[i] customers by shares[i] (rows)."""
    raw = counts[:, None] * shares
    parts = np.floor(raw).astype(int)
    need = counts - parts.sum(axis=1)
    order = np.argsort(-(raw - parts), axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(len(counts))[:, None], order] = np.arange(shares.shape[1])
    return parts + (rank < need[:, None])


def share_grid(k):
    ticks = round(1 / STEP)
    return [np.array(h + (ticks - sum(h),)) / ticks for h in product(range(ticks + 1), repeat=k - 1)
            if sum(h) <= ticks]


class Setting:
    """One design and customer mix: evaluates (overload %, soak MWh/day, mean share per interval) of
    per-composition shares."""

    def __init__(self, designs, rho):
        w = weights(config.N, rp.mix(rho))
        total = sum(w.values())
        self.keys = list(w)
        self.w = np.array([w[key] for key in self.keys]) / total
        self.n_m = np.array([key[0] for key in self.keys])
        self.n_l = np.array([key[1] for key in self.keys])
        self.designs = designs
        self.days = {s: len(rp.solar_season(s)) for s in rp.SEASONS}

    def __call__(self, shares):
        m, l = split(self.n_m, shares), split(self.n_l, shares)
        amps, n, kw = config.CURRENT, config.N, config.load_kw()
        over = np.zeros(len(self.keys), bool)
        for design in self.designs.values():
            for _, _, cov in segments(list(design)):
                mc, lc = m[:, cov].sum(axis=1), l[:, cov].sum(axis=1)
                over |= amps["S"] * (n - mc - lc) + amps["M"] * mc + amps["L"] * lc > config.T
        to_mw = rp.UPTAKE * config.VIC_CUSTOMERS / n / 1000
        per_interval = to_mw * self.w @ ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l)    # MW
        soak = 0.0
        for s, design in self.designs.items():
            load = np.full(config.SLOTS, rp.base_mw(s))                         # normal load of plan customers
            for (a, b), mw in zip(design, per_interval):
                load[in_range(a, b)] += mw * rp.season_weight(s)                  # load counted for soak
            soak += rp.net(load, rp.solar_season(s), design, s) * self.days[s]
        mean_share = self.w @ (m + l) / (self.w @ (self.n_m + self.n_l))       # share of M + L per interval
        return (100 * self.w @ over, soak / sum(self.days.values()), *mean_share)


def optimised(designs, rho, strategy):
    """(overload %, soak MWh/day, mean share per interval) of a per-circuit optimised strategy."""
    over, soak = rp.combine(designs, rho, strategy)
    days = {s: len(rp.solar_season(s)) for s in rp.SEASONS}
    share = 0.0
    for s, design in designs.items():
        _, _, w, alloc = rp.evaluate(design, s, rho, strategy)
        counts = sum(w[key] * (m + l) for key, (m, l) in alloc.items())
        share = share + counts / counts.sum() * days[s] / sum(days.values())
    return (over, soak, *share)


def points(k, designs, rho, rng):
    """{family: array of (overload %, soak, mean share per interval)} for one design."""
    setting = Setting(designs, rho)
    ones = np.ones((len(setting.keys), 1))
    out = {"share rules": np.array([setting(ones * s) for s in share_grid(k)])}
    if k > 1:
        out["random"] = np.array([setting(rng.dirichlet(np.ones(k), size=len(setting.keys)))
                                  for _ in range(N_RANDOM)])
        for st in ("lowest overload", "best soak"):
            out[st] = np.array([optimised(designs, rho, st)])
    return out


def interval_mw(rho, shares=None, design=None, season=None):
    """Added VIC load per interval counted for soak (MW, times w_s): a share rule (shares) or the best-soak
    split of design in season."""
    kw, n = config.load_kw(), config.N
    to_mw = rp.UPTAKE * config.VIC_CUSTOMERS / n / 1000 * rp.season_weight(season)
    if shares is None:
        _, _, w, alloc = rp.evaluate(design, season, rho, "best soak")
        return to_mw * sum(w[key] * ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l) for key, (m, l) in alloc.items())
    setting = Setting({}, rho)
    shares = np.ones((len(setting.keys), 1)) * np.asarray(shares)
    m, l = split(setting.n_m, shares), split(setting.n_l, shares)
    return to_mw * setting.w @ ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l)


def load_figure(designs, rho):
    """Solar and plan load per season: 1 interval, 2 intervals and 3 intervals (best soak)."""
    fig, axes = plt.subplots(1, len(rp.SEASONS), figsize=(16, 4.2), sharey=True)
    for ax, season in zip(axes, rp.SEASONS):
        x = rp.solar_season(season)
        ax.fill_between(SLOT, np.percentile(x, 10, axis=0), np.percentile(x, 90, axis=0), color="#bbbbbb",
                        alpha=0.5, lw=0, label="Rooftop, 10–90% of days")
        ax.plot(SLOT, x.mean(axis=0), color="#333333", lw=1.6, label="Rooftop, mean")
        cases = {"1 interval": (designs[1][season], interval_mw(rho, design=designs[1][season], season=season), "-", rp.COLORS[1]),
                 "2 intervals": (designs[2][season], interval_mw(rho, design=designs[2][season], season=season), "-", rp.COLORS[2]),
                 "3 intervals, best soak": (designs[3][season], interval_mw(rho, design=designs[3][season], season=season), "-", rp.COLORS[3])}
        for label, (design, mw, ls, color) in cases.items():
            load = np.full(config.SLOTS, rp.base_mw(season))
            for (a, b), v in zip(design, mw):
                load[in_range(a, b)] += v
            soak = rp.net(load, x, design, season)
            wide = label == "2 intervals"                                       # drawn wide underneath 3
            ax.plot(SLOT, np.where(rp.counted(design, season), load, np.nan), ls=ls, lw=5 if wide else 2, alpha=0.5 if wide else 1, zorder=2 if wide else 3,
                    color=color, label=label)
            print(f"  {season:<7} {label:<24} split {' / '.join(f'{v:.0f}' for v in mw)} MW  soak {soak:7.1f} MWh/day"
                  f" ({rp.pct(soak, season):.2f}% of the season's 08-16 rooftop solar)")
        ax.set_xlim(6, 18)
        ax.set_ylim(top=1.3 * max(np.percentile(rp.solar_season(t), 90, axis=0).max() for t in rp.SEASONS))
        ax.set_title(season, loc="left", fontweight="bold")
        ax.set_xlabel("Hour")
        rp.style(ax)
    axes[0].set_ylabel("MW")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(LOAD_PLOT, dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    designs = {k: {s: rp.choose(k, s) for s in rp.SEASONS} for k in (1, 2, 3)}
    ideal = rp.ideal()                                                          # rooftop solar 08-16, MWh/day
    dist = lambda p: np.hypot(p[:, 0] / 100, 1 - p[:, 1] / ideal)
    print(f"ideal: 0% overload, {ideal:.1f} MWh/day")
    base = rp.UPTAKE
    x0, x1 = ZOOM_X
    fig, grid = plt.subplots(2, len(SCALES), figsize=(15, 8.6))
    for ax, zoom, f in zip(grid[0], grid[1], SCALES):
        rp.UPTAKE = f
        rho = rp.TODAY
        rng = np.random.default_rng(SEED)                                       # same random strategies per panel
        pts = {k: points(k, designs[k], rho, rng) for k in (1, 2, 3)}
        title = f"u = {100 * rp.UPTAKE:.0f}%"
        print(f"{title} (uptake {rp.UPTAKE:.1%}), today's mix")
        for k in (3, 2, 1):                                                     # 1 interval drawn on top
            lab = f"{k} interval{'s' * (k > 1)}"
            for fam, p in pts[k].items():
                style = {"lowest overload": dict(s=60, marker="o", edgecolor="black", lw=0.8, zorder=4),
                         "best soak": dict(s=55, marker="D", edgecolor="black", lw=0.8, zorder=4),
                         "share rules": dict(s=16, lw=0, zorder=3),
                         "random": dict(s=10, alpha=0.25, lw=0, zorder=2)}[fam] if k > 1 else dict(s=70, zorder=4)
                for a in (ax, zoom):
                    a.scatter(p[:, 0], rp.pct(p[:, 1]), color=rp.COLORS[k], **style)
            allp = np.vstack(list(pts[k].values()))
            j = dist(allp).argmin()
            for a in (ax, zoom):
                a.scatter(allp[j, 0], rp.pct(allp[j, 1]), s=300, facecolor="none", edgecolor=rp.COLORS[k], lw=2.2,
                          zorder=5)
            fams = {"share rule": pts[k]["share rules"]}
            if k > 1:
                fams.update({st: pts[k][st] for st in ("random", "lowest overload", "best soak")})
            for fam, p in fams.items():                                         # best of each family
                b = p[dist(p).argmin()]
                print(f"  {lab:<12} {fam:<16} shares {' / '.join(f'{v:.0%}' for v in b[2:]):<16}"
                      f" overload {b[0]:6.2f}%  soak {b[1]:6.1f} ({rp.pct(b[1]):.1f}%)  distance {dist(p).min():.3f}")
        ax.scatter(0, 100, s=220, marker="*", color="#d62728", zorder=6)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Circuits that overload (%)")
        ax.set_xlim(-3, 103)
        near = np.vstack([p[:, :2] for k in (2, 3) for p in pts[k].values()])
        near = rp.pct(near[near[:, 0] <= x1, 1])
        y0, y1 = near.min() - 0.3, near.max() + 0.3
        ax.set_ylim(top=102)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec="#8a8984"))
        zoom.set_xlim(x0, x1)
        zoom.set_ylim(y0, y1)
        zoom.set_xlabel("Circuits that overload (%)")
        rp.style(ax)
        rp.style(zoom)
    rp.UPTAKE = base
    fig.supylabel(LABEL, fontsize=10)
    mark = lambda label, **kw: Line2D([], [], ls="", label=label, **kw)
    handles = ([mark(f"{k} interval{'s' * (k > 1)}", marker="s", ms=8, color=rp.COLORS[k]) for k in (1, 2, 3)]
               + [mark("share rule", marker="o", ms=4, color="#52514e"),
                  mark("random", marker="o", ms=4, color="#52514e", alpha=0.3),
                  mark("lowest overload", marker="o", ms=8, mfc="white", mec="black"),
                  mark("best soak", marker="D", ms=7, mfc="white", mec="black"),
                  mark("ideal", marker="*", ms=14, color="#d62728"),
                  mark("optimised", marker="o", ms=15, mfc="none", mec="#52514e", mew=2)])
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")
    print("Added load and soak at today's mix")
    if not (GROSS or FILL):
        load_figure(designs, rp.TODAY)
    print(f"graph: {LOAD_PLOT}")


if __name__ == "__main__":
    main()

"""Load scale: the calibrated uptake times a factor k (1.0 = calibrated). Overload does not depend on k.
Figures: the total load of two intervals against mean rooftop solar per season, and soak against k.

Run from the repo root:  python3 -m play.run_load_scale
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from play import config
from play import run_participation as rp
from play.soak import SLOT, in_range

K = [0.10, 0.13, 0.16, 0.20]                      # uptakes
SHADES = ["#fbd3bd", "#f6ad86", "#f0905c", "#e4743e", "#c75a27", "#9c3d12"]   # one hue, light -> dark with k
OUT = config.OUT_DIR / "load_scale"
K_REPORT = [0.10, 0.13, 0.20]                     # uptakes shown in the report
REPORT = config.OUT_DIR / "report"
ZOOM_X = (-1, 20)                                 # overload % of the zoomed row of the report figure
ZOOM_DEPTH = 6                                    # net soak points shown below the top of the 2 and 3 curves
CACHE = config.OUT_DIR / "load_scale" / "tradeoff.npz"
BEST = "--best" in sys.argv


def plan_load(design, season):
    """Total load counted for soak (MW per slot) of the lowest-overload allocation at today's mix."""
    kw = config.load_kw()
    to_mw = rp.UPTAKE * config.VIC_CUSTOMERS / config.N / 1000 * rp.season_weight(season)
    _, _, w, alloc = rp.evaluate(design, season, rp.TODAY, "lowest overload")
    load = np.full(config.SLOTS, rp.base_mw(season))
    for (a, b), (mw) in zip(design, to_mw * sum(w[key] * ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l)
                                                for key, (m, l) in alloc.items())):
        load[in_range(a, b)] += mw
    return load


def flat_load(design, season):
    """Plan load (MW per slot) at today's mix with the added load split exactly evenly between the intervals,
    so it is flat over non-overlapping intervals; drawn to show the size of the load against the solar."""
    kw, p = config.load_kw(), rp.mix(rp.TODAY)
    added = rp.UPTAKE * config.VIC_CUSTOMERS / 1000 * rp.season_weight(season) * (
        p["M"] * (kw["M"] - kw["S"]) + p["L"] * (kw["L"] - kw["S"]))                # MW, all intervals
    load = np.full(config.SLOTS, rp.base_mw(season))
    for a, b in design:
        load[in_range(a, b)] += added / len(design)
    return load


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = rp.UPTAKE
    designs = {k: {s: rp.choose(k, s) for s in rp.SEASONS} for k in (1, 2, 3)}
    loads, soak = {}, {}
    for f in K:
        rp.UPTAKE = f
        loads[f] = {s: flat_load(designs[2][s], s) for s in rp.SEASONS}
        soak[f] = {k: rp.combine(d, rp.TODAY, "lowest overload")[1] for k, d in designs.items()}
        print(f"u={f:.0%}: summer peak {loads[f]['Summer'].max():5.0f} MW;  soak "
              + "  ".join(f"{k} int {soak[f][k]:7.0f}" for k in (1, 2, 3)), flush=True)
    rp.UPTAKE = base

    fig, axes = plt.subplots(1, len(rp.SEASONS), figsize=(14, 3.4), sharey=True)
    for ax, season in zip(axes, rp.SEASONS):
        ax.plot(SLOT, rp.solar_season(season).mean(axis=0), color="#333333", lw=1.8, label="Rooftop solar")
        for f, c in zip(K, SHADES):
            ax.plot(SLOT, np.where(rp.counted(designs[2][season], season), loads[f][season], np.nan), lw=1.8, color=c,
                    label=f"u = {100 * f:.0f}%")
        ax.set_xlim(6, 18)
        ax.set_xticks([6, 9, 12, 15, 18])
        ax.set_title(season, loc="left", fontweight="bold", fontsize=10)
        rp.style(ax)
    axes[0].set_ylabel("MW")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=7)
    fig.supxlabel("Hour", fontsize=10, y=0.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "load_scale_shapes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k in (1, 2, 3):
        ax.plot(K, [soak[f][k] for f in K], "-o", ms=4, lw=2, color=rp.COLORS[k], label=f"{k} interval{'s' * (k > 1)}")
    ax.set_xlabel("Load scale k")
    ax.set_ylabel("Net soak (MWh/day)")
    ax.legend(frameon=False, fontsize=8)
    rp.style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "load_scale_soak.png", dpi=150)
    plt.close(fig)
    report_figures(designs, loads, base)
    print(f"figures in {OUT} and {REPORT}")


def report_figures(designs, loads, base):
    """Report: total load per season for K_REPORT, and overload against net soak as rho grows per scale."""
    load_figure(designs, loads, base)
    with Pool(len(K_REPORT) * 3) as pool:
        out = pool.map(scale_result, [(f, k) for f in K_REPORT for k in (1, 2, 3)])
    res = {(f, k): r for (f, k), r in zip([(f, k) for f in K_REPORT for k in (1, 2, 3)], out)}
    np.savez(CACHE, **{f"{f}_{k}": np.stack(r) for (f, k), r in res.items()})
    tradeoff_figure(res)


def load_figure(designs, loads, base):
    """Total load of two intervals per season for each uptake of K_REPORT, against mean rooftop solar."""
    shades = dict(zip(K_REPORT, ("#f0905c", "#c75a27", "#7a2e0b")))
    fig, axes = plt.subplots(1, len(rp.SEASONS), figsize=(14, 3.4), sharey=True)
    for ax, season in zip(axes, rp.SEASONS):
        ax.plot(SLOT, rp.solar_season(season).mean(axis=0), color="#333333", lw=1.8, label="Rooftop solar")
        for f in K_REPORT:
            ax.plot(SLOT, np.where(rp.counted(designs[2][season], season), loads[f][season], np.nan), lw=1.8,
                    color=shades[f], label=f"u = {100 * f:.0f}%")              # drawn where soak is counted
        ax.set_xlim(6, 18)
        ax.set_xticks([6, 9, 12, 15, 18])
        ax.set_title(season, loc="left", fontweight="bold", fontsize=10)
        rp.style(ax)
    axes[0].set_ylabel("MW")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.supxlabel("Hour", fontsize=10, y=0.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(REPORT / "fig2b_load_scale.png", dpi=150)
    plt.close(fig)


def scale_result(job):
    """(lowest overload, best soak) arrays of (overload %, soak, excess) over RHO for uptake f and K intervals;
    excess = soak - net soak (MWh/day)."""
    f, k = job
    rp.UPTAKE = f
    d = {s: rp.choose(k, s) for s in rp.SEASONS}

    def run(strategy):
        rows = []
        for rho in rp.RHO:
            over, soak = rp.combine(d, rho, strategy)
            rp.NET_SOAK = True
            net = rp.combine(d, rho, strategy)[1]
            rp.NET_SOAK = False
            rows.append((over, soak, soak - net))
        return np.array(rows)

    lo = run("lowest overload")
    best = run("best soak") if k > 1 and BEST else lo                          # --best: also best soak (slow)
    i = np.flatnonzero(np.isclose(rp.RHO, rp.TODAY))[0]
    cost = 100 * (best[:, 1] - lo[:, 1]) / np.abs(best[:, 1])
    cap = np.flatnonzero(lo[:, 0] <= 5)[-1]
    peak = lo[:, 1].argmax()
    print(f"u={f:.0%} K={k}: today {lo[i, 0]:.2f}% / {lo[i, 1]:,.0f} ({rp.pct(lo[i, 1]):.1f}%);  peak rho "
          f"{rp.RHO[peak]:.1%} ({rp.pct(lo[peak, 1]):.1f}%);  cap 5%: rho {rp.RHO[cap]:.1%} -> "
          f"{lo[cap, 1]:,.0f};  max soak given up {cost.max():.2f}%;  max overload gap "
          f"{np.max(best[:, 0] - lo[:, 0]):.2f} pts;  best soak today {best[i, 0]:.2f}% / {rp.pct(best[i, 1]):.1f}%",
          flush=True)
    return lo, best


def uptake_title(f):
    return f"u = {100 * f:.0f}%"


def tradeoff_figure(res, panels=K_REPORT, title=uptake_title, path=REPORT / "fig2_tradeoff_scale.png",
                    ylabel=None, names=("soak", "excess load")):
    """Overload against net soak as rho grows (lowest overload), one panel per res key (load scale by default);
    bottom: zoom on 2 and 3."""
    today = np.isclose(rp.RHO, rp.TODAY)
    x0, x1 = ZOOM_X
    fig, axes = plt.subplots(2, len(panels), figsize=(14, 7.6))
    for j, f in enumerate(panels):
        now = [rp.pct(res[(f, k)][0][today, 1])[0] for k in (2, 3)]
        y0, y1 = min(now) - ZOOM_DEPTH / 2, max(now) + ZOOM_DEPTH / 2           # zoom box around today's mix
        for row, ks in ((0, (1, 2, 3)), (1, (2, 3))):                          # bottom: zoom on 2 and 3
            ax = axes[row, j]
            for k in ks:
                lo = res[(f, k)][0]
                y = rp.pct(lo[:, 1])
                ax.plot(lo[:, 0], y, "-o", ms=3 + row, lw=2, color=rp.COLORS[k], label=f"{k} interval{'s' * (k > 1)}")
                if row == 0 and lo.shape[1] > 2:                                # excess load, same unit
                    ax.plot(lo[:, 0], rp.pct(lo[:, 2]), "--", lw=1.6, color=rp.COLORS[k])
                ax.scatter(lo[today, 0], y[today], s=90, facecolor="none", edgecolor=rp.COLORS[k], lw=2, zorder=5)
            ax.set_xlabel("Circuits that overload (%)")
            rp.style(ax)
        axes[0, j].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec="#8a8984"))
        axes[1, j].set_xlim(x0, x1)
        axes[1, j].set_xticks(np.arange(0, x1 + 1, 5))
        axes[1, j].set_ylim(y0, y1)
        axes[0, j].set_title(title(f) if callable(title) else title.format(f), loc="left", fontweight="bold",
                             fontsize=10)
    excess = res[(panels[0], 1)][0].shape[1] > 2
    fig.supylabel(ylabel or ("Soak and excess load (% of rooftop solar, 10–15)" if excess else rp.SOAK_LABEL),
                  fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if res[(panels[0], 1)][0].shape[1] > 2:
        handles += [Line2D([], [], color="#52514e", lw=2, ls="-"), Line2D([], [], color="#52514e", lw=1.6, ls="--")]
        labels += list(names)
    axes[1, 0].legend(handles, labels, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:                                                    # replot both report figures
        base = rp.UPTAKE
        designs = {k: rp.year_designs(k) for k in (1, 2, 3)}
        loads = {}
        for f in K_REPORT:
            rp.UPTAKE = f
            loads[f] = {s: flat_load(designs[2][s], s) for s in rp.SEASONS}
        rp.UPTAKE = base
        load_figure(designs, loads, base)
        z = np.load(CACHE)
        tradeoff_figure({(f, k): tuple(z[f"{f}_{k}"]) for f in K_REPORT for k in (1, 2, 3)})
    else:
        main()

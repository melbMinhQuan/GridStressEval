"""Figure 6 with plain soak (excess not subtracted) summed over 10:00-15:00 for every design, as a share of the
rooftop solar 10:00-15:00 (rp.SOAK_HOURS = "window"). Lowest overload, one panel per uptake, zoom on today's
mix. Figure 4 in the same mode: python3 -m play.run_allocation_scatter --window.

Run from the repo root:  python3 -m play.run_fill_figures [--load]   (--load: load shapes per season)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play import run_participation as rp
from play.run_load_scale import K_REPORT, tradeoff_figure
from play.soak import SLOT, in_range

rp.SOAK_HOURS, rp.SOAK_WINDOW, rp.REFERENCE, rp.NET_SOAK = "window", (10, 15), (10, 15), False   # also in workers
rp.SOAK_LABEL = "Soak (% of rooftop solar, 10–15)"
PLOT = config.ROOT / "doc" / "vis" / "fig6_10to15.png"
LOAD_PLOT = config.ROOT / "doc" / "vis" / "load_10to15.png"


def plan_load(design, season):
    """Plan load (MW per slot) of the lowest-overload allocation at today's mix: small customers' load plus the
    interval load; NaN outside the counted hours 10-15."""
    kw = config.load_kw()
    to_mw = rp.UPTAKE * config.VIC_CUSTOMERS / config.N / 1000
    _, _, w, alloc = rp.evaluate(design, season, rp.TODAY, "lowest overload")
    per = to_mw * sum(w[key] * ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l) for key, (m, l) in alloc.items())
    load = np.full(config.SLOTS, rp.base_mw(season))
    for (a, b), mw in zip(design, per):
        load[in_range(a, b)] += mw
    return np.where(in_range(*rp.SOAK_WINDOW), load, np.nan)


def load_figure():
    """Per season: rooftop solar and the plan load of 1, 2 and 3 intervals over the counted hours 10-15."""
    fig, axes = plt.subplots(1, len(rp.SEASONS), figsize=(16, 4.2), sharey=True)
    for ax, season in zip(axes, rp.SEASONS):
        x = rp.solar_season(season)
        mean = x.mean(axis=0)
        ax.fill_between(SLOT, np.percentile(x, 10, axis=0), np.percentile(x, 90, axis=0), color="#bbbbbb",
                        alpha=0.5, lw=0, label="Rooftop, 10–90% of days")
        ax.plot(SLOT, mean, color="#333333", lw=1.6, label="Rooftop, mean")
        for k, lw in ((2, 5), (1, 2), (3, 2)):
            ax.plot(SLOT, plan_load(rp.choose(k, season), season), lw=lw, alpha=0.5 if k == 2 else 1,
                    color=rp.COLORS[k], label=f"{k} interval{'s' * (k > 1)}")
        ax.set_xlim(6, 18)
        ax.set_title(season, loc="left", fontweight="bold")
        ax.set_xlabel("Hour")
        rp.style(ax)
    axes[0].set_ylabel("MW")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(n) for n in ("Rooftop, 10–90% of days", "Rooftop, mean", "1 interval", "2 intervals",
                                       "3 intervals")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=8,
               loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(LOAD_PLOT, dpi=150)
    plt.close(fig)
    print(f"graph: {LOAD_PLOT}")


def curve(job):
    f, k = job
    rp.UPTAKE = f
    return np.array([rp.combine(rp.year_designs(k), rho, "lowest overload") for rho in rp.RHO])


def main():
    jobs = [(f, k) for f in K_REPORT for k in (1, 2, 3)]
    with Pool(len(jobs)) as pool:
        out = pool.map(curve, jobs)
    res = {job: (lo, lo) for job, lo in zip(jobs, out)}
    today = np.isclose(rp.RHO, rp.TODAY)
    for (f, k), (lo, _) in res.items():
        print(f"u={f:.0%} K={k}: today overload {lo[today, 0][0]:5.1f}%  soak {rp.pct(lo[today, 1])[0]:5.1f}%  "
              f"peak {rp.pct(lo[:, 1]).max():5.1f}% at rho {rp.RHO[lo[:, 1].argmax()]:.1%}")
    tradeoff_figure(res, path=PLOT)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    load_figure() if "--load" in sys.argv else main()

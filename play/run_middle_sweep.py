"""Fix 08-11 and 11-14; add a third 3 h interval and sweep its start. Compare with 2 intervals.

Two soak metrics (rows):
    rooftop: min(total circuit demand, rooftop share), summed over interval hours
    added  : min(added load, curtailment share), summed over interval hours
Each circuit uses its own best allocation (play.frontier.pareto_front): compositions move from their
best-soak to their best safe allocation, cheapest first. (Share rules are not used: rounding each share
separately favours designs with more intervals.) Violation is trinomial-weighted (All-On).

Run from the repo root:  python3 -m play.run_middle_sweep
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config

# ================================================================ INPUTS (this run)
FIXED = [(8, 11), (11, 14)]
MIDDLE_STARTS = [s for s in np.arange(7, 14.01, 0.5) if s not in (8, 11)]   # third interval; skip duplicates
LENGTH = 3
UPTAKE = 0.1
PLOT = config.OUT_DIR / "middle_sweep.png"
# ==================================================================================

METRICS = {"rooftop": ("rooftop_actual", "total"), "added": ("curtailment", "added")}


def _front(job):
    metric, design = job
    source, demand = METRICS[metric]
    import play.frontier as frontier
    from play.run_rooftop_fronts import total_table
    from play.soak import soak_table
    # set on every job: worker processes are reused across metrics
    frontier.soak_table = total_table if demand == "total" else soak_table
    config.SOLAR_SOURCE = source                                      # after imports, every job
    return [(safe, soak, mix.tolist()) for safe, soak, mix in frontier.pareto_front(design, UPTAKE)]


def label(s):
    return f"{int(s):02d}:{round(s % 1 * 60):02d}-{int(s + LENGTH):02d}:{round((s + LENGTH) % 1 * 60):02d}"


def main():
    designs = {"2": FIXED} | {label(s): sorted(FIXED + [(float(s), float(s) + LENGTH)]) for s in MIDDLE_STARTS}
    jobs = [(m, d) for m in METRICS for d in designs.values()]
    with ProcessPoolExecutor() as pool:
        fronts = dict(zip(((m, tuple(d)) for m, d in jobs), pool.map(_front, jobs, chunksize=1)))

    names = {"rooftop": "rooftop solar, total demand", "added": "curtailment, added load"}
    colors = {"rooftop": "#D9822B", "added": "#2A7FB8"}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    for metric in METRICS:
        two = fronts[(metric, tuple(designs["2"]))]
        base_soak = two[-1][1]
        rows = [(s, fronts[(metric, tuple(designs[label(s)]))][-1]) for s in MIDDLE_STARTS]
        gain = [(f[1] / base_soak - 1) * 100 for _, f in rows]
        top.plot(MIDDLE_STARTS, gain, "-o", color=colors[metric], lw=2.2, label=names[metric])
        print(f"\n{metric}: 2 intervals safe {two[-1][0]:.1%}, soak {base_soak:.1f}")
        for (s, f), g in zip(rows, gain):
            print(f"  + {label(s)}  safe {f[0]:6.1%}  soak {f[1]:7.1f}  ({g:+.1f}%)")
        if metric == "added":
            bottom.plot(MIDDLE_STARTS, [f[0] * 100 for _, f in rows], "-o", color="#444", lw=2.2,
                        label="3 intervals")
            bottom.axhline(two[-1][0] * 100, color="#444", ls="--", lw=1.2, label="2 intervals (08-11 | 11-14)")
    for ax in (top, bottom):
        ax.axvspan(7.75, 11.25, color="#888", alpha=0.10, lw=0)      # third interval inside 08-14: no new hours
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.text(9.5, top.get_ylim()[1] * 0.92, "inside 08:00-14:00\n(no new hours)", ha="center", fontsize=9, color="#555")
    top.axhline(0, color="#999", lw=1)
    top.set_ylabel("Soak gain over 2 intervals (%)")
    top.set_title("Soak at the safest allocation", loc="left", fontweight="bold")
    top.legend(frameon=False, loc="upper left")
    bottom.set_ylabel("Circuits without overload (%)")
    bottom.set_title("Highest achievable safety", loc="left", fontweight="bold")
    bottom.set_xlabel("Third interval (3 h)")
    bottom.set_xticks(MIDDLE_STARTS, [label(s) for s in MIDDLE_STARTS], rotation=45, ha="right", fontsize=8)
    bottom.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Adding a third interval to 08-11 | 11-14 (each circuit picks its own allocation, uptake {UPTAKE:.0%})",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

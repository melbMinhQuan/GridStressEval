"""E6 Daylight saving: interval times as local clock time (AEDT in summer) instead of market time.
Best 3 h designs inside 08-15 for 2 and 3 intervals, at the safest allocation."""
import matplotlib.pyplot as plt
import numpy as np

from play.experiments.common import best_design, design_grid, name, safest_points, save, style

WINDOW, LENGTH, UPTAKES = (8, 15), 3, [0.1, 0.5]
MODES = {"market time (AEST)": {"LOCAL_TIME": False}, "local clock time (DST)": {"LOCAL_TIME": True}}


def run():
    out = {}
    for mode, overrides in MODES.items():
        for k in (2, 3):
            designs = design_grid(k, *WINDOW, LENGTH)
            for u in UPTAKES:
                res = safest_points([(overrides, d, u) for d in designs])
                i, v, s = best_design(designs, res)
                out[(mode, k, u)] = (designs[i], v, s)
    print(f"E6 daylight saving: best {LENGTH} h designs inside {WINDOW[0]}-{WINDOW[1]} (safest allocation)")
    for (mode, k, u), (d, v, s) in out.items():
        print(f"  {mode:<24} K={k} uptake {u:>4.0%}  {name(d):<40} violation {v:7.3%}  soak {s:6.1f}")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"K={k}, {u:.0%}" for k in (2, 3) for u in UPTAKES]
    x = np.arange(len(labels))
    for i, (mode, c) in enumerate(zip(MODES, ["#2A7FB8", "#D9822B"])):
        vals = [out[(mode, k, u)][2] for k in (2, 3) for u in UPTAKES]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=c, label=mode)
        ax.bar_label(bars, labels=[name(out[(mode, k, u)][0]) for k in (2, 3) for u in UPTAKES],
                     rotation=90, padding=3, fontsize=6.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(out[key][2] for key in out) * 1.9)
    ax.set_ylabel("Best soak per circuit (kWh/day)")
    ax.set_title("E6  Market time vs local clock time (labels: best design)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    save(fig, "e06_dst.png")
    return out


if __name__ == "__main__":
    run()

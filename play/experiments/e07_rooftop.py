"""E7 Rooftop solar: soak against the VIC1 rooftop-solar shape (same annual energy as the
curtailment) instead of utility curtailment. Best 3 h designs inside 08-15 for 1, 2, 3 intervals."""
import matplotlib.pyplot as plt
import numpy as np

from play.experiments.common import best_design, design_grid, name, safest_points, save, style

WINDOW, LENGTH, UPTAKES = (8, 15), 3, [0.01, 0.1, 0.5]
SOURCES = {"utility curtailment": {"SOLAR_SOURCE": "curtailment"}, "rooftop shape": {"SOLAR_SOURCE": "rooftop"}}


def run():
    out = {}
    for source, overrides in SOURCES.items():
        for k in (1, 2, 3):
            designs = design_grid(k, *WINDOW, LENGTH)
            for u in UPTAKES:
                res = safest_points([(overrides, d, u) for d in designs])
                i, v, s = best_design(designs, res)
                out[(source, k, u)] = (designs[i], v, s)
    print(f"E7 solar source: best {LENGTH} h designs inside {WINDOW[0]}-{WINDOW[1]} (safest allocation)")
    for source in SOURCES:
        for u in UPTAKES:
            s2, s3 = out[(source, 2, u)][2], out[(source, 3, u)][2]
            for k in (1, 2, 3):
                d, v, s = out[(source, k, u)]
                print(f"  {source:<20} uptake {u:>4.0%} K={k}  {name(d):<40} violation {v:7.3%}  soak {s:6.1f}")
            print(f"  {'':<20} {'':>11} gain 3 vs 2: {s3 / s2 - 1:+.1%}")

    fig, axes = plt.subplots(1, len(UPTAKES), figsize=(5 * len(UPTAKES), 4.6))
    for ax, u in zip(axes, UPTAKES):
        x = np.arange(3)
        for i, (source, c) in enumerate(zip(SOURCES, ["#2A7FB8", "#D9822B"])):
            ax.bar(x + (i - 0.5) * 0.38, [out[(source, k, u)][2] for k in (1, 2, 3)], 0.36, color=c, label=source)
        ax.set_xticks(x, ["1 interval", "2 intervals", "3 intervals"])
        ax.set_title(f"Uptake {u:.0%}", loc="left", fontweight="bold")
        style(ax)
    axes[0].set_ylabel("Best soak per circuit (kWh/day)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("E7  Utility curtailment vs rooftop-solar timing (best design per K)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "e07_rooftop.png")
    return out


if __name__ == "__main__":
    run()

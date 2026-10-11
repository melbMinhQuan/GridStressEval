"""E2 Customer mix: All-On violation for 1, 2 and 3 intervals over a grid of p_M and p_L."""
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.experiments.common import DESIGNS, min_peak_table, save, violation, weights

P_M = np.round(np.arange(0, 0.51, 0.05), 2)
P_L = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
SHOW = ["1 x 3h  11-14", "2 x 3h  09-12 | 12-15", "3 x 2h  09-11 | 11-13 | 13-15"]


def run():
    keys = [(m, l) for m in range(config.N + 1) for l in range(config.N + 1 - m)]
    grids = {}
    for label in SHOW:
        peaks = min_peak_table(DESIGNS[label], keys)
        grid = np.zeros((len(P_L), len(P_M)))
        for i, pl in enumerate(P_L):
            for j, pm in enumerate(P_M):
                grid[i, j] = violation(peaks, weights(config.N, {"S": 1 - pm - pl, "M": pm, "L": pl}), config.T)
        grids[label] = grid

    print("E2 customer mix: P(violation) (%), rows p_L, columns p_M")
    for label, grid in grids.items():
        print(f"  {label}")
        print("    p_L\\p_M " + "".join(f"{pm:>7.2f}" for pm in P_M))
        for pl, row in zip(P_L, grid):
            print(f"    {pl:>8.3f} " + "".join(f"{v * 100:>7.2f}" for v in row))

    fig, axes = plt.subplots(1, len(SHOW), figsize=(6 * len(SHOW), 4.8), layout="constrained")
    for ax, (label, grid) in zip(axes, grids.items()):
        im = ax.imshow(np.log10(np.maximum(grid * 100, 1e-3)), origin="lower", aspect="auto",
                       cmap="Reds", vmin=-3, vmax=2)
        for i in range(len(P_L)):
            for j in range(len(P_M)):
                v = grid[i, j] * 100
                ax.text(j, i, f"{v:.0f}" if v >= 1 else f"{v:.1g}" if v >= 0.001 else "0",
                        ha="center", va="center", fontsize=6.5, color="white" if v > 10 else "#222")
        ax.set_xticks(range(len(P_M)), [f"{p:.2f}" for p in P_M], rotation=60, fontsize=7)
        ax.set_yticks(range(len(P_L)), [f"{p:g}" for p in P_L], fontsize=7)
        ax.set_xlabel("p_M")
        ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("p_L")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2], labels=["0.001", "0.01", "0.1", "1", "10", "100"])
    cbar.set_label("P(violation) %")
    fig.suptitle("E2  Violation (%) by customer mix (All-On)", x=0.01, ha="left", fontweight="bold")
    save(fig, "e02_mix.png")
    return grids


if __name__ == "__main__":
    run()

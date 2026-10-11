"""P(no upgrade) = 1 - O_viol over the share of heavy (large) customers x = p_L and the customers per line N,
for one shared window, two assigned windows and three assigned windows (the middle one overlapping both).

p_M = 30% (today's), p_S = 1 - p_M - x; limit T = 25 A x N (1000 A at N = 40); lowest-overload allocation, All-On.

Run from the repo root:  python3 -m play.run_upgrade_heatmap
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from play import config
from play.experiments.common import min_peak, weights

X = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]      # share of heavy customers p_L
N_VALUES = [10, 20, 30, 40, 50, 60, 80]                                     # customers per line
P_M = 0.30
AMPS_PER_CUSTOMER = 25
DESIGNS = {"One window": [(11, 14)], "Two windows": [(9.5, 12.5), (12.5, 15.5)],
           "Three windows": [(9.5, 12.5), (11, 14), (12.5, 15.5)]}
PLOT = config.OUT_DIR / "report" / "fig3_upgrade_heatmap.png"


def no_upgrade(design):
    """(len(X), len(N_VALUES)) P(no upgrade) %."""
    out = np.zeros((len(X), len(N_VALUES)))
    for j, n in enumerate(N_VALUES):
        limit = AMPS_PER_CUSTOMER * n
        peaks = {}
        for i, x in enumerate(X):
            w = weights(n, {"S": 1 - P_M - x, "M": P_M, "L": x})
            total = sum(w.values())
            for key in w:
                if key not in peaks:
                    peaks[key] = min_peak(design, *key, n=n)[0]
            out[i, j] = 100 * sum(wi for key, wi in w.items() if peaks[key] <= limit) / total
    return out


def panel(ax, grid, title, cmap, fmt):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=100)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5, color="white" if v > 55 else "#222222")
    ax.set_xticks(range(len(N_VALUES)), N_VALUES)
    ax.set_yticks(range(len(X)), [f"{x:.1%}".replace(".0%", "%") for x in X])
    ax.set_xlabel("Customers per line N")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    return im


def ratio_panel(ax, three, one):
    """Three windows' P(no upgrade) divided by one window's, log colour scale; inf where one window is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(one > 0, three / np.where(one > 0, one, 1), np.where(three > 0, np.inf, np.nan))
    finite = r[np.isfinite(r)]
    norm = LogNorm(vmin=max(finite.min(), 1), vmax=finite.max())
    im = ax.imshow(np.where(np.isinf(r), finite.max(), r), origin="lower", aspect="auto", cmap="Greens",
                   norm=norm)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r[i, j]
            text = "–" if np.isnan(v) else "∞" if np.isinf(v) else f"{v:.1f}" if v < 10 else f"{v:,.0f}"
            dark = np.isinf(v) or (np.isfinite(v) and norm(v) > 0.55)
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="white" if dark else "#222222")
    ax.set_xticks(range(len(N_VALUES)), N_VALUES)
    ax.set_yticks(range(len(X)), [f"{x:.1%}".replace(".0%", "%") for x in X])
    ax.set_xlabel("Customers per line N")
    ax.set_title("Three ÷ one", loc="left", fontweight="bold", fontsize=10)
    return im, r


def main():
    grids = {name: no_upgrade(d) for name, d in DESIGNS.items()}
    one, two, three = grids.values()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True, layout="constrained")
    axes = axes.ravel()
    pct = lambda v: f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}"
    for ax, (name, grid) in zip(axes, grids.items()):
        im = panel(ax, grid, name, "Blues", pct)
    fig.colorbar(im, ax=axes[:2], shrink=0.9, label="P(no upgrade) (%)")
    im, r = ratio_panel(axes[3], three, one)
    fig.colorbar(im, ax=axes[3], shrink=0.85, label="×")
    for ax in axes[::2]:
        ax.set_ylabel("Heavy customers p_L")
    print(f"largest difference between three and two windows: {np.abs(three - two).max():.4f} points")
    print("three / one")
    print("  x \\ N  " + "".join(f"{n:>7}" for n in N_VALUES))
    for x, row in zip(X, r):
        print(f"  {x:6.1%} " + "".join(f"{v:7.1f}" for v in row))
    fig.savefig(PLOT, dpi=150, bbox_inches="tight")
    for name, grid in grids.items():
        print(name)
        print("  x \\ N  " + "".join(f"{n:>7}" for n in N_VALUES))
        for x, row in zip(X, grid):
            print(f"  {x:6.1%} " + "".join(f"{v:7.1f}" for v in row))
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

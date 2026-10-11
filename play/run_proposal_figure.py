"""Figure 1 of the proposal (doc/references/New.md): Victorian power draw of the plan customers against mean
rooftop solar. Left: everyone in 11-14. Right: the proposal, heavy hitters that share a phase moved to 9-12 and
12-15. Premise model of play/run_research.py (household load I0, moderate users, 40 kW heavy hitters); the
uptake is set so the single slot peaks at the mean midday rooftop peak (as in the report).

Run from the repo root:  python3 -m play.run_proposal_figure
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play import run_participation as rp
from play import run_research as rr
from play.soak import SLOT, in_range

OUT = config.ROOT / "doc" / "vis"
PLOT = OUT / "fig1_proposal.png"
LAYERS = {"Household": "#c9c8c4", "Moderate users": rp.COLORS[2], "Heavy hitters": rp.COLORS[3]}


def layers(strategy):
    """Expected kW per phase on each slot: household load, moderate users, heavy hitters."""
    kw = {name: np.zeros(config.SLOTS) for name in LAYERS}
    kw["Household"][:] = rr.N * rr.I0 * rr.V / 1000
    for n_m, n_h, w in rr.compositions(rr.N, rr.P_M, rr.P_H):
        mods, heavy, _ = rr.allocate(strategy, n_m, n_h)
        for (a, b), count in mods.items():
            kw["Moderate users"][in_range(a, b)] += w * count * rr.DI_M * rr.V / 1000
        for a, b in heavy:
            kw["Heavy hitters"][in_range(a, b)] += w * rr.DI_H * rr.V / 1000
    return kw


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    solar = np.vstack([rp.solar_season(s) for s in rp.SEASONS])                  # every day of 2025, MW
    mean = solar.mean(axis=0)
    kw = {s: layers(s) for s in ("policy", "proposal")}
    phases = rp.midday_peak() * 1000 / sum(kw["policy"].values()).max()         # single slot peaks at it
    u = phases * rr.N / config.VIC_CUSTOMERS
    moved = 1 - kw["proposal"]["Heavy hitters"][in_range(12, 14)].mean() / kw["policy"]["Heavy hitters"][in_range(12, 14)].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    titles = {"policy": "One slot", "proposal": "Proposal"}
    win = in_range(8, 16)
    for ax, (s, title) in zip(axes, titles.items()):
        bottom = np.zeros(config.SLOTS)
        for name, color in LAYERS.items():
            top = bottom + phases * kw[s][name] / 1000                          # MW
            ax.fill_between(SLOT, bottom, top, step="post", color=color, lw=0, label=name)
            bottom = top
        ax.plot(SLOT, mean, color="#333333", lw=1.8, label="Rooftop solar")
        ax.set_xlim(6, 18)
        ax.set_xticks([6, 9, 12, 15, 18])
        ax.set_xlabel("Hour")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        rp.style(ax)
        soaked = np.minimum(bottom[win], solar[:, win]).sum(axis=1).mean() * config.DT
        excess = np.maximum(bottom[win] - solar[:, win], 0).sum(axis=1).mean() * config.DT
        print(f"{s:<9} peak {bottom.max():5.0f} MW  soaked {soaked:6.0f}  excess {excess:6.0f}  "
              f"net {soaked - excess:6.0f} MWh/day (08-16, every day of 2025)")
    axes[0].set_ylabel("MW")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOT, dpi=150)
    plt.close(fig)
    print(f"uptake {u:.1%} ({u * config.VIC_CUSTOMERS:,.0f} customers); heavy-hitter load moved out of 12-14: "
          f"{moved:.0%}\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

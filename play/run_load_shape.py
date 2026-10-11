"""Load shape: customers draw their extra current for d hours from the start of their 3-hour window.

d = 3 is All-On. With d < 3 a customer's consumption is the block [s_k, s_k + d), so the model of
play/run_participation.py is applied to the blocks instead of the windows. Windows keep the report's
structure (3 hours inside 08:00-16:00, starts every 0.5 h; 2 windows do not overlap; of 3, the middle one
overlaps both), and for each d and season the window starts are chosen for the lowest overload at today's
mix, then the most rooftop solar inside the blocks. Lowest-overload allocation, rooftop solar, uptake 50%.

Run from the repo root:  python3 -m play.run_load_shape
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
from play.circuit import segments
from play.experiments.common import weights

D_VALUES = [3.0, 2.5, 2.0, 1.5, 1.0]              # hours of consumption from the window start
CAP = 5                                           # overload cap for the hosting read-off, %
EXAMPLE = (13, 1)                                 # circuit composition (n_M, n_L) drawn in the example
OUT = config.OUT_DIR / "load_shape"


def blocks(design, d):
    return tuple((a, a + d) for a, _ in design)


def overload(block_designs, mix):
    """O_viol (%) with the lowest peak per composition; a circuit overloads if it does in any season."""
    w = weights(config.N, mix)
    total = sum(w.values())
    bad = sum(wi for key, wi in w.items()
              if any(rp.candidates(b, *key)[1][0] > config.T for b in block_designs.values()))
    return 100 * bad / total


def choose(k, season, d):
    """Window set: lowest overload of its blocks at today's mix, then most rooftop solar in the blocks."""
    profile = rp.solar_season(season).mean(axis=0)
    w = weights(config.N, rp.mix(rp.TODAY))
    total = sum(w.values())
    best = None
    for windows in rp.placements(k):
        b = blocks(windows, d)
        over = sum(wi for key, wi in w.items() if rp.candidates(b, *key)[1][0] > config.T) / total
        score = (round(over, 9), -rp.solar_sum(b, profile)[0])
        if best is None or score < best[0]:
            best = (score, windows)
        rp.candidates.cache_clear()                                             # keeps memory bounded
    return best[1]


def current(design, medium, large):
    """Circuit current over the day (A) for one allocation."""
    amps, n = config.CURRENT, config.N
    out = np.full(config.SLOTS, float(n * amps["S"]))
    for (a, b), m, l in zip(design, medium, large):
        mask = (rp.SLOT >= a) & (rp.SLOT < b)
        out[mask] += (amps["M"] - amps["S"]) * m + (amps["L"] - amps["S"]) * l
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, chosen = {}, {}
    for d in D_VALUES:
        chosen[d] = {k: {s: choose(k, s, d) for s in rp.SEASONS} for k in (1, 2, 3)}
        for k in (1, 2, 3):
            b = {s: blocks(v, d) for s, v in chosen[d][k].items()}
            over_today, soak_today = rp.combine(b, rp.TODAY, "lowest overload")
            caps = [rho for rho in rp.RHO if overload(b, rp.mix(rho)) <= CAP]
            rows[(d, k)] = (over_today, soak_today, max(caps) if caps else 0.0)
            print(f"d={d:3.1f} h K={k}: windows {rp.name(chosen[d][k]['Summer']):<36} (summer)  overload {over_today:6.2f}%"
                  f"  soak {soak_today:7.1f} MWh/day  largest rho with overload <= {CAP}%: {rows[(d, k)][2]:.1%}",
                  flush=True)

    x = np.array(D_VALUES)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for k in (1, 2, 3):
        lab = f"{k} interval{'s' * (k > 1)}"
        for ax, i in zip(axes, (0, 2, 1)):                                    # overload, hosting, soak
            vals = [rows[(d, k)][i] * (100 if i == 2 else 1) for d in D_VALUES]
            ax.plot(x, vals, "-o", ms=5, lw=2, color=rp.COLORS[k], label=lab)
    titles = ("Overload at today's mix", f"Largest ρ with overload ≤ {CAP}%", "Rooftop solar soaked, today's mix")
    ylabels = ("Circuits that overload (%)", "ρ = p_M + p_L (%)", "MWh/day")
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("Hours of consumption from the window start, d")
        ax.set_ylabel(yl)
        ax.invert_xaxis()
        rp.style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Load shape: consumption lasts d hours from the start of a customer's 3-hour window (d = 3: All-On)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "load_shape_summary.png", dpi=150)
    plt.close(fig)

    # example circuit: lowest-peak allocation of EXAMPLE under 2 and 3 windows, All-On and d = 1.5 h
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, d in zip(axes, (3.0, 1.5)):
        for k in (2, 3):
            b = blocks(chosen[d][k]["Summer"], d)
            _, peak, med, lar = rp.candidates(b, *EXAMPLE)
            ax.plot(rp.SLOT, current(b, med[0], lar[0]), lw=2.2, color=rp.COLORS[k],
                    label=f"{k} windows, peak {peak[0]:.0f} A")
            print(f"  example d={d:g} h, {k} windows {rp.name(chosen[d][k]['Summer'])}: "
                  f"M {list(med[0])}, L {list(lar[0])}, peak {peak[0]:.0f} A")
        ax.axhline(config.T, color="#d62728", ls=":", lw=1.2)
        ax.text(6.1, config.T + 8, f"limit T = {config.T} A", color="#d62728", fontsize=8)
        ax.set_xlim(6, 18)
        ax.set_title(f"d = {d:g} h" + (" (All-On)" if d == 3 else ""), loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("Hour (AEST)")
        ax.legend(frameon=False, fontsize=8, loc="center left")
        rp.style(ax)
    axes[0].set_ylabel(f"Circuit current (A), {EXAMPLE[0]} M + {EXAMPLE[1]} L customers")
    fig.tight_layout()
    fig.savefig(OUT / "load_shape_example.png", dpi=150)
    plt.close(fig)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()

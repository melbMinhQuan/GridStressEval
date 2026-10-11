"""Excess load: total load of the plan circuits not covered by rooftop solar over 08:00-16:00, for 1, 2 and 3
intervals (lowest overload), as the uptake u, the customers per circuit N, p_M and p_L grow. One knob moves at
a time from the calibrated setting (u calibrated, N = 40, p = 0.69 / 0.30 / 0.01).

N: the number of circuits on the plan stays at the calibrated count, so N more customers per circuit means
N / 40 times as many customers on the plan; the circuit limit is 25 A x N (1000 A at N = 40).

Run from the repo root:  python3 -m play.run_excess
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

N0, T0 = config.N, config.T
UPTAKES = np.round(np.arange(0.02, 0.401, 0.02), 2)
NS = [10, 20, 30, 40, 50, 60, 70, 80]
P_MS = np.round(np.arange(0.0, 0.601, 0.05), 2)                # p_L = 1%
P_LS = np.round(np.arange(0.0, 0.0601, 0.005), 3)              # p_M = 30%
PLOT = config.OUT_DIR / "report" / "fig5_excess.png"


def excess(designs, p):
    """(overload %, net soak, excess) with excess = soak - net soak, MWh/day."""
    rp.NET_SOAK = False
    _, soak = rp.combine(designs, p, "lowest overload")
    rp.NET_SOAK = True
    over, net = rp.combine(designs, p, "lowest overload")
    return over, net, soak - net


def set_n(n):
    """n customers per circuit with the calibrated number of plan circuits."""
    config.N, config.T = n, 25 * n
    rp.candidates.cache_clear()


def main():
    u0, today = rp.UPTAKE, rp.mix(rp.TODAY)
    designs = {k: {s: rp.choose(k, s) for s in rp.SEASONS} for k in (1, 2, 3)}
    sweeps = {"u": [], "N": [], "pM": [], "pL": []}
    for k, d in designs.items():
        for u in UPTAKES:
            rp.UPTAKE = u
            sweeps["u"].append((k, 100 * u, *excess(d, today)))
        rp.UPTAKE = u0
        for n in NS:
            set_n(n)
            rp.UPTAKE = u0 * n / N0                                         # same circuits, n customers each
            sweeps["N"].append((k, n, *excess(d, today)))
        set_n(N0)
        config.T = T0
        rp.UPTAKE = u0
        for pm in P_MS:
            sweeps["pM"].append((k, 100 * pm, *excess(d, {"S": 0.99 - pm, "M": pm, "L": 0.01})))
        for pl in P_LS:
            sweeps["pL"].append((k, 100 * pl, *excess(d, {"S": 0.70 - pl, "M": 0.30, "L": pl})))
        print(f"{k} interval(s) done", flush=True)

    labels = {"u": ("Uptake u (%)", 100 * u0), "N": ("Customers per circuit N", N0),
              "pM": ("p_M (%), p_L = 1%", 30), "pL": ("p_L (%), p_M = 30%", 1)}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, (key, (xl, now)) in zip(axes, labels.items()):
        a = np.array(sweeps[key])
        for k in (1, 2, 3):
            r = a[a[:, 0] == k]
            ax.plot(r[:, 1], rp.pct(r[:, 4]), "-o", ms=3, lw=2, color=rp.COLORS[k],
                    label=f"{k} interval{'s' * (k > 1)}")
        ax.axvline(now, color="#8a8984", ls=":", lw=1)
        ax.set_xlabel(xl)
        rp.style(ax)
        for k in (1, 2, 3):
            r = a[a[:, 0] == k]
            i = np.argmin(abs(r[:, 1] - now))
            print(f"{key:<3} K={k}: at {now:g}: excess {r[i, 4]:7.0f} MWh/day ({rp.pct(r[i, 4]):5.1f}%), "
                  f"range {rp.pct(r[0, 4]):5.1f}% .. {rp.pct(r[-1, 4]):6.1f}%, overload {r[i, 2]:.1f}%")
    for key in sweeps:                                                          # largest gap between designs
        a = np.array(sweeps[key])
        e = np.stack([rp.pct(a[a[:, 0] == k][:, 4]) for k in (1, 2, 3)])
        print(f"{key:<3}: designs differ by at most {np.ptp(e, axis=0).max():.2f} pts of rooftop solar")
        print(f"{key:<3}: 2 and 3 intervals differ by at most {np.abs(e[1] - e[2]).max():.2f} pts")
    axes[0].set_ylabel("Excess load (% of rooftop solar, 08–16)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    plt.close(fig)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

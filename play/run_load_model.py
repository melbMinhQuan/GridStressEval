"""Section 3.3: a stochastic load model.

Each day every assigned medium and large customer uses its 3-hour window for l in {0, 1, 2, 3} hours with
probabilities pi = (pi_0, pi_1, pi_2, pi_3), starting uniformly at any 30-minute position where the block
fits inside the window. pi_3 is swept with pi_0 = pi_1 = pi_2 = (1 - pi_3) / 3; pi_3 = 1 is All-On.

Windows and allocation are those of the report (play/run_participation.py: seasonal windows, lowest-overload
All-On allocation, today's mix), fixed in advance. Per circuit composition n and season, the daily overload
probability P_s(n) is estimated over DAYS simulated days; P(n) averages the seasons by days.
    daily overload    sum_n w(n) P(n)                      share of circuit-days that overload
    upgrade share     sum_n w(n) 1[P(n) > UPGRADE]          circuits overloading on more than 1% of days
    soak              expected load profile (closed form), seasonal weights, rooftop solar, uptake 50%

Run from the repo root:  python3 -m play.run_load_model
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
from play.soak import SLOT, in_range

PI3 = [0.0, 0.25, 0.5, 0.75, 1.0]                 # probability of using the full 3 hours
DEFAULT = 0.25                                    # every length equally likely
DAYS = 4000                                       # simulated days per composition and season
UPGRADE = 0.01                                    # a circuit needs an upgrade if it overloads on > 1% of days
MIN_WEIGHT = 1e-9                                 # compositions below this weight are skipped
HALF = 0.5                                        # start grid, hours
SEED = 7
PLOT = config.OUT_DIR / "report" / "fig4_load_model.png"


def lengths(pi3):
    return np.array([(1 - pi3) / 3] * 3 + [pi3])


def daily_overload(design, medium, large, pi, rng):
    """P(circuit overloads on a day) for one allocation (medium, large customers per interval)."""
    n, amps = config.N, config.CURRENT
    t0, nslot = rp.WINDOW[0], int((rp.WINDOW[1] - rp.WINDOW[0]) / HALF)
    width = int(rp.LENGTH / HALF)                                              # half-hours per window
    cur = np.full((DAYS, nslot), float(n * amps["S"]))
    x = np.arange(width)
    for (s, _), m, l in zip(design, medium, large):
        off = int(round((s - t0) / HALF))
        for count, extra in ((m, amps["M"] - amps["S"]), (l, amps["L"] - amps["S"])):
            if count == 0:
                continue
            b = 2 * rng.choice(4, size=(DAYS, count), p=pi)                    # block length, half-hours
            start = (rng.random((DAYS, count)) * (width + 1 - b)).astype(int)  # uniform over fitting starts
            active = (x >= start[..., None]) & (x < (start + b)[..., None])
            cur[:, off:off + width] += extra * active.sum(axis=1)
    return (cur.max(axis=1) > config.T).mean()


def active_probability(pi):
    """P(a customer is active) at each 5-minute slot of its window (closed form)."""
    x = (np.arange(int(rp.LENGTH / config.DT)) + 0.5) * config.DT
    p = np.zeros_like(x)
    for ell in (1, 2, 3):
        starts = np.arange(0, rp.LENGTH - ell + 1e-9, HALF)
        for st in starts:
            p += pi[ell] / len(starts) * ((x >= st) & (x < st + ell))
    return p


def evaluate(designs, pi):
    """(daily overload %, upgrade share %, soak MWh/day) for one design per season."""
    days = {s: len(rp.solar_season(s)) for s in rp.SEASONS}
    total_days = sum(days.values())
    kw = config.load_kw()
    prob = active_probability(pi)
    p_n, soak = {}, 0.0
    for si, (season, design) in enumerate(designs.items()):
        _, _, w, alloc = rp.evaluate(design, season, rp.TODAY, "lowest overload")
        for key, (m, l) in alloc.items():
            if w[key] < MIN_WEIGHT:
                continue
            rng = np.random.default_rng([key[0], key[1], si, SEED])                # same draws for every design
            p_n[key] = p_n.get(key, 0.0) + days[season] / total_days * daily_overload(design, m, l, pi, rng)
        to_mw = rp.UPTAKE * config.VIC_CUSTOMERS / config.N / 1000 * rp.season_weight(season)
        per_interval = to_mw * sum(w[key] * ((kw["M"] - kw["S"]) * m + (kw["L"] - kw["S"]) * l)
                                   for key, (m, l) in alloc.items())
        load = np.full(config.SLOTS, rp.base_mw(season))                       # normal load of plan customers
        for (a, b), mw in zip(design, per_interval):
            load[in_range(a, b)] += mw * prob
        soak += rp.net(load, rp.solar_season(season), design) * days[season]
    weight = {key: w[key] for key in p_n}
    total = sum(weight.values())
    daily = 100 * sum(weight[key] * p for key, p in p_n.items()) / total
    upgrade = 100 * sum(weight[key] for key, p in p_n.items() if p > UPGRADE) / total
    return daily, upgrade, soak / total_days


def main():
    designs = {k: {s: rp.choose(k, s) for s in rp.SEASONS} for k in (1, 2, 3)}
    res = {}
    for pi3 in PI3:
        for k, d in designs.items():
            res[(pi3, k)] = evaluate(d, lengths(pi3))
            print(f"pi3={pi3:4.2f} K={k}: daily overload {res[(pi3, k)][0]:7.3f}%  circuits overloading on >1% of days "
                  f"{res[(pi3, k)][1]:6.2f}%  soak {res[(pi3, k)][2]:7.1f} MWh/day ({rp.pct(res[(pi3, k)][2]):.1f}%)",
                  flush=True)

    x = np.array(PI3)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    titles = ("Circuit-days that overload", f"Circuits overloading on more than {UPGRADE:.0%} of days",
              "Net soak")
    ylabels = ("% of circuit-days", "% of circuits", "% of rooftop solar, 08–16")
    for i, (ax, t, yl) in enumerate(zip(axes, titles, ylabels)):
        for k in (1, 2, 3):
            ax.plot(x, [res[(p, k)][i] if i < 2 else rp.pct(res[(p, k)][i]) for p in PI3], "-o", ms=5, lw=2, color=rp.COLORS[k],
                    label=f"{k} interval{'s' * (k > 1)}")
        ax.axvline(DEFAULT, color="#52514e", ls=":", lw=1)
        ax.set_title(t, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("π₃: probability of using the full 3 hours (π₀ = π₁ = π₂)")
        ax.set_ylabel(yl)
        rp.style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(DEFAULT, axes[0].get_ylim()[1], " default", va="top", fontsize=8, color="#52514e")
    fig.suptitle("Stochastic load: each day a customer uses 0, 1, 2 or 3 hours of its window, starting anywhere "
                 "(π₃ = 1: All-On)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

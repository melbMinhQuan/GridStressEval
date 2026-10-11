"""Fixed allocation rule for 3 intervals: a share alpha of each circuit's medium and large customers goes to
the middle interval, the rest is split evenly between the outer two (largest remainder, M and L separately).
alpha = 0 is the 2-interval design. Today's mix, seasonal intervals of play/run_participation.py, All-On.

    overload     share of circuits whose peak exceeds T in any season
    soak         mean_d sum_t min(L(t), X_d(t)) dt, X = VIC1 utility solar curtailment
    system peak  mean over days of max_t (D_d(t) + L(t)), D = VIC1 operational demand (hourly, held)
                 L(t) = u (N_VIC / N) E_n[added load at t]; reported as the rise over no plan

Run from the repo root:  python3 -m play.run_middle_share
"""
import csv
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
from play.soak import in_range
from play.solar import _profile_mw

SHARES = np.round(np.arange(0, 0.601, 0.05), 2)   # share of M + L customers in the middle interval
UPTAKES = (0.1, 0.3, 0.5, 1.0)
UPTAKE_COLORS = ("#f6c3a3", "#f0905c", "#d9622b", "#9c3d12")   # one hue, light -> dark with uptake
PLOT = config.OUT_DIR / "report" / "fig3_middle_share.png"


def split(count, shares):
    """Largest-remainder split of count customers by shares."""
    raw = count * np.asarray(shares)
    parts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - parts))[: count - parts.sum()]:
        parts[i] += 1
    return parts


def demand():
    """(days, slots) VIC1 operational demand MW, hourly values held for each 5-minute slot."""
    with open(config.ROOFTOP_DEMAND_CSV, newline="") as f:
        hourly = np.array([float(r["demand_mw"]) for r in csv.DictReader(f)]).reshape(-1, 24)
    return np.repeat(hourly, config.SLOTS // 24, axis=1)


def rule(design, shares, w):
    """({(n_M, n_L): overloads}, expected added kW per circuit at each slot) for a fixed share rule."""
    amps, n, kw = config.CURRENT, config.N, config.load_kw()
    segs = segments(list(design))
    over, load = {}, np.zeros(config.SLOTS)
    for (n_m, n_l), wi in w.items():
        m, l = split(n_m, shares), split(n_l, shares)
        over[(n_m, n_l)] = max(amps["S"] * (n - m[c].sum() - l[c].sum()) + amps["M"] * m[c].sum()
                               + amps["L"] * l[c].sum() for _, _, c in segs) > config.T
        for k, (a, b) in enumerate(design):
            load[in_range(a, b)] += wi * ((kw["M"] - kw["S"]) * m[k] + (kw["L"] - kw["S"]) * l[k])
    return over, load


def main():
    _, days = _profile_mw(rp.SOLAR)
    months = np.array([int(d[5:7]) for d in days])
    season_of = {s: np.isin(months, ms) for s, ms in rp.SEASONS.items()}
    curtail, _ = _profile_mw(rp.SOLAR)
    dem = demand()
    base_peak = dem.max(axis=1)
    w = weights(config.N, rp.mix(rp.TODAY))
    total = sum(w.values())
    w = {key: v / total for key, v in w.items()}
    designs = {s: rp.choose(3, s) for s in rp.SEASONS}

    res = []                                          # per share: overload %, {u: (soak, mean peak rise, max peak rise)}
    for a in SHARES:
        shares = [(1 - a) / 2, a, (1 - a) / 2]
        flags, loads = {}, {}
        for s, d in designs.items():
            flags[s], loads[s] = rule(d, shares, w)
        over = 100 * sum(wi for key, wi in w.items() if any(flags[s][key] for s in rp.SEASONS))
        by_u = {}
        for u in UPTAKES:
            plan = np.zeros_like(dem)
            for s, mask in season_of.items():
                plan[mask] = u * config.VIC_CUSTOMERS / config.N * loads[s] / 1000          # MW
            soak = np.minimum(plan, curtail).sum(axis=1).mean() * config.DT
            peak = (dem + plan).max(axis=1)
            by_u[u] = (soak, (peak - base_peak).mean(), peak.max() - base_peak.max())
        res.append((over, by_u))
        print(f"middle {a:4.0%}: overload {over:5.1f}%  " + "  ".join(
            f"u={u:.0%}: soak {v[0]:5.0f}, peak +{v[1]:4.0f} MW (max +{v[2]:4.0f})" for u, v in by_u.items()))

    x = SHARES * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(x, [r[0] for r in res], "-o", ms=4, lw=2, color=rp.COLORS[3])
    axes[0].set_ylabel("Circuits that overload (%)")
    axes[0].set_title("Overload (any uptake)", loc="left", fontweight="bold", fontsize=10)
    for u, color in zip(UPTAKES, UPTAKE_COLORS):
        axes[1].plot(x, [r[1][u][0] for r in res], "-o", ms=3.5, lw=2, color=color, label=f"uptake {u:.0%}")
        axes[2].plot(x, [r[1][u][1] for r in res], "-o", ms=3.5, lw=2, color=color, label=f"uptake {u:.0%}")
    axes[1].set_ylabel("Curtailed solar soaked (MWh/day)")
    axes[1].set_title("Solar soak", loc="left", fontweight="bold", fontsize=10)
    axes[2].set_ylabel("Rise in mean daily VIC peak demand (MW)")
    axes[2].set_title("System peak demand", loc="left", fontweight="bold", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Share of M + L customers in the middle interval (%)")
        ax.axvline(0, color="#2a78d6", lw=1, ls=":")
        rp.style(ax)
    axes[0].text(0.5, axes[0].get_ylim()[1], " 0% = 2 intervals", color="#2a78d6", fontsize=8, va="top")
    fig.suptitle("3 intervals with a fixed rule: what the middle-interval share changes (today's mix)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"no plan: mean daily peak {base_peak.mean():,.0f} MW, annual peak {base_peak.max():,.0f} MW")
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

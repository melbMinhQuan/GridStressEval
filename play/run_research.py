"""Research: when does a three-slot scheme differ from two slots? (doc/three_slot_research.tex)

Premise model (doc/references/New.md): a circuit is one phase of a distribution line with N customers. On a
cloudy day (no local solar) every customer draws a base current I0; moderate free-window users add dI_M
inside their slot; heavy hitters (two level-2 EV chargers, 40 kW) add dI_H. The phase limit T is set so that
an average phase can host one heavy hitter but not two.

Slots: the mandated 11-14 (middle) and two non-overlapping relief slots 9-12 and 12-15.

Strategies (circuit level):
    policy        everyone in 11-14
    proposal      everyone in 11-14; if a phase has >= 2 heavy hitters they move to 9-12 / 12-15
    split         every free-window user in 9-12 or 12-15, lowest peak (the two-slot upper bound)
    timers        proposal, and a third heavy hitter goes to 11-14; heavy sessions last d hours from the
                  slot opening (smart-charging timers)
    adaptive(a)   proposal, plus a share 1 - a of moderate users moved to the outer slots (system lever)
    cap           everyone in 11-14, free power capped at CAP_KW per customer

Run from the repo root:  python3 -m play.run_research
"""
import csv
import sys
from math import lgamma, log, exp
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

# ---------------------------------------------------------------- premise model (illustrative, per phase)
N = 30                                            # customers per phase
V = 240
I0 = 5.0                                          # base current per customer on a cloudy midday, A (1.2 kW)
DI_M = 15.0                                       # moderate free-window user, A (3.6 kW, e.g. hot water)
DI_H = 40_000 / V                                 # heavy hitter, A (40 kW: two level-2 EV chargers)
P_M = 0.30                                        # share of moderate users
P_H = 0.02                                        # share of heavy hitters (default)
HEADROOM = 1.25                                   # T = base + mean moderate load + 1.25 heavy hitters
CAP_KW = 7.0                                      # cap on free power per customer, kW
MID, MORN, AFT = (11.0, 14.0), (9.0, 12.0), (12.0, 15.0)
X_VALUES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]
N_VALUES = [10, 20, 30, 40, 50, 60]
UPTAKES = [0.1, 0.25, 0.5, 0.75, 1.0]
ALPHAS = np.round(np.arange(0, 1.0001, 0.1), 2)   # share of moderate users kept in the middle slot
GRID = np.arange(8, 16, 1 / 12)                   # 5-minute grid, hours
OUT = config.OUT_DIR / "research"
COL = {"policy": "#8a8984", "proposal": "#2a78d6", "split": "#1baf7a", "timers": "#eb6834",
       "adaptive": "#9c3d12", "cap": "#c9a227", "even": "#7b5cc4"}
LABEL = {"policy": "Policy: everyone 11-14", "proposal": "Proposal: heavy hitters to 9-12 / 12-15",
         "split": "Two-slot split: everyone 9-12 / 12-15", "timers": "Three slots + timers (third heavy in 11-14)",
         "adaptive": "Three slots, adaptive middle share", "cap": f"Cap: free power <= {CAP_KW:g} kW",
         "even": "Three slots, users split 1/3 each"}


def limit(n):
    return n * I0 + P_M * n * DI_M + HEADROOM * DI_H


def compositions(n, p_m, p_h):
    """[(n_M, n_H, weight)] with n_M moderate users and n_H heavy hitters (trinomial)."""
    out = []
    for n_m in range(n + 1):
        for n_h in range(n + 1 - n_m):
            lw = lgamma(n + 1) - lgamma(n_m + 1) - lgamma(n_h + 1) - lgamma(n - n_m - n_h + 1)
            for count, p in ((n_m, p_m), (n_h, p_h), (n - n_m - n_h, 1 - p_m - p_h)):
                if count:
                    lw = lw + count * log(p) if p > 0 else -np.inf
            if lw > log(1e-14):
                out.append((n_m, n_h, exp(lw)))
    total = sum(w for *_, w in out)
    return [(m, h, w / total) for m, h, w in out]


def profile(n, mods, heavy, d=3.0, cap_a=None):
    """Phase current on GRID. mods: {slot: count}; heavy: list of slots (one per heavy hitter)."""
    cur = np.full(len(GRID), n * I0)
    for (a, b), count in mods.items():
        cur[(GRID >= a) & (GRID < b)] += count * (DI_M if cap_a is None else min(DI_M, cap_a))
    for (a, b) in heavy:
        cur[(GRID >= a) & (GRID < min(b, a + d))] += DI_H if cap_a is None else min(DI_H, cap_a)
    return cur


def allocate(strategy, n_m, n_h, alpha=1.0):
    """(moderate users per slot, heavy hitters' slots, customers moved out of 11-14)."""
    if strategy in ("policy", "cap"):
        return {MID: n_m}, [MID] * n_h, 0
    heavy = [MID] * n_h if n_h < 2 else [MORN if i % 2 == 0 else AFT for i in range(n_h)]
    if strategy == "timers" and n_h >= 3:
        heavy = [MORN, AFT, MID] + [(MORN, AFT, MID)[i % 3] for i in range(n_h - 3)]
    moved = sum(s != MID for s in heavy)
    if strategy == "adaptive":
        keep = int(round(alpha * n_m))
        out = n_m - keep
        return {MID: keep, MORN: (out + 1) // 2, AFT: out // 2}, heavy, moved + out
    return {MID: n_m}, heavy, moved


def split_peak(n, n_m, n_h, d=3.0):
    """Lowest peak with every user in 9-12 or 12-15 (the two-slot upper bound)."""
    best = None
    for m1 in range(n_m + 1):
        for h1 in range(n_h + 1):
            cur = profile(n, {MORN: m1, AFT: n_m - m1}, [MORN] * h1 + [AFT] * (n_h - h1), d)
            if best is None or cur.max() < best[0]:
                best = (cur.max(), m1, h1)
    return best


def circuit(strategy, n=N, p_m=P_M, p_h=P_H, d=3.0, alpha=1.0):
    """(P(no upgrade) %, customers moved %, expected extra kW per slot per phase)."""
    t = limit(n)
    ok = moved = 0.0
    kw = {}
    for n_m, n_h, w in compositions(n, p_m, p_h):
        if strategy == "split":
            peak, m1, h1 = split_peak(n, n_m, n_h, d)
            mods, heavy, mv = {MORN: m1, AFT: n_m - m1}, [MORN] * h1 + [AFT] * (n_h - h1), n_m + n_h
        else:
            mods, heavy, mv = allocate(strategy, n_m, n_h, alpha)
            cap_a = CAP_KW * 1000 / V if strategy == "cap" else None
            peak = profile(n, mods, heavy, d, cap_a).max()
        ok += w * (peak <= t)
        moved += w * mv / n
        for slot, c in mods.items():
            per = DI_M if strategy != "cap" else min(DI_M, CAP_KW * 1000 / V)
            kw[slot] = kw.get(slot, 0.0) + w * c * per * V / 1000
        for a, b in heavy:                                                      # session block [a, a + d)
            per = DI_H if strategy != "cap" else min(DI_H, CAP_KW * 1000 / V)
            block = (a, min(b, a + d))
            kw[block] = kw.get(block, 0.0) + w * per * V / 1000
    return 100 * ok, 100 * moved, kw


# ---------------------------------------------------------------- system level (Victoria)
def demand():
    with open(config.ROOFTOP_DEMAND_CSV, newline="") as f:
        hourly = np.array([float(r["demand_mw"]) for r in csv.DictReader(f)]).reshape(-1, 24)
    return np.repeat(hourly, config.SLOTS // 24, axis=1)


def state_load(kw, u):
    """Added Victorian load (MW per 5-minute slot) for expected extra kW per slot per phase."""
    phases = u * config.VIC_CUSTOMERS / N
    load = np.zeros(config.SLOTS)
    for (a, b), v in kw.items():
        load[in_range(a, b)] += phases * v / 1000
    return load


def system(kw, u, dem, solar_by_season, days_by_season):
    """(rise in mean daily peak MW, rise in annual peak MW, largest 5-minute step MW, soak MWh/day)."""
    load = state_load(kw, u)
    base = dem.max(axis=1)
    peak = (dem + load).max(axis=1)
    step = np.abs(np.diff(load)).max()
    soak = sum(np.minimum(load * rp.season_weight(s), x).sum(axis=1).mean() * config.DT * days_by_season[s]
               for s, x in solar_by_season.items()) / sum(days_by_season.values())
    return (peak - base).mean(), peak.max() - base.max(), step, soak


# ---------------------------------------------------------------- diagnosis on the report's model
def diagnosis():
    """Lowest peak of every composition for 2 and 3 slots in the report's model (All-On)."""
    d2, d3 = ((10.0, 13.0), (13.0, 16.0)), ((10.0, 13.0), (11.0, 14.0), (13.0, 16.0))
    keys = [(m, l) for m in range(config.N + 1) for l in range(config.N + 1 - m)]
    return np.array([(rp.candidates(d2, *k)[1][0], rp.candidates(d3, *k)[1][0]) for k in keys])


def style(ax):
    rp.style(ax)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"premise: N={N}, I0={I0} A, dI_M={DI_M} A, dI_H={DI_H:.0f} A, T={limit(N):.0f} A "
          f"(base {N * I0:.0f} + moderates {P_M * N * DI_M:.0f} + {HEADROOM} heavy)")

    # R1 diagnosis
    pk = diagnosis()
    print(f"R1: compositions where 3 slots lower the lowest peak: {(pk[:, 1] < pk[:, 0] - 1e-9).sum()} of {len(pk)}")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(pk[:, 0], pk[:, 1], s=10, color=COL["timers"], alpha=0.6)
    lim = [pk.min() - 20, pk.max() + 20]
    ax.plot(lim, lim, color="#52514e", lw=1, ls=":")
    ax.set_xlabel("Lowest peak, 2 slots (A)")
    ax.set_ylabel("Lowest peak, 3 slots (A)")
    ax.set_title("Every circuit: 3 slots = 2 slots", loc="left", fontweight="bold", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "r1_diagnosis.png", dpi=150)
    plt.close(fig)

    # R2 premise model: strategies at the default mix
    rows = {}
    for s in ("policy", "proposal", "split", "cap"):
        rows[s] = circuit(s)
    for d in (3.0, 1.0):
        rows[f"timers d={d:g}"] = circuit("timers", d=d)
    rows["proposal d=1"] = circuit("proposal", d=1.0)
    dem = demand()
    solar = {s: rp.solar_season(s) for s in rp.SEASONS}
    days = {s: len(x) for s, x in solar.items()}
    rows["three slots, even"] = circuit("adaptive", alpha=1 / 3)
    table = {}
    for name, (ok, mv, kw_) in rows.items():
        peak, _, step, soak = system(kw_, 0.5, dem, solar, days)
        d = 1.0 if "d=1" in name else 3.0
        free = (min(DI_H, CAP_KW * 1000 / V) if name == "cap" else DI_H) * V / 1000 * d
        table[name] = (ok, mv, soak, peak, step, free)
        print(f"R2 {name:<18} P(no upgrade) {ok:6.2f}%  moved {mv:5.2f}%  soak {soak:6.0f} MWh/day  "
              f"peak +{peak:5.0f} MW  step {step:5.0f} MW  free energy per heavy hitter {free:5.0f} kWh/day")

    # R3 Q1 heatmaps: P(no upgrade) over heavy share and customers per phase
    heat = {}
    for s, d in (("policy", 3.0), ("proposal", 3.0), ("timers", 1.0)):
        heat[(s, d)] = np.array([[circuit(s, n=n, p_h=x, d=d)[0] for n in N_VALUES] for x in X_VALUES])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, layout="constrained")
    titles = {("policy", 3.0): "Policy: everyone 11-14", ("proposal", 3.0): "Proposal: 2 relief slots",
              ("timers", 1.0): "3 slots + timers (heavy sessions 1 h)"}
    for ax, (key, grid) in zip(axes, heat.items()):
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=100)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > 55 else "#222222")
        ax.set_xticks(range(len(N_VALUES)), N_VALUES)
        ax.set_yticks(range(len(X_VALUES)), [f"{x:.1%}".replace(".0%", "%") for x in X_VALUES])
        ax.set_xlabel("Customers per phase N")
        ax.set_title(f"{titles[key]}: P(no upgrade) %", loc="left", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Heavy hitters x (moderate users 30%)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="P(no upgrade) (%)")
    fig.savefig(OUT / "r3_heatmaps.png", dpi=150)
    plt.close(fig)
    for key, grid in heat.items():
        print(f"R3 {key}: N=30 column", np.round(grid[:, N_VALUES.index(30)], 1))

    # R4 timers: P(no upgrade) against heavy share for session lengths
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for s, d, ls in (("policy", 3.0, "-"), ("proposal", 3.0, "-"), ("proposal", 1.0, "--"), ("timers", 1.0, "-"),
                     ("timers", 1.5, ":"), ("split", 3.0, ":")):
        vals = [circuit(s, p_h=x, d=d)[0] for x in X_VALUES]
        ax.plot(np.array(X_VALUES) * 100, vals, ls=ls, marker="o", ms=4, lw=2, color=COL[s],
                label=LABEL[s] + (", All-On" if d == 3 else f", sessions {d:g} h"))
        print(f"R4 {s} d={d}: " + " ".join(f"{v:.1f}" for v in vals))
    ax.set_xlabel("Heavy hitters x (%)")
    ax.set_ylabel("P(no upgrade) (%)")
    ax.set_title(f"Phase of {N} customers: avoiding an upgrade", loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "r4_timers.png", dpi=150)
    plt.close(fig)

    # R5 system level: peak, ramp and soak against uptake; adaptive middle share
    kw = {s: circuit(s)[2] for s in ("policy", "proposal", "split", "cap")}
    kw["even"] = circuit("adaptive", alpha=1 / 3)[2]
    adaptive = {a: circuit("adaptive", alpha=a) for a in ALPHAS}
    sysres = {s: [] for s in ("policy", "proposal", "split", "cap", "even", "adaptive")}
    chosen = []
    for u in UPTAKES:
        for s in ("policy", "proposal", "split", "cap", "even"):
            sysres[s].append(system(kw[s], u, dem, solar, days))
        cand = [(system(adaptive[a][2], u, dem, solar, days), a) for a in ALPHAS]
        best = min(cand, key=lambda c: (round(c[0][0], 1), c[0][2]))           # lowest peak rise, then ramp
        sysres["adaptive"].append(best[0])
        chosen.append(best[1])
        print(f"R5 u={u:.0%}: " + "  ".join(f"{s} peak+{sysres[s][-1][0]:5.0f} MW step {sysres[s][-1][2]:5.0f} MW "
                                             f"soak {sysres[s][-1][3]:6.0f}" for s in sysres)
              + f"  (adaptive middle share {best[1]:.0%}, overload ok {adaptive[best[1]][0]:.1f}%, "
                f"moved {adaptive[best[1]][1]:.1f}%)")
    x = np.array(UPTAKES) * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for s, vals in sysres.items():
        vals = np.array(vals)
        for ax, i in zip(axes, (0, 2, 3)):
            ax.plot(x, vals[:, i], "-o", ms=4, lw=2, color=COL[s], label=LABEL[s])
    for i, a in enumerate(chosen):
        axes[0].annotate(f"{a:.0%}", (x[i], sysres["adaptive"][i][0]), xytext=(4, -10), textcoords="offset points",
                         fontsize=7.5, color=COL["adaptive"])
    for ax, t, yl in zip(axes, ("Rise in mean daily VIC peak", "Largest 5-minute step in added load",
                                "Rooftop solar soaked (seasonal weights)"), ("MW", "MW", "MWh/day")):
        ax.set_title(t, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("Uptake u (% of phases on the plan)")
        ax.set_ylabel(yl)
        style(ax)
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "r5_system.png", dpi=150)
    plt.close(fig)

    # R6 state load shapes at u = 50%
    fig, ax = plt.subplots(figsize=(8, 4.6))
    hours = SLOT
    for s in ("policy", "proposal", "split"):
        ax.plot(hours, state_load(kw[s], 0.5), lw=2, color=COL[s], label=LABEL[s])
    ax.plot(hours, state_load(adaptive[0.3][2], 0.5), lw=2, color=COL["adaptive"], label=LABEL["adaptive"] + " (30%)")
    ax.set_xlim(8, 16)
    ax.set_xlabel("Hour (AEST)")
    ax.set_ylabel("Added Victorian load (MW), uptake 50%")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "r6_shapes.png", dpi=150)
    plt.close(fig)
    # R7 disruption-safety frontier at the default mix; marker size: soak at u = 50%
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    curve = np.array([(adaptive[a][1], adaptive[a][0]) for a in ALPHAS])
    ax.plot(curve[:, 0], curve[:, 1], "-", lw=1.5, color=COL["adaptive"], zorder=2,
            label="Three slots, middle share 100% (proposal) to 0%")
    for a in (1.0, 0.5, 0.0):
        i = list(ALPHAS).index(a)
        ax.annotate(f"middle {a:.0%}", curve[i], xytext=(6, -12), textcoords="offset points", fontsize=7.5,
                    color=COL["adaptive"])
    points = {"policy": "policy", "proposal": "proposal", "split": "split", "cap": "cap",
              "three slots, even": "even", "timers d=1": "timers"}
    for name, key in points.items():
        ok, mv, soak = table[name][:3]
        ax.scatter(mv, ok, s=soak / 25, color=COL[key], edgecolor="white", zorder=4,
                   label=f"{LABEL[key]}" + (", sessions 1 h" if key == "timers" else "") + f" (soak {soak:,.0f})")
    ax.set_xlabel("Customers moved out of the mandated 11-14 slot (%)")
    ax.set_ylabel("P(no upgrade) (%)")
    ax.set_title(f"Safety against disruption (heavy hitters {P_H:.0%}; marker area: soak at 50% uptake, MWh/day)",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "r7_frontier.png", dpi=150)
    plt.close(fig)
    for a in ALPHAS:
        print(f"R7 middle {a:.0%}: P(no upgrade) {adaptive[a][0]:.2f}%  moved {adaptive[a][1]:.2f}%")
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()

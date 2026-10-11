"""Sweeps for 1, 2 and 3 intervals (seasonal intervals and model of play/run_participation.py,
lowest-overload strategy, today's mix unless swept):

    T    circuit limit, A                       (utility: network capacity needed)
    p_L  large customers (EVs), p_M fixed       (government: EV growth)
    u    uptake, share of circuits on the plan  (government: plan reach; overload does not depend on u)
    q    On-Off: each interval customer uses its window on a day with probability q; overload is the
         probability a circuit overloads on a given day, soak scales the added load by q
    N    customers per circuit, T = 25 A x N    (rural vs urban circuits)

Run from the repo root:  python3 -m play.run_sweeps
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
from play.experiments.common import onoff_safe_probability

T_VALUES = np.arange(800, 1301, 20)
PL_VALUES = np.round(np.arange(0, 0.0601, 0.0025), 4)
U_VALUES = np.round(np.arange(0.05, 1.001, 0.05), 2)
Q_VALUES = np.round(np.arange(0.1, 1.001, 0.1), 2)
N_VALUES = [10, 20, 30, 40, 60, 80]
AMPS_PER_CUSTOMER = 25
CAP = 5                                           # overload cap drawn and read off, %
TODAY = {"S": 0.69, "M": 0.30, "L": 0.01}
OUT = config.OUT_DIR / "sweeps"
DEFAULTS = dict(T=config.T, N=config.N, UPTAKE=rp.UPTAKE, Q=rp.Q)


def reset():
    config.T, config.N = DEFAULTS["T"], DEFAULTS["N"]
    rp.UPTAKE, rp.Q = DEFAULTS["UPTAKE"], DEFAULTS["Q"]
    rp.candidates.cache_clear()


def point(designs, p=TODAY):
    """(overload %, soak MWh/day), All-On."""
    return rp.combine(designs, p, "lowest overload")


def onoff_point(designs, q):
    """(P(circuit overloads on a given day) %, soak MWh/day) with usage probability q."""
    rp.Q = q
    days = {h: len(rp.solar_season(h)) for h in rp.SEASONS}
    over = soak = 0.0
    for h in rp.SEASONS:
        _, s, w, alloc = rp.evaluate(designs[h], h, TODAY, "lowest overload")
        bad = sum(wi * (1 - onoff_safe_probability(list(designs[h]), *alloc[key], q)) for key, wi in w.items())
        over += bad * days[h]
        soak += s * days[h]
    total = sum(days.values())
    return over / total * 100, soak / total


def sweep(designs, values, run):
    """{K: array of run(design, value) over values}, defaults restored around every run."""
    out = {}
    for k, d in designs.items():
        rows = []
        for v in values:
            reset()
            rows.append(run(d, v))
        out[k] = np.array(rows)
    reset()
    return out


def by_t(d, v):
    config.T = v
    return point(d)


def by_pl(d, v):
    return point(d, {"S": 1 - TODAY["M"] - v, "M": TODAY["M"], "L": v})


def by_u(d, v):
    rp.UPTAKE = v
    return point(d)


def by_n(d, v):
    config.N, config.T = v, AMPS_PER_CUSTOMER * v
    rp.candidates.cache_clear()
    return point(d)


def run_all(designs):
    return {"T": (T_VALUES, sweep(designs, T_VALUES, by_t)),
            "p_L": (PL_VALUES, sweep(designs, PL_VALUES, by_pl)),
            "u": (U_VALUES, sweep(designs, U_VALUES, by_u)),
            "q": (Q_VALUES, sweep(designs, Q_VALUES, onoff_point)),
            "N": (np.array(N_VALUES), sweep(designs, N_VALUES, by_n))}


LABELS = {
    "T": ("Circuit limit T (A)", 1, "Utility: circuit capacity"),
    "p_L": ("Large customers p_L (%)", 100, "Government: EV growth (p_M = 30%)"),
    "u": ("Uptake u (% of circuits on the plan)", 100, "Government: plan uptake"),
    "q": ("Usage probability q (On-Off)", 1, "Realism: customers use the window on a share q of days"),
    "N": ("Customers per circuit N (T = 25 A x N)", 1, "Circuit size: rural to urban"),
}


def figure(res):
    fig, axes = plt.subplots(2, len(res), figsize=(5 * len(res), 8.2))
    for col, (key, (values, curves)) in enumerate(res.items()):
        xlabel, scale, title = LABELS[key]
        for k, arr in curves.items():
            kw = dict(color=rp.COLORS[k], lw=2, ms=3.5, marker="o", label=f"{k} interval{'s' * (k > 1)}")
            axes[0, col].plot(values * scale, arr[:, 0], **kw)
            axes[1, col].plot(values * scale, arr[:, 1], **kw)
        soak = np.concatenate([arr[:, 1] for arr in curves.values()])
        mid = (soak.max() + soak.min()) / 2
        if soak.max() - soak.min() < 0.1 * mid:                                 # small differences: span >= +-5%
            axes[1, col].set_ylim(mid * 0.95, mid * 1.05)
        axes[0, col].axhline(CAP, color="#52514e", ls=":", lw=1)
        axes[0, col].set_title(title, loc="left", fontweight="bold", fontsize=10)
        axes[0, col].set_ylabel("P(overload on a given day) (%)" if key == "q" else "Circuits that overload (%)")
        axes[1, col].set_ylabel("Rooftop solar soaked (MWh/day)")
        axes[1, col].set_xlabel(xlabel)
        for ax in axes[:, col]:
            rp.style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].text(axes[0, 0].get_xlim()[1], CAP, f"{CAP}% cap ", ha="right", va="bottom", fontsize=8,
                    color="#52514e")
    fig.suptitle("What 1, 2 and 3 intervals allow (top: overload, bottom: solar soak; today's mix unless swept)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sweeps.png", dpi=130)
    plt.close(fig)


def readoffs(res):
    """Numbers to quote."""
    values, curves = res["T"]
    print("T: smallest limit with overload <= cap:",
          {k: f"{values[np.flatnonzero(a[:, 0] <= CAP)[0]]} A" for k, a in curves.items()})
    values, curves = res["p_L"]
    print("p_L: largest share of large customers with overload <= cap:",
          {k: f"{values[np.flatnonzero(a[:, 0] <= CAP)[-1]]:.2%}" if (a[:, 0] <= CAP).any() else "none"
           for k, a in curves.items()})
    values, curves = res["u"]
    for u in (0.1, 0.5, 1.0):
        i = np.flatnonzero(np.isclose(values, u))[0]
        print(f"u = {u:.0%}: soak", {k: f"{a[i, 1]:,.0f}" for k, a in curves.items()})
    values, curves = res["q"]
    for q in (0.3, 0.5, 1.0):
        i = np.flatnonzero(np.isclose(values, q))[0]
        print(f"q = {q:.1f}: daily overload", {k: f"{a[i, 0]:.3g}%" for k, a in curves.items()},
              "soak", {k: f"{a[i, 1]:,.0f}" for k, a in curves.items()})
    values, curves = res["N"]
    for i, n in enumerate(values):
        print(f"N = {n}: overload", {k: f"{a[i, 0]:.3g}%" for k, a in curves.items()})


def main():
    designs = {k: {h: rp.choose(k, h) for h in rp.SEASONS} for k in (1, 2, 3)}
    res = run_all(designs)
    figure(res)
    readoffs(res)
    print(f"figure: {OUT / 'sweeps.png'}")


if __name__ == "__main__":
    main()

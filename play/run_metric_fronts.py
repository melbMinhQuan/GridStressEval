"""Pareto fronts with continuous safety metrics instead of the All-On 0/1 overload.

Safety metrics (trinomial-weighted over compositions, per share rule alpha):
    peak : E[max_g A_g] / T                      expected peak loading (lower is better)
    onoff: P(A > T) when each M/L customer's extra load is on for its whole interval with prob. Q
Soak: added load inside intervals, capped by the circuit's curtailment share (All-On), w(n)-weighted.
Each panel compares a 2-interval design with its 3-interval counterpart; fronts use mixing.

Run from the repo root:  python3 -m play.run_metric_fronts
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import compositions, weight

# ================================================================ INPUTS (this run)
Q = 0.5                                           # On-Off: probability a customer's extra load is on
UPTAKE = config.UPTAKE
MIN_WEIGHT = 1e-9                                 # On-Off skips compositions lighter than this
PAIRS = [("A2: 08-11 | 11-14", [(8, 11), (11, 14)], "A3: 08-11 | 10:30-13:30 | 11-14", [(8, 11), (10.5, 13.5), (11, 14)]),
         ("B2: 09-12 | 12-15", [(9, 12), (12, 15)], "B3: 08-11 | 09-12 | 12-15", [(8, 11), (9, 12), (12, 15)]),
         ("C2: 08-11 | 12-15", [(8, 11), (12, 15)], "C3: 09-11 | 11-13 | 13-15", [(9, 11), (11, 13), (13, 15)])]
SINGLE = ("1: 11-14", [(11, 14)])
PLOT = config.OUT_DIR / "metric_fronts.png"
# ==================================================================================


def _rules_eval(design):
    """For every share rule: (E[peak]/T, P_onoff(violation), soak, alpha)."""
    from play.circuit import segments
    from play.experiments.common import onoff_safe_probability
    from play.run_report import added_table, rules, split, window_segments
    from play.solar import excess_per_circuit
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    excess = excess_per_circuit(None, UPTAKE)
    segs = window_segments(design)
    tables = [added_table(a, b, excess) for a, b, _ in segs]
    amps, n, limit = config.CURRENT, config.N, config.T
    out = []
    for alpha in rules(len(design)):
        peak_sum = viol = soak = 0.0
        for (_, n_m, n_l), wi in zip(comps, w):
            med, lar = split(n_m, alpha), split(n_l, alpha)
            peak = s = 0.0
            for (_, _, cov), table in zip(segs, tables):
                m, l = med[cov].sum(), lar[cov].sum()
                peak = max(peak, amps["S"] * (n - m - l) + amps["M"] * m + amps["L"] * l)
                s += table[m, l]
            peak_sum += wi * peak
            soak += wi * s
            if wi >= MIN_WEIGHT:
                viol += wi * (1 - onoff_safe_probability(design, med, lar, Q))
        out.append((peak_sum / limit, viol, soak, alpha))
    return out


def main():
    from play.run_strategy_pareto import mixed_front
    designs = [SINGLE] + [(lab, d) for two, d2, three, d3 in PAIRS for lab, d in ((two, d2), (three, d3))]
    cache = PLOT.with_suffix(".pkl")
    key = (Q, UPTAKE, str(config.P_TYPE), config.T, tuple(lab for lab, _ in designs))
    res = None
    if cache.exists():
        saved = pickle.loads(cache.read_bytes())
        res = saved["res"] if saved["key"] == key else None
    if res is None:
        with ProcessPoolExecutor() as pool:
            res = dict(zip((lab for lab, _ in designs), pool.map(_rules_eval, [d for _, d in designs], chunksize=1)))
        cache.write_bytes(pickle.dumps(dict(key=key, res=res)))

    metrics = [("peak", 0, "Expected peak loading E[max A]/T  ←  better"),
               ("onoff", 1, f"Circuits without overload, On-Off q={Q:g} (%)  →  better")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, (mname, idx, xlabel) in zip(axes, metrics):
        print(f"\n{mname}")
        for ax, (two, _, three, _) in zip(row, PAIRS):
            for lab, color, wide in ((SINGLE[0], "#6B6B6B", False), (two, "#2A7FB8", True), (three, "#D9822B", False)):
                pts = [(r[idx], r[2], r[3]) for r in res[lab]]
                hull = mixed_front(pts)                           # minimises metric, maximises soak
                to_x = (lambda v: v) if mname == "peak" else (lambda v: (1 - v) * 100)
                ax.scatter([to_x(p[0]) for p in pts], [p[1] for p in pts], s=10, color=color, alpha=0.25, lw=0)
                xs = [to_x(1 - x) for (x, _), _ in hull]
                ys = [y for (_, y), _ in hull]
                ax.plot(xs, ys, "-o", color=color, lw=6 if wide else 2.2, ms=8 if wide else (9 if lab == SINGLE[0] else 4),
                        alpha=0.55 if wide else 1, zorder=3 if wide else 4, label=lab)
                print(f"  {lab:<34} " + "  ->  ".join(f"{to_x(1 - x):.3f}, soak {y:.1f}" for (x, y), _ in hull))
            if mname == "peak":
                ax.invert_xaxis()
            ax.set_xlabel(xlabel)
            ax.set_title(f"{two.split(':')[0]} vs {three.split(':')[0]}", loc="left", fontweight="bold")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            ax.grid(color="#e6e6e6")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        row[0].set_ylabel("Expected solar soak per circuit (kWh/day)  ↑")
    fig.suptitle(f"Fronts with continuous safety metrics (uptake {UPTAKE:.0%}; top: expected peak loading, "
                 f"bottom: On-Off violation, q = {Q:g})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

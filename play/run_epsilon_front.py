"""epsilon-constraint Pareto front over interval design AND allocation, with hypervolume.

For each allowed overload share eps:
    S_K(eps) = max over designs d with K intervals, share rules alpha (mixing allowed)
               of O_soak  subject to  O_viol <= eps.
Soak: added load (P_j - P_S) inside intervals, capped by the circuit's curtailment share, w(n)-weighted.
Violation: trinomial-weighted, total current. Hypervolume: area dominated by the front above the
reference point (safe share 0, soak of the 1-interval design), so HV_1 = 0.

Run from the repo root:  python3 -m play.run_epsilon_front
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock, compositions, weight

# ================================================================ INPUTS (this run)
WINDOW, LENGTH, STEP_START = (8, 15), 3, 0.5      # designs: intervals inside WINDOW, start grid
KS = (1, 2, 3)
UPTAKES = [0.01, 0.1]
GRID = np.linspace(0, 1, 1001)                    # safe share levels (1 - eps)
PLOT = config.OUT_DIR / "epsilon_front.png"
# ==================================================================================

COLORS = {1: "#6B6B6B", 2: "#2A7FB8", 3: "#D9822B"}


def designs(k):
    starts = np.arange(WINDOW[0], WINDOW[1] - LENGTH + 1e-9, STEP_START)
    return [[(float(s), float(s) + LENGTH) for s in c] for c in combinations(starts, k)]


def _hull(job):
    """Best mixed-rule front of one design: [(safe, soak, alpha)], most soak first."""
    design, uptake = job
    from play.run_report import added_table, evaluate, rules, window_segments
    from play.run_strategy_pareto import mixed_front
    from play.solar import excess_per_circuit
    comps = compositions()
    w = np.array([weight(*c) for c in comps])
    excess = excess_per_circuit(None, uptake)
    segs = window_segments(design)
    tables = [added_table(a, b, excess) for a, b, _ in segs]
    pts = [(*evaluate(a, segs, tables, comps, w), a) for a in rules(len(design))]
    return [(x, y, a.tolist()) for (x, y), a in mixed_front(pts)]


def best_at(hull, safe):
    """Max soak of one design with safe share >= `safe` (mixing along its hull); -inf if unreachable."""
    xs = np.array([h[0] for h in hull])
    ys = np.array([h[1] for h in hull])
    out = np.full(len(safe), -np.inf)
    out[safe <= xs[0]] = ys[0]                                    # constraint slack: most-soak end
    inside = (safe > xs[0]) & (safe <= xs[-1] + 1e-12)
    out[inside] = np.interp(safe[inside], xs, ys)
    return out


def main():
    jobs = [(d, u) for u in UPTAKES for k in KS for d in designs(k)]
    cache = PLOT.with_suffix(".pkl")                              # per-design fronts, reused if inputs match
    key = (WINDOW, LENGTH, STEP_START, tuple(UPTAKES), config.SOLAR_SOURCE, str(config.P_TYPE), config.T)
    hulls = None
    if cache.exists():
        saved = pickle.loads(cache.read_bytes())
        hulls = saved["hulls"] if saved["key"] == key else None
    if hulls is None:
        print(f"{len(jobs)} design evaluations ...")
        with ProcessPoolExecutor() as pool:
            hulls = dict(zip(((tuple(d), u) for d, u in jobs), pool.map(_hull, jobs, chunksize=1)))
        cache.write_bytes(pickle.dumps(dict(key=key, hulls=hulls)))

    fig, axes = plt.subplots(1, len(UPTAKES), figsize=(7.5 * len(UPTAKES), 5.6), squeeze=False)
    for ax, u in zip(axes[0], UPTAKES):
        fronts, who = {}, {}
        for k in KS:
            ds = designs(k)
            vals = np.array([best_at(hulls[(tuple(d), u)], GRID) for d in ds])     # (designs, grid)
            fronts[k] = vals.max(axis=0)
            who[k] = vals.argmax(axis=0)
        ref = fronts[1][np.isfinite(fronts[1])].max()                             # 1-interval soak
        print(f"\nuptake {u:.0%}  (reference soak = 1 interval, {ref:.1f} kWh/day)")
        for k in KS:
            f = fronts[k]
            ok = np.isfinite(f)
            hv = np.trapz(np.clip(f[ok] - ref, 0, None), GRID[ok])
            ax.plot(GRID[ok] * 100, f[ok], color=COLORS[k], lw=2.8,
                    label=f"{k} interval{'s' * (k > 1)} (HV {hv:.2f})")
            ax.scatter(GRID[ok][-1] * 100, f[ok][-1], s=50, color=COLORS[k], zorder=5)
            print(f"  {k} interval(s): reaches {GRID[ok][-1]:.1%} safe; soak {f[ok][0]:.1f} at no constraint, "
                  f"{f[ok][-1]:.1f} at the safest; hypervolume {hv:.3f}")
            prev = None
            for i in np.flatnonzero(ok):                                          # designs along the front
                d = designs(k)[who[k][i]]
                if d != prev:
                    print(f"     from {GRID[i]:6.1%} safe: {' | '.join(f'{clock(a)}-{clock(b)}' for a, b in d)}")
                    prev = d
        ax.set_xlabel("Required share of circuits without overload, 1 − ε (%)")
        ax.set_ylabel("Max expected soak per circuit (kWh/day)")
        ax.set_title(f"Uptake {u:.0%}", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="center left")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"ε-constraint front: best design and allocation for each safety requirement "
                 f"({LENGTH} h intervals inside {clock(WINDOW[0])}-{clock(WINDOW[1])})", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"\ngraph: {PLOT}")


if __name__ == "__main__":
    main()

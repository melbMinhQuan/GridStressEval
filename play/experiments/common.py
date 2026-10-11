"""Shared tools for the experiments: designs, fast min-peak search, On-Off violation,
expected added-load profiles, and parallel design sweeps with config overrides."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import comb, lgamma, log, exp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.circuit import clock, segments, splits
from play.frontier import evaluate, pareto_front
from play.soak import in_range, soak_table
from play.solar import excess_per_circuit

OUT = config.OUT_DIR / "experiments"

DESIGNS = {                                       # representative designs used across experiments
    "1 x 3h  11-14": [(11, 14)],
    "2 x 3h  09-12 | 12-15": [(9, 12), (12, 15)],
    "3 x 3h  08-11 | 09-12 | 12-15 (overlap)": [(8, 11), (9, 12), (12, 15)],
    "3 x 2h  09-11 | 11-13 | 13-15": [(9, 11), (11, 13), (13, 15)],
    "3 x 3h  08-11 | 11-14 | 14-17": [(8, 11), (11, 14), (14, 17)],
}
COLORS = ["#6B6B6B", "#2A7FB8", "#D9822B", "#2E8B57", "#8E44AD"]
LINES = [dict(ls="-", marker="o"), dict(ls="-", marker="o"), dict(ls="--", marker="s", mfc="none"),
         dict(ls="-", marker="o"), dict(ls="--", marker="s", mfc="none")]   # dashed: same result as the design above


# ---------------------------------------------------------------- compositions
def log_weight(n_s, n_m, n_l, p):
    """log w(n) (trinomial); -inf when a probability is 0 and its count is positive."""
    out = lgamma(n_s + n_m + n_l + 1) - lgamma(n_s + 1) - lgamma(n_m + 1) - lgamma(n_l + 1)
    for count, prob in ((n_s, p["S"]), (n_m, p["M"]), (n_l, p["L"])):
        if count:
            if prob <= 0:
                return -np.inf
            out += count * log(prob)
    return out


def weights(n, p, floor=1e-13):
    """{(n_M, n_L): w(n)} for compositions with w >= floor."""
    out = {}
    for n_m in range(n + 1):
        for n_l in range(n + 1 - n_m):
            lw = log_weight(n - n_m - n_l, n_m, n_l, p)
            if lw > log(floor):
                out[(n_m, n_l)] = exp(lw)
    return out


# ---------------------------------------------------------------- All-On minimum peak
def min_peak(intervals, n_m, n_l, n=None, current=None):
    """Lowest peak current over all allocations, and that allocation (medium, large per interval)."""
    n = n or config.N
    current = current or config.CURRENT
    k = len(intervals)
    medium, large = np.array(splits(n_m, k)), np.array(splits(n_l, k))
    peak = np.zeros((len(medium), len(large)))
    for _, _, cov in segments(intervals):
        m = medium[:, cov].sum(axis=1)[:, None]
        l = large[:, cov].sum(axis=1)[None, :]
        peak = np.maximum(peak, current["S"] * (n - m - l) + current["M"] * m + current["L"] * l)
    i, j = np.unravel_index(peak.argmin(), peak.shape)
    return peak[i, j], medium[i], large[j]


def min_peak_table(intervals, keys, n=None, current=None):
    """{(n_M, n_L): lowest peak} for the given compositions."""
    return {key: min_peak(intervals, *key, n=n, current=current)[0] for key in keys}


def violation(peaks, w, limit):
    return sum(wi for key, wi in w.items() if peaks[key] > limit)


# ---------------------------------------------------------------- On-Off
def binom_pmf(n, q):
    return np.array([comb(n, y) * q ** y * (1 - q) ** (n - y) for y in range(n + 1)])


def onoff_safe_probability(intervals, medium, large, q, n=None, current=None, limit=None):
    """P(no overload) when each M/L customer's extra load is on (whole interval) with probability q.

    Extra current of interval k: Z_k = (I_M-I_S) Y_M + (I_L-I_S) Y_L, Y ~ Binomial(customers, q),
    independent across intervals; segment g carries sum_{k in C(g)} Z_k on top of N I_S."""
    n, current, limit = n or config.N, current or config.CURRENT, limit or config.T
    d_m, d_l = current["M"] - current["S"], current["L"] - current["S"]
    values, probs = [], []
    for m, l in zip(medium, large):
        pm, pl = binom_pmf(int(m), q), binom_pmf(int(l), q)
        z = (d_m * np.arange(len(pm))[:, None] + d_l * np.arange(len(pl))[None, :]).ravel()
        pz = (pm[:, None] * pl[None, :]).ravel()
        uniq, inv = np.unique(z, return_inverse=True)
        values.append(uniq)
        probs.append(np.bincount(inv, weights=pz))
    k = len(intervals)
    grids = np.meshgrid(*values, indexing="ij")
    joint = probs[0]
    for pk in probs[1:]:
        joint = np.multiply.outer(joint, pk)
    ok = np.ones(joint.shape, bool)
    for _, _, cov in segments(intervals):
        ok &= n * current["S"] + sum(grids[c] for c in cov) <= limit
    return joint[ok].sum()


# ---------------------------------------------------------------- expected added load
def safest_added_profile(intervals, uptake):
    """Expected added kW per slot at the safest end of the front (as in frontier.safest_soak_profile)."""
    k, segs = len(intervals), segments(intervals)
    excess = excess_per_circuit(None, uptake)
    tables = [soak_table(a, b, excess) for a, b, _ in segs]
    load = config.load_kw()
    extra = {j: load[j] - load["S"] for j in ("M", "L")}
    profile = np.zeros(config.SLOTS)
    for (n_m, n_l), w in weights(config.N, config.P_TYPE).items():
        over, soak, medium, large = evaluate(segs, tables, k, n_m, n_l)
        score = np.where(over, -np.inf, soak) if (~over).any() else soak
        i, j = np.unravel_index(score.argmax(), soak.shape)
        for a, b, cov in segs:
            profile[in_range(a, b)] += w * (extra["M"] * medium[i, cov].sum() + extra["L"] * large[j, cov].sum())
    return profile


# ---------------------------------------------------------------- parallel sweeps
def apply(overrides):
    for key, value in overrides.items():
        setattr(config, key, value)


def _safest(job):
    overrides, intervals, uptake = job
    apply(overrides)
    safe, soak, _ = pareto_front(intervals, uptake)[-1]
    return 1 - safe, soak


def safest_points(jobs):
    """[(violation, soak)] at the safest allocation for jobs = [(overrides, intervals, uptake)]."""
    with ProcessPoolExecutor() as pool:
        return list(pool.map(_safest, jobs, chunksize=1))


def design_grid(k, earliest, latest, length, step=0.5):
    starts = np.arange(earliest, latest - length + 1e-9, step)
    return [[(float(s), float(s) + length) for s in c] for c in combinations(starts, k)]


def best_design(designs, results):
    """Safest design, then most soak: (index, violation, soak)."""
    i = min(range(len(designs)), key=lambda j: (round(results[j][0], 10), -results[j][1]))
    return i, *results[i]


# ---------------------------------------------------------------- output
def name(intervals):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in intervals)


def style(ax):
    ax.grid(color="#e6e6e6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    print(f"graph: {OUT / filename}")

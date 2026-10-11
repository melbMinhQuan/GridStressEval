"""Pareto front of expected solar soak vs share of circuits without overload."""
import numpy as np

from play import config
from play.circuit import compositions, segments, splits, weight
from play.soak import in_range, soak_table
from play.solar import excess_per_circuit


def evaluate(segs, tables, k, n_m, n_l):
    """Overload flag and soak for every allocation, shaped (medium splits, large splits)."""
    n, amps = config.N, config.CURRENT
    medium, large = np.array(splits(n_m, k)), np.array(splits(n_l, k))
    peak = np.zeros((len(medium), len(large)))
    soak = np.zeros_like(peak)
    for (_, _, cov), table in zip(segs, tables):
        m = medium[:, cov].sum(axis=1)[:, None]
        l = large[:, cov].sum(axis=1)[None, :]
        peak = np.maximum(peak, amps["S"] * (n - m - l) + amps["M"] * m + amps["L"] * l)
        soak += table[m, l]
    return peak > config.T, soak, medium, large


def pareto_front(intervals, uptake, uniform=False):
    """[(share without overload, expected soak kWh/day, share of M+L per interval)], most soak first.
    uniform=True weights every composition equally instead of by w(n)."""
    k = len(intervals)
    segs = segments(intervals)
    excess = excess_per_circuit(None, uptake)
    tables = [soak_table(a, b, excess) for a, b, _ in segs]
    comps = compositions()
    w = np.full(len(comps), 1 / len(comps)) if uniform else np.array([weight(*c) for c in comps])

    best, best_over, safe, best_mix, safe_mix = [], [], [], [], []
    for _, n_m, n_l in comps:
        over, soak, medium, large = evaluate(segs, tables, k, n_m, n_l)
        mix = lambda i: medium[i[0]] + large[i[1]]                         # M+L customers per interval
        i = np.unravel_index(np.lexsort((over.ravel(), -soak.ravel()))[0], soak.shape)  # safe wins ties
        best.append(soak[i]); best_over.append(over[i]); best_mix.append(mix(i))
        if (~over).any():
            j = np.unravel_index(np.where(over, -np.inf, soak).argmax(), soak.shape)
            safe.append(soak[j]); safe_mix.append(mix(j))
        else:
            safe.append(np.nan); safe_mix.append(mix(i))
    best, safe, best_over = np.array(best), np.array(safe), np.array(best_over)
    best_mix, safe_mix = np.array(best_mix, float), np.array(safe_mix, float)

    p_over, soak, mix = w @ best_over, w @ best, w @ best_mix
    customers = w @ best_mix.sum(axis=1)
    front = [(1 - p_over, soak, mix / customers)]
    fixable = best_over & ~np.isnan(safe)
    cost = np.where(fixable, best - safe, np.inf)
    for i in np.argsort(cost):                                             # cheapest soak loss first
        if not fixable[i]:
            break
        p_over -= w[i]
        soak -= w[i] * cost[i]
        mix += w[i] * (safe_mix[i] - best_mix[i])
        front.append((1 - p_over, soak, mix / customers))
    return front


def safest_soak_profile(intervals, uptake):
    """Expected soak per slot (kW) at the safest end of the front: each composition uses its
    best safe allocation (or its best allocation if none is safe), weighted by w(n)."""
    k = len(intervals)
    segs = segments(intervals)
    excess = excess_per_circuit(None, uptake)
    tables = [soak_table(a, b, excess) for a, b, _ in segs]
    load = config.load_kw()
    extra = {j: load[j] - load["S"] for j in ("M", "L")}
    by_load = {}                                                           # added kW per segment -> total w(n)
    for n in compositions():
        _, n_m, n_l = n
        over, soak, medium, large = evaluate(segs, tables, k, n_m, n_l)
        score = np.where(over, -np.inf, soak) if (~over).any() else soak
        i, j = np.unravel_index(score.argmax(), soak.shape)
        added = tuple(extra["M"] * medium[i, cov].sum() + extra["L"] * large[j, cov].sum() for _, _, cov in segs)
        by_load[added] = by_load.get(added, 0.0) + weight(*n)

    profile = np.zeros(config.SLOTS)
    for added, w in by_load.items():
        d = np.zeros(config.SLOTS)
        for (a, b, _), kw in zip(segs, added):
            d[in_range(a, b)] = kw
        profile += w * np.minimum(d, excess).mean(axis=0)                 # min per day, then mean
    return profile

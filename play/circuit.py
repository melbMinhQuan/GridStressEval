"""One circuit: segments, allocations, current, overload, and violation probability."""
from itertools import product
from math import factorial

from play import config


def clock(h):
    return f"{int(h):02d}:{round(h % 1 * 60):02d}"


def segments(intervals):
    """[(start, end, covering interval indices)] for time inside at least one interval."""
    edges = sorted({t for iv in intervals for t in iv})
    segs = [(a, b, [k for k, (s, e) in enumerate(intervals) if s <= a and b <= e])
            for a, b in zip(edges, edges[1:])]
    return [seg for seg in segs if seg[2]]          # time outside every interval is ignored


def splits(count, k):
    """Every way to put `count` identical customers into k intervals."""
    if k == 1:
        return [(count,)]
    return [(first,) + rest for first in range(count + 1) for rest in splits(count - first, k - 1)]


def active(covering, alloc):
    """S_g, M_g, L_g during a segment covered by the intervals in `covering`."""
    m = sum(alloc["M"][k] for k in covering)
    l = sum(alloc["L"][k] for k in covering)
    return {"S": config.N - m - l, "M": m, "L": l}


def total(counts, per):
    return sum(counts[j] * per[j] for j in counts)


def peak(segs, alloc):
    """max_g A_g, in A."""
    return max(total(active(c, alloc), config.CURRENT) for _, _, c in segs)


def weight(n_s, n_m, n_l):
    """Trinomial probability w(n)."""
    p, n = config.P_TYPE, config.N
    return (factorial(n) / (factorial(n_s) * factorial(n_m) * factorial(n_l))
            * p["S"] ** n_s * p["M"] ** n_m * p["L"] ** n_l)


def compositions():
    """Every (n_S, n_M, n_L) with n_S + n_M + n_L = N."""
    n = config.N
    return [(n - n_m - n_l, n_m, n_l) for n_m in range(n + 1) for n_l in range(n + 1 - n_m)]


def auto_allocation(n_m, n_l, segs, k):
    """Allocation with the lowest peak current (first one found on ties)."""
    return min(({"M": list(m), "L": list(l)} for m, l in product(splits(n_m, k), splits(n_l, k))),
               key=lambda a: peak(segs, a))


def violation_probability(segs, k):
    """sum_n w(n) r(n, auto allocation of n)."""
    p, cache = 0.0, {}
    for n in compositions():
        w = weight(*n)
        if w < 1e-15:                               # negligible compositions
            continue
        _, n_m, n_l = n
        if (n_m, n_l) not in cache:
            cache[(n_m, n_l)] = peak(segs, auto_allocation(n_m, n_l, segs, k)) > config.T
        p += w * cache[(n_m, n_l)]
    return p

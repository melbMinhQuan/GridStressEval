"""Load profiles over the day and solar soak = min(added load, curtailment)."""
import numpy as np

from play import config
from play.circuit import active, total

SLOT = np.arange(config.SLOTS) * config.DT        # start hour of each slot


def in_range(a, b):
    return (SLOT >= a) & (SLOT < b)


def load_profiles(segs, alloc):
    """(base, added) kW per slot: everyone's small load, and the load the free intervals add."""
    load = config.load_kw()
    base, added = np.zeros(config.SLOTS), np.zeros(config.SLOTS)
    for a, b, cov in segs:
        base[in_range(a, b)] = config.N * load["S"]
        added[in_range(a, b)] = total(active(cov, alloc), load) - config.N * load["S"]
    return base, added


def kwh(series, a, b):
    """Energy of a kW-per-slot series between hours a and b."""
    return series[in_range(a, b)].sum() * config.DT


def soak_table(a, b, excess):
    """F[m, l] = mean over days of sum_{t in [a, b)} min(added load, excess) dt,
    with m medium and l large customers at full load in [a, b)."""
    load, n = config.load_kw(), config.N
    extra = {j: load[j] - load["S"] for j in ("M", "L")}
    x = excess[:, in_range(a, b)]                                             # (days, slots)
    table = np.zeros((n + 1, n + 1))
    for m in range(n + 1):
        added = extra["M"] * m + extra["L"] * np.arange(n + 1 - m)            # kW for l = 0..N-m
        table[m, : n + 1 - m] = np.minimum(added[:, None, None], x[None]).sum(axis=2).mean(axis=1) * config.DT
    return table

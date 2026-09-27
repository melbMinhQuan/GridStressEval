"""Exact allocation tables for one, two or three labelled intervals."""

import numpy as np


def allocation_tables(customers, currents, intervals):
    """Dynamic programming over remaining M/L counts; no random sampling.

    Each subproblem uses the original connected population size. Removing an
    assigned high-current customer does not remove their small current elsewhere.
    """
    s, m, l = (currents[j] for j in 'SML')
    counts = np.arange(customers+1)
    demand = customers*s + (m-s)*counts[:, None] + (l-s)*counts[None, :]
    peaks = [None, demand.astype(float)]
    choices = [None, None]
    for k in range(2, intervals+1):
        peak = np.full(demand.shape, np.inf)
        choice = np.zeros((*demand.shape, 2), dtype=int)
        for nm in range(customers+1):
            for nl in range(customers-nm+1):
                first = demand[:nm+1, :nl+1]
                candidates = np.maximum(first, peaks[-1][nm::-1, nl::-1])
                m1, l1 = np.unravel_index(np.argmin(candidates), candidates.shape)
                peak[nm, nl] = candidates[m1, l1]
                choice[nm, nl] = (m1, l1)
        peaks.append(peak)
        choices.append(choice)
    return peaks[-1], choices


def recover_allocation(nm, nl, intervals, choices):
    medium, large = [], []
    for k in range(intervals, 1, -1):
        m1, l1 = (int(v) for v in choices[k][nm, nl])
        medium.append(m1)
        large.append(l1)
        nm, nl = nm-m1, nl-l1
    return medium+[nm], large+[nl]

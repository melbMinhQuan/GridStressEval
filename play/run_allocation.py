"""One circuit: allocation per interval, energy and solar soak per hour, violation probability.

Run from the repo root:  python3 -m play.run_allocation   (or python3 play/run_allocation.py)
Shared model inputs live in play/config.py; the inputs below are for this run only.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_allocation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from play import config
from play.circuit import active, auto_allocation, clock, peak, segments, total, violation_probability
from play.plots import plot_allocation
from play.soak import kwh, load_profiles
from play.solar import excess_per_circuit

# ================================================================ INPUTS (this run)
NUM_INTERVALS = 3                                 # 1, 2 or 3 (times in config.INTERVAL_SETS)
COMPOSITION = {"S": 32, "M": 7, "L": 1}           # this circuit; must sum to config.N
ALLOCATION = {"M": [3, 2, 2], "L": [0, 1, 0]}     # customers per interval, or None = auto (lowest peak)
SOLAR_DAY = "2025-11-06"                          # one day of 2025, or None = average over all days
ENERGY_STEP = 1.0                                 # section 2: row width in hours
ENERGY_SUMS = [(10.5, 13.5)]                      # section 2: extra totals over these ranges
PLOT = config.OUT_DIR / "allocation.png"
# ==================================================================================


def main():
    n, limit = config.N, config.T
    intervals = config.INTERVAL_SETS[NUM_INTERVALS]
    k = len(intervals)
    segs = segments(intervals)
    assert sum(COMPOSITION.values()) == n, "COMPOSITION must sum to N"
    alloc = ALLOCATION or auto_allocation(COMPOSITION["M"], COMPOSITION["L"], segs, k)
    assert sum(alloc["M"]) == COMPOSITION["M"] and sum(alloc["L"]) == COMPOSITION["L"], \
        "ALLOCATION must place every M and L customer"

    print(f"Circuit {COMPOSITION['S']} S, {COMPOSITION['M']} M, {COMPOSITION['L']} L | "
          f"{k} interval(s) | allocation: {'auto' if ALLOCATION is None else 'manual'}\n")

    print("1. Allocation (full-load customers per interval; S = everyone else)")
    for i, (a, b) in enumerate(intervals):
        m, l = alloc["M"][i], alloc["L"][i]
        print(f"   Interval {i + 1} {clock(a)}-{clock(b)} : {n - m - l:>2} S, {m:>2} M, {l:>2} L")

    print("\n   Segments of the day (overlaps add up)")
    for a, b, cov in segs:
        c = active(cov, alloc)
        amps = total(c, config.CURRENT)
        names = " + ".join(f"I{i + 1}" for i in cov)
        flag = "  OVER" if amps > limit else ""
        print(f"   {clock(a)}-{clock(b)}  {names:<9} {c['S']:>2} S, {c['M']:>2} M, {c['L']:>2} L"
              f"   {amps:>5} A{flag}")

    excess = excess_per_circuit(SOLAR_DAY)                        # (days, slots) kW
    base, added = load_profiles(segs, alloc)
    soaked = np.minimum(added, excess).mean(axis=0)               # min per day, then mean over days
    available = excess.mean(axis=0)

    day = "average day of 2025" if SOLAR_DAY is None else SOLAR_DAY
    print(f"\n2. Energy per {ENERGY_STEP:g} h, one circuit, {day} (kWh)")
    print(f"   uptake {config.UPTAKE:.0%} -> {config.circuits():,.0f} circuits share the curtailment")
    print(f"   {'time':<11} {'consumed':>8} {'added':>7} {'excess':>7} {'soak':>7}   soak / added")
    start, end = segs[0][0], segs[-1][1]
    h = start
    while h < end - 1e-9:
        e = min(h + ENERGY_STEP, end)
        add, sk = kwh(added, h, e), kwh(soaked, h, e)
        bar = "█" * round(20 * sk / add) if add else ""
        print(f"   {clock(h)}-{clock(e)} {kwh(base + added, h, e):>8.1f} {add:>7.1f} "
              f"{kwh(available, h, e):>7.1f} {sk:>7.1f}   {bar:<20} {sk / add if add else 0:.0%}")
        h = e
    print(f"   {'total':<11} {kwh(base + added, start, end):>8.1f} {kwh(added, start, end):>7.1f} "
          f"{kwh(available, start, end):>7.1f} {kwh(soaked, start, end):>7.1f}")
    for a, b in ENERGY_SUMS:
        print(f"   {clock(a)}-{clock(b)} {kwh(base + added, a, b):>8.1f} {kwh(added, a, b):>7.1f} "
              f"{kwh(available, a, b):>7.1f} {kwh(soaked, a, b):>7.1f}")

    plot_allocation(PLOT, intervals, base, added, excess, soaked,
                    f"Solar soak, {day} (uptake {config.UPTAKE:.0%}, {config.circuits():,.0f} circuits)")
    print(f"   graph: {PLOT}")

    top = peak(segs, alloc)
    print(f"\n3. This circuit overloads: {'YES' if top > limit else 'no'} (peak {top} A, limit {limit} A)")
    print(f"   P(violation), all circuits, auto allocation: {violation_probability(segs, k):.4%}")


if __name__ == "__main__":
    main()

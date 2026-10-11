"""Soak gain of 3 over 2 intervals for each composition, both using their soak-maximising allocation
(overload ignored when choosing; reported). Also shows one chosen composition in detail.

Run from the repo root:  python3 -m play.run_composition_gain   (or python3 play/run_composition_gain.py)
"""
import sys
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_composition_gain.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from play import config
from play.circuit import clock, compositions, segments, weight
from play.frontier import evaluate
from play.soak import soak_table
from play.solar import excess_per_circuit

# ================================================================ INPUTS (this run)
EXAMPLE = (10, 10, 20)                            # (n_S, n_M, n_L), must sum to config.N
UPTAKE = config.UPTAKE
PAIRS = {                                         # (2-interval design, 3-interval design)
    "default (config)": (config.INTERVAL_SETS[2], config.INTERVAL_SETS[3]),
    "best from sweep": ([(9, 12), (12, 15)], [(8, 11), (9, 12), (12, 15)]),
}
TOP = 5
# ==================================================================================


def name(intervals):
    return " | ".join(f"{clock(a)}-{clock(b)}" for a, b in intervals)


def max_soak(intervals, excess):
    """{(n_M, n_L): (soak, overloads, medium split, large split)} for the soak-maximising allocation."""
    segs = segments(intervals)
    tables = [soak_table(a, b, excess) for a, b, _ in segs]
    out = {}
    for _, n_m, n_l in compositions():
        over, soak, medium, large = evaluate(segs, tables, len(intervals), n_m, n_l)
        i, j = np.unravel_index(soak.argmax(), soak.shape)
        out[(n_m, n_l)] = (soak[i, j], bool(over[i, j]), medium[i], large[j])
    return out


def describe(n, r2, r3, two, three):
    for label, (soak, over, med, lar), design in (("2 intervals", r2, two), ("3 intervals", r3, three)):
        alloc = ", ".join(f"{clock(a)}-{clock(b)}: {m} M + {l} L" for (a, b), m, l in zip(design, med, lar))
        print(f"    {label}: soak {soak:6.1f} kWh/day  {'OVERLOADS' if over else 'safe':<9}  {alloc}")


def main():
    assert sum(EXAMPLE) == config.N, "EXAMPLE must sum to N"
    excess = excess_per_circuit(None, UPTAKE)
    print(f"uptake {UPTAKE:.0%} ({config.circuits(UPTAKE):,.0f} circuits); soak-maximising allocation for each")
    for pair, (two, three) in PAIRS.items():
        s2, s3 = max_soak(two, excess), max_soak(three, excess)
        rows = []
        for n_s, n_m, n_l in compositions():
            a, b = s2[(n_m, n_l)][0], s3[(n_m, n_l)][0]
            rows.append(((n_s, n_m, n_l), b - a, b / a - 1 if a > 0 else 0.0))
        print(f"\n{pair}: 2 = {name(two)}   3 = {name(three)}")

        n = EXAMPLE
        print(f"  example {n[0]} S, {n[1]} M, {n[2]} L  (probability {weight(*n):.2e})")
        describe(n, s2[n[1:]], s3[n[1:]], two, three)
        r = next(x for x in rows if x[0] == n)
        print(f"    gain {r[1]:+.1f} kWh/day ({r[2]:+.1%})")

        for title, key in (("absolute", 1), ("relative", 2)):
            print(f"  top {TOP} compositions by {title} gain:")
            for comp, gain, rel in sorted(rows, key=lambda x: -x[key])[:TOP]:
                print(f"    {comp[0]:>2} S, {comp[1]:>2} M, {comp[2]:>2} L   gain {gain:+6.1f} kWh/day ({rel:+6.1%})  "
                      f"probability {weight(*comp):.1e}")
            best = sorted(rows, key=lambda x: -x[key])[0][0]
            describe(best, s2[best[1:]], s3[best[1:]], two, three)

        w = np.array([weight(*c) for c, _, _ in rows])
        gains = np.array([g for _, g, _ in rows])
        print(f"  expected gain over the customer mix: {w @ gains:+.2f} kWh/day per circuit")


if __name__ == "__main__":
    main()

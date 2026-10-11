"""Every point of a Pareto front as a table: which composition moved to its safe allocation and
what that did to the totals. Saves CSV to play/out/.

Run from the repo root:  python3 -m play.run_front_table [--uniform] [--uptake 0.1]
"""
import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from play import config
from play.circuit import compositions, segments, weight
from play.frontier import evaluate
from play.soak import soak_table
from play.solar import excess_per_circuit

DESIGNS = {2: [(9, 12), (12, 15)], 3: [(8, 11), (9, 12), (12, 15)]}


def split(med, lar):
    return " / ".join(f"{m}M+{l}L" for m, l in zip(med, lar))


def front_table(intervals, uptake, uniform):
    k, segs = len(intervals), segments(intervals)
    excess = excess_per_circuit(None, uptake)
    tables = [soak_table(a, b, excess) for a, b, _ in segs]
    comps = compositions()
    w = np.full(len(comps), 1 / len(comps)) if uniform else np.array([weight(*c) for c in comps])
    info = []
    for c in comps:
        over, soak, med, lar = evaluate(segs, tables, k, c[1], c[2])
        i = np.unravel_index(np.lexsort((over.ravel(), -soak.ravel()))[0], soak.shape)
        safe = None
        if over[i] and (~over).any():
            j = np.unravel_index(np.where(over, -np.inf, soak).argmax(), soak.shape)
            safe = (soak[j], split(med[j[0]], lar[j[1]]))
        info.append((c, soak[i], bool(over[i]), split(med[i[0]], lar[i[1]]), safe))

    p_over = sum(wi * x[2] for wi, x in zip(w, info))
    total = sum(wi * x[1] for wi, x in zip(w, info))
    rows = [dict(point=0, moved="(start: every composition at best soak)", weight="", best_alloc="",
                 safe_alloc="", soak_lost=0.0, safe_pct=100 * (1 - p_over), expected_soak=total)]
    movers = sorted((x[1] - x[4][0], idx) for idx, x in enumerate(info) if x[4])
    for n, (cost, idx) in enumerate(movers, 1):
        c, s, _, best_alloc, (_, safe_alloc) = info[idx]
        p_over -= w[idx]
        total -= w[idx] * cost
        rows.append(dict(point=n, moved=f"{c[0]}S {c[1]}M {c[2]}L", weight=w[idx], best_alloc=best_alloc,
                         safe_alloc=safe_alloc, soak_lost=cost, safe_pct=100 * (1 - p_over), expected_soak=total))
    counts = dict(safe_at_best=sum(not x[2] for x in info), movers=len(movers),
                  never_safe=sum(x[2] and not x[4] for x in info))
    return rows, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--uptake", type=float, default=config.UPTAKE)
    args = ap.parse_args()
    for k, iv in DESIGNS.items():
        rows, counts = front_table(iv, args.uptake, args.uniform)
        print(f"\n{k} intervals {iv}, uptake {args.uptake:.0%}, {'uniform' if args.uniform else 'trinomial'} weights")
        print(f"  compositions: {counts['safe_at_best']} safe at best soak, {counts['movers']} on the curve, "
              f"{counts['never_safe']} never safe")
        print(f"  {'pt':>3} {'moved':<14} {'weight':>9} {'best alloc (overloads)':<26} {'safe alloc':<26}"
              f"{'soak lost':>10} {'safe %':>8} {'E[soak]':>9}")
        for r in rows:
            wt = f"{r['weight']:.2e}" if r["weight"] != "" else ""
            print(f"  {r['point']:>3} {r['moved'][:14]:<14} {wt:>9} {r['best_alloc']:<26} {r['safe_alloc']:<26}"
                  f"{r['soak_lost']:>10.2f} {r['safe_pct']:>8.3f} {r['expected_soak']:>9.3f}")
        path = config.OUT_DIR / f"front_{k}int_{'uniform' if args.uniform else 'weighted'}_{args.uptake:g}.csv"
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"  csv: {path}")


if __name__ == "__main__":
    main()

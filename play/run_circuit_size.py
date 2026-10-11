"""Figure 7 for customers per circuit: overload against net soak as rho grows (lowest overload), one panel per
N, at the calibrated load (k = 1); the circuit limit is 25 A x N (1000 A at N = 40). Two ways to grow N:
    circuits   the plan circuits stay at the calibrated count: N / 40 times as many customers on the plan
    customers  the customers on the plan stay at the calibrated count: 40 / N times as many circuits

Run from the repo root:  python3 -m play.run_circuit_size [customers]   (--plot: replot from the cache)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from play import config
from play import run_participation as rp
from play.run_load_scale import REPORT, tradeoff_figure

NS = [20, 40, 80]                                 # customers per circuit, one panel each
N0 = config.N
FIXED = "customers" if "customers" in sys.argv else "circuits"                 # what stays fixed as N grows
CACHE = config.OUT_DIR / "load_scale" / f"tradeoff_circuit_size_{FIXED}.npz"
PLOT = REPORT / f"fig2c_tradeoff_n_fixed_{FIXED}.png"


def size_result(job):
    """(overload %, net soak) over RHO for N customers per circuit and K intervals (lowest overload)."""
    n, k = job
    u0 = rp.UPTAKE
    config.N, config.T = n, 25 * n
    if FIXED == "circuits":
        rp.UPTAKE = u0 * n / N0                                                 # same circuits, n customers each
    d = {s: rp.choose(k, s) for s in rp.SEASONS}
    rows = []
    for rho in rp.RHO:
        rp.candidates.cache_clear()                                            # bounds memory at large N
        rows.append(rp.combine(d, rho, "lowest overload"))
    lo = np.array(rows)
    i = np.flatnonzero(np.isclose(rp.RHO, rp.TODAY))[0]
    cap = np.flatnonzero(lo[:, 0] <= 5)
    print(f"[{FIXED}] N={n} K={k}: today {lo[i, 0]:.2f}% / {rp.pct(lo[i, 1]):.1f}%;  peak {rp.pct(lo[:, 1]).max():.1f}% at rho "
          f"{rp.RHO[lo[:, 1].argmax()]:.1%};  cap 5%: rho {rp.RHO[cap[-1]] if len(cap) else 0:.1%}", flush=True)
    return lo, lo


def main():
    jobs = [(n, k) for n in NS for k in (1, 2, 3)]
    with Pool(len(jobs)) as pool:
        out = pool.map(size_result, jobs)
    res = dict(zip(jobs, out))
    np.savez(CACHE, **{f"{n}_{k}": np.stack(r) for (n, k), r in res.items()})
    tradeoff_figure(res, NS, "N = {}", PLOT)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        z = np.load(CACHE)
        tradeoff_figure({(n, k): tuple(z[f"{n}_{k}"]) for n in NS for k in (1, 2, 3)}, NS, "N = {}", PLOT)
    else:
        main()

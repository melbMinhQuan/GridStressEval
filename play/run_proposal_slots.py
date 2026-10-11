"""Figure 6 with the proposal's fixed slots in every season: 1 interval 11-14; 2 intervals 9-12, 12-15;
3 intervals 9-12, 11-14, 12-15. Soak is counted over 09:00-15:00 for every design (one interval: small customers
only outside 11-14) and given as a share of the rooftop solar between 09:00 and 15:00. Lowest overload, one
panel per uptake, zoom on today's mix below.

Run from the repo root:  python3 -m play.run_proposal_slots
"""
import sys
from multiprocessing import Pool
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from play import config
from play import run_participation as rp
from play.run_load_scale import K_REPORT, tradeoff_figure

SLOTS = {1: ((11.0, 14.0),), 2: ((9.0, 12.0), (12.0, 15.0)), 3: ((9.0, 12.0), (11.0, 14.0), (12.0, 15.0))}
rp.choose = lambda k, season: SLOTS[k]            # same slots in every season (also in worker processes)
rp.SOAK_HOURS = "two"                             # count 09-15 for every design
rp.WINDOW = (9, 15)                               # reference: rooftop solar 09-15
rp.SOAK_LABEL = "Net soak (% of rooftop solar, 09–15)"
PLOT = config.ROOT / "doc" / "vis" / "fig6_proposal_slots.png"


def curve(job):
    f, k = job
    rp.UPTAKE = f
    lo = np.array([rp.combine(rp.year_designs(k), rho, "lowest overload") for rho in rp.RHO])
    return lo


def main():
    jobs = [(f, k) for f in K_REPORT for k in (1, 2, 3)]
    with Pool(len(jobs)) as pool:
        out = pool.map(curve, jobs)
    res = {job: (lo, lo) for job, lo in zip(jobs, out)}
    today = np.isclose(rp.RHO, rp.TODAY)
    for (f, k), (lo, _) in res.items():
        print(f"u={f:.0%} K={k}: today overload {lo[today, 0][0]:5.1f}%  net soak {rp.pct(lo[today, 1])[0]:5.1f}%  "
              f"peak {rp.pct(lo[:, 1]).max():5.1f}% at rho {rp.RHO[lo[:, 1].argmax()]:.1%}")
    tradeoff_figure(res, path=PLOT)
    print(f"graph: {PLOT}")


if __name__ == "__main__":
    main()

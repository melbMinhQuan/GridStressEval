"""E4/E5 Same daily energy: the window 09-15 is split into K equal slots; every M/L customer needs
H hours of full-power use per day inside its slot.
  E4 inflexible: full power for H h at an uncontrolled time in the slot. Overload uses the full
     current over the whole slot (worst case); soak uses the expected load, power x H / slot length.
  E5 flexible (smart charging): the H h of energy is spread evenly over the slot, so the extra
     current is (I_j - I_S) x H / slot length for both overload and soak."""
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.experiments.common import min_peak_table, safest_points, save, style, violation, weights

SPAN, H = (9, 15), 2.0
UPTAKES = [0.01, 0.1, 0.5]


def slots(k):
    length = (SPAN[1] - SPAN[0]) / k
    return [(SPAN[0] + i * length, SPAN[0] + (i + 1) * length) for i in range(k)], length


def scaled_current(factor):
    c = config.CURRENT
    return {"S": c["S"], "M": c["S"] + (c["M"] - c["S"]) * factor, "L": c["S"] + (c["L"] - c["S"]) * factor}


def run():
    w = weights(config.N, config.P_TYPE)
    rows = []
    for k in (1, 2, 3):
        intervals, length = slots(k)
        current = scaled_current(H / length)
        jobs = [({"CURRENT": current}, intervals, u) for u in UPTAKES]
        flexible = safest_points(jobs)                                 # (violation, soak) per uptake
        inflexible_violation = violation(min_peak_table(intervals, w.keys()), w, config.T)
        rows.append((k, length, intervals, inflexible_violation, flexible))

    print(f"E4/E5 same daily energy: {H:g} h of full-power use per M/L customer, window "
          f"{SPAN[0]}-{SPAN[1]} split into K slots")
    print(f"  {'K':>2} {'slot':>5} {'E4 viol':>9} {'E5 viol':>9}" + "".join(f"{f'soak {u:.0%}':>10}" for u in UPTAKES))
    for k, length, _, v4, flex in rows:
        print(f"  {k:>2} {length:>4g}h {v4:>9.3%} {flex[0][0]:>9.3%}" + "".join(f"{s:>10.1f}" for _, s in flex))
    print("  soak (kWh/day per circuit) is the same for E4 and E5: expected load = power x H / slot length")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    ks = [r[0] for r in rows]
    x = np.arange(len(ks))
    a.bar(x - 0.2, [max(r[3], 1e-7) * 100 for r in rows], 0.4, color="#C0392B", label="E4 inflexible (fixed power)")
    a.bar(x + 0.2, [max(r[4][0][0], 1e-7) * 100 for r in rows], 0.4, color="#2E8B57", label="E5 flexible (spread energy)")
    a.set_yscale("log")
    a.set_xticks(x, [f"{k} x {r[1]:g}h" for k, r in zip(ks, rows)])
    a.set_ylabel("P(violation) (%, log)  ↓ better")
    a.set_title("Violation, same energy per customer", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=8)
    for i, u in enumerate(UPTAKES):
        b.bar(x + (i - 1) * 0.27, [r[4][i][1] for r in rows], 0.25, label=f"uptake {u:.0%}",
              color=["#9ecae1", "#3182bd", "#08519c"][i])
    b.set_xticks(x, [f"{k} x {r[1]:g}h" for k, r in zip(ks, rows)])
    b.set_ylabel("Solar soak per circuit (kWh/day)  ↑ better")
    b.set_title("Soak, same energy per customer", loc="left", fontweight="bold")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        style(ax)
    fig.suptitle(f"E4/E5  Window {SPAN[0]}-{SPAN[1]} split into K slots, {H:g} h of full-power energy per customer",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "e04_energy.png")
    return rows


if __name__ == "__main__":
    run()

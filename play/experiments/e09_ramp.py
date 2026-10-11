"""E9 Grid ramp: total added load across all participating circuits,
    D_state(t) = C x E[D(t)]  (MW),  C = uptake x VIC customers / N,
and its largest 5-minute step (how many MW switch on or off at once)."""
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.experiments.common import COLORS, name, safest_added_profile, save, style
from play.soak import SLOT

UPTAKE = 0.1
RAMP_DESIGNS = {
    "2 x 3h  09-12 | 12-15": [(9, 12), (12, 15)],
    "2 x 3h  08-11 | 11-14": [(8, 11), (11, 14)],
    "3 x 3h  08-11 | 09-12 | 12-15": [(8, 11), (9, 12), (12, 15)],
    "3 x 3h  08-11 | 10-13 | 12-15": [(8, 11), (10, 13), (12, 15)],
    "3 x 2h  09-11 | 11-13 | 13-15": [(9, 11), (11, 13), (13, 15)],
    "3 x 3h  08-11 | 11-14 | 14-17": [(8, 11), (11, 14), (14, 17)],
}


def run():
    circuits = config.circuits(UPTAKE)
    out = {}
    for label, d in RAMP_DESIGNS.items():
        mw = safest_added_profile(d, UPTAKE) * circuits / 1000
        steps = np.diff(np.r_[0, mw, 0])
        out[label] = (mw, np.abs(steps).max(), mw.max())
    print(f"E9 grid ramp (uptake {UPTAKE:.0%}, {circuits:,.0f} circuits): added state load, largest 5-min step")
    for label, (_, step, peak) in out.items():
        print(f"  {label:<36} largest step {step:6.1f} MW   peak added load {peak:6.1f} MW")

    fig, ax = plt.subplots(figsize=(10, 5))
    hours = SLOT + config.DT / 2
    for (label, (mw, step, _)), c in zip(out.items(), COLORS + ["#C0392B"]):
        ax.step(hours, mw, where="mid", color=c, lw=2, label=f"{label}  (max step {step:.0f} MW)")
    ax.set_xlim(7, 18)
    ax.set_xlabel("Time of day (AEST)")
    ax.set_ylabel("Added load, all participating circuits (MW)")
    ax.set_title(f"E9  State-level added load and ramps (uptake {UPTAKE:.0%})", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, "e09_ramp.png")
    return out


if __name__ == "__main__":
    run()

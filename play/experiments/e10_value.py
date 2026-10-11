"""E10 Value: circuit upgrades avoided compared with one interval,
    avoided = (P1 - PK) x C,   C = uptake x VIC customers / N,
priced at an ILLUSTRATIVE cost per upgrade (replace with DNSP data)."""
import matplotlib.pyplot as plt
import numpy as np

from play import config
from play.experiments.common import COLORS, DESIGNS, min_peak_table, save, style, violation, weights

UPTAKES = [0.01, 0.1, 0.5, 1.0]
COST_PER_UPGRADE = 100_000                       # AUD, illustrative placeholder only


def run():
    w = weights(config.N, config.P_TYPE)
    p = {label: violation(min_peak_table(d, w.keys()), w, config.T) for label, d in DESIGNS.items()}
    base = p["1 x 3h  11-14"]
    print(f"E10 upgrades avoided vs one interval (cost {COST_PER_UPGRADE:,} AUD per upgrade is a placeholder)")
    print(f"  {'design':<42}{'P(viol)':>9}" + "".join(f"{f'u={u:.0%}':>16}" for u in UPTAKES))
    rows = {}
    for label, pk in p.items():
        avoided = [(base - pk) * config.circuits(u) for u in UPTAKES]
        rows[label] = avoided
        print(f"  {label:<42}{pk:>9.3%}" + "".join(
            f"{a:>7,.0f} ${a * COST_PER_UPGRADE / 1e6:>6,.0f}M" for a in avoided))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(UPTAKES))
    labels = [label for label in rows if label != "1 x 3h  11-14"]
    for i, label in enumerate(labels):
        ax.bar(x + (i - (len(labels) - 1) / 2) * 0.2, rows[label], 0.19, color=COLORS[1 + i], label=label)
    ax.set_xticks(x, [f"uptake {u:.0%}\n({config.circuits(u):,.0f} circuits)" for u in UPTAKES])
    ax.set_ylabel("Circuit upgrades avoided vs one interval")
    ax.set_title(f"E10  Upgrades avoided (x {COST_PER_UPGRADE:,} AUD per upgrade, illustrative)", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, "e10_value.png")
    return rows


if __name__ == "__main__":
    run()

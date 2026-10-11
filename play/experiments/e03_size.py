"""E3 Circuit size: N customers with the limit scaled T = 25 N (A)."""
import matplotlib.pyplot as plt

from play import config
from play.experiments.common import COLORS, DESIGNS, LINES, min_peak_table, save, style, violation, weights

SIZES = [20, 40, 60, 80]
AMPS_PER_CUSTOMER = 25


def run():
    table = {label: [] for label in DESIGNS}
    for n in SIZES:
        w = weights(n, config.P_TYPE)
        for label, d in DESIGNS.items():
            table[label].append(violation(min_peak_table(d, w.keys(), n=n), w, AMPS_PER_CUSTOMER * n))
    print(f"E3 circuit size: P(violation) with T = {AMPS_PER_CUSTOMER} A x N")
    print(f"  {'design':<42}" + "".join(f"{f'N={n}':>10}" for n in SIZES))
    for label, row in table.items():
        print(f"  {label:<42}" + "".join(f"{v:>10.3%}" for v in row))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, row), c, line in zip(table.items(), COLORS, LINES):
        ax.plot(SIZES, [max(v, 1e-7) * 100 for v in row], color=c, lw=2, ms=8, label=label, **line)
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xlabel(f"customers per circuit N (limit T = {AMPS_PER_CUSTOMER} A x N)")
    ax.set_ylabel("P(violation) (%, log)  ↓ better")
    ax.set_title("E3  Circuit size", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, "e03_size.png")
    return table


if __name__ == "__main__":
    run()

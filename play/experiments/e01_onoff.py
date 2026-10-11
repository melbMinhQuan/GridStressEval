"""E1 On-Off: each M/L customer's extra load is on for its whole interval with probability q
(base load always on). The operator allocates with the All-On min-peak rule. q = 1 is All-On."""
import matplotlib.pyplot as plt

from play import config
from play.experiments.common import (COLORS, DESIGNS, LINES, min_peak, onoff_safe_probability, save, style,
                                     weights)

Q = [0.1, 0.25, 0.5, 0.75, 1.0]


def run():
    w = weights(config.N, config.P_TYPE)
    table = {}
    for label, d in DESIGNS.items():
        alloc = {key: min_peak(d, *key)[1:] for key in w}
        table[label] = [sum(wi * (1 - onoff_safe_probability(d, *alloc[key], q)) for key, wi in w.items())
                        for q in Q]
    print("E1 On-Off: P(violation) vs probability q that a customer's extra load is on")
    print(f"  {'design':<42}" + "".join(f"{f'q={q:g}':>10}" for q in Q))
    for label, row in table.items():
        print(f"  {label:<42}" + "".join(f"{v:>10.3%}" for v in row))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, row), c, line in zip(table.items(), COLORS, LINES):
        ax.plot(Q, [max(v, 1e-7) * 100 for v in row], color=c, lw=2, ms=8, label=label, **line)
    ax.set_yscale("log")
    ax.set_xlabel("q = probability a medium/large customer's extra load is on")
    ax.set_ylabel("P(violation) (%, log)  ↓ better")
    ax.set_title("E1  On-Off: violation vs usage probability", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, "e01_onoff.png")
    return table


if __name__ == "__main__":
    run()

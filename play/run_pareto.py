"""Pareto fronts of expected solar soak vs circuits without overload, for 1, 2 and 3 intervals.

Run from the repo root:  python3 -m play.run_pareto [--uniform] [--best]
  --uniform  every composition counts equally (no trinomial weights)
  --best     use the best designs from the design sweep instead of config.INTERVAL_SETS
Shared model inputs live in play/config.py; the inputs below are for this run only.
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):                             # allow `python3 play/run_pareto.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from play import config
from play.frontier import pareto_front, safest_soak_profile
from play.plots import plot_fronts, plot_soak_profile, plot_summary
from play.solar import excess_per_circuit

# ================================================================ INPUTS (this run)
UPTAKES = [0.01, config.UPTAKE, 0.5]              # one panel per uptake
PLOT = config.OUT_DIR / "pareto.png"
PLOT_ZOOM = config.OUT_DIR / "pareto_zoom.png"    # 2 vs 3 intervals only
PLOT_SUMMARY = config.OUT_DIR / "summary.png"     # safest point of each front
PLOT_PROFILE = config.OUT_DIR / "soak_profile.png"  # mean soak over the day, 2 and 3 intervals
BEST_SETS = {1: [(11, 14)], 2: [(9, 12), (12, 15)], 3: [(8, 11), (9, 12), (12, 15)]}
# ==================================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", action="store_true", help="weight every composition equally")
    parser.add_argument("--best", action="store_true", help="best designs from the sweep")
    args = parser.parse_args()
    sets = BEST_SETS if args.best else config.INTERVAL_SETS
    tag = ("_best" if args.best else "") + ("_uniform" if args.uniform else "")
    plot, zoom, summary = (p.with_name(p.stem + tag + p.suffix) for p in (PLOT, PLOT_ZOOM, PLOT_SUMMARY))
    weighting = "every composition equal" if args.uniform else "trinomial weights"
    print(f"designs: {', '.join(' | '.join(f'{a:g}-{b:g}' for a, b in iv) for iv in sets.values())}; {weighting}")

    fronts = {}
    for uptake in UPTAKES:
        fronts[uptake] = {k: pareto_front(iv, uptake, args.uniform) for k, iv in sets.items()}
        print(f"uptake {uptake:.0%} ({config.circuits(uptake):,.0f} circuits)")
        for k, front in fronts[uptake].items():
            for name, (safe, soak, mix) in (("most soak", front[0]), ("most safe", front[-1])):
                shares = ", ".join(f"{v:.0%}" for v in mix)
                print(f"  {k} interval(s) {name:<9}  no overload {safe:7.2%}  soak {soak:7.2f} kWh/day  "
                      f"M+L per interval: {shares}")
    plot_fronts(plot, fronts, UPTAKES, (1, 2, 3),
                f"Pareto front, All-On, 2025 VIC1 curtailment ({weighting})")
    plot_fronts(zoom, fronts, UPTAKES, (2, 3), f"Pareto front, 2 vs 3 intervals, zoomed ({weighting})")
    plot_summary(summary, fronts, UPTAKES, args.uniform)
    if args.uniform or args.best:
        print(f"graph: {plot}\ngraph: {zoom}\ngraph: {summary}")
        return
    profiles = {u: {k: safest_soak_profile(config.INTERVAL_SETS[k], u) for k in (2, 3)} for u in UPTAKES}
    curtailment = {u: excess_per_circuit(None, u).mean(axis=0) for u in UPTAKES}
    plot_soak_profile(PLOT_PROFILE, profiles, curtailment, UPTAKES)
    print(f"graph: {PLOT}\ngraph: {PLOT_ZOOM}\ngraph: {PLOT_SUMMARY}\ngraph: {PLOT_PROFILE}")


if __name__ == "__main__":
    main()

"""Run every experiment:  python3 -m play.experiments  [e01 e02 ...]"""
import importlib
import sys
import time

ALL = ["e01_onoff", "e02_mix", "e03_size", "e04_energy", "e06_dst", "e07_rooftop", "e08_evening",
       "e09_ramp", "e10_value"]

if __name__ == "__main__":
    chosen = [m for m in ALL if not sys.argv[1:] or m.split("_")[0] in sys.argv[1:]]
    for module in chosen:
        start = time.time()
        print(f"\n{'=' * 100}\n{module}\n{'=' * 100}")
        importlib.import_module(f"play.experiments.{module}").run()
        print(f"({time.time() - start:.0f} s)")

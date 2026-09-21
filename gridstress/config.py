"""Shared model defaults and project paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "results.json"
EXPERIMENTS_DIRECTORY = ROOT / "data" / "experiments"
FIGURES_DIRECTORY = ROOT / "figures"
CACHE_DIRECTORY = ROOT / "runtime" / "cache"

LOAD_CURRENT = {"S": 20, "M": 40, "L": 100}
THRESHOLD = 1000
NUM_CUSTOMERS = 40
P_ON = 0.5
SAMPLES = 10_000
SEED = 42
MAX_CUSTOMERS = 100
MAX_SAMPLES = 100_000


def compositions(total_customers):
    """Yield each non-negative (S, M, L) triple with the requested total."""
    for small in range(total_customers + 1):
        for medium in range(total_customers - small + 1):
            yield small, medium, total_customers - small - medium

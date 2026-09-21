"""Reusable Monte Carlo samples for the local interactive viewer.

Samples are shared across allocations (common random numbers). Each allocation
still has independent customers and trials; estimates across allocations are
correlated, which makes slider comparisons smoother. This uses a different
random stream layout from model_full.py, but the same probability model.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


CACHE_DIRECTORY = Path(__file__).resolve().parent / "cache"
SEED = 42
MAX_CUSTOMERS = 100
MAX_SAMPLES = 100_000


@lru_cache(maxsize=3)
def active_counts(customers, samples, probability_percent):
    """One active-count bank for every group size, shape (3, n+1, k).

    The disk cache holds only the most recently generated bank. The three
    most recent banks are retained in memory; neither cache grows unbounded.
    """
    path = CACHE_DIRECTORY / "latest_active_counts.npz"
    key = np.array([1, SEED, customers, samples, probability_percent])
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                if np.array_equal(saved["key"], key):
                    counts = saved["counts"]
                    if counts.shape == (3, customers + 1, samples):
                        counts.setflags(write=False)
                        return counts
        except (OSError, ValueError, KeyError):
            pass

    counts = np.zeros((3, customers + 1, samples), dtype=np.uint16)
    for group, seed in enumerate(np.random.SeedSequence(SEED).spawn(3)):
        rng = np.random.default_rng(seed)
        for size in range(1, customers + 1):
            counts[group, size] = (
                counts[group, size - 1]
                + (rng.random(samples) < probability_percent / 100)
            )
    CACHE_DIRECTORY.mkdir(exist_ok=True)
    temporary = CACHE_DIRECTORY / "latest_active_counts.pending.npz"
    np.savez_compressed(temporary, key=key, counts=counts)
    temporary.replace(path)
    counts.setflags(write=False)
    return counts


def validate_settings(settings):
    ranges = {
        "small": (0, 500), "medium": (0, 500), "large": (0, 500),
        "customers": (0, MAX_CUSTOMERS), "probability": (0, 100),
        "threshold": (0, 50_000), "samples": (1000, MAX_SAMPLES),
    }
    if set(settings) != set(ranges):
        raise ValueError("Expected loads, customers, probability, threshold and samples")
    for name, (minimum, maximum) in ranges.items():
        value = settings[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")


def evaluate(settings):
    validate_settings(settings)
    return _evaluate(**settings)


@lru_cache(maxsize=32)
def _evaluate(small, medium, large, customers, probability, threshold, samples):
    counts = active_counts(customers, samples, probability)
    # Promote before weighting: uint16 would overflow at higher loads.
    weighted = [counts[i].astype(np.int32) * weight
                for i, weight in enumerate((small, medium, large))]
    rows = []
    for s in range(customers + 1):
        for m in range(customers - s + 1):
            l = customers - s - m
            loads = weighted[0][s] + weighted[1][m] + weighted[2][l]
            violations = int(np.count_nonzero(loads > threshold))
            rows.append({"S": s, "M": m, "L": l,
                         "iterations": samples, "violation_count": violations,
                         "violation_probability": violations / samples})
    return rows

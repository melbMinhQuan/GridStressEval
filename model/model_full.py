"""Monte Carlo model: independent on/off states for every customer."""

import json
from pathlib import Path

import numpy as np


# Edit these inputs, then run this file to regenerate results.json.
load_current = {"S": 20, "M": 40, "L": 100}
threshold = 1000
num_customers = 40
p_on = 0.5
iterations = 10_000  # Samples per customer combination.
random_seed = 42
results_path = Path(__file__).resolve().parent / "results.json"


def evaluate_all_combinations(
    total_customers=num_customers,
    currents=None,
    limit=threshold,
    probability_on=p_on,
    samples=iterations,
    seed=random_seed,
):
    """Return estimated P(load > limit) for every fixed-total allocation.

    Binomial draws count independently active customers within each group.
    Each sample uses fresh states; group sizes remain fixed for all samples.
    Memory use is proportional to samples, not the number of combinations.
    """
    currents = load_current if currents is None else currents
    if not isinstance(total_customers, int) or total_customers < 0:
        raise ValueError("total_customers must be a non-negative integer")
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not 0 <= probability_on <= 1:
        raise ValueError("probability_on must be between 0 and 1")
    if set(currents) != {"S", "M", "L"} or any(
        not np.isfinite(value) or value < 0 for value in currents.values()
    ):
        raise ValueError("currents must contain finite non-negative S, M, L loads")
    if not np.isfinite(limit):
        raise ValueError("limit must be finite")

    rng = np.random.default_rng(seed)
    results = []
    for small in range(total_customers + 1):
        for medium in range(total_customers - small + 1):
            large = total_customers - small - medium
            active_load = (
                currents["S"] * rng.binomial(small, probability_on, samples)
                + currents["M"] * rng.binomial(medium, probability_on, samples)
                + currents["L"] * rng.binomial(large, probability_on, samples)
            )
            violation_count = int(np.count_nonzero(active_load > limit))
            results.append({
                "S": small,
                "M": medium,
                "L": large,
                "iterations": samples,
                "violation_count": violation_count,
                "violation_probability": violation_count / samples,
            })
    return results


def run_model():
    """Run configured inputs and save results with their input metadata."""
    results = evaluate_all_combinations(
        num_customers, load_current, threshold, p_on, iterations, random_seed
    )
    payload = {
        "inputs": {
            "load_current": load_current,
            "threshold": threshold,
            "num_customers": num_customers,
            "p_on": p_on,
            "iterations": iterations,
            "random_seed": random_seed,
        },
        "results": results,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluated {len(results)} combinations, {iterations:,} samples each.")
    print(f"Saved results: {results_path}")
    return payload


if __name__ == "__main__":
    run_model()

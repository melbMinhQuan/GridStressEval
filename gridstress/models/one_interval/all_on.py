"""Deterministic single-interval All-On model."""
from gridstress.config import (
    LOAD_CURRENT as load_current,
    THRESHOLD as threshold,
    NUM_CUSTOMERS as num_customers,
    compositions,
    TYPE_PROBABILITIES,
)
from gridstress.results import overall_probability


def evaluate_all_combinations():
    """Return every allocation where S + M + L equals num_customers."""
    results = []

    for small, medium, large in compositions(num_customers):
        total_load = (small * load_current["S"] + medium * load_current["M"]
                      + large * load_current["L"])
        results.append({"S": small, "M": medium, "L": large,
                        "total_load": total_load, "violation": total_load > threshold})

    return results


if __name__ == "__main__":
    combinations = evaluate_all_combinations()
    violation_count = sum(result["violation"] for result in combinations)

    print(f"Evaluated {len(combinations)} combinations (all customers ON).")
    print(f"Violations: {violation_count}")
    print(f"Non-violations: {len(combinations) - violation_count}")
    risk = overall_probability(combinations, TYPE_PROBABILITIES)['violation_probability']
    print(f"Trinomial-weighted violation probability: {risk:.6%}; p_type={TYPE_PROBABILITIES}")

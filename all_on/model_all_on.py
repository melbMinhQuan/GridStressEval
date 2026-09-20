# Model inputs: every customer is ON.
load_current = {"S": 20, "M": 40, "L": 100}
threshold = 1000
num_customers = 40


def evaluate_all_combinations():
    """Return every allocation where S + M + L equals num_customers."""
    results = []

    for small in range(num_customers + 1):
        for medium in range(num_customers - small + 1):
            large = num_customers - small - medium
            total_load = (
                small * load_current["S"]
                + medium * load_current["M"]
                + large * load_current["L"]
            )

            results.append(
                {
                    "S": small,
                    "M": medium,
                    "L": large,
                    "total_load": total_load,
                    "violation": total_load > threshold,
                }
            )

    return results


if __name__ == "__main__":
    combinations = evaluate_all_combinations()
    violation_count = sum(result["violation"] for result in combinations)

    print(f"Evaluated {len(combinations)} combinations (all customers ON).")
    print(f"Violations: {violation_count}")
    print(f"Non-violations: {len(combinations) - violation_count}")

"""Saved results and common viewer metadata."""

import json

from gridstress.config import RESULTS_PATH


def load_results():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def viewer_settings(payload):
    inputs = payload["inputs"]
    return {
        "small": inputs["load_current"]["S"],
        "medium": inputs["load_current"]["M"],
        "large": inputs["load_current"]["L"],
        "customers": inputs["num_customers"],
        "probability": round(inputs["p_on"] * 100),
        "threshold": inputs["threshold"],
        "samples": inputs.get("samples_per_seed", inputs["iterations"]),
    }


def overall_probability(rows):
    violations = sum(row["violation_count"] for row in rows)
    trials = sum(row["iterations"] for row in rows)
    return {
        "violation_count": violations,
        "iterations": trials,
        "violation_probability": violations / trials,
        "composition_weighting": "uniform",
    }

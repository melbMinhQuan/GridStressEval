"""Saved results and common viewer metadata."""

import json
from math import exp, fsum, lgamma, log, isclose

from gridstress.config import RESULTS_PATH, SEED, TYPE_PROBABILITIES
from gridstress.probabilities import (
    class_probabilities, CLASSES, PROBABILITY_FIELDS, TYPE_FIELDS,
    customer_type_probabilities,
)


def load_results():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def viewer_settings(payload):
    inputs = payload["inputs"]
    probabilities = class_probabilities(inputs["p_on"])
    return {
        "small": inputs["load_current"]["S"],
        "medium": inputs["load_current"]["M"],
        "large": inputs["load_current"]["L"],
        "customers": inputs["num_customers"],
        **{field: probabilities[name] * 100
           for name, field in zip(CLASSES, PROBABILITY_FIELDS)},
        "threshold": inputs["threshold"],
        **({"samples": inputs.get("samples_per_seed", inputs.get("iterations"))}
           if "iterations" in inputs else {}),
    }


def simulation_inputs(settings):
    """Canonical metadata for a fresh viewer simulation (probabilities as fractions)."""
    return {
        "load_current": {name: settings[key] for name, key in
                         zip(CLASSES, ("small", "medium", "large"))},
        "num_customers": settings["customers"],
        "threshold": settings["threshold"],
        "p_on": {name: settings[field] / 100
                 for name, field in zip(CLASSES, PROBABILITY_FIELDS)},
        "iterations": settings["samples"],
        "random_seed": SEED,
    }


def type_settings(payload):
    probabilities = customer_type_probabilities(payload['inputs'].get('p_type', TYPE_PROBABILITIES))
    return {field: probabilities[name] * 100 for name, field in zip(CLASSES, TYPE_FIELDS)}


def split_viewer_settings(settings):
    """Keep population weights out of the conditional simulation's cache key."""
    settings = dict(settings)
    if any(field in settings for field in TYPE_FIELDS):
        if not all(field in settings for field in TYPE_FIELDS):
            raise ValueError("Provide all three customer type probabilities")
        probabilities = {name: settings.pop(field) / 100
                         for name, field in zip(CLASSES, TYPE_FIELDS)}
    else:
        probabilities = TYPE_PROBABILITIES
    return settings, customer_type_probabilities(probabilities)


def composition_weights(rows, type_probabilities):
    """Trinomial masses for the listed compositions; zero-probability types are allowed."""
    probabilities = customer_type_probabilities(type_probabilities)
    weights = []
    for row in rows:
        total = sum(row[name] for name in CLASSES)
        if any(row[name] > 0 and probabilities[name] == 0 for name in CLASSES):
            weights.append(0.0)
            continue
        log_mass = lgamma(total + 1) - fsum(lgamma(row[name] + 1) for name in CLASSES)
        log_mass += fsum(row[name] * log(probabilities[name])
                         for name in CLASSES if row[name] > 0)
        weights.append(exp(log_mass))
    if not isclose(fsum(weights), 1, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Overall risk requires the full set of compositions for one customer total")
    return weights


def overall_probability(rows, type_probabilities):
    """Weight conditional risks by the customer population, not simulation frequency."""
    probabilities = customer_type_probabilities(type_probabilities)
    weights = composition_weights(rows, probabilities)
    # All-on rows are deterministic; Monte Carlo rows contain estimated conditional risk.
    risk = fsum(weight * (row['violation_probability'] if 'violation_probability' in row
                         else float(row['violation'])) for weight, row in zip(weights, rows))
    return {
        "violation_probability": min(1.0, max(0.0, risk)),
        "composition_weighting": "trinomial",
        "type_probabilities": probabilities,
    }

"""Exact binomial tail sums for every customer composition (no sampling)."""

from functools import lru_cache
from math import comb

import numpy as np

from gridstress.config import compositions
from gridstress.models.one_interval.validation import validate_settings
from gridstress.probabilities import normalize_settings
from gridstress.results import viewer_settings


def exact_settings(payload):
    settings = viewer_settings(payload)
    settings.pop("samples", None)
    return settings


def exact_inputs(settings):
    return {
        "method": "exact_binomial",
        "load_current": dict(zip(("S", "M", "L"),
                                 (settings["small"], settings["medium"], settings["large"]))),
        "num_customers": settings["customers"],
        "threshold": settings["threshold"],
        "p_on": {j: settings[f"probability_{name}"] / 100
                 for j, name in zip(("S", "M", "L"), ("small", "medium", "large"))},
    }


def evaluate(settings):
    settings = normalize_settings(settings)
    if "samples" in settings:
        raise ValueError("Exact calculation does not accept samples")
    validate_settings(settings)
    return _evaluate(**settings)


@lru_cache(maxsize=32)
def _evaluate(small, medium, large, customers, probability_small,
              probability_medium, probability_large, threshold):
    probabilities = [p / 100 for p in
                     (probability_small, probability_medium, probability_large)]
    masses = [[np.array([comb(n, k) * q**k * (1-q)**(n-k)
                         for k in range(n+1)])
               for n in range(customers+1)] for q in probabilities]
    rows = []
    for s, m, l in compositions(customers):
        # Sum the L tail directly, avoiding cancellation from 1 - CDF.
        tail = np.append(np.cumsum(masses[2][l][::-1])[::-1], 0.0)
        remaining = threshold - (small * np.arange(s+1)[:, None]
                                 + medium * np.arange(m+1)[None, :])
        first_violating = np.searchsorted(large * np.arange(l+1), remaining, side="right")
        risk = float(np.sum(masses[0][s][:, None] * masses[1][m][None, :]
                            * tail[first_violating]))
        rows.append(dict(S=s, M=m, L=l, violation_probability=min(1.0, max(0.0, risk))))
    return rows

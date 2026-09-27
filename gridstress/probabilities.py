"""Class On-probabilities, including support for older shared-p datasets."""

from collections.abc import Mapping
from math import isfinite, isclose, fsum
from numbers import Real

CLASSES = ("S", "M", "L")
PROBABILITY_FIELDS = ("probability_small", "probability_medium", "probability_large")
TYPE_FIELDS = ("type_small", "type_medium", "type_large")


def class_probabilities(value):
    """Return S/M/L probabilities in [0, 1]; a scalar applies to all classes."""
    probabilities = dict(value) if isinstance(value, Mapping) else dict.fromkeys(CLASSES, value)
    if set(probabilities) != set(CLASSES) or any(
        not isinstance(p, Real) or not isfinite(p) or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise ValueError("On probabilities must be finite S/M/L values between 0 and 1")
    return {name: float(probabilities[name]) for name in CLASSES}


def normalize_settings(settings):
    """Expand an old shared-percentage request without modifying the caller."""
    settings = dict(settings)
    if "probability" in settings:
        if any(field in settings for field in PROBABILITY_FIELDS):
            raise ValueError("Use either one shared probability or three class probabilities")
        probability = settings.pop("probability")
        settings.update(dict.fromkeys(PROBABILITY_FIELDS, probability))
    return settings


def customer_type_probabilities(value):
    """Validate trinomial probabilities; unlike On probabilities, these sum to one."""
    if not isinstance(value, Mapping) or set(value) != set(CLASSES) or any(
        not isinstance(p, Real) or not isfinite(p) or not 0 <= p <= 1
        for p in value.values()
    ):
        raise ValueError("Customer type probabilities must be finite S/M/L values between 0 and 1")
    total = fsum(value.values())
    if not isclose(total, 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Customer type probabilities must sum to 100%")
    return {name: float(value[name]) / total for name in CLASSES}

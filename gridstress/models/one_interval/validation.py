"""Input bounds for the active single-interval exact model."""

from gridstress.config import MAX_CUSTOMERS
from gridstress.probabilities import PROBABILITY_FIELDS


def validate_settings(settings):
    ranges = {
        'small': (0, 500), 'medium': (0, 500), 'large': (0, 500),
        'customers': (0, MAX_CUSTOMERS), 'threshold': (0, 50_000),
        **dict.fromkeys(PROBABILITY_FIELDS, (0, 100)),
    }
    if set(settings) != set(ranges):
        raise ValueError('Expected currents, customer count, three On probabilities and threshold')
    for name, (minimum, maximum) in ranges.items():
        value = settings[name]
        allowed_types = (int, float) if name in PROBABILITY_FIELDS else (int,)
        if type(value) not in allowed_types or not minimum <= value <= maximum:
            kind = 'a number' if name in PROBABILITY_FIELDS else 'an integer'
            raise ValueError(f'{name} must be {kind} from {minimum} to {maximum}')

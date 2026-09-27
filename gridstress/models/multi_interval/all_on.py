"""Exact optimal allocation and one random assignment per All-On composition."""

from math import isfinite, fsum
import numpy as np

from gridstress.config import LOAD_CURRENT, NUM_CUSTOMERS, THRESHOLD, SEED, compositions
from gridstress.results import composition_weights
from gridstress.models.multi_interval.allocation import allocation_tables, recover_allocation


def summarize(rows, type_probabilities):
    """Keep small safe/rescued masses instead of subtracting near-one risks."""
    weights = composition_weights(rows, type_probabilities)
    total = fsum(weights)
    masses = {category: fsum(w for r, w in zip(rows, weights) if r['category'] == category)/total
              for category in ('already_safe', 'rescued', 'still_violating')}
    result = dict(single_violation_probability=fsum((masses['rescued'], masses['still_violating'])),
                optimal_violation_probability=masses['still_violating'],
                single_safe_probability=masses['already_safe'],
                optimal_safe_probability=fsum((masses['already_safe'], masses['rescued'])),
                reduction=masses['rescued'], probability_mass=masses)
    result['random_points_reversed'] = sum(r['single_violation'] and not r['random_violation'] for r in rows)
    result['random_violating_compositions'] = sum(r['random_violation'] for r in rows)
    if rows[0]['intervals'] == 2:
        result['two_violation_probability'] = result['optimal_violation_probability']
        result['two_safe_probability'] = result['optimal_safe_probability']
    return result


def evaluate(total_customers=NUM_CUSTOMERS, currents=None, threshold=THRESHOLD, intervals=2, random_seed=SEED):
    currents = dict(LOAD_CURRENT if currents is None else currents)
    if type(total_customers) is not int or total_customers < 0:
        raise ValueError("Customer count must be a nonnegative integer")
    if set(currents) != set("SML") or any(not isfinite(v) or v < 0 for v in currents.values()):
        raise ValueError("Provide nonnegative finite S/M/L currents")
    if currents['S'] > min(currents['M'], currents['L']):
        raise ValueError("Small current must not exceed medium or large current")
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be nonnegative and finite")
    if type(intervals) is not int or intervals not in (1, 2, 3):
        raise ValueError('Choose one, two or three intervals')
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError('Random seed must be a nonnegative integer')
    rng = np.random.default_rng(random_seed)
    peaks, choices = allocation_tables(total_customers, currents, intervals)
    s, m, l = (currents[j] for j in 'SML')
    rows = []
    for ns, nm, nl in compositions(total_customers):
        shared = s*ns + m*nm + l*nl
        medium, large = recover_allocation(nm, nl, intervals, choices)
        interval_currents = [m*mt+l*lt+s*(total_customers-mt-lt) for mt, lt in zip(medium, large)]
        random_medium = np.bincount(rng.integers(intervals, size=nm), minlength=intervals).tolist()
        random_large = np.bincount(rng.integers(intervals, size=nl), minlength=intervals).tolist()
        random_currents = [m*mt+l*lt+s*(total_customers-mt-lt)
                           for mt, lt in zip(random_medium, random_large)]
        peak = float(peaks[nm, nl])
        single_violation, optimal_violation = shared > threshold, peak > threshold
        category = ('still_violating' if optimal_violation else
                    'rescued' if single_violation else 'already_safe')
        row = dict(S=ns, M=nm, L=nl, intervals=intervals,
                   medium_allocation=medium, large_allocation=large, interval_currents=interval_currents,
                   shared_current=shared, optimal_peak=peak, single_violation=single_violation,
                   optimal_violation=optimal_violation, category=category,
                   random_medium_allocation=random_medium, random_large_allocation=random_large,
                   random_interval_currents=random_currents, random_peak=max(random_currents),
                   random_violation=int(max(random_currents) > threshold))
        for t, (mt, lt, at) in enumerate(zip(medium, large, interval_currents), 1):
            row.update({f'medium_interval_{t}': mt, f'large_interval_{t}': lt, f'current_interval_{t}': at})
        if intervals == 2:
            row['two_violation'] = optimal_violation
        rows.append(row)
    return rows

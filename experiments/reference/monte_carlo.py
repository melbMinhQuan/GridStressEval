"""Reference five-seed experiment; explicitly running this replaces published data."""

import json
from datetime import datetime
from time import perf_counter

from gridstress.models.one_interval.reference.monte_carlo import evaluate_all_combinations
from gridstress.config import (
    LOAD_CURRENT as load_current, THRESHOLD as threshold,
    NUM_CUSTOMERS as num_customers, P_ON as p_on,
    EXPERIMENTS_DIRECTORY, RESULTS_PATH, TYPE_PROBABILITIES,
)
from gridstress.plotting.probabilities import plot_results
from gridstress.results import overall_probability
from gridstress.probabilities import class_probabilities


def main():
    directory = EXPERIMENTS_DIRECTORY / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    seeds = [42, 43, 44, 45, 46]
    samples = 100_000
    runs = []
    pooled = None
    for seed in seeds:
        start = perf_counter()
        rows = evaluate_all_combinations(num_customers, load_current, threshold,
                                         p_on, samples, seed)
        seconds = perf_counter() - start
        inputs = dict(load_current=load_current, threshold=threshold,
                      num_customers=num_customers, p_on=class_probabilities(p_on),
                      p_type=dict(TYPE_PROBABILITIES),
                      iterations=samples, random_seed=seed)
        payload = dict(inputs=inputs, results=rows)
        (directory / f'seed_{seed}.json').write_text(json.dumps(payload, indent=2))
        run = dict(seed=seed, seconds=seconds,
                   overall_probability=overall_probability(rows, TYPE_PROBABILITIES)['violation_probability'])
        runs.append(run)
        print(json.dumps(run), flush=True)
        if pooled is None:
            pooled = [dict(r) for r in rows]
        else:
            for target, row in zip(pooled, rows):
                target['iterations'] += row['iterations']
                target['violation_count'] += row['violation_count']
    for row in pooled:
        row['violation_probability'] = row['violation_count'] / row['iterations']
    payload = dict(inputs=dict(inputs, iterations=samples*len(seeds), random_seed=None,
                               random_seeds=seeds, samples_per_seed=samples),
                   results=pooled,
                   overall=overall_probability(pooled, TYPE_PROBABILITIES))
    summary = dict(runs=runs, simulation_seconds=sum(r['seconds'] for r in runs),
                   overall=payload['overall'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    (directory / 'pooled_results.json').write_text(json.dumps(payload, indent=2))
    previous = RESULTS_PATH
    if previous.exists():
        (directory / 'previous_results.json').write_bytes(previous.read_bytes())
    previous.write_text(json.dumps(payload, indent=2))
    for path in plot_results(payload):
        print(path, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Experiment saved to {directory}', flush=True)


if __name__ == '__main__':
    main()

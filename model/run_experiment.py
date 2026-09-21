"""Run five independent seeds and publish pooled results for the viewer."""

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from model_full import evaluate_all_combinations, load_current, threshold, num_customers, p_on
import plot_full


def main():
    root = Path(__file__).resolve().parent
    directory = root / 'experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True)
    seeds = [42, 43, 44, 45, 46]
    samples = 100_000
    runs = []
    pooled = None
    for seed in seeds:
        start = perf_counter()
        rows = evaluate_all_combinations(samples=samples, seed=seed)
        seconds = perf_counter() - start
        inputs = dict(load_current=load_current, threshold=threshold,
                      num_customers=num_customers, p_on=p_on,
                      iterations=samples, random_seed=seed)
        payload = dict(inputs=inputs, results=rows)
        (directory / f'seed_{seed}.json').write_text(json.dumps(payload, indent=2))
        run = dict(seed=seed, seconds=seconds,
                   overall_probability=sum(r['violation_count'] for r in rows) / (len(rows)*samples))
        runs.append(run)
        print(json.dumps(run), flush=True)
        if pooled is None:
            pooled = [dict(r) for r in rows]
        else:
            for target, row in zip(pooled, rows):
                assert all(target[k] == row[k] for k in ('S', 'M', 'L'))
                target['iterations'] += row['iterations']
                target['violation_count'] += row['violation_count']
    for row in pooled:
        row['violation_probability'] = row['violation_count'] / row['iterations']
    total = sum(r['iterations'] for r in pooled)
    violations = sum(r['violation_count'] for r in pooled)
    payload = dict(inputs=dict(inputs, iterations=samples*len(seeds), random_seed=None,
                               random_seeds=seeds, samples_per_seed=samples),
                   results=pooled,
                   overall=dict(violation_count=violations, iterations=total,
                                violation_probability=violations/total,
                                composition_weighting='uniform'))
    summary = dict(runs=runs, simulation_seconds=sum(r['seconds'] for r in runs),
                   overall=payload['overall'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    (directory / 'pooled_results.json').write_text(json.dumps(payload, indent=2))
    previous = root / 'results.json'
    if previous.exists():
        (directory / 'previous_results.json').write_bytes(previous.read_bytes())
    previous.write_text(json.dumps(payload, indent=2))
    for path in plot_full.plot_results(payload):
        print(path, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Experiment saved to {directory}', flush=True)


if __name__ == '__main__':
    main()

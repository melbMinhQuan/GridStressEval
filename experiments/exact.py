"""Publish exact results and figures, preserving the previous results file."""

from datetime import datetime
import json
from time import perf_counter

from gridstress.config import RESULTS_PATH, EXPERIMENTS_DIRECTORY
from gridstress.models.one_interval.exact import evaluate, exact_settings, exact_inputs
from gridstress.results import load_results, overall_probability
from gridstress.plotting.probabilities import plot_results


def main():
    previous = load_results()
    settings = exact_settings(previous)
    start = perf_counter()
    rows = evaluate(settings)
    elapsed = perf_counter() - start
    inputs = exact_inputs(settings)
    inputs.update({key: previous['inputs'][key] for key in ('p_type', 'p_type_source')
                   if key in previous['inputs']})
    payload = dict(inputs=inputs, results=rows,
                   overall=overall_probability(rows, inputs['p_type']))
    backup = EXPERIMENTS_DIRECTORY / ('exact_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    (backup / 'previous_results.json').write_bytes(RESULTS_PATH.read_bytes())
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot_results(payload)
    print(f'{len(rows)} compositions calculated in {elapsed:.4f} s')
    print(f'Overall probability: {payload["overall"]["violation_probability"]:.12%}')
    print(f'Previous results preserved: {backup}')


if __name__ == '__main__':
    main()

"""Publish All-On optimal/random allocation results for one to three intervals."""

from collections import Counter
import argparse
import json

import matplotlib.pyplot as plt

from gridstress.config import ROOT, FIGURES_DIRECTORY, LOAD_CURRENT, NUM_CUSTOMERS, THRESHOLD, TYPE_PROBABILITIES
from gridstress.models.multi_interval.all_on import evaluate, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    intervals = args.intervals
    rows = evaluate(intervals=intervals, random_seed=args.random_seed)
    categories = ('already_safe', 'rescued', 'still_violating')
    counts = Counter(row['category'] for row in rows)
    summary = dict(summarize(rows, TYPE_PROBABILITIES), compositions=len(rows), counts=dict(counts))
    payload = dict(inputs=dict(method='exact_all_on_allocation', intervals=intervals,
                               random_allocation='single_independent_uniform_assignment',
                               random_seed=args.random_seed, num_customers=NUM_CUSTOMERS,
                               load_current=LOAD_CURRENT, threshold=THRESHOLD,
                               p_type=TYPE_PROBABILITIES, p_type_source='illustrative'),
                   summary=summary, results=rows)
    suffix = '' if intervals == 2 else f'_{intervals}_intervals'
    path = ROOT / f'data/multi_interval_results{suffix}.json'
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    directory = FIGURES_DIRECTORY / 'multi_interval'
    directory.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for category, label, colour in zip(categories, ('Already safe', 'Made safe by allocation', 'Still violating'),
                                      ('#3465a4', '#26966f', '#cf4446')):
        selected = [r for r in rows if r['category'] == category]
        ax.scatter([r['S'] for r in selected], [r['M'] for r in selected], [r['L'] for r in selected],
                   c=colour, s=16, depthshade=False, label=f'{label} ({len(selected)})')
    ax.set_xlabel('Small customers (S)', labelpad=10)
    ax.set_ylabel('Medium customers (M)', labelpad=10)
    ax.text2D(-.06, .5, 'Large customers (L)', transform=ax.transAxes, rotation=90, va='center')
    ax.view_init(elev=25, azim=45)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.10), frameon=False)
    fig.savefig(directory / f'allocation_3d{suffix}.png', dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

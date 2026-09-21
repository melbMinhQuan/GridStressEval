"""Regression checks for composition counts, threshold semantics and saved data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gridstress.config import ROOT, compositions
from gridstress.models.all_on import evaluate_all_combinations as all_on
from gridstress.models.monte_carlo import evaluate_all_combinations as monte_carlo
from gridstress.models import interactive
from gridstress.results import load_results, overall_probability, viewer_settings


class ModelTests(unittest.TestCase):
    def test_all_on_count(self):
        rows = all_on()
        self.assertEqual(len(rows), 861)
        self.assertEqual(sum(r['violation'] for r in rows), 840)
        self.assertEqual(len(set(compositions(40))), 861)

    def test_deterministic_endpoints_and_equality(self):
        currents = dict(S=20, M=40, L=100)
        rows = monte_carlo(total_customers=2, currents=currents, limit=100,
                           probability_on=1, samples=10)
        for row in rows:
            expected = sum(currents[k] * row[k] for k in currents) > 100
            self.assertEqual(row['violation_probability'], float(expected))
        rows = monte_carlo(total_customers=2, probability_on=0, samples=10)
        self.assertTrue(all(r['violation_count'] == 0 for r in rows))
        self.assertEqual(len(monte_carlo(total_customers=0, samples=10)), 1)

    def test_batch_reproduces_archived_run(self):
        archived = ROOT / 'data/experiments/20260921_223118/previous_results.json'
        saved = json.loads(archived.read_text())
        inputs = saved['inputs']
        rows = monte_carlo(inputs['num_customers'], inputs['load_current'],
                           inputs['threshold'], inputs['p_on'],
                           inputs['iterations'], inputs['random_seed'])
        self.assertEqual(rows, saved['results'])

    def test_published_pool_matches_archived_seeds(self):
        saved = load_results()
        archive = ROOT / 'data/experiments/20260921_223118'
        runs = [json.loads((archive / f'seed_{seed}.json').read_text())['results']
                for seed in saved['inputs']['random_seeds']]
        for index, row in enumerate(saved['results']):
            self.assertEqual(row['violation_count'], sum(r[index]['violation_count'] for r in runs))
            self.assertEqual(row['iterations'], 500000)
        self.assertEqual(overall_probability(saved['results']), saved['overall'])
        self.assertEqual(viewer_settings(saved)['samples'], 100000)

    def test_interactive_matches_all_on_at_probability_one(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(interactive, 'CACHE_DIRECTORY', Path(directory)):
                interactive.active_counts.cache_clear()
                interactive._evaluate.cache_clear()
                settings = dict(small=20, medium=40, large=100, customers=2,
                                probability=100, threshold=100, samples=1000)
                rows = interactive.evaluate(settings)
                for row in rows:
                    expected = 20*row['S'] + 40*row['M'] + 100*row['L'] > 100
                    self.assertEqual(row['violation_probability'], float(expected))
                interactive.active_counts.cache_clear()
                interactive._evaluate.cache_clear()

    def test_invalid_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            monte_carlo(samples=0)
        with self.assertRaises(ValueError):
            interactive.evaluate(dict(small=20, medium=40, large=100, customers=101,
                                      probability=50, threshold=1000, samples=1000))


if __name__ == '__main__':
    unittest.main()

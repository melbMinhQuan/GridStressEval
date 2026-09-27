"""Class probabilities: analytical checks, cache separation and compatibility."""

import copy
from math import comb
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gridstress.models.one_interval.reference import interactive
from gridstress.models.one_interval.reference.monte_carlo import evaluate_all_combinations
from gridstress.probabilities import class_probabilities, normalize_settings
from gridstress.results import load_results, viewer_settings, simulation_inputs


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = patch.object(interactive, 'CACHE_DIRECTORY', Path(self.directory.name))
        self.cache.start()
        interactive.active_counts.cache_clear()
        interactive._evaluate.cache_clear()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.cache.stop)
        self.addCleanup(interactive.active_counts.cache_clear)
        self.addCleanup(interactive._evaluate.cache_clear)

    def test_scalar_and_equal_mapping_are_identical(self):
        shared = evaluate_all_combinations(total_customers=4, probability_on=0.5, samples=1000)
        separate = evaluate_all_combinations(total_customers=4,
                    probability_on=dict(S=0.5, M=0.5, L=0.5), samples=1000)
        self.assertEqual(shared, separate)
        legacy = dict(small=20, medium=40, large=100, customers=4,
                      probability=50, threshold=100, samples=1000)
        self.assertEqual(interactive.evaluate(legacy), interactive.evaluate(normalize_settings(legacy)))

    def test_each_class_can_be_on_alone(self):
        for selected in ('S', 'M', 'L'):
            with self.subTest(selected=selected):
                probabilities = {name: int(name == selected) for name in ('S', 'M', 'L')}
                settings = dict(small=20, medium=40, large=100, customers=3,
                                threshold=100, samples=1000,
                                probability_small=100*probabilities['S'],
                                probability_medium=100*probabilities['M'],
                                probability_large=100*probabilities['L'])
                batch = evaluate_all_combinations(total_customers=3, limit=100,
                                                  probability_on=probabilities, samples=1000)
                self.assertEqual(batch, interactive.evaluate(settings))
                for row in batch:
                    expected = row[selected] * {'S': 20, 'M': 40, 'L': 100}[selected] > 100
                    self.assertEqual(row['violation_probability'], float(expected))

    def test_unequal_probabilities_match_exact_small_model(self):
        probabilities = dict(S=0.2, M=0.5, L=0.8)
        settings = dict(small=20, medium=40, large=100, customers=3,
                        threshold=100, samples=100000,
                        probability_small=20, probability_medium=50, probability_large=80)
        for rows in (evaluate_all_combinations(total_customers=3, limit=100,
                       probability_on=probabilities, samples=100000), interactive.evaluate(settings)):
            for row in rows:
                expected = 0
                for s in range(row['S'] + 1):
                    for m in range(row['M'] + 1):
                        for l in range(row['L'] + 1):
                            if 20*s + 40*m + 100*l <= 100:
                                continue
                            mass = 1
                            for name, active in zip(('S', 'M', 'L'), (s, m, l)):
                                n, p = row[name], probabilities[name]
                                mass *= comb(n, active) * p**active * (1-p)**(n-active)
                            expected += mass
                self.assertAlmostEqual(row['violation_probability'], expected, delta=0.008)

    def test_cache_keys_cover_all_three_probabilities_and_disk_reload(self):
        first = interactive.active_counts(3, 1000, 20, 50, 80)
        for group, probabilities in enumerate(((30, 50, 80), (20, 60, 80), (20, 50, 90))):
            changed = interactive.active_counts(3, 1000, *probabilities)
            self.assertFalse(np.array_equal(first[group], changed[group]))
            for other in set(range(3)) - {group}:
                np.testing.assert_array_equal(first[other], changed[other])
        interactive.active_counts.cache_clear()
        np.testing.assert_array_equal(changed, interactive.active_counts(3, 1000, 20, 50, 90))

    def test_saved_metadata_and_export_roundtrip(self):
        saved = load_results()
        defaults = viewer_settings(saved)
        self.assertEqual([defaults[k] for k in ('probability_small', 'probability_medium',
                                               'probability_large')], [50, 50, 50])
        modified = copy.deepcopy(saved)
        modified['inputs']['p_on'] = dict(S=0.2, M=0.5, L=0.8)
        settings = viewer_settings(modified)
        from gridstress.models.one_interval.exact import exact_inputs
        exported = exact_inputs(settings) if 'samples' not in settings else simulation_inputs(settings)
        self.assertEqual(exported['p_on'], modified['inputs']['p_on'])
        self.assertEqual(viewer_settings({'inputs': exported}), settings)

    def test_invalid_class_probabilities(self):
        for value in (dict(S=0.5, M=0.5), dict(S=0.5, M=0.5, L=2),
                      dict(S=float('nan'), M=0.5, L=0.5)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                class_probabilities(value)
        settings = dict(small=20, medium=40, large=100, customers=3, threshold=100,
                        samples=1000, probability_small=20, probability_medium=50,
                        probability_large=101)
        with self.assertRaises(ValueError):
            interactive.evaluate(settings)
        with self.assertRaises(ValueError):
            normalize_settings(dict(settings, probability=50))


if __name__ == '__main__':
    unittest.main()

from itertools import product
import unittest

from gridstress.models.multi_interval.all_on import evaluate, summarize


class MultiIntervalTests(unittest.TestCase):
    def test_one_to_three_against_every_individual_assignment(self):
        for currents in (dict(S=20, M=40, L=100), dict(S=0, M=0, L=40), dict(S=20, M=20, L=20)):
            for k in (1, 2, 3):
                for row in evaluate(4, currents, 100, intervals=k):
                    people = [currents['M']]*row['M']+[currents['L']]*row['L']
                    peaks = []
                    for assignment in product(range(k), repeat=len(people)):
                        loads = [row['S']*currents['S']+sum(c if t == slot else currents['S']
                                  for c, t in zip(people, assignment)) for slot in range(k)]
                        peaks.append(max(loads))
                    self.assertEqual(row['optimal_peak'], min(peaks))
                    self.assertIn(row['random_violation'], (0, 1))
                    self.assertEqual(row['random_violation'], int(max(row['random_interval_currents']) > 100))
                    self.assertEqual(sum(row['random_medium_allocation']), row['M'])
                    self.assertEqual(sum(row['random_large_allocation']), row['L'])
                    self.assertEqual(len(row['random_interval_currents']), k)
                    for m, l, a in zip(row['random_medium_allocation'], row['random_large_allocation'], row['random_interval_currents']):
                        self.assertEqual(a, m*currents['M']+l*currents['L']+(4-m-l)*currents['S'])
                    self.assertEqual(sum(row['medium_allocation']), row['M'])
                    self.assertEqual(sum(row['large_allocation']), row['L'])
                    self.assertEqual(len(row['interval_currents']), k)
                    self.assertEqual(max(row['interval_currents']), row['optimal_peak'])
                    self.assertEqual(sum(row['interval_currents']), row['shared_current']+(k-1)*4*currents['S'])

    def test_more_intervals_and_random_dominance(self):
        results = [evaluate(intervals=k) for k in (1, 2, 3)]
        for one, two, three in zip(*results):
            self.assertGreaterEqual(one['optimal_peak'], two['optimal_peak'])
            self.assertGreaterEqual(two['optimal_peak'], three['optimal_peak'])
            for row in (one, two, three):
                self.assertLessEqual(int(row['optimal_violation']), row['random_violation'])
                self.assertLessEqual(row['optimal_peak'], row['random_peak'])
        self.assertEqual(sum(r['category']=='rescued' for r in results[2]), 112)
        for k in (0, 4, 2.5, True):
            with self.assertRaises(ValueError):
                evaluate(intervals=k)

    def test_random_redraw_reproducibility(self):
        first = evaluate(intervals=3, random_seed=42)
        self.assertEqual(first, evaluate(intervals=3, random_seed=42))
        second = evaluate(intervals=3, random_seed=43)
        self.assertNotEqual([r['random_medium_allocation'] for r in first],
                            [r['random_medium_allocation'] for r in second])
        self.assertEqual([r['optimal_peak'] for r in first], [r['optimal_peak'] for r in second])
        for row in evaluate(intervals=1):
            self.assertEqual(row['random_violation'], int(row['optimal_violation']))
            self.assertEqual(row['random_interval_currents'], row['interval_currents'])

    def test_tiny_rescued_probability_preserved(self):
        rows = evaluate()
        result = summarize(rows, dict(S=.1, M=.3, L=.6))
        self.assertGreater(result['reduction'], 0)
        self.assertLess(result['reduction'], 1e-15)
        self.assertGreater(result['two_safe_probability'], result['single_safe_probability'])
        self.assertAlmostEqual(sum(result['probability_mass'].values()), 1)

    def test_exhaustive_individual_assignment(self):
        for row in evaluate(total_customers=5, threshold=140):
            currents = [40]*row['M'] + [100]*row['L']
            peaks = []
            for allocation in product((0, 1), repeat=len(currents)):
                a1 = 20*row['S'] + sum(c if t == 0 else 20 for c, t in zip(currents, allocation))
                a2 = 20*row['S'] + sum(c if t == 1 else 20 for c, t in zip(currents, allocation))
                peaks.append(max(a1, a2))
            self.assertEqual(row['optimal_peak'], min(peaks))
            self.assertLessEqual(row['optimal_peak'], row['shared_current'])
            self.assertEqual(row['current_interval_1']+row['current_interval_2'], row['shared_current']+5*20)

    def test_current_parameters_and_example(self):
        rows = evaluate()
        self.assertEqual(len(rows), 861)
        row = next(r for r in rows if (r['S'], r['M'], r['L']) == (34, 4, 2))
        self.assertEqual(row['shared_current'], 1040)
        self.assertEqual(row['optimal_peak'], 920)
        self.assertEqual(row['category'], 'rescued')
        self.assertTrue(all(not r['two_violation'] or r['single_violation'] for r in rows))

    def test_equality_empty_and_all_small(self):
        self.assertEqual(evaluate(total_customers=0)[0]['optimal_peak'], 0)
        row = evaluate(total_customers=2, currents=dict(S=20, M=40, L=100), threshold=120)[0]
        self.assertEqual(row['optimal_peak'], 120)
        self.assertFalse(row['two_violation'])
        self.assertEqual(evaluate()[-1]['optimal_peak'], 800)

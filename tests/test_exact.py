"""Compare exact count aggregation to independent individual-state enumeration."""

from itertools import product
from math import prod
import unittest

from gridstress.models.one_interval.exact import evaluate, exact_inputs, exact_settings
from gridstress.results import overall_probability


class ExactTests(unittest.TestCase):
    def settings(self, **updates):
        return dict(dict(small=20, medium=40, large=100, customers=4,
                         probability_small=23.4, probability_medium=60,
                         probability_large=85, threshold=100), **updates)

    def test_against_individual_patterns(self):
        for currents in [(20, 40, 100), (0, 40, 0), (100, 20, 40)]:
            settings = self.settings(**dict(zip(('small', 'medium', 'large'), currents)))
            for row in evaluate(settings):
                people = [(current, q) for j, current, q in
                          zip('SML', currents, (.234, .6, .85)) for _ in range(row[j])]
                expected = sum(prod(q if on else 1-q for on, (_, q) in zip(states, people))
                               for states in product((0, 1), repeat=4)
                               if sum(on*c for on, (c, _) in zip(states, people)) > 100)
                self.assertAlmostEqual(row['violation_probability'], expected, places=13)

    def test_endpoints_equality_empty(self):
        for q in (0, 100):
            for row in evaluate(self.settings(probability_small=q, probability_medium=q,
                                              probability_large=q)):
                expected = float(q == 100 and 20*row['S']+40*row['M']+100*row['L'] > 100)
                self.assertEqual(row['violation_probability'], expected)
        self.assertEqual(evaluate(self.settings(customers=0))[0]['violation_probability'], 0)
        self.assertEqual(evaluate(self.settings(customers=1, probability_large=100))[0]['violation_probability'], 0)

    def test_metadata_and_no_samples(self):
        settings = self.settings()
        self.assertEqual(exact_settings({'inputs': exact_inputs(settings)}), settings)
        with self.assertRaises(ValueError):
            evaluate(dict(settings, samples=1000))

    def test_weighted_against_population_enumeration(self):
        settings = self.settings(customers=3)
        types = dict(S=.6, M=.3, L=.1)
        states = [(0, .6*.766+.3*.4+.1*.15), (20, .6*.234), (40, .3*.6), (100, .1*.85)]
        expected = sum(prod(p for _, p in outcome) for outcome in product(states, repeat=3)
                       if sum(c for c, _ in outcome) > 100)
        actual = overall_probability(evaluate(settings), types)['violation_probability']
        self.assertAlmostEqual(actual, expected, places=13)

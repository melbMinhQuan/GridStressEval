"""Population risk must use multinomial masses, not uniform composition weights."""

from itertools import product
import math
import unittest

from gridstress.config import compositions
from gridstress.probabilities import customer_type_probabilities
from gridstress.results import composition_weights, overall_probability, split_viewer_settings


class TrinomialTests(unittest.TestCase):
    def rows(self, n):
        return [dict(S=s, M=m, L=l, violation_probability=0.0) for s, m, l in compositions(n)]

    def test_weights_sum_to_one_including_zero_counts_and_extreme_mix(self):
        for n in (0, 1, 40, 100):
            for mix in (dict(S=1/3, M=1/3, L=1/3), dict(S=.8, M=.15, L=.05), dict(S=0, M=0, L=1)):
                self.assertAlmostEqual(sum(composition_weights(self.rows(n), mix)), 1, places=10)

    def test_equal_type_probabilities_are_not_uniform_compositions(self):
        rows = self.rows(3)
        weights = composition_weights(rows, dict(S=1/3, M=1/3, L=1/3))
        lookup = {(r['S'], r['M'], r['L']):w for r,w in zip(rows,weights)}
        self.assertAlmostEqual(lookup[(1,1,1)], 6/27)
        self.assertAlmostEqual(lookup[(3,0,0)], 1/27)

    def test_degenerate_mix_selects_one_composition(self):
        rows = self.rows(4)
        for row in rows:
            row['violation_probability'] = .8 if row['L'] == 4 else .1
        self.assertAlmostEqual(overall_probability(rows, dict(S=0, M=0, L=1))['violation_probability'], .8)

    def test_trinomial_weighting_matches_direct_individual_outcome_enumeration(self):
        n, threshold = 3, 100
        mix = dict(S=.6, M=.3, L=.1)
        on = dict(S=.2, M=.5, L=.8)
        currents = dict(S=20, M=40, L=100)
        rows = self.rows(n)
        for row in rows:
            probability = 0
            for s,m,l in product(range(row['S']+1),range(row['M']+1),range(row['L']+1)):
                if 20*s+40*m+100*l > threshold:
                    probability += math.prod(math.comb(row[k],active) * on[k]**active *
                                             (1-on[k])**(row[k]-active)
                                             for k,active in zip(('S','M','L'),(s,m,l)))
            row['violation_probability'] = probability
        outcomes = [(current if active else 0, mix[k]*(on[k] if active else 1-on[k]))
                    for k,current in currents.items() for active in (0,1)]
        direct = sum(math.prod(x[1] for x in trial) for trial in product(outcomes, repeat=n)
                     if sum(x[0] for x in trial)>threshold)
        self.assertAlmostEqual(overall_probability(rows,mix)['violation_probability'], direct)

    def test_all_on_weights_and_probability_endpoints(self):
        rows = [dict(S=s,M=m,L=l,violation=(100*l>100)) for s,m,l in compositions(2)]
        self.assertAlmostEqual(overall_probability(rows,dict(S=.6,M=.3,L=.1))['violation_probability'], .01)
        for row in rows:
            row['violation'] = True
        self.assertAlmostEqual(overall_probability(rows,dict(S=.6,M=.3,L=.1))['violation_probability'], 1)

    def test_weights_do_not_depend_on_how_many_trials_each_composition_received(self):
        rows = self.rows(1)
        for row in rows:
            row['violation_probability'] = float(row['L'] == 1)
            row['iterations'] = 100 if row['L'] else 10000
        self.assertAlmostEqual(overall_probability(rows,dict(S=.6,M=.3,L=.1))['violation_probability'], .1)

    def test_invalid_mix_and_separation_from_simulation_settings(self):
        for p in (dict(S=.5,M=.5,L=.5),dict(S=-.1,M=.6,L=.5),dict(S=0,M=0,L=0)):
            with self.assertRaises(ValueError): customer_type_probabilities(p)
        settings, mix = split_viewer_settings(dict(customers=40,type_small=60,type_medium=30,type_large=10))
        self.assertEqual(settings,dict(customers=40))
        self.assertEqual(mix,dict(S=.6,M=.3,L=.1))


if __name__ == '__main__':
    unittest.main()

"""Parameter-to-plot regression tests for both Streamlit viewers."""

import json
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from gridstress.models.one_interval.exact import evaluate as exact
from gridstress.models.multi_interval.all_on import evaluate as allocation
from gridstress.results import overall_probability


class VisualizationInputTests(unittest.TestCase):
    def run_app(self, path):
        return AppTest.from_file(path, default_timeout=20).run()

    def submit(self, app, values):
        for key, value in values.items():
            app.number_input(key=key).set_value(value)
        next(b for b in app.button if b.label == 'Update visualization').click().run()
        self.assertFalse(app.exception)
        return app

    def test_single_interval_every_input_and_export(self):
        app = self.run_app('apps/streamlit_app.py')
        base = dict(small=20, medium=40, large=100, customers=4, threshold=100,
                    probability_small=20., probability_medium=50., probability_large=80.)
        changes = dict(small=30, medium=60, large=120, customers=6, threshold=120,
                       probability_small=23.4, probability_medium=0., probability_large=100.)
        for key, value in changes.items():
            with self.subTest(key=key):
                settings = dict(base, **{key: value})
                self.submit(app, settings)
                self.assertFalse(app.error)
                data = json.loads(app.get('plotly_chart')[0].proto.spec)['data'][0]
                rows = exact(settings)
                self.assertEqual(list(zip(data['x'], data['y'], data['z'])),
                                 [(r['S'], r['M'], r['L']) for r in rows])
                self.assertEqual(data['marker']['color'], [r['violation_probability'] for r in rows])
        old_colours = data['marker']['color']
        for mix in ((33.3, 33.4, 33.3), (100., 0., 0.), (0., 100., 0.), (0., 0., 100.)):
            values = dict(zip(('type_small', 'type_medium', 'type_large'), mix))
            with patch.object(st, 'download_button', wraps=st.download_button) as download:
                self.submit(app, values)
                payload = json.loads(download.call_args.args[1])
            self.assertFalse(app.error)
            for j, value in zip('SML', mix):
                self.assertAlmostEqual(payload['inputs']['p_type'][j], value/100)
            self.assertEqual(payload['results'], rows)
            self.assertEqual(payload['overall'], overall_probability(rows, payload['inputs']['p_type']))
            self.assertEqual(json.loads(app.get('plotly_chart')[0].proto.spec)['data'][0]['marker']['color'], old_colours)

    def test_multi_interval_every_input_and_export(self):
        app = self.run_app('apps/multi_interval_app.py')
        base = dict(small=20, medium=40, large=100, customers=4, threshold=100)
        for k in (2, 3):
            app.radio[0].set_value(k).run()
            for key, value in dict(small=10, medium=60, large=120, customers=6, threshold=120).items():
                with self.subTest(k=k, key=key):
                    settings = dict(base, **{key: value})
                    with patch.object(st, 'download_button', wraps=st.download_button) as download:
                        self.submit(app, settings)
                        payload = json.loads(download.call_args.args[1])
                    self.assertFalse(app.error)
                    expected = allocation(settings['customers'],
                                          dict(zip('SML', [settings[n] for n in ('small', 'medium', 'large')])),
                                          settings['threshold'], k, 42)
                    self.assertEqual(payload['results'], expected)
                    self.assertEqual(payload['inputs']['intervals'], k)
                    self.assertEqual(payload['inputs']['random_seed'], 42)
                    for t, chart in enumerate(app.tabs[1].get('plotly_chart')):
                        traces = json.loads(chart.proto.spec)['data']
                        field = 'shared_current' if t == 0 else f'current_interval_{t}'
                        for violates, trace in enumerate(traces):
                            selected = [r for r in expected if (r[field] > settings['threshold']) == bool(violates)]
                            self.assertEqual(trace['customdata'], [[r[field], trace['customdata'][i][1]]
                                                                  for i, r in enumerate(selected)])

    def test_invalid_values_and_recovery(self):
        for path in ('apps/streamlit_app.py', 'apps/multi_interval_app.py'):
            app = self.run_app(path)
            defaults = {n.key: n.value for n in app.number_input}
            invalid = [(key, v) for key in ('small', 'medium', 'large') for v in (-1, 501)]
            invalid += [('customers', -1), ('customers', 101), ('threshold', -1), ('threshold', 50001)]
            invalid += [(key, v) for key in ('type_small', 'type_medium', 'type_large') for v in (-.1, 100.1)]
            invalid += [('type_small', 30.)]  # incorrect total
            if path.endswith('/streamlit_app.py'):
                invalid += [(key, v) for key in ('probability_small', 'probability_medium', 'probability_large')
                            for v in (-.1, 100.1)]
            else:
                invalid += [('small', 101)]  # small cannot exceed medium/large
            for key, value in invalid:
                with self.subTest(path=path, key=key, value=value):
                    self.submit(app, dict(defaults, **{key: value}))
                    self.assertTrue(app.error)
                    self.assertFalse(app.get('plotly_chart'))
            self.submit(app, defaults)
            self.assertFalse(app.error)
            self.assertTrue(app.get('plotly_chart'))

    def test_multi_type_mix_changes_weights_not_assignments(self):
        app = self.run_app('apps/multi_interval_app.py')
        with patch.object(st, 'download_button', wraps=st.download_button) as download:
            self.submit(app, dict(type_small=33.3, type_medium=33.4, type_large=33.3))
            before = json.loads(download.call_args.args[1])
        with patch.object(st, 'download_button', wraps=st.download_button) as download:
            self.submit(app, dict(type_small=100., type_medium=0., type_large=0.))
            after = json.loads(download.call_args.args[1])
        self.assertEqual(before['results'], after['results'])
        self.assertEqual(after['summary']['optimal_violation_probability'], 0)
        self.assertNotEqual(before['summary']['optimal_violation_probability'], 0)
        with patch.object(st, 'download_button', wraps=st.download_button) as download:
            next(b for b in app.button if b.label == 'New random assignment').click().run()
            redrawn = json.loads(download.call_args.args[1])
        self.assertEqual(redrawn['inputs']['random_seed'], 43)
        self.assertEqual([r['optimal_peak'] for r in redrawn['results']],
                         [r['optimal_peak'] for r in after['results']])
        self.assertNotEqual([r['random_medium_allocation'] for r in redrawn['results']],
                            [r['random_medium_allocation'] for r in after['results']])

    def test_valid_boundaries(self):
        for path in ('apps/streamlit_app.py', 'apps/multi_interval_app.py'):
            app = self.run_app(path)
            if path.endswith('/multi_interval_app.py'):
                app.radio[0].set_value(3).run()
            for n, current, threshold in ((0, 0, 0), (100, 500, 50000), (100, 0, 0), (1, 500, 0)):
                settings = dict(customers=n, small=current, medium=current, large=current, threshold=threshold)
                if path.endswith('/streamlit_app.py'):
                    settings.update(probability_small=100., probability_medium=100., probability_large=100.)
                self.submit(app, settings)
                self.assertFalse(app.error)
                self.assertTrue(app.get('plotly_chart'))

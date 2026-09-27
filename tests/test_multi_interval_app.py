"""Regression coverage for interval selection and allocation comparison tabs."""

import unittest

from streamlit.testing.v1 import AppTest


class MultiIntervalAppTests(unittest.TestCase):
    def test_interval_selection_and_comparison(self):
        app = AppTest.from_file('apps/multi_interval_app.py', default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual([t.label for t in app.tabs],
                         ['Allocation comparison', 'Interval graphs', 'Optimal vs random'])
        self.assertEqual(app.radio[0].options, ['2', '3'])
        for k, rescued, graphs in ((2, 44, 3), (3, 112, 4)):
            app.radio[0].set_value(k).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.metric[0].value, str(rescued))
            self.assertEqual(len(app.tabs[1].get('plotly_chart')), graphs)
            self.assertEqual(len(app.tabs[2].get('plotly_chart')), 2)
        app.number_input(key='customers').set_value(0)
        next(b for b in app.button if b.label == 'Update visualization').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.metric[0].value, '0')

    def test_redraw_changes_only_random_assignment(self):
        import json
        app = AppTest.from_file('apps/multi_interval_app.py', default_timeout=20).run()
        charts = app.tabs[2].get('plotly_chart')
        before = [json.loads(c.proto.spec)['data'][0] for c in charts]
        next(b for b in app.button if b.label == 'New random assignment').click().run()
        self.assertFalse(app.exception)
        after = [json.loads(c.proto.spec)['data'][0] for c in app.tabs[2].get('plotly_chart')]
        self.assertEqual(before[0], after[0])
        self.assertTrue(before[1]['text'] != after[1]['text'])
        self.assertTrue(set(after[1]['customdata']) <= {0, 1})

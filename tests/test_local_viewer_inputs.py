"""Exercise the actual local HTTP endpoints on a temporary loopback port."""

import base64
import json
from http.server import ThreadingHTTPServer
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from apps.local_viewer import Handler
from gridstress.models.one_interval.exact import evaluate


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


class LocalViewerInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def get_json(self, path):
        with urlopen(self.url+path, timeout=20) as response:
            return json.load(response)

    def test_config_decimal_probabilities_and_export(self):
        config = self.get_json('/api/config')
        self.assertNotIn('samples', config)
        self.assertAlmostEqual(sum(config[k] for k in ('type_small', 'type_medium', 'type_large')), 100)
        settings = dict(small=15, medium=45, large=90, customers=4, threshold=100,
                        probability_small=23.4, probability_medium=56.7, probability_large=89.1)
        query = dict(settings, type_small=33.3, type_medium=33.4, type_large=33.3)
        payload = self.get_json('/api/evaluate?'+urlencode(query))
        self.assertEqual(payload['results'], evaluate(settings))
        self.assertAlmostEqual(payload['inputs']['p_on']['S'], .234)
        self.assertEqual(payload['inputs']['p_type'], dict(S=.333, M=.334, L=.333))
        self.assertTrue(base64.b64decode(payload['png']).startswith(b'\x89PNG'))
        query.update(type_small=100, type_medium=0, type_large=0)
        changed = self.get_json('/api/evaluate?'+urlencode(query))
        self.assertEqual(payload['results'], changed['results'])
        all_small = next(r for r in changed['results'] if r['S']==4)
        self.assertEqual(changed['overall']['violation_probability'], all_small['violation_probability'])

    def test_invalid_requests_are_400(self):
        config = self.get_json('/api/config')
        for changes in (dict(customers=-1), dict(customers=101), dict(small=501),
                        dict(probability_small='nan'), dict(probability_large=100.1),
                        dict(threshold=''), dict(type_small=''), dict(type_small=0),
                        dict(customers='4.5'), dict(samples=1000)):
            with self.subTest(changes=changes):
                with self.assertRaises(HTTPError) as error:
                    self.get_json('/api/evaluate?'+urlencode(dict(config, **changes)))
                self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.get_json('/api/evaluate?'+urlencode(config)+'&threshold=100')
        self.assertEqual(error.exception.code, 400)

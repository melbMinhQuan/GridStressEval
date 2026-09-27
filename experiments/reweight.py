"""Reweight saved conditional results without repeating the simulation."""

import argparse
from datetime import datetime
import json

from gridstress.config import RESULTS_PATH, EXPERIMENTS_DIRECTORY
from gridstress.probabilities import CLASSES, customer_type_probabilities
from gridstress.results import load_results, overall_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type-probabilities', nargs=3, type=float, required=True,
                        metavar=('S', 'M', 'L'), help='Type probabilities as fractions summing to one')
    parser.add_argument('--illustrative', action='store_true',
                        help='Mark the supplied mix as an illustrative assumption')
    args = parser.parse_args()
    probabilities = customer_type_probabilities(dict(zip(CLASSES, args.type_probabilities)))
    payload = load_results()
    previous = EXPERIMENTS_DIRECTORY / ('reweight_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    previous.mkdir(parents=True)
    (previous / 'previous_results.json').write_bytes(RESULTS_PATH.read_bytes())
    payload['inputs']['p_type'] = probabilities
    payload['inputs']['p_type_source'] = 'illustrative' if args.illustrative else 'user_supplied'
    payload['overall'] = overall_probability(payload['results'], probabilities)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload['overall'], indent=2))
    print(f'Previous data preserved in {previous}')


if __name__ == '__main__':
    main()

"""Select hysteresis thresholds on validation seeds; no training is performed.

Default previews the grid. --execute runs rule evaluation and writes a JSON
that evaluate_matrix.py requires before formal test execution.
"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from paper_experiments import make_env, rollout, threshold_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-densities', nargs='+', type=float, default=[.22, .26, .30])
    parser.add_argument('--close-densities', nargs='+', type=float, default=[.16, .20, .24])
    parser.add_argument('--approach-distances', nargs='+', type=int, default=[60, 80, 100])
    parser.add_argument('--densities', nargs='+', type=int, default=[350, 520, 850])
    parser.add_argument('--validation-seed-start', type=int, default=10000)
    parser.add_argument('--validation-seeds', type=int, default=10)
    parser.add_argument('--out', type=Path, default=Path('paper_plan/threshold_validation.json'))
    parser.add_argument('--execute', action='store_true', help='Run rule validation; never trains')
    args = parser.parse_args()
    if args.validation_seeds <= 0:
        parser.error('validation-seeds must be positive')
    candidates = [(o, c, d) for o, c, d in product(args.open_densities,
                    args.close_densities, args.approach_distances) if 0 <= c < o <= 1 and d >= 0]
    if not candidates:
        parser.error('No valid open/close/approach combinations')
    print(f'{len(candidates)} rule candidates x {len(args.densities)} densities x '
          f'{args.validation_seeds} validation scenes; execute={args.execute}')
    if not args.execute:
        return
    envs = {density: make_env(density, 150, 400, 5) for density in args.densities}
    rows = []
    for open_density, close_density, approach in candidates:
        scores = []
        for density, env in envs.items():
            for seed in range(args.validation_seed_start,
                              args.validation_seed_start + args.validation_seeds):
                outcome = rollout(env, threshold_policy(open_density, close_density, approach),
                                  seed, 500)
                scores.append(outcome['validation_reward'])
        rows.append({'open_density': open_density, 'close_density': close_density,
                     'approach_distance': approach, 'validation_reward': float(np.mean(scores))})
        print(rows[-1], flush=True)
    winner = max(rows, key=lambda item: item['validation_reward'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**winner, 'validation_seed_start': args.validation_seed_start,
                                    'validation_seeds': args.validation_seeds,
                                    'densities': args.densities}, indent=2), encoding='utf-8')
    with args.out.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()

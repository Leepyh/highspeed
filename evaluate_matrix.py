"""Evaluate a completed experiment_matrix manifest on common held-out scenes.

Default is a command preview. --execute runs evaluation only; it never trains.
Every training seed remains a separate result directory for hierarchical analysis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_FLAG = {'maddpg': '--model', 'idqn': '--idqn-model',
              'qmix': '--qmix-model', 'mappo': '--mappo-model'}


def commands(manifest, output_root, test_start, test_count, threshold_config=None):
    jobs = json.loads(Path(manifest).read_text(encoding='utf-8'))
    groups = {}
    for job in jobs:
        groups.setdefault((job['group'], job['seed']), []).append(job)
    for (group, seed), members in sorted(groups.items()):
        config = json.loads(Path(members[0]['config']).read_text(encoding='utf-8'))
        ca, control = config['ca'], config['env']
        densities = ([350, 850] if group == 'train_520' else
                     [520] if group.startswith(('emergency_weight_', 'segments_', 'interval_'))
                     else [350, 520, 850])
        command = [sys.executable, str(HERE / 'paper_experiments.py'),
                   '--densities', *map(str, densities),
                   '--seeds', *map(str, range(test_start, test_start + test_count)),
                   '--warmup', str(ca['warm_up_steps']), '--window', str(ca['max_steps']),
                   '--interval', str(control['decision_interval']),
                   '--segments', str(control['n_segments']),
                   '--observation-density-range', *map(str, ca['n_cars_range']),
                   '--out', str(output_root / group / f'seed_{seed:02d}')]
        for job in members:
            checkpoint = Path(job['output']) / 'best_validation.pth'
            command.extend([MODEL_FLAG[job['algorithm']], str(checkpoint)])
        if threshold_config:
            command.extend(['--threshold-config', str(threshold_config)])
        yield group, seed, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('paper_test_results'))
    parser.add_argument('--test-seed-start', type=int, default=1000)
    parser.add_argument('--test-scenarios', type=int, default=10)
    parser.add_argument('--threshold-config', type=Path,
                        help='Independent validation output from tune_threshold.py')
    parser.add_argument('--execute', action='store_true', help='Run evaluation; no training')
    args = parser.parse_args()
    if args.test_scenarios <= 0:
        parser.error('test-scenarios must be positive')
    if args.execute and (not args.threshold_config or not args.threshold_config.is_file()):
        parser.error('Formal execution requires an existing --threshold-config from validation')
    if args.execute:
        calibration = json.loads(args.threshold_config.read_text(encoding='utf-8'))
        test_seeds = set(range(args.test_seed_start,
                               args.test_seed_start + args.test_scenarios))
        rule_seeds = set(range(calibration['validation_seed_start'],
                               calibration['validation_seed_start'] + calibration['validation_seeds']))
        model_seeds = set(range(10000, 10010))
        if test_seeds & (rule_seeds | model_seeds):
            parser.error('Test seeds overlap validation seeds')
    jobs = list(commands(args.manifest, args.output_root.resolve(),
                         args.test_seed_start, args.test_scenarios, args.threshold_config))
    print(f'{len(jobs)} paired evaluation batches; execute={args.execute}')
    for group, seed, command in jobs:
        print(f'{group} training_seed={seed}: {" ".join(command)}')
        if args.execute:
            subprocess.run(command, check=True, cwd=HERE)


if __name__ == '__main__':
    main()

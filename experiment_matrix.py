"""Prepare the paper experiment matrix. Never trains unless --execute is passed.

Main: four algorithms x ten training seeds on mixed 350/520/850 traffic.
Generalization: four algorithms x ten seeds trained only at 520 cars.
Ablation and sensitivity: MADDPG, independently retrained per configuration.
Validation/test seed ranges are disjoint from training seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config_maddpg import ENV_CONFIG

HERE = Path(__file__).resolve().parent
ALGORITHMS = ('maddpg', 'idqn', 'qmix', 'mappo')


def base_config():
    return {'ca': {'n_cars_range': [350, 850], 'n_cars_choices': [350, 520, 850],
                   'P_emergency_range': [1.0, 1.0], 'max_emergency': 1,
                   'warm_up_steps': 150, 'max_steps': 400,
                   'inject_emergency_after_warmup': True},
            'env': {'n_segments': 5, 'n_agents': 5, 'decision_interval': 5,
                    'shared_team_reward': True}}


def variants(groups):
    results = []
    if 'main' in groups:
        results.append(('main_mixed', base_config(), ALGORITHMS, 10))
    if 'generalization' in groups:
        config = base_config()
        config['ca']['n_cars_choices'] = [520]
        results.append(('train_520', config, ALGORITHMS, 10))
    if 'ablation' in groups:
        weights = ENV_CONFIG['reward_weights']
        keep = {'emergency_speed_weight', 'global_avg_speed', 'congestion_relief_weight',
                'speed_improvement_weight', 'global_congestion_penalty', 'local_speed'}
        ablations = {
            'full': {},
            'no_emergency': {'emergency_speed_weight': 0},
            'no_resource': {'correct_close_reward': 0, 'emergency_lane_usage_penalty': 0},
            'no_switch': {'local_action_smoothness': 0},
            'efficiency_only': {key: 0 for key in weights if key not in keep},
        }
        for name, overrides in ablations.items():
            config = base_config()
            config['env']['reward_weights'] = overrides
            results.append((f'ablation_{name}', config, ('maddpg',), 10))
    if 'emergency_weight' in groups:
        base = ENV_CONFIG['reward_weights']['emergency_speed_weight']
        for weight in (.1, .3, .5, .7, .9):
            config = base_config()
            config['ca']['n_cars_choices'] = [520]
            config['env']['reward_weights'] = {'emergency_speed_weight': base * weight / (1-weight)}
            results.append((f'emergency_weight_{weight:.1f}', config, ('maddpg',), 5))
    if 'segments' in groups:
        for segments in (2, 5, 10):
            config = base_config()
            config['ca']['n_cars_choices'] = [520]
            config['env'].update(n_segments=segments, n_agents=segments)
            results.append((f'segments_{segments}', config, ('maddpg',), 5))
    if 'interval' in groups:
        for interval in (5, 10, 20):
            config = base_config()
            config['ca']['n_cars_choices'] = [520]
            config['env']['decision_interval'] = interval
            results.append((f'interval_{interval}', config, ('maddpg',), 5))
    return results


def make_plan(plan_dir, groups, episodes, seeds):
    jobs = []
    for name, config, methods, default_seeds in variants(groups):
        config_path = plan_dir / 'configs' / f'{name}.json'
        for method in methods:
            for seed in range(default_seeds if seeds is None else seeds):
                output = plan_dir / 'runs' / name / method / f'seed_{seed:02d}'
                if method == 'maddpg':
                    command = [sys.executable, str(HERE / 'run_training.py'),
                               '--experiment-config', str(config_path), '--episodes', str(episodes),
                               '--seed', str(seed), '--no-resume', '--save-freq', '50',
                               '--model-dir', str(output), '--log-dir', str(output)]
                    postcommand = [sys.executable, str(HERE / 'validate_maddpg_checkpoints.py'),
                                   '--experiment-config', str(config_path),
                                   '--model-dir', str(output),
                                   '--validation-densities',
                                   *map(str, config['ca']['n_cars_choices'])]
                else:
                    command = [sys.executable, str(HERE / 'train_baselines.py'),
                               '--algorithm', method, '--experiment-config', str(config_path),
                               '--episodes', str(episodes), '--seed', str(seed),
                               '--validation-densities', *map(str, config['ca']['n_cars_choices']),
                               '--output', str(output), '--execute']
                    postcommand = None
                jobs.append({'group': name, 'algorithm': method, 'seed': seed,
                             'config': str(config_path.relative_to(plan_dir)),
                             'output': str(output.relative_to(plan_dir)),
                             'command': command, 'postcommand': postcommand})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=('main', 'generalization', 'ablation',
                        'emergency_weight', 'segments', 'interval'), default=['main', 'generalization',
                        'ablation', 'emergency_weight', 'segments', 'interval'])
    parser.add_argument('--plan-dir', type=Path, default=Path('paper_plan'))
    parser.add_argument('--episodes', type=int, default=700)
    parser.add_argument('--seeds-per-group', type=int, default=None,
                        help='Override counts for a small protocol check')
    parser.add_argument('--write-plan', action='store_true', help='Write JSON configs and manifest only')
    parser.add_argument('--execute', action='store_true', help='Explicitly launch all training jobs')
    args = parser.parse_args()
    if args.episodes <= 0 or (args.seeds_per_group is not None and args.seeds_per_group <= 0):
        parser.error('Episode and seed counts must be positive')
    plan_dir = args.plan_dir.resolve()
    jobs = make_plan(plan_dir, args.groups, args.episodes, args.seeds_per_group)
    counts = {}
    for job in jobs:
        counts[job['group']] = counts.get(job['group'], 0) + 1
    print(json.dumps({'jobs': len(jobs), 'groups': counts, 'plan_dir': str(plan_dir),
                      'training_started': args.execute}, ensure_ascii=False, indent=2))
    if not (args.write_plan or args.execute):
        return
    (plan_dir / 'configs').mkdir(parents=True, exist_ok=True)
    for name, config, _, _ in variants(args.groups):
        (plan_dir / 'configs' / f'{name}.json').write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (plan_dir / 'manifest.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    if args.execute:
        for index, job in enumerate(jobs, 1):
            (plan_dir / job['output']).mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(jobs)}] {job['group']} {job['algorithm']} seed={job['seed']}", flush=True)
            subprocess.run(job['command'], check=True, cwd=HERE)
            if job['postcommand']:
                subprocess.run(job['postcommand'], check=True, cwd=HERE)


if __name__ == '__main__':
    main()

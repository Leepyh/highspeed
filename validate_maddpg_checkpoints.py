"""Select a MADDPG checkpoint using held-out validation seeds, never test seeds."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from env_maddpg import MultiAgentEmergencyLaneEnv
from evaluate import create_maddpg_agent
from train_baselines import experiment_config, validation


class Policy:
    def __init__(self, agent):
        self.agent = agent

    def act(self, obs, deterministic=True):
        return self.agent.select_action(obs, evaluate=True, deterministic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--validation-seed-start', type=int, default=10000)
    parser.add_argument('--validation-seeds', type=int, default=10)
    parser.add_argument('--validation-densities', nargs='+', type=int)
    args = parser.parse_args()
    if args.validation_seeds <= 0:
        parser.error('validation-seeds must be positive')
    ca, control = experiment_config(args.experiment_config)
    env = MultiAgentEmergencyLaneEnv(ca, control)
    candidates = sorted(args.model_dir.glob('maddpg_episode_*.pth'))
    final = args.model_dir / 'maddpg_final.pth'
    if final.is_file():
        candidates.append(final)
    if not candidates:
        parser.error('No periodic or final MADDPG checkpoints found')
    seeds = range(args.validation_seed_start,
                  args.validation_seed_start + args.validation_seeds)
    densities = args.validation_densities or ca.get('n_cars_choices') or [520]
    rows = []
    best_path, best_score = None, -float('inf')
    for path in candidates:
        agent = create_maddpg_agent(env, str(path))
        if agent is None:
            raise RuntimeError(f'Cannot load {path}')
        match = re.search(r'episode_(\d+)', path.name)
        checkpoint = int(match.group(1)) if match else 'final'
        result = validation(Policy(agent), ca, control, seeds, densities)
        score = sum(r['reward'] for r in result) / len(result)
        for row in result:
            rows.append({'checkpoint': checkpoint, 'path': str(path), **row})
        if score > best_score:
            best_path, best_score = path, score
        print(f'{path.name}: validation_reward={score:.3f}', flush=True)
    output = args.model_dir / 'validation.csv'
    with output.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    shutil.copy2(best_path, args.model_dir / 'best_validation.pth')
    print(f'Selected {best_path.name} as best_validation.pth ({best_score:.3f})')


if __name__ == '__main__':
    main()

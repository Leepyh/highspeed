"""Train I-DQN, QMIX or MAPPO on the same five-agent binary-action task.

Training never starts without --execute. Validation seeds are independent of
training; test seeds are reserved for paper_experiments.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from config_maddpg import CA_CONFIG, ENV_CONFIG
from env_maddpg import MultiAgentEmergencyLaneEnv
from learning_baselines import ALGORITHMS, save_checkpoint


def experiment_config(path):
    ca, control = copy.deepcopy(CA_CONFIG), copy.deepcopy(ENV_CONFIG)
    if not path:
        ca.update(n_cars_range=[350, 850], n_cars_choices=[350, 520, 850],
                  P_emergency_range=[1.0, 1.0], max_emergency=1,
                  warm_up_steps=150, max_steps=400,
                  inject_emergency_after_warmup=True)
        control['decision_interval'] = 5
        control['shared_team_reward'] = True
    if path:
        with open(path, encoding='utf-8') as handle:
            overrides = json.load(handle)
        if set(overrides) - {'ca', 'env'}:
            raise ValueError('Only ca and env sections are supported')
        for section, target in (('ca', ca), ('env', control)):
            for key, value in overrides.get(section, {}).items():
                if key not in target and not (section == 'ca' and key == 'n_cars_choices'):
                    raise ValueError(f'Unknown {section} key: {key}')
                if key == 'reward_weights':
                    if set(value) - set(target[key]):
                        raise ValueError(f'Unknown reward weights: {set(value) - set(target[key])}')
                    target[key].update(value)
                else:
                    target[key] = value
    if control['n_agents'] != control['n_segments'] or ca['L'] % control['n_segments']:
        raise ValueError('n_agents must equal n_segments and divide road length')
    if control['decision_interval'] <= 0 or ca['max_steps'] % control['decision_interval']:
        raise ValueError('max_steps must be divisible by positive decision_interval')
    if ca['n_cars_range'][1] + ca['max_emergency'] > ca['L']:
        raise ValueError('Maximum traffic demand exceeds road initialization capacity')
    return ca, control


def validation(agent, ca_config, env_config, seeds, densities):
    py_state, np_state = random.getstate(), np.random.get_state()
    rows = []
    try:
        for density in densities:
            ca = copy.deepcopy(ca_config)
            ca['n_cars_choices'] = [density]
            env = MultiAgentEmergencyLaneEnv(ca, copy.deepcopy(env_config))
            env.trace_steps = True
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                obs = env.reset()
                done = False
                reward = 0.0
                while not done:
                    actions = agent.act(obs, deterministic=True)
                    obs, rewards, done, _ = env.step(actions)
                    reward += float(np.mean(rewards))
                trace = env.step_trace
                distance = 0
                travel_time = None
                for step in trace:
                    distance += step['emergency_speed'] or 0
                    if travel_time is None and distance >= 500:
                        travel_time = step['step']
                rows.append({'density': density, 'seed': seed, 'reward': reward,
                             'social_speed': float(np.mean([r['social_speed'] for r in trace])),
                             'social_speed_variance': float(np.mean(
                                 [r['social_speed_variance'] for r in trace])),
                             'emergency_travel_time': travel_time,
                             'emergency_completed': travel_time is not None})
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return rows


def run(args):
    ca, control = experiment_config(args.experiment_config)
    if not args.execute:
        print(json.dumps({'algorithm': args.algorithm, 'seed': args.seed,
                          'episodes': args.episodes, 'validation_every': args.validation_every,
                          'ca': ca, 'env': control, 'output': str(args.output)},
                         ensure_ascii=False, indent=2))
        print('DRY RUN: add --execute to train')
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    env = MultiAgentEmergencyLaneEnv(ca, control)
    agent = ALGORITHMS[args.algorithm](env.n_agents, env.obs_dim,
                                       env.n_actions, seed=args.seed)
    rewards_path = args.output / 'training.csv'
    validation_path = args.output / 'validation.csv'
    val_seeds = list(range(args.validation_seed_start,
                           args.validation_seed_start + args.validation_seeds))
    val_densities = args.validation_densities or ca.get('n_cars_choices') or [520]
    best_score = -float('inf')
    with rewards_path.open('w', newline='', encoding='utf-8-sig') as rewards_file, \
         validation_path.open('w', newline='', encoding='utf-8-sig') as validation_file:
        train_writer = csv.DictWriter(rewards_file,
                                      fieldnames=['episode', 'reward', 'updates_loss', 'decisions'])
        val_writer = csv.DictWriter(validation_file,
                                    fieldnames=['episode', 'density', 'seed', 'reward',
                                                'social_speed', 'social_speed_variance',
                                                'emergency_travel_time', 'emergency_completed'])
        train_writer.writeheader(); val_writer.writeheader()
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            total_reward, decisions, losses = 0.0, 0, []
            epsilon = max(.05, 1 - .95 * episode / max(1, args.episodes * .8))
            while not done:
                actions = agent.act(obs, deterministic=False, epsilon=epsilon)
                next_obs, rewards, done, _ = env.step(actions)
                agent.store(obs, actions, rewards, next_obs, done)
                if args.algorithm != 'mappo':
                    loss = agent.update()
                    if loss is not None:
                        losses.append(loss)
                total_reward += float(np.mean(rewards))
                decisions += 1
                obs = next_obs
            if args.algorithm == 'mappo':
                losses.append(agent.update())
            train_writer.writerow({'episode': episode, 'reward': total_reward,
                                   'updates_loss': float(np.mean(losses)) if losses else '',
                                   'decisions': decisions})
            rewards_file.flush()
            if episode % args.validation_every == 0 or episode == args.episodes:
                rows = validation(agent, ca, control, val_seeds, val_densities)
                for row in rows:
                    val_writer.writerow({'episode': episode, **row})
                validation_file.flush()
                score = float(np.mean([r['reward'] for r in rows]))
                if score > best_score:
                    best_score = score
                    save_checkpoint(args.output / 'best_validation.pth', args.algorithm,
                                    agent, env, episode, args.seed)
                print(f'episode={episode} validation_reward={score:.3f} best={best_score:.3f}',
                      flush=True)
    save_checkpoint(args.output / 'final.pth', args.algorithm, agent, env,
                    args.episodes, args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=ALGORITHMS, required=True)
    parser.add_argument('--experiment-config', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=700)
    parser.add_argument('--validation-every', type=int, default=50)
    parser.add_argument('--validation-seed-start', type=int, default=10000)
    parser.add_argument('--validation-seeds', type=int, default=10)
    parser.add_argument('--validation-densities', nargs='+', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Required to start training')
    args = parser.parse_args()
    if min(args.episodes, args.validation_every, args.validation_seeds) <= 0:
        parser.error('Episodes, validation frequency and validation seeds must be positive')
    run(args)


if __name__ == '__main__':
    main()

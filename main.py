#!/usr/bin/env python
"""Run one short scenario with the current segmented emergency-lane model.

Historical clearance-distance figures were produced by a different CA API;
its original driver is preserved in analysis/legacy_main.py.
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from config_maddpg import CA_CONFIG
from env_maddpg import MultiAgentEmergencyLaneEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description='Current highspeed CA scenario')
    parser.add_argument('--cars', type=int, default=520)
    parser.add_argument('--emergency-prob', type=float, default=0.4)
    parser.add_argument('--policy', choices=('open', 'close'), default='close')
    parser.add_argument('--decisions', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    if not 0 <= args.emergency_prob <= 1:
        parser.error('--emergency-prob must be in [0, 1]')
    if not 0 <= args.cars <= CA_CONFIG['L'] - CA_CONFIG['max_emergency']:
        parser.error('--cars exceeds the road capacity')
    if args.decisions < 1:
        parser.error('--decisions must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    ca_config = dict(CA_CONFIG)
    ca_config['n_cars_range'] = (args.cars, args.cars)
    ca_config['P_emergency_range'] = (args.emergency_prob, args.emergency_prob)
    env = MultiAgentEmergencyLaneEnv(ca_config=ca_config)
    env.reset()

    actions = np.full(env.n_agents, int(args.policy == 'open'), dtype=int)
    rewards, social_speeds, emergency_speeds, lane2_social = [], [], [], []
    for _ in range(args.decisions):
        _, step_rewards, done, _ = env.step(actions)
        ca = env.ca
        rewards.append(float(np.mean(step_rewards)))
        social_speeds.append(float(np.mean(ca.vel[:ca.n_cars])) if ca.n_cars else 0.0)
        lane2_social.append(int(np.count_nonzero(ca.lane[:ca.n_cars] == 2)))
        if ca.n_emergency:
            emergency_speeds.append(float(np.mean(ca.vel[ca.emergency_ids])))
        if done:
            break

    print(f'seed={args.seed} cars={args.cars} emergency_cars={env.ca.n_emergency}')
    print(f'policy={args.policy} decisions={len(rewards)}')
    print(f'mean_reward={np.mean(rewards):.3f}')
    print(f'mean_social_speed={np.mean(social_speeds):.3f}')
    print('mean_emergency_speed=' +
          (f'{np.mean(emergency_speeds):.3f}' if emergency_speeds else 'n/a'))
    print(f'mean_social_cars_in_emergency_lane={np.mean(lane2_social):.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

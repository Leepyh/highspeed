#!/usr/bin/env python
"""Regression checks for segmented control, rewards, and CA occupancy."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import install_gym_stub, install_torch_stub, seed_all  # noqa: E402

install_torch_stub()
install_gym_stub()

from config_maddpg import CA_CONFIG, ENV_CONFIG  # noqa: E402
from env_maddpg import MultiAgentEmergencyLaneEnv  # noqa: E402
from high import EmergencyVehicleCA  # noqa: E402


def check_segment_gates():
    ca_config = dict(CA_CONFIG)
    ca_config['P_emergency_range'] = (0.0, 0.0)
    env = MultiAgentEmergencyLaneEnv(ca_config=ca_config)

    for rebuild in range(2):
        if rebuild:
            env.reset()
        for segment in range(env.n_segments):
            pos = segment * env.segment_length + env.segment_length // 2
            assert env.ca.segment_of(pos) == segment
            env.ca.pos[0] = pos  # Gate reads the position; no CA step is taken here.
            actions = np.zeros(env.n_agents, dtype=int)
            env._install_emergency_lane_gate(actions)
            assert not env.ca._can_enter_emergency_lane(0)
            actions[segment] = 1
            env._install_emergency_lane_gate(actions)
            assert env.ca._can_enter_emergency_lane(0)


def check_switch_penalty():
    env_config = dict(ENV_CONFIG)
    env_config['reward_weights'] = {key: 0.0 for key in ENV_CONFIG['reward_weights']}
    env_config['reward_weights']['local_action_smoothness'] = -0.05
    env = MultiAgentEmergencyLaneEnv(env_config=env_config)
    env.reset()
    env.step(np.zeros(env.n_agents, dtype=int))
    next_obs, rewards, _, _ = env.step(np.ones(env.n_agents, dtype=int))
    assert np.allclose(rewards, -0.05), rewards
    assert all(obs[-1] == 1.0 for obs in next_obs)


def check_road_occupancy():
    for seed in (0, 1, 42, 99):
        seed_all(seed)
        ca = EmergencyVehicleCA(
            L=CA_CONFIG['L'], n_lanes=CA_CONFIG['n_lanes'], n_cars=520,
            v_max=CA_CONFIG['v_max'], D=CA_CONFIG['D'], P1=CA_CONFIG['P1'],
            P2=CA_CONFIG['P2'], dev=CA_CONFIG['dev'], emergency_prob=0.5,
            max_emergency=CA_CONFIG['max_emergency'],
            v_init_min=CA_CONFIG['v_initial_range'][0],
            v_init_max=CA_CONFIG['v_initial_range'][1],
            yield_distance=CA_CONFIG['emergency_yield_distance'],
        )
        for step in range(200):
            ca.step()
            ids = np.arange(ca.total_vehicles)
            assert np.array_equal(ca.road[ca.lane[ids], ca.pos[ids]], ids), (seed, step)
            assert np.count_nonzero(ca.road >= 0) == ca.total_vehicles, (seed, step)


def main():
    check_segment_gates()
    print('PASS segment gates before and after reset')
    check_switch_penalty()
    print('PASS action switch penalty and next observation')
    check_road_occupancy()
    print('PASS road occupancy across four seeds and 200 steps each')


if __name__ == '__main__':
    main()

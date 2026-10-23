"""Small algorithm invariants; no episode training is started."""

import io
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_matrix import make_plan  # noqa: E402
from learning_baselines import ALGORITHMS, Mixer, load_checkpoint, save_checkpoint  # noqa: E402
from train_baselines import experiment_config  # noqa: E402
from env_maddpg import MultiAgentEmergencyLaneEnv  # noqa: E402


def main():
    torch.manual_seed(1)
    mixer = Mixer(15, 3)
    state = torch.randn(4, 15)
    q = torch.randn(4, 3)
    higher = q.clone(); higher[:, 1] += 2
    assert torch.all(mixer(higher, state) >= mixer(q, state) - 1e-6)

    base_ca, base_control = experiment_config(None)
    base_ca.update(n_cars_choices=[350], warm_up_steps=2, max_steps=10)
    outcomes = []
    for team_reward in (False, True):
        control_copy = copy.deepcopy(base_control)
        control_copy['shared_team_reward'] = team_reward
        random.seed(123); np.random.seed(123)
        check_env = MultiAgentEmergencyLaneEnv(copy.deepcopy(base_ca), control_copy)
        random.seed(456); np.random.seed(456)
        check_env.reset()
        _, reward, _, _ = check_env.step(np.array([0, 1, 0, 1, 0]))
        outcomes.append((reward, check_env.ca.pos.copy()))
    assert np.allclose(outcomes[1][0], np.mean(outcomes[0][0]))
    assert np.array_equal(outcomes[0][1], outcomes[1][1])

    injected_ca = copy.deepcopy(base_ca)
    injected_ca['inject_emergency_after_warmup'] = True
    random.seed(123); np.random.seed(123)
    injected = MultiAgentEmergencyLaneEnv(injected_ca, copy.deepcopy(base_control))
    random.seed(456); np.random.seed(456)
    injected.reset()
    assert injected.ca.n_emergency == 1
    assert injected.ca.sim_step == 0
    assert injected.ca.vel[injected.ca.emergency_ids[0]] == injected.ca.v_max
    assert injected.ca.total_vehicles == injected.ca.n_cars + 1

    ca, control = experiment_config(None)
    ca['warm_up_steps'], ca['max_steps'] = 2, 10
    env = MultiAgentEmergencyLaneEnv(ca, control)
    obs = env.reset()
    for name, cls in ALGORITHMS.items():
        kwargs = {'seed': 1, 'batch_size': 4} if name != 'mappo' else {'seed': 1}
        agent = cls(env.n_agents, env.obs_dim, **kwargs)
        for t in range(5):
            actions = agent.act(obs, deterministic=False, epsilon=.2)
            assert actions.shape == (env.n_agents,) and set(actions).issubset({0, 1})
            agent.store(obs, actions, np.ones(env.n_agents), obs, t == 4)
        loss = agent.update()
        assert loss is not None and np.isfinite(loss), (name, loss)
        buffer = io.BytesIO()
        save_checkpoint(buffer, name, agent, env, 1, 1)
        buffer.seek(0)
        loaded, metadata = load_checkpoint(buffer)
        assert metadata['algorithm'] == name
        assert np.array_equal(agent.act(obs), loaded.act(obs))
        print(f'PASS {name} update and checkpoint')

    jobs = make_plan(Path('paper_plan'), ['main'], 700, None)
    assert len(jobs) == 40
    assert {job['algorithm'] for job in jobs} == {'maddpg', 'idqn', 'qmix', 'mappo'}
    print('PASS team reward, QMIX monotonicity and four-algorithm ten-seed matrix')


if __name__ == '__main__':
    main()

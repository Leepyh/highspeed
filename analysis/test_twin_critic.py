#!/usr/bin/env python
"""Unit test for the twin-critic (TD3 clipped double-Q) implementation.

    python analysis/test_twin_critic.py

Checks
------
A. twin critics are constructed and are NOT identical to each other
   (otherwise min(Q1,Q2) would be meaningless).
B. the TD target really uses  min(Q1_target, Q2_target)  - verified end-to-end
   through learn() by making target critics return known constants and reading
   back the critic loss (= mean squared target when the online critic returns 0).
C. save/load round-trip preserves both critics.
D. an OLD checkpoint (no critic2_state_dicts) still loads into a twin agent.
E. enabling twin critics does NOT change the Actor outputs, so evaluation of
   existing models is bit-for-bit unchanged.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from maddpg_v2 import MADDPGAgent            # noqa: E402

N_AGENTS, OBS_DIM, N_ACT = 5, 45, 2


def cfg(twin, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return dict(
        actor_hidden_dims=[128, 64], critic_hidden_dims=[256, 128, 64],
        actor_lr=3e-3, critic_lr=1e-3, gamma=0.9, tau=0.005,
        buffer_capacity=5000, batch_size=32, policy_freq=1, train_every=1,
        max_grad_norm=1.0, entropy_coef=0.0, entropy_coef_start=0.0,
        entropy_coef_end=0.0, entropy_anneal=False, entropy_anneal_episodes=1,
        epsilon_schedule='exp', epsilon_decay=0.998, epsilon_decay_episodes=1,
        epsilon_min=0.0, target_action_mode='greedy', target_flip_p=0.0,
        actor_update_mode='counterfactual', twin_critics=twin)


class ConstCritic(nn.Module):
    """Returns a constant Q; carries a dummy parameter so backward() works."""

    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, obs, actions):
        return (self.w * 0 + self.value).expand(actions.size(0)).unsqueeze(-1)


def fill_buffer(agent, n=32, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
        nobs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
        acts = rng.integers(0, N_ACT, size=N_AGENTS)
        rews = np.zeros(N_AGENTS, dtype=np.float32)
        agent.store_transition(obs, acts, rews, nobs, False)


def test_a():
    a = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(True))
    if not a.twin_critics or len(a.critics2) != N_AGENTS:
        return False, 'twin critics not constructed'
    diff = max(float((p1 - p2).abs().max())
               for c1, c2 in zip(a.critics, a.critics2)
               for p1, p2 in zip(c1.parameters(), c2.parameters()))
    return diff > 0, f'max |Q1_init - Q2_init| = {diff:.2e} (must be > 0)'


def test_b():
    """TD target must be r + gamma*min(Q1,Q2) -> loss = (gamma*min)^2 when Q_online=0."""
    a = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(True))
    fill_buffer(a)
    for i in range(N_AGENTS):
        a.critics[i] = ConstCritic(0.0).to(a.device)
        a.critic_optimizers[i] = torch.optim.Adam(a.critics[i].parameters(), lr=0.0)
        a.critic_targets[i] = ConstCritic(100.0).to(a.device)
        a.critics2[i] = ConstCritic(0.0).to(a.device)
        a.critic_optimizers2[i] = torch.optim.Adam(a.critics2[i].parameters(), lr=0.0)
        a.critic_targets2[i] = ConstCritic(10.0).to(a.device)
    loss, _ = a.learn()
    gamma = a.config['gamma']
    expect = (gamma * 10.0) ** 2          # min(100, 10) = 10
    wrong = (gamma * 100.0) ** 2          # if min were NOT applied
    ok = abs(loss - expect) < 1e-4
    return ok, (f'critic loss={loss:.4f}  expect(min)={expect:.4f}  '
                f'would-be(no min)={wrong:.4f}')


def test_cd():
    a = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(True))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'twin.pth')
        a.save(p, episode=7)
        b = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(True, seed=99))
        b.load(p)
        same = all(float((x - y).abs().max()) == 0.0
                   for c1, c2 in zip(a.critics2, b.critics2)
                   for x, y in zip(c1.parameters(), c2.parameters()))
        if not same:
            return False, 'critic2 not restored by load()'

        # D: an old-format checkpoint (drop critic2_state_dicts) must still load
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        ckpt.pop('critic2_state_dicts', None)
        p_old = os.path.join(d, 'old.pth')
        torch.save(ckpt, p_old)
        c = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(True, seed=7))
        try:
            c.load(p_old)
        except Exception as e:                       # noqa: BLE001
            return False, f'old checkpoint failed to load: {e}'
        warm = all(float((x - y).abs().max()) == 0.0
                   for c1, c2 in zip(c.critics, c.critics2)
                   for x, y in zip(c1.parameters(), c2.parameters()))
        return warm, ('critic2 restored from twin checkpoint; '
                      'old checkpoint warm-started critic2 from critic1 '
                      f'(match={warm})')


def test_e(tmpdir):
    """Actor outputs must be identical with twin on/off (evaluation unchanged)."""
    a = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(True, seed=5))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'm.pth')
        a.save(p, episode=1)
        obs = torch.zeros(64, N_AGENTS, OBS_DIM, device=a.device)
        with torch.no_grad():
            p_on = a._actor_probs(obs)

        b = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg(False, seed=5))
        b.load(p)
        with torch.no_grad():
            p_off = b._actor_probs(obs)
        diff = float((p_on - p_off).abs().max())
        act_on = p_on.argmax(-1).cpu().numpy()
        act_off = p_off.argmax(-1).cpu().numpy()
        same_actions = bool((act_on == act_off).all())
        return diff == 0.0 and same_actions, (
            f'max |p_open(twin) - p_open(no twin)| = {diff:.2e}; '
            f'argmax actions identical = {same_actions}')


def main():
    print('=' * 84)
    print('TWIN CRITIC TEST (TD3 clipped double-Q)')
    print('=' * 84)
    checks = []
    ok, msg = test_a(); print(f'A) construction        : {"PASS" if ok else "FAIL"}  {msg}'); checks.append(ok)
    ok, msg = test_b(); print(f'B) target uses min()   : {"PASS" if ok else "FAIL"}  {msg}'); checks.append(ok)
    ok, msg = test_cd(); print(f'C/D) save+load compat  : {"PASS" if ok else "FAIL"}  {msg}'); checks.append(ok)
    ok, msg = test_e(None); print(f'E) eval unchanged      : {"PASS" if ok else "FAIL"}  {msg}'); checks.append(ok)
    print('-' * 84)
    print('RESULT:', 'PASS' if all(checks) else 'FAIL')
    return 0 if all(checks) else 1


if __name__ == '__main__':
    sys.exit(main())

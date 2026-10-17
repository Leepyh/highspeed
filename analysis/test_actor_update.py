#!/usr/bin/env python
"""Unit test for the rewritten actor update.

    python analysis/test_actor_update.py

Why
---
The old actor loss fed the critic a **soft probability vector**:

    joint_soft = [pi_0, pi_1, ..., pi_{n-1}]     # blended "actions"
    q = critic(obs, joint_soft)                  # critic never saw such input

but the critic is trained only on **one-hot** joint actions
(`F.one_hot(action_tensor)` in the critic update).  So the actor was reading
the critic at an input it had never been fitted on.  In the real project that
showed up as: policy probability stuck at p(open) ~= 0.43 forever, argmax
acting like noise, reward peaking around episode 600-800 and then drifting down.

The new loss evaluates the critic at the two **one-hot counterfactual**
actions and maximises  p(close)*Q(close) + p(open)*Q(open).

The environment is not used at all: every critic is replaced by a stub with a
known value, so the test isolates the update rule.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from maddpg_v2 import MADDPGAgent            # noqa: E402

N_AGENTS, OBS_DIM, N_ACT = 5, 45, 2
TEST_LR = 3e-3          # 100x the project lr, so the test converges in ~300 steps


class OneHotCritic(nn.Module):
    """Well-behaved critic: Q = gap * onehot(agent i chose OPEN)."""

    def __init__(self, idx, gap):
        super().__init__()
        self.idx, self.gap = idx, gap
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, obs, actions):
        joint = actions.view(actions.size(0), N_AGENTS, N_ACT)
        return (self.w * 0 + self.gap * joint[:, self.idx, 1]).unsqueeze(-1)


class BlindCritic(nn.Module):
    """Models the real pathology: the critic is only meaningful on one-hot
    joint actions, and returns a constant (zero gradient) anywhere else."""

    def __init__(self, idx, gap):
        super().__init__()
        self.idx, self.gap = idx, gap
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, obs, actions):
        joint = actions.view(actions.size(0), N_AGENTS, N_ACT)
        # NOTE: a softmax vector also sums to 1, so "sums to 1" is NOT enough -
        # a real one-hot row has max == 1.
        is_onehot = ((joint.max(dim=-1).values - 1.0).abs() < 1e-6).all(dim=-1)
        val = self.gap * joint[:, self.idx, 1]
        val = torch.where(is_onehot, val, torch.zeros_like(val))
        return (self.w * 0 + val).unsqueeze(-1)


def build(critic_cls, gap, entropy_coef, update_mode, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = dict(actor_hidden_dims=[128, 64], critic_hidden_dims=[256, 128, 64],
               actor_lr=TEST_LR, critic_lr=1e-3, gamma=0.95, tau=0.005,
               buffer_capacity=5000, batch_size=64, policy_freq=1,
               train_every=1, max_grad_norm=1.0,
               entropy_coef=entropy_coef, entropy_coef_start=entropy_coef,
               entropy_coef_end=entropy_coef, entropy_anneal=False,
               entropy_anneal_episodes=1, epsilon_schedule='exp',
               epsilon_decay=0.998, epsilon_decay_episodes=1, epsilon_min=0.0,
               target_action_mode='greedy', target_flip_p=0.0,
               actor_update_mode=update_mode)
    agent = MADDPGAgent(N_AGENTS, OBS_DIM, N_ACT, cfg)
    for i in range(N_AGENTS):
        agent.critics[i] = critic_cls(i, gap).to(agent.device)
        agent.critic_targets[i] = critic_cls(i, gap).to(agent.device)
        agent.critic_optimizers[i] = torch.optim.Adam(
            agent.critics[i].parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(64):
        obs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
        nobs = rng.normal(size=(N_AGENTS, OBS_DIM)).astype(np.float32)
        acts = rng.integers(0, N_ACT, size=N_AGENTS)
        rews = rng.normal(size=N_AGENTS).astype(np.float32)
        agent.store_transition(obs, acts, rews, nobs, False)
    return agent


def mean_p_open(agent, n=512):
    obs = torch.zeros(n, N_AGENTS, OBS_DIM, device=agent.device)
    with torch.no_grad():
        return float(agent._actor_probs(obs)[..., 1].mean())


def run(critic_cls, gap, entropy_coef, update_mode, steps=300):
    agent = build(critic_cls, gap, entropy_coef, update_mode)
    p0 = mean_p_open(agent)
    for _ in range(steps):
        agent.learn()
    return p0, mean_p_open(agent)


def main():
    print('=' * 82)
    print('ACTOR UPDATE TEST (env not used; critics replaced by stubs)')
    print('=' * 82)
    checks = []

    # A. the new update must actually follow the critic's preference
    p0, pA = run(OneHotCritic, 50.0, 0.0, 'counterfactual')
    print(f'A) counterfactual update, ideal critic, gap=+50   '
          f'p(open) {p0:.4f} -> {pA:.4f}   expect -> 1')
    checks.append(('A new update learns the preference', pA > 0.90))

    # B. the old update on a one-hot-blind critic: no gradient at all
    p0, pB = run(BlindCritic, 50.0, 0.0, 'legacy')
    print(f'B) LEGACY update, one-hot-blind critic          '
          f'p(open) {p0:.4f} -> {pB:.4f}   expect ~unchanged (this is the bug)')
    checks.append(('B legacy update is blind (bug reproduced)', abs(pB - p0) < 0.02))

    # C. the new update still works with the same blind critic
    p0, pC = run(BlindCritic, 50.0, 0.0, 'counterfactual')
    print(f'C) NEW update, same one-hot-blind critic        '
          f'p(open) {p0:.4f} -> {pC:.4f}   expect -> 1 (fix works)')
    checks.append(('C new update fixes the blind critic', pC > 0.90))

    # D. the entropy coefficient must control how decisive the policy can get
    print('-' * 82)
    print('D) entropy annealing: gap=+2, counterfactual update')
    res = {}
    for coef in (0.0, 1.0, 5.0):
        p0, p = run(OneHotCritic, 2.0, coef, 'counterfactual', steps=600)
        res[coef] = p
        theory = 1.0 / (1.0 + np.exp(-2.0 / coef)) if coef > 0 else 1.0
        print(f'     entropy_coef={coef:<4} p(open) {p0:.4f} -> {p:.4f}'
              f'   (theory sigma(gap/coef) = {theory:.4f})')
    checks.append(('D entropy coef lowers decisiveness',
                   res[0.0] > res[1.0] > res[5.0]))
    checks.append(('D coef=5 keeps policy near-uniform (>0.45)',
                   res[5.0] > 0.45))

    print('-' * 82)
    ok = True
    for name, passed in checks:
        print(f'  [{"PASS" if passed else "FAIL"}] {name}')
        ok &= bool(passed)
    print('=' * 82)
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

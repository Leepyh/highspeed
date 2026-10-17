#!/usr/bin/env python
"""Decompose the episode reward of each strategy into its five terms.

Answers "where does the advantage come from?" instead of only "who wins".
The per-step reward is the sum of

    emergency_speed  +  resource  +  efficiency  +  speed_gain  +  control

`env.reward_breakdown()` recomputes those parts, but only if the
`prev_global_speed` baseline is the one that was in force *during* `step()`
(`step()` overwrites it right after using it).  This script saves that value
before the call and temporarily restores it for the breakdown, then puts the
real value back - the environment is left exactly as it was found.

    python analysis/reward_decompose.py --policies MADDPG_argmax,AlwaysClose \
        --episodes 30 --out analysis/decompose_a.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_paired import (make_env, load_maddpg, MaddpgPolicy, AlwaysOpen,   # noqa: E402
                         AlwaysClose, RandomPolicy, Periodic, Greedy,
                         OracleDensity)

TERMS = ['emergency_speed', 'resource', 'efficiency', 'speed_gain', 'control']


def build_policy(name, agent, seed_start, oracle_threshold):
    if name == 'MADDPG_argmax':
        return MaddpgPolicy(agent, 'argmax')
    if name == 'MADDPG_sample':
        return MaddpgPolicy(agent, 'sample')
    if name == 'AlwaysOpen':
        return AlwaysOpen()
    if name == 'AlwaysClose':
        return AlwaysClose()
    if name == 'Random':
        return RandomPolicy(seed_start)
    if name == 'Periodic':
        return Periodic()
    if name == 'Greedy':
        return Greedy()
    if name == 'OracleDensity':
        return OracleDensity(oracle_threshold)
    raise SystemExit(f'unknown policy {name}')


def decompose(policy, seeds):
    env = make_env()
    acc = {t: 0.0 for t in TERMS}
    n_agent_steps = 0
    total = 0.0
    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        obs = env.reset()
        done = False
        step = 0
        while not done:
            rs, ns = random.getstate(), np.random.get_state()
            actions = np.asarray(policy(env, obs, step), dtype=int).reshape(-1)
            random.setstate(rs)
            np.random.set_state(ns)

            baseline = env.prev_global_speed          # value used inside step()
            obs, rewards, done, info = env.step(actions)
            real_prev = env.prev_global_speed         # step() overwrote it

            env.prev_global_speed = baseline          # restore for exact breakdown
            _, parts = env.reward_breakdown(actions)
            env.prev_global_speed = real_prev         # put the real value back

            for p in parts:
                for t in TERMS:
                    acc[t] += p.get(t, 0.0)
            n_agent_steps += len(parts)
            total += float(np.sum(rewards))
            step += 1
    n = max(n_agent_steps, 1)
    return {
        'per_agent_step': {t: acc[t] / n for t in TERMS},
        'share_pct': {t: 100.0 * acc[t] / total if total else 0.0 for t in TERMS},
        'total_agent_steps': n_agent_steps,
        'total_reward_sum': total,
        'mean_step_reward_per_agent': total / n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--policies', required=True)
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--seed-start', type=int, default=1000)
    ap.add_argument('--model', default=os.path.join(BASE, 'models_matd3',
                                                    'maddpg_best.pth'))
    ap.add_argument('--oracle-threshold', type=int, default=500)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)

    wanted = a.policies.split(',')
    agent = None
    if any(p.startswith('MADDPG') for p in wanted):
        agent, _ = load_maddpg(a.model)

    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    out = {}
    for name in wanted:
        pol = build_policy(name, agent, a.seed_start, a.oracle_threshold)
        r = decompose(pol, seeds)
        out[name] = r
        pts = r['per_agent_step']
        print(f"  {name:<16} " + "  ".join(
            f"{t}={pts[t]:+7.3f}" for t in TERMS) +
            f"   | sum/step={r['mean_step_reward_per_agent']:7.3f}", flush=True)

    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    print(f"[decompose] written: {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

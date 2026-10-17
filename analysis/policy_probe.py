#!/usr/bin/env python
"""Probe what the trained actor actually computes, state by state.

For every decision of an episode this records, per agent:
    action (argmax), p(open) from the actor, local social-lane speed,
    whether an emergency vehicle is inside the segment, segment step index.

That answers "what did the model learn" far better than a single reward number:
does the action depend on the state at all, and on which part of it?

    python analysis/policy_probe.py --model models_matd3/maddpg_best.pth \
        --episodes 24 --out analysis/policy_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eval_paired import make_env, load_maddpg          # noqa: E402

V_MAX = 5.0


def run_probe(model_path, n_episodes, seed_start=1000, mode='argmax'):
    import torch
    agent, meta = load_maddpg(model_path)
    env = make_env()

    recs = []
    for k in range(n_episodes):
        seed = seed_start + k
        random.seed(seed)
        np.random.seed(seed)
        obs = env.reset()
        params = dict(env.current_params)
        done = False
        step = 0
        ep = {'seed': seed, 'n_cars': params['n_cars'],
              'P_emergency': params['P_emergency'], 'actions': []}

        while not done:
            rs, ns = random.getstate(), np.random.get_state()
            obs_all = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                                      device=agent.device).unsqueeze(0)
            with torch.no_grad():
                probs = agent._actor_probs(obs_all)[0].cpu().numpy()
            if mode == 'argmax':
                actions = probs.argmax(axis=1).astype(int)
            else:
                actions = np.array([np.random.choice(2, p=probs[i])
                                    for i in range(env.n_agents)])
            random.setstate(rs)
            np.random.set_state(ns)

            em_segments = set(env.ca.segment_emergency_ids(
                env.segment_length, env.n_segments))
            for i in range(env.n_agents):
                seginfo = env._get_segment_info(i)
                ep['actions'].append({
                    'step': step,
                    'agent': i,
                    'action': int(actions[i]),
                    'p_open': float(probs[i][1]),
                    'local_speed': float(env._local_congestion_speed(i)),
                    'has_emergency': bool(i in em_segments),
                    'lane1_density': float(seginfo[1]),
                })

            obs, rewards, done, info = env.step(actions)
            step += 1
        ep['n_decisions'] = step
        ep['n_emergency'] = int(env.ca.n_emergency)
        recs.append(ep)
    return recs, meta


def summarise(recs):
    flat = [a for ep in recs for a in ep['actions']]
    p = np.array([a['p_open'] for a in flat])
    act = np.array([a['action'] for a in flat], dtype=int)
    spd = np.array([a['local_speed'] for a in flat])
    em = np.array([a['has_emergency'] for a in flat], dtype=bool)
    ag = np.array([a['agent'] for a in flat], dtype=int)
    st = np.array([a['step'] for a in flat], dtype=int)
    nc = np.array([ep['n_cars'] for ep in recs for _ in ep['actions']], dtype=float)

    out = {
        'n_decisions': int(len(flat)),
        'mean_p_open': float(p.mean()),
        'std_p_open': float(p.std()),
        'min_p_open': float(p.min()),
        'max_p_open': float(p.max()),
        'argmax_open_rate': float(act.mean()),
        'p_open_std_over_agents': [float(p[ag == i].std()) for i in range(5)],
        'mean_p_open_per_agent': [float(p[ag == i].mean()) for i in range(5)],
        'argmax_open_rate_per_agent': [float(act[ag == i].mean()) for i in range(5)],
        'open_rate_when_emergency': float(act[em].mean()) if em.any() else None,
        'open_rate_when_no_emergency': float(act[~em].mean()) if (~em).any() else None,
        'p_open_when_emergency': float(p[em].mean()) if em.any() else None,
        'p_open_when_no_emergency': float(p[~em].mean()) if (~em).any() else None,
    }

    # open rate vs local social-lane speed
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    speed_bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (spd >= lo) & (spd < hi)
        if m.sum() == 0:
            continue
        speed_bins.append({'lo': lo, 'hi': hi, 'n': int(m.sum()),
                           'open_rate': float(act[m].mean()),
                           'mean_p_open': float(p[m].mean())})
    out['by_local_speed'] = speed_bins

    # open rate vs step index (time within the episode)
    edges_t = list(range(0, 200, 20))
    time_bins = []
    for lo, hi in zip(edges_t[:-1], edges_t[1:]):
        m = (st >= lo) & (st < hi)
        if m.sum() == 0:
            continue
        time_bins.append({'lo': lo, 'hi': hi, 'n': int(m.sum()),
                          'open_rate': float(act[m].mean()),
                          'mean_p_open': float(p[m].mean())})
    out['by_step'] = time_bins

    # p(open) vs n_cars
    dens_bins = []
    for lo, hi in ((450, 500), (500, 530), (530, 560), (560, 601)):
        m = (nc >= lo) & (nc < hi)
        if m.sum() == 0:
            continue
        dens_bins.append({'lo': lo, 'hi': hi, 'n': int(m.sum()),
                          'open_rate': float(act[m].mean()),
                          'mean_p_open': float(p[m].mean())})
    out['by_n_cars'] = dens_bins

    # how often would the action flip if p crossed 0.5 in the opposite way?
    out['frac_decisions_p_near_half'] = float(np.mean(np.abs(p - 0.5) < 0.05))
    out['entropy_mean'] = float(np.mean(
        -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))))
    out['entropy_max'] = float(np.log(2))

    hist, edges = np.histogram(p, bins=np.linspace(0.0, 1.0, 51))
    out['p_hist'] = hist.tolist()
    out['p_hist_edges'] = edges.tolist()
    out['p_deciles'] = [float(np.quantile(p, q)) for q in
                        (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(
        BASE, 'models_matd3', 'maddpg_best.pth'))
    ap.add_argument('--episodes', type=int, default=24)
    ap.add_argument('--mode', choices=['argmax', 'sample'], default='argmax')
    ap.add_argument('--out', default=os.path.join(HERE, 'policy_probe.json'))
    a = ap.parse_args(argv)

    recs, meta = run_probe(a.model, a.episodes, mode=a.mode)
    s = summarise(recs)
    s['meta'] = meta
    s['mode'] = a.mode
    s['n_episodes'] = a.episodes
    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)

    print(f"[probe] mode={a.mode}  episodes={a.episodes}  "
          f"decisions={s['n_decisions']}")
    print(f"[probe] mean p(open) = {s['mean_p_open']:.4f} "
          f"(std {s['std_p_open']:.4f}, range "
          f"{s['min_p_open']:.4f}..{s['max_p_open']:.4f})")
    print(f"[probe] policy entropy = {s['entropy_mean']:.4f} "
          f"/ ln2 = {s['entropy_max']:.4f}")
    print(f"[probe] argmax open rate = {s['argmax_open_rate']:.4f}")
    print(f"[probe] decisions with |p-0.5|<0.05 : "
          f"{s['frac_decisions_p_near_half']:.3f}")
    print(f"[probe] per-agent argmax open rate  : "
          f"{[round(x, 4) for x in s['argmax_open_rate_per_agent']]}")
    print(f"[probe] open rate  emergency-in-seg : "
          f"{s['open_rate_when_emergency']}  vs none: "
          f"{s['open_rate_when_no_emergency']}")
    print('[probe] open rate by local social-lane speed:')
    for b in s['by_local_speed']:
        print(f"          {b['lo']:.1f}-{b['hi']:.1f}  n={b['n']:>6}  "
              f"open={b['open_rate']:.4f}  p={b['mean_p_open']:.4f}")
    print('[probe] open rate by decision-step index:')
    for b in s['by_step']:
        print(f"          {b['lo']:>3}-{b['hi']:<3}  n={b['n']:>6}  "
              f"open={b['open_rate']:.4f}  p={b['mean_p_open']:.4f}")
    print(f"[probe] written: {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

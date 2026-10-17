#!/usr/bin/env python
"""Paired-seed fair evaluation harness for the MADDPG emergency-lane controller.

Why this file exists
--------------------
The shipped `evaluate.py` calls `env.reset()` n times per strategy without
controlling the RNG.  Each `reset()` samples `n_cars ~ U{450..600}` and
`P_emergency ~ U(0.3, 0.5)`, so *every strategy is graded on a different set
of traffic scenarios*.  With only 50 episodes that sampling noise is large
enough to reorder the leaderboard.

This harness removes that noise by **pairing**: for episode i every strategy
runs on the exact same scenario.

    random.seed(seed); np.random.seed(seed); env.reset()      # scenario fixed

`high.py` draws from BOTH `random` (positions, speeds, lane changes) and
`np.random` (emergency-vehicle count).  A stochastic policy also draws from
those same global streams, which would desynchronise the scenarios.  So the
global RNG state is saved and restored around every policy call; policy
randomness therefore never leaks into the environment.

Everything here is READ-ONLY with respect to the existing project:
no checkpoint, log, config or source file is modified or deleted.

Modes
-----
    screen     evaluate every checkpoint in a directory (deterministic policy)
    full       evaluate the full strategy set (MADDPG + 6 baselines)
    landscape  measure reward(open) - reward(close) at fixed densities

Examples
--------
    python analysis/eval_paired.py screen --model-dir models_matd3 \
        --seeds 24 --out analysis/screen.json

    python analysis/eval_paired.py full --model models_matd3/maddpg_best.pth \
        --seeds 60 --policies MADDPG_argmax,MADDPG_sample,AlwaysOpen,... \
        --out analysis/full.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from env_maddpg import MultiAgentEmergencyLaneEnv          # noqa: E402
from config_maddpg import CA_CONFIG, ENV_CONFIG            # noqa: E402

V_MAX = CA_CONFIG['v_max']
EM_FLOOR = CA_CONFIG.get('emergency_min_acceptable_speed', 2.0)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def make_env(n_cars_range=None):
    ca = dict(CA_CONFIG)                      # copy: never mutate the global config
    if n_cars_range is not None:
        ca['n_cars_range'] = (int(n_cars_range[0]), int(n_cars_range[1]))
    return MultiAgentEmergencyLaneEnv(ca_config=ca, env_config=dict(ENV_CONFIG))


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
class Policy:
    name = 'policy'
    display = 'policy'

    def __call__(self, env, obs, step):
        raise NotImplementedError


class AlwaysOpen(Policy):
    name = 'AlwaysOpen'
    display = 'Always Open\n(全段恒开放)'

    def __call__(self, env, obs, step):
        return np.ones(env.n_agents, dtype=int)


class AlwaysClose(Policy):
    name = 'AlwaysClose'
    display = 'Always Close\n(全段恒关闭)'

    def __call__(self, env, obs, step):
        return np.zeros(env.n_agents, dtype=int)


class RandomPolicy(Policy):
    name = 'Random'
    display = 'Random\n(随机策略)'

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, obs, step):
        return self.rng.integers(0, 2, size=env.n_agents)


class Periodic(Policy):
    name = 'Periodic'
    display = 'Periodic\n(周期开关, T=10)'

    def __init__(self, period=10):
        self.period = period

    def __call__(self, env, obs, step):
        return np.full(env.n_agents, (step // self.period) % 2, dtype=int)


class Greedy(Policy):
    """Open the segment whenever its social-lane speed is below the threshold."""
    name = 'Greedy'
    display = 'Greedy\n(本段堵则开)'

    def __call__(self, env, obs, step):
        thr = env.env_config['congestion_threshold'] / env.ca_config['v_max']
        a = np.zeros(env.n_agents, dtype=int)
        for i in range(env.n_agents):
            if env._local_congestion_speed(i) < thr:
                a[i] = 1
        return a


class OracleDensity(Policy):
    """Privileged reference: knows the true density and applies the optimum.

    Upper-bound reference only - it reads `n_cars`, which the agents cannot
    observe directly.  Used to show how much of the available reward the
    learned policy leaves on the table.
    """
    name = 'OracleDensity'
    display = 'Oracle\n(密度规则, 上界参考)'

    def __init__(self, threshold=500):
        self.threshold = threshold

    def __call__(self, env, obs, step):
        n_cars = env.current_params['n_cars']
        return np.full(env.n_agents, 1 if n_cars >= self.threshold else 0, dtype=int)


class MaddpgPolicy(Policy):
    """Wraps a trained MADDPGAgent.

    mode='argmax'  -> deterministic action  argmax_a pi(a|s)   (the learned policy)
    mode='sample'  -> a ~ pi(.|s)  with eps=0, i.e. exactly what
                      MADDPGAgent.select_action(evaluate=True) does today.
    """

    def __init__(self, agent, mode):
        self.agent = agent
        self.mode = mode
        self.name = 'MADDPG_argmax' if mode == 'argmax' else 'MADDPG_sample'
        self.display = ('MADDPG 确定性策略\n(argmax, 本文模型)' if mode == 'argmax'
                        else 'MADDPG 随机采样\n(evaluate.py 口径)')

    def __call__(self, env, obs, step):
        if self.mode == 'sample':
            # identical semantics to MADDPGAgent.select_action(evaluate=True)
            return self.agent.select_action(obs, evaluate=True)
        import torch
        obs_all = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                                  device=self.agent.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.agent._actor_probs(obs_all)[0].cpu().numpy()
        return probs.argmax(axis=1).astype(int)


def load_maddpg(model_path):
    """Build a MADDPGAgent and load a checkpoint (inference only)."""
    import torch
    from maddpg_v2 import MADDPGAgent
    env = make_env()
    cfg = {
        'actor_hidden_dims': [128, 64],
        'critic_hidden_dims': [256, 128, 64],
        'actor_lr': 3e-5, 'critic_lr': 1e-3,
        'gamma': 0.95, 'tau': 0.005,
        'buffer_capacity': 100, 'batch_size': 128,
        'policy_freq': 2, 'max_grad_norm': 1.0,
    }
    agent = MADDPGAgent(env.n_agents, env.obs_dim, ENV_CONFIG['n_actions'], cfg)
    ckpt = agent.load(model_path)          # load() also syncs the target nets
    agent.epsilon = 0.0
    meta = {
        'path': model_path,
        'episode': ckpt.get('episode'),
        'best_reward': ckpt.get('best_reward'),
        'avg_reward': ckpt.get('avg_reward'),
        'train_step': ckpt.get('train_step'),
        'epsilon': ckpt.get('epsilon'),
        'device': str(agent.device),
        'torch': torch.__version__,
    }
    return agent, meta


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------
def rollout(env, policy, seeds, verbose=False):
    """Run `policy` on the scenarios identified by `seeds`.

    Returns one row per episode.  All policies called with the same `seeds`
    see byte-identical traffic scenarios.
    """
    rows = []
    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        obs = env.reset()
        params = dict(env.current_params)

        done = False
        step = 0
        ep_reward = 0.0
        soc, em, lane2, openr, stall = [], [], [], [], []
        n_emergency_seen = 0

        while not done:
            # ---- policy call with the global RNG streams frozen ----
            rstate = random.getstate()
            nstate = np.random.get_state()
            actions = np.asarray(policy(env, obs, step), dtype=int).reshape(-1)
            random.setstate(rstate)
            np.random.set_state(nstate)
            # --------------------------------------------------------

            obs, rewards, done, info = env.step(actions)
            ca = env.ca
            n = ca.n_cars
            vmax = env.ca_config['v_max']

            openr.append(float(np.mean(actions == 1)))
            soc.append(float(np.mean(ca.vel[:n])) if n else 0.0)
            lane2.append(float(np.sum(ca.lane[:ca.total_vehicles] == 2)) / max(n, 1))
            if ca.n_emergency > 0:
                speeds = ca.vel[ca.emergency_ids]
                em.append(float(np.mean(speeds)))
                stall.append(float(np.mean(speeds <= 0)))
                n_emergency_seen = max(n_emergency_seen, ca.n_emergency)

            ep_reward += float(np.mean(rewards))
            step += 1

        row = {
            'seed': int(s),
            'n_cars': int(params['n_cars']),
            'P_emergency': float(params['P_emergency']),
            'steps': int(step),
            'reward': ep_reward,
            'social_speed': float(np.mean(soc)),
            'emergency_speed': float(np.mean(em)) if em else None,
            'emergency_stall_frac': float(np.mean(stall)) if stall else None,
            'lane2_frac': float(np.mean(lane2)),
            'open_rate': float(np.mean(openr)),
            'n_emergency_max': int(n_emergency_seen),
        }
        rows.append(row)
        if verbose and (len(rows) % 10 == 0):
            print(f"      ... {len(rows)}/{len(seeds)} episodes", flush=True)
    return rows


def aggregate(rows):
    rew = np.array([r['reward'] for r in rows], dtype=float)
    soc = np.array([r['social_speed'] for r in rows], dtype=float)
    lane2 = np.array([r['lane2_frac'] for r in rows], dtype=float)
    openr = np.array([r['open_rate'] for r in rows], dtype=float)

    em_rows = [r for r in rows if r['emergency_speed'] is not None]
    if em_rows:
        em = np.array([r['emergency_speed'] for r in em_rows], dtype=float)
        st = np.array([r['emergency_stall_frac'] for r in em_rows], dtype=float)
        succ = np.array([1.0 if r['emergency_speed'] >= EM_FLOOR else 0.0
                         for r in em_rows], dtype=float)
    else:
        em = st = succ = np.array([])

    def m(a, d=0.0):
        return float(np.mean(a)) if a.size else d

    return {
        'n_episodes': len(rows),
        'emergency_episodes': len(em_rows),
        'mean_reward': m(rew),
        'std_reward': float(np.std(rew)) if rew.size else 0.0,
        'sem_reward': float(np.std(rew, ddof=1) / np.sqrt(rew.size)) if rew.size > 1 else 0.0,
        'mean_social_speed': m(soc),
        'mean_emergency_speed': m(em),
        'min_emergency_speed': float(np.min(em)) if em.size else 0.0,
        'mean_emergency_stall_frac': m(st),
        'success_rate': m(succ),
        'mean_lane2_frac': m(lane2),
        'mean_open_rate': m(openr),
        '_episodes': rows,
    }


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def _seed_list(n, start=1000):
    return list(range(start, start + n))


def mode_screen(args):
    model_dir = args.model_dir
    files = sorted(f for f in os.listdir(model_dir) if f.endswith('.pth'))
    if args.exclude_resume:
        files = [f for f in files if 'resume' not in f.lower()]
    if args.only:
        want = set(args.only.split(','))
        files = [f for f in files if f.replace('.pth', '') in want or f in want]
    if args.shard:
        k, of = args.shard
        files = files[k::of]
    print(f"[screen] {len(files)} checkpoint(s) on {args.seeds} paired episodes")
    print(f"[screen] files: {files}")

    seeds = _seed_list(args.seeds, args.seed_start)
    out = {}
    for f in files:
        path = os.path.join(model_dir, f)
        t0 = time.time()
        agent, meta = load_maddpg(path)
        env = make_env()
        pol = MaddpgPolicy(agent, args.mode)
        rows = rollout(env, pol, seeds)
        agg = aggregate(rows)
        agg['meta'] = meta
        agg['seconds'] = round(time.time() - t0, 1)
        out[f.replace('.pth', '')] = agg
        print(f"  {f:<28} ep={meta['episode']!s:>5} "
              f"best={meta['best_reward']!s:>9} "
              f"reward={agg['mean_reward']:8.2f} "
              f"soc={agg['mean_social_speed']:.3f} "
              f"em={agg['mean_emergency_speed']:.3f} "
              f"open={agg['mean_open_rate']:.3f} "
              f"lane2={agg['mean_lane2_frac']:.4f} "
              f"({agg['seconds']}s)", flush=True)

    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"[screen] written: {args.out}")
    return out


def mode_full(args):
    seeds = _seed_list(args.seeds, args.seed_start)
    wanted = args.policies.split(',')
    print(f"[full] {len(wanted)} policies x {len(seeds)} paired episodes")

    agent = None
    if any(p.startswith('MADDPG') for p in wanted):
        agent, meta = load_maddpg(args.model)
        print(f"[full] model: {args.model}")
        print(f"[full] checkpoint meta: {meta}")

    registry = {
        'MADDPG_argmax': lambda: MaddpgPolicy(agent, 'argmax'),
        'MADDPG_sample': lambda: MaddpgPolicy(agent, 'sample'),
        'AlwaysOpen': AlwaysOpen,
        'AlwaysClose': AlwaysClose,
        'Random': lambda: RandomPolicy(args.seed_start),
        'Periodic': Periodic,
        'Greedy': Greedy,
        'OracleDensity': lambda: OracleDensity(args.oracle_threshold),
    }

    results = {}
    for name in wanted:
        if name not in registry:
            raise SystemExit(f"unknown policy: {name}")
        t0 = time.time()
        env = make_env()
        pol = registry[name]()
        rows = rollout(env, pol, seeds)
        agg = aggregate(rows)
        agg['display'] = pol.display
        agg['seconds'] = round(time.time() - t0, 1)
        results[name] = agg
        print(f"  {name:<16} reward={agg['mean_reward']:9.2f} "
              f"(sem {agg['sem_reward']:6.2f})  "
              f"soc={agg['mean_social_speed']:.3f}  "
              f"em={agg['mean_emergency_speed']:.3f}  "
              f"succ={agg['success_rate']:.3f}  "
              f"open={agg['mean_open_rate']:.3f}  "
              f"lane2={agg['mean_lane2_frac']:.4f}  "
              f"({agg['seconds']}s)", flush=True)

    payload = {
        'meta': {
            'n_episodes': len(seeds),
            'seeds': seeds,
            'model': args.model,
            'paired': True,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'env': {'n_cars_range': list(CA_CONFIG['n_cars_range']),
                    'P_emergency_range': list(CA_CONFIG['P_emergency_range']),
                    'max_steps': CA_CONFIG['max_steps'],
                    'warm_up_steps': CA_CONFIG['warm_up_steps'],
                    'decision_interval': ENV_CONFIG['decision_interval']},
        },
        'results': results,
    }
    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"[full] written: {args.out}")
    return payload


def mode_landscape(args):
    """Delta = mean reward(all open) - mean reward(all close) at fixed density."""
    seeds = _seed_list(args.seeds, args.seed_start)
    densities = [int(x) for x in args.densities.split(',')]
    if args.shard:
        k, of = args.shard
        densities = densities[k::of]
    out = {}
    for n in densities:
        t0 = time.time()
        rec = {}
        for name, pol in (('open', AlwaysOpen()), ('close', AlwaysClose())):
            env = make_env(n_cars_range=(n, n))
            rows = rollout(env, pol, seeds)
            rec[name] = aggregate(rows)
        delta = rec['open']['mean_reward'] - rec['close']['mean_reward']
        out[str(n)] = {
            'n_cars': n,
            'open_reward': rec['open']['mean_reward'],
            'close_reward': rec['close']['mean_reward'],
            'delta': delta,
            'optimal': 'OPEN' if delta > 0 else 'CLOSE',
            'open_social': rec['open']['mean_social_speed'],
            'close_social': rec['close']['mean_social_speed'],
            'open_em': rec['open']['mean_emergency_speed'],
            'close_em': rec['close']['mean_emergency_speed'],
            'n_episodes': len(seeds),
            'seconds': round(time.time() - t0, 1),
        }
        print(f"  n_cars={n:4d}  open={rec['open']['mean_reward']:8.1f}  "
              f"close={rec['close']['mean_reward']:8.1f}  "
              f"delta={delta:+9.1f}  -> {out[str(n)]['optimal']}  "
              f"({out[str(n)]['seconds']}s)", flush=True)

    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"[landscape] written: {args.out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, default_seeds=24):
        sp.add_argument('--seeds', type=int, default=default_seeds,
                        help=f'配对场景数（默认 {default_seeds}）；'
                             f'正式对比建议 >=160，筛查用 24 即可')
        sp.add_argument('--seed-start', type=int, default=1000)
        sp.add_argument('--out', required=True)
        sp.add_argument('--shard', type=lambda s: tuple(int(x) for x in s.split('/')),
                        default=None, help='k/of sharding, e.g. 0/4')

    s1 = sub.add_parser('screen')
    common(s1, default_seeds=24)
    s1.add_argument('--model-dir', default=os.path.join(BASE, 'models_matd3'))
    s1.add_argument('--mode', choices=['argmax', 'sample'], default='argmax')
    s1.add_argument('--exclude-resume', action='store_true', default=True)
    s1.add_argument('--only', default=None)

    s2 = sub.add_parser('full')
    common(s2, default_seeds=160)
    s2.add_argument('--model', default=os.path.join(BASE, 'models_matd3', 'maddpg_best.pth'))
    s2.add_argument('--policies',
                    default='MADDPG_argmax,MADDPG_sample,AlwaysOpen,AlwaysClose,'
                            'Random,Periodic,Greedy,OracleDensity')
    s2.add_argument('--oracle-threshold', type=int, default=500)

    s3 = sub.add_parser('landscape')
    common(s3, default_seeds=24)
    s3.add_argument('--densities', default='450,480,520,560,600')

    args = p.parse_args(argv)
    if args.cmd == 'screen':
        mode_screen(args)
    elif args.cmd == 'full':
        mode_full(args)
    else:
        mode_landscape(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

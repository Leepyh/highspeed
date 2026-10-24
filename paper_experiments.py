"""Reproducible, paired traffic evaluation for the 1014 manuscript.

No model is trained here. Missing learning checkpoints are reported, never
replaced with synthetic results. Run ``python paper_experiments.py --help``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from config_maddpg import CA_CONFIG, ENV_CONFIG
from env_maddpg import MultiAgentEmergencyLaneEnv
from experiment_paths import resolve_input_path, missing_checkpoints_message


def threshold_policy(open_density=0.26, close_density=0.22, approach=80):
    """Local social density with hysteresis and emergency approach override.

    This baseline uses the same local density and emergency position sources as
    the environment observation. The parameters must be chosen on validation
    seeds before a formal test comparison.
    """
    state = None

    def act(env, _obs, _step):
        nonlocal state
        if state is None:
            state = np.zeros(env.n_agents, dtype=int)
        result = state.copy()
        ca = env.ca
        for seg in range(env.n_agents):
            start = seg * env.segment_length
            end = start + env.segment_length
            positions = ca.pos[:ca.n_cars]
            density = np.count_nonzero((positions >= start) & (positions < end)) / (2 * env.segment_length)
            if density >= open_density:
                result[seg] = 1
            elif density <= close_density:
                result[seg] = 0
            for vid in ca.emergency_ids:
                forward = (start - int(ca.pos[vid])) % ca.L
                if start <= ca.pos[vid] < end or forward <= approach:
                    result[seg] = 0
        state = result
        return result

    return act


def make_env(n_cars, warmup, window, interval, observation_range=(350, 850), segments=5):
    ca = dict(CA_CONFIG)
    ca.update(n_cars_range=tuple(observation_range), n_cars_choices=[n_cars],
              P_emergency_range=(1.0, 1.0),
              max_emergency=1, warm_up_steps=warmup, max_steps=window,
              inject_emergency_after_warmup=True)
    cfg = dict(ENV_CONFIG)
    cfg['decision_interval'] = interval
    cfg['shared_team_reward'] = True
    cfg['n_segments'] = cfg['n_agents'] = segments
    env = MultiAgentEmergencyLaneEnv(ca_config=ca, env_config=cfg)
    env.trace_steps = True
    return env


def check_model_protocol(path, env):
    """Reject a checkpoint whose training traffic protocol cannot be verified."""
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint.get('config', {}).get('experiment_protocol')
    if not saved:
        raise ValueError('Checkpoint lacks experiment_protocol; its training conditions cannot be verified')
    expected = {'warm_up_steps': env.ca_config['warm_up_steps'],
                'max_steps': env.ca_config['max_steps'],
                'decision_interval': env.env_config['decision_interval'],
                'max_emergency': env.ca_config['max_emergency'],
                'inject_emergency_after_warmup': True,
                'shared_team_reward': env.env_config['shared_team_reward'],
                'n_segments': env.n_segments,
                'n_cars_range': list(env.ca_config['n_cars_range']),
                'P_emergency_range': list(env.ca_config['P_emergency_range'])}
    mismatches = {key: (saved.get(key), value) for key, value in expected.items()
                  if saved.get(key) != value}
    if mismatches:
        raise ValueError(f'Checkpoint training protocol differs from evaluation: {mismatches}')
    return saved


def check_baseline_protocol(saved, env):
    ca, control = saved['protocol']['ca'], saved['protocol']['env']
    expected = [('warm_up_steps', ca['warm_up_steps'], env.ca_config['warm_up_steps']),
                ('max_steps', ca['max_steps'], env.ca_config['max_steps']),
                ('max_emergency', ca['max_emergency'], env.ca_config['max_emergency']),
                ('inject_emergency_after_warmup',
                 ca.get('inject_emergency_after_warmup', False), True),
                ('shared_team_reward', control.get('shared_team_reward', False),
                 env.env_config['shared_team_reward']),
                ('P_emergency_range', list(ca['P_emergency_range']),
                 list(env.ca_config['P_emergency_range'])),
                ('n_cars_range', list(ca['n_cars_range']), list(env.ca_config['n_cars_range'])),
                ('decision_interval', control['decision_interval'], env.env_config['decision_interval']),
                ('n_segments', control['n_segments'], env.n_segments)]
    differences = {key: (trained, evaluated) for key, trained, evaluated in expected
                   if trained != evaluated}
    if differences:
        raise ValueError(f'Baseline training protocol differs from evaluation: {differences}')


def rollout(env, policy, seed, target_distance, keep_trace=False):
    random.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    env.ca.record_lane_events = True
    initial = {'social_positions': env.ca.pos[:env.ca.n_cars].tolist(),
               'social_lanes': env.ca.lane[:env.ca.n_cars].tolist(),
               'social_speeds': env.ca.vel[:env.ca.n_cars].tolist(),
               'emergency_position': int(env.ca.pos[env.ca.emergency_ids[0]]),
               'emergency_injection_step': 0,
               'emergency_initial_speed': int(env.ca.vel[env.ca.emergency_ids[0]])}
    done = False
    decision = 0
    episode_reward = 0.0
    while not done:
        py_state, np_state = random.getstate(), np.random.get_state()
        actions = np.asarray(policy(env, obs, decision), dtype=int)
        random.setstate(py_state)
        np.random.set_state(np_state)
        obs, rewards, done, _ = env.step(actions)
        episode_reward += float(np.mean(rewards))
        decision += 1

    trace = env.step_trace
    distance = 0
    travel_time = None
    for row in trace:
        distance += row['emergency_speed']
        row['emergency_cumulative_distance'] = distance
        if travel_time is None and distance >= target_distance:
            travel_time = row['step']
    events = env.ca.lane_events
    changes = Counter(e['reason'] for e in events if e['vehicle_type'] == 'social')
    counts = Counter(e['step'] for e in events if e['vehicle_type'] == 'social')
    for row in trace:
        row['lane_changes'] = counts[row['step']]
    social = [e for e in events if e['vehicle_type'] == 'social']
    result = {
        'seed': seed, 'n_cars': env.ca.n_cars, 'emergency_count': env.ca.n_emergency,
        'validation_reward': episode_reward,
        'social_speed': float(np.mean([r['social_speed'] for r in trace])),
        'social_speed_variance': float(np.mean([r['social_speed_variance'] for r in trace])),
        'emergency_travel_time': travel_time,
        'emergency_completed': travel_time is not None,
        'emergency_distance_at_window': distance,
        'lane_changes': len(social),
        'lane_changes_normalized': len(social) * 10000 / (env.ca.n_cars * len(trace)),
        'lane_change_reasons': dict(changes),
    }
    if keep_trace:
        result.update(initial_state=initial, trace=trace, lane_events=events)
    return result


def summarize(rows):
    output = []
    for density in sorted({r['n_cars'] for r in rows}):
        for strategy in sorted({r['strategy'] for r in rows}):
            group = [r for r in rows if r['n_cars'] == density and r['strategy'] == strategy]
            if not group:
                continue
            item = {'n_cars': density, 'strategy': strategy, 'n': len(group),
                    'emergency_completed': sum(r['emergency_completed'] for r in group)}
            for key in ('social_speed', 'social_speed_variance', 'lane_changes_normalized',
                        'emergency_travel_time'):
                values = [r[key] for r in group if r[key] is not None]
                complete = key != 'emergency_travel_time' or len(values) == len(group)
                item[key + '_mean'] = float(np.mean(values)) if values and complete else None
                item[key + '_sd'] = (float(np.std(values, ddof=1))
                                     if len(values) > 1 and complete else None)
            output.append(item)
    return output


def paired_differences(rows, reference='AlwaysClose'):
    by_key = {(r['n_cars'], r['seed'], r['strategy']): r for r in rows}
    out = []
    for r in rows:
        if r['strategy'] == reference:
            continue
        base = by_key.get((r['n_cars'], r['seed'], reference))
        if base is None:
            continue
        item = {'n_cars': r['n_cars'], 'seed': r['seed'], 'strategy': r['strategy'],
                'reference': reference}
        for key in ('social_speed', 'social_speed_variance', 'emergency_travel_time'):
            item[key + '_difference'] = (r[key] - base[key]
                                          if r[key] is not None and base[key] is not None else None)
        out.append(item)
    return out


def summarize_differences(rows):
    """Paired scenario intervals; censored travel times remain missing."""
    out = []
    for density in sorted({r['n_cars'] for r in rows}):
        for strategy in sorted({r['strategy'] for r in rows}):
            group = [r for r in rows if r['n_cars'] == density and r['strategy'] == strategy]
            for metric in ('social_speed', 'social_speed_variance', 'emergency_travel_time'):
                values = np.array([r[metric + '_difference'] for r in group
                                   if r[metric + '_difference'] is not None], dtype=float)
                n = len(values)
                complete = metric != 'emergency_travel_time' or n == len(group)
                mean = float(np.mean(values)) if n and complete else None
                sd = float(np.std(values, ddof=1)) if n > 1 and complete else None
                # Student t interval; scipy is optional for this project.
                try:
                    from scipy.stats import t
                    critical = float(t.ppf(0.975, n - 1)) if n > 1 else None
                except ImportError:
                    table = [None, 12.706, 4.303, 3.182, 2.776, 2.571,
                             2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
                             2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
                             2.101, 2.093, 2.086, 2.080, 2.074, 2.069,
                             2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
                    critical = table[n - 1] if 1 < n <= 31 else (1.96 if n > 31 else None)
                half = critical * sd / np.sqrt(n) if critical is not None and sd is not None else None
                out.append({'n_cars': density, 'strategy': strategy, 'metric': metric,
                            'n_complete_pairs': n, 'mean_difference': mean, 'sd_difference': sd,
                            'ci95_low': mean - half if half is not None else None,
                            'ci95_high': mean + half if half is not None else None})
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--densities', nargs='+', type=int, default=[350, 520, 850])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1000, 1010)))
    parser.add_argument('--warmup', type=int, default=150)
    parser.add_argument('--window', type=int, default=400)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--segments', type=int, default=5)
    parser.add_argument('--target-distance', type=int, default=500)
    parser.add_argument('--observation-density-range', nargs=2, type=int,
                        default=[350, 850], metavar=('LOW', 'HIGH'),
                        help='Training density range used to normalize observations')
    parser.add_argument('--open-density', type=float, default=0.26)
    parser.add_argument('--close-density', type=float, default=0.22)
    parser.add_argument('--approach-distance', type=int, default=80)
    parser.add_argument('--threshold-config', type=Path,
                        help='JSON calibrated on independent validation seeds')
    parser.add_argument('--model', type=Path, help='Current-code MADDPG checkpoint; absent models are never inferred')
    for method in ('idqn', 'qmix', 'mappo'):
        parser.add_argument(f'--{method}-model', type=Path,
                            help=f'Validated {method.upper()} checkpoint')
    parser.add_argument('--out', type=Path, default=Path('paper_results'))
    parser.add_argument('--check-only', action='store_true',
                        help='Check input file paths only; do not run simulation or training')
    args = parser.parse_args()
    missing = []
    for key in ('model', 'idqn_model', 'qmix_model', 'mappo_model'):
        value = getattr(args, key)
        if value is not None:
            path = resolve_input_path(value, checkpoint=True)
            setattr(args, key, path)
            if not path.is_file():
                missing.append(('--' + key.replace('_', '-'), path))
            elif path != value.resolve():
                print(f'Resolved --{key.replace("_", "-")}: {path}')
    if missing:
        parser.error(missing_checkpoints_message(missing))
    if args.threshold_config:
        args.threshold_config = resolve_input_path(args.threshold_config)
        if not args.threshold_config.is_file():
            parser.error(f'Threshold configuration not found: {args.threshold_config}')
        with args.threshold_config.open(encoding='utf-8') as handle:
            calibrated = json.load(handle)
        args.open_density = calibrated['open_density']
        args.close_density = calibrated['close_density']
        args.approach_distance = calibrated['approach_distance']
    if any(n <= 0 or n + 1 > CA_CONFIG['L'] for n in args.densities):
        parser.error('Each density must fit the road with one emergency vehicle')
    if args.observation_density_range[0] > args.observation_density_range[1]:
        parser.error('Observation density range must be ascending')
    if args.window <= 0 or args.interval <= 0 or args.warmup < 0 or args.target_distance <= 0:
        parser.error('Window, interval and target distance must be positive; warmup nonnegative')
    if args.window % args.interval:
        parser.error('Window must be divisible by interval for a common observation window')
    if args.segments <= 0 or CA_CONFIG['L'] % args.segments:
        parser.error('Segments must be positive and divide road length')
    if not 0 <= args.close_density < args.open_density <= 1 or args.approach_distance < 0:
        parser.error('Require 0 <= close density < open density <= 1 and nonnegative approach')
    if args.check_only:
        print('Input paths and arguments OK. Checkpoint contents have not been validated. '
              'No simulation or training started.')
        if not any(getattr(args, key) for key in ('model', 'idqn_model', 'qmix_model', 'mappo_model')):
            print('No learning checkpoints requested: evaluation will contain three rule policies only.')
        return
    args.out.mkdir(parents=True, exist_ok=True)
    records, detailed = [], []
    for density in args.densities:
        env = make_env(density, args.warmup, args.window, args.interval,
                       args.observation_density_range, args.segments)
        policies = {'AlwaysClose': lambda e, o, t: np.zeros(e.n_agents, dtype=int),
                    'AlwaysOpen': lambda e, o, t: np.ones(e.n_agents, dtype=int)}
        if args.model:
            if not args.model.is_file():
                parser.error(f'Model not found: {args.model}')
            try:
                check_model_protocol(args.model, env)
            except (ValueError, KeyError) as exc:
                parser.error(str(exc))
            from evaluate import create_maddpg_agent, make_policy_maddpg
            agent = create_maddpg_agent(env, str(args.model))
            if agent is None:
                parser.error(f'Cannot load model: {args.model}')
            policies['MADDPG'] = make_policy_maddpg(agent)
        for method in ('idqn', 'qmix', 'mappo'):
            path = getattr(args, method + '_model')
            if path:
                if not path.is_file():
                    parser.error(f'Model not found: {path}')
                from learning_baselines import load_checkpoint
                baseline, saved = load_checkpoint(path)
                if saved['algorithm'] != method:
                    parser.error(f'{path} contains {saved["algorithm"]}, expected {method}')
                try:
                    check_baseline_protocol(saved, env)
                except (ValueError, KeyError) as exc:
                    parser.error(str(exc))
                policies[method.upper()] = (
                    lambda e, o, t, agent=baseline: agent.act(o, deterministic=True))
        for seed in args.seeds:
            for name in [*policies, 'DensityEmergencyThreshold']:
                policy = (threshold_policy(args.open_density, args.close_density,
                                           args.approach_distance)
                          if name == 'DensityEmergencyThreshold' else policies[name])
                row = rollout(env, policy, seed, args.target_distance, keep_trace=True)
                detail = {k: row.pop(k) for k in ('initial_state', 'trace', 'lane_events')}
                row['strategy'] = name
                records.append(row)
                detailed.append({'n_cars': density, 'seed': seed, 'strategy': name, **detail})
                print(f'{density} cars, seed {seed}, {name}: '
                      f'speed={row["social_speed"]:.3f}, completed={row["emergency_completed"]}', flush=True)
    write_csv(args.out / 'episodes.csv', records)
    write_csv(args.out / 'summary.csv', summarize(records))
    differences = paired_differences(records)
    write_csv(args.out / 'paired_differences.csv', differences)
    write_csv(args.out / 'paired_summary.csv', summarize_differences(differences))
    with (args.out / 'traces.json').open('w', encoding='utf-8') as handle:
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}
        json.dump({'settings': settings, 'episodes': detailed},
                  handle, ensure_ascii=False)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""检查点筛查与挑选流水线（2026-10-01 新增，**这是本次最重要的工具**）

为什么需要它
------------
实测（analysis/训练策略对比与改进.md §7）：**同一训练过程内部，不同检查点的
argmax 策略质量散落在 2920–3382 之间，波动 ±400**，远大于训练策略改动带来的差异。
按旧的"训练奖励 / 单回合最高分"口径会挑到差的那一个：

    runA 若按训练奖励挑 -> ep800（实测 2923，等于恒关闭）
    runA 按 argmax 筛查 -> ep200（实测 3382）
    **差距 +459，比任何调参收益都大。**

所以"挑检查点"必须变成固定流程，而不是靠"最后一版"或"训练日志里的 Avg"。

它做什么
--------
1. 找出目录下所有检查点（自动排除 resume / 已筛过的 BEST_* 副本）；
2. 用**同一批配对场景**跑**确定性 argmax** 策略（`eval_paired` 的同一套口径），
   因此不同检查点之间也是**配对可比**的；
3. 按平均奖励排序，打印表格，标出最优；
4. 可选：与一个参考模型（例如旧最优）做配对显著性检验 —— 直接回答"新的到底有没有更好"；
5. 可选：把最优检查点复制成一个稳定文件名，便于后续评估。

用法
----
    # 单个种子
    python tools/screen_run.py --run-dir models_v4/seed0 --seeds 24 --out analysis/screen_v4_s0.json

    # 多随机种子：一条命令筛查全部 seed*/
    python tools/screen_run.py --root models_v4 --seeds 24 --out analysis/screen_v4.json

    # 与旧最优对比（配对，同一批场景）
    python tools/screen_run.py --root models_v4 --seeds 40 --reference models_matd3/maddpg_best.pth

    # 太慢就分片并行（4 个进程），再合并
    python tools/screen_run.py --root models_v4 --seeds 24 --shard 0/4 --out analysis/scr0.json
    python tools/screen_run.py --merge-glob "analysis/scr*.json" --out analysis/screen_v4.json

性能参考：单进程约 2.4 s/（检查点·场景）。24 个场景 x 20 个检查点 ~ 20 分钟。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(BASE, 'analysis')
for p in (BASE, HERE, ANALYSIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_paired import (make_env, load_maddpg, MaddpgPolicy,      # noqa: E402
                         rollout, aggregate)

SKIP_MARKERS = ('resume', 'best_avg', 'best_peak', 'final', 'BEST_')


def find_checkpoints(run_dir, include_all=False):
    out = []
    for f in sorted(os.listdir(run_dir)):
        if not f.endswith('.pth'):
            continue
        low = f.lower()
        if not include_all and any(m.lower() in low for m in SKIP_MARKERS):
            # best_avg/best_peak/final 默认跳过：它们的轮次与 episode_N 重复，
            # 且"best"是训练奖励口径的产物，正是我们要绕开的东西
            continue
        out.append(os.path.join(run_dir, f))
    return out


def collect_jobs(args):
    jobs = []          # (run_label, path)
    if args.run_dir:
        for p in find_checkpoints(args.run_dir, args.include_all):
            jobs.append((os.path.basename(os.path.normpath(args.run_dir)), p))
    if args.root:
        for d in sorted(os.listdir(args.root)):
            sub = os.path.join(args.root, d)
            if not os.path.isdir(sub):
                continue
            for p in find_checkpoints(sub, args.include_all):
                jobs.append((d, p))
    if args.shard:
        k, of = args.shard
        jobs = jobs[k::of]
    return jobs


def screen(jobs, seeds, reference_path, verbose=True):
    results, ref_rows = {}, None
    if reference_path:
        if not os.path.exists(reference_path):
            print(f'[screen] reference not found: {reference_path}')
        else:
            agent, meta = load_maddpg(reference_path)
            env = make_env()
            ref_rows = rollout(env, MaddpgPolicy(agent, 'argmax'), seeds)
            print(f"[screen] reference = {reference_path}  ep={meta['episode']}  "
                  f"mean reward = {np.mean([r['reward'] for r in ref_rows]):.2f}")

    for i, (run, path) in enumerate(jobs, 1):
        agent, meta = load_maddpg(path)
        env = make_env()
        rows = rollout(env, MaddpgPolicy(agent, 'argmax'), seeds)
        agg = aggregate(rows)
        key = f'{run}/{os.path.basename(path)}'
        results[key] = {'run': run, 'file': os.path.basename(path),
                        'path': path, 'episode': meta.get('episode'),
                        'mean_reward': agg['mean_reward'],
                        'sem_reward': agg['sem_reward'],
                        'mean_social_speed': agg['mean_social_speed'],
                        'mean_emergency_speed': agg['mean_emergency_speed'],
                        'mean_open_rate': agg['mean_open_rate'],
                        'mean_lane2_frac': agg['mean_lane2_frac'],
                        '_episodes': rows}
        if ref_rows is not None:
            ref = {r['seed']: r['reward'] for r in ref_rows}
            cur = {r['seed']: r['reward'] for r in rows}
            ks = sorted(set(ref) & set(cur))
            d = np.array([cur[k] - ref[k] for k in ks])
            sem = float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
            tc = 2.1 if d.size < 60 else 2.0
            results[key].update({
                'pair_diff_vs_ref': float(d.mean()) if d.size else None,
                'pair_ci95': [float(d.mean() - tc * sem), float(d.mean() + tc * sem)] if d.size else None,
                'pair_win_rate': float(np.mean(d > 0)) if d.size else None,
                'pair_significant': bool(d.size > 1 and (d.mean() - tc * sem > 0
                                                         or d.mean() + tc * sem < 0)),
            })
        if verbose:
            extra = ''
            if 'pair_diff_vs_ref' in results[key] and results[key]['pair_diff_vs_ref'] is not None:
                r = results[key]
                extra = (f"  vs_ref {r['pair_diff_vs_ref']:+7.1f} "
                         f"[{r['pair_ci95'][0]:+.0f},{r['pair_ci95'][1]:+.0f}]"
                         f"{'  SIG' if r['pair_significant'] else '     '}")
            print(f"  [{i}/{len(jobs)}] {run:<10} ep={meta.get('episode')!s:>5} "
                  f"reward={agg['mean_reward']:8.2f} (sem {agg['sem_reward']:5.1f}) "
                  f"soc={agg['mean_social_speed']:.3f} em={agg['mean_emergency_speed']:.3f} "
                  f"open={agg['mean_open_rate']:.3f} lane2={agg['mean_lane2_frac']:.4f}{extra}",
                  flush=True)
    return results


def report(results, reference_path):
    if not results:
        print('[screen] nothing screened')
        return None
    order = sorted(results.items(), key=lambda kv: -kv[1]['mean_reward'])
    print('\n' + '=' * 104)
    print('检查点筛查结果（按平均奖励降序；配对 argmax 评估）')
    print('=' * 104)
    print(f"{'run/file':<44}{'ep':>6}{'reward':>10}{'vs_ref':>10}{'sig':>6}")
    print('-' * 104)
    for k, v in order:
        d = v.get('pair_diff_vs_ref')
        print(f"{k:<44}{(v['episode'] if v['episode'] is not None else -1):>6}"
              f"{v['mean_reward']:>10.2f}"
              f"{(f'{d:+.1f}' if d is not None else '-'):>10}"
              f"{('SIG' if v.get('pair_significant') else ''):>6}")
    print('=' * 104)

    best_k, best_v = order[0]
    print(f"\n★ 最优检查点: {best_k}")
    print(f"  平均奖励 {best_v['mean_reward']:.2f} (sem {best_v['sem_reward']:.1f})，"
          f"episode {best_v['episode']}")
    if reference_path and best_v.get('pair_diff_vs_ref') is not None:
        d, ci = best_v['pair_diff_vs_ref'], best_v['pair_ci95']
        verdict = '显著更优' if best_v['pair_significant'] and d > 0 else (
            '显著更差' if best_v['pair_significant'] and d < 0 else '不显著（仍需更多场景/种子）')
        print(f"  相对参考模型: {d:+.1f}  95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]  -> {verdict}")

    # 按 run 汇总（多种子时看每个种子的最好成绩）
    runs = {}
    for k, v in results.items():
        runs.setdefault(v['run'], []).append(v)
    if len(runs) > 1:
        print('\n各 run（种子）的最好成绩：')
        for r, vs in sorted(runs.items()):
            b = max(vs, key=lambda x: x['mean_reward'])
            print(f"  {r:<12} best ep={b['episode']!s:>5}  reward={b['mean_reward']:8.2f}"
                  f"   ({len(vs)} 个检查点)")
    return best_v


def main(argv=None):
    ap = argparse.ArgumentParser(description='检查点筛查与挑选')
    ap.add_argument('--run-dir', default=None, help='单个 run 目录')
    ap.add_argument('--root', default=None, help='多个种子目录的父目录（筛查其下每个子目录）')
    ap.add_argument('--seeds', type=int, default=24, help='配对场景数（筛查用，默认 24）')
    ap.add_argument('--seed-start', type=int, default=1000)
    ap.add_argument('--reference', default=None, help='参考模型路径（做配对显著性对比）')
    ap.add_argument('--include-all', action='store_true',
                    help='连 best_avg/best_peak/final 一起筛查')
    ap.add_argument('--shard', type=lambda s: tuple(int(x) for x in s.split('/')), default=None)
    ap.add_argument('--merge-glob', default=None, help='合并多个分片结果')
    ap.add_argument('--copy-best', default=None, help='把最优检查点复制到该路径')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    if args.merge_glob:
        merged = {}
        for fp in sorted(glob.glob(args.merge_glob)):
            with open(fp, encoding='utf-8') as fh:
                merged.update(json.load(fh)['results'])
        payload = {'seeds': None, 'reference': None, 'results': merged}
        print(f'[screen] merged {len(merged)} entries from {args.merge_glob}')
    else:
        jobs = collect_jobs(args)
        if not jobs:
            print('[screen] no checkpoints found - check --run-dir / --root')
            return 1
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        print(f'[screen] {len(jobs)} checkpoints x {len(seeds)} paired scenarios')
        results = screen(jobs, seeds, args.reference)
        payload = {'seeds': seeds, 'reference': args.reference,
                   'seed_start': args.seed_start, 'results': results}

    best = report(payload['results'], args.reference)

    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f'\n[screen] written: {args.out}')

    if args.copy_best and best:
        import shutil
        dst = args.copy_best
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        print(f'[screen] PRE : dst exists={os.path.exists(dst)}')
        shutil.copyfile(best['path'], dst)
        print(f'[screen] copied best -> {dst}')
        print(f'[screen] POST: dst exists={os.path.exists(dst)} '
              f'size={os.path.getsize(dst)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

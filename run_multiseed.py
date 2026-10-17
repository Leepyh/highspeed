#!/usr/bin/env python
"""多随机种子训练启动器（2026-10-01 新增）

为什么需要它
------------
实测发现：**同一训练过程不同检查点之间的 argmax 策略质量波动达 ±400**，
远大于训练策略改动带来的差异。单次训练的曲线因此不足以支撑任何结论。
参考实现普遍跑 5–10 个种子取均值（见 analysis/训练策略对比与改进.md §5）。

本脚本把"跑 N 个种子"变成一条命令，每个种子写到独立目录，互不干扰。

用法
----
    # 先看会执行什么命令，不真的跑
    python run_multiseed.py --n-seeds 5 --episodes 500 --dry-run

    # 5 个种子，并行 3 个进程
    python run_multiseed.py --n-seeds 5 --episodes 500 --parallel 3 --root ./models_v5 --log-root ./logs_v5

    # 关掉 twin critic 做 A/B
    python run_multiseed.py --n-seeds 5 --episodes 500 --parallel 3 --no-twin-critics ^
        --root ./models_v5_notwin --log-root ./logs_v5_notwin

健壮性（2026-10-02 修复）
------------------------
1. **开跑前先校验参数**：先执行 `run_training.py --help`，确认要传的每个开关都被接受，
   有任何一个不认识就立刻停下并打印缺失项 —— 不会再出现"5 个种子全部 2 秒退出、
   只留下一个 returncode=2"的情况。
2. **每个子进程的控制台输出落盘**到 `<log_dir>/run_console.log`；
   失败时直接把最后几行打出来，不用再去翻 manifest。
3. 每个种子的命令都带 `--no-resume`，保证是全新训练，不会误续到别的实验。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_TRAINING = os.path.join(HERE, 'run_training.py')

# 本脚本可能传给 run_training.py 的全部开关（用于开跑前校验）
REQUIRED_FLAGS = ['--episodes', '--seed', '--no-resume', '--model-dir', '--log-dir']
OPTIONAL_FLAGS = ['--twin-critics', '--no-twin-critics', '--actor-update-mode',
                  '--target-action-mode', '--entropy-coef', '--entropy-coef-end',
                  '--entropy-anneal-episodes']
KNOWN_FLAGS = REQUIRED_FLAGS + OPTIONAL_FLAGS


def supported_flags():
    """执行 run_training.py --help，解析出它认识的开关集合。"""
    r = subprocess.run([sys.executable, RUN_TRAINING, '--help'],
                       capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    text = (r.stdout or '') + (r.stderr or '')
    return set(re.findall(r'(--[a-zA-Z][a-zA-Z0-9-]*)', text)), text


def preflight(verbose=True):
    """开跑前校验。返回 (ok, 缺失的开关列表, help 文本)。"""
    have, text = supported_flags()
    missing = [f for f in KNOWN_FLAGS if f not in have]
    if verbose:
        print(f'[preflight] run_training.py 认识 {len(have)} 个开关；'
              f'本脚本需要 {len(KNOWN_FLAGS)} 个')
        if missing:
            print(f'[preflight] ✗ 缺失: {missing}')
        else:
            print('[preflight] ✓ 全部开关都被接受')
    return (not missing), missing, text


def build_cmd(seed, args, model_dir, log_dir):
    cmd = [sys.executable, RUN_TRAINING,
           '--episodes', str(args.episodes),
           '--seed', str(seed),
           '--no-resume',
           '--model-dir', model_dir,
           '--log-dir', log_dir]
    if args.twin_critics is not None:
        cmd.append('--twin-critics' if args.twin_critics else '--no-twin-critics')
    if args.actor_update_mode:
        cmd += ['--actor-update-mode', args.actor_update_mode]
    if args.target_action_mode:
        cmd += ['--target-action-mode', args.target_action_mode]
    if args.entropy_coef is not None:
        cmd += ['--entropy-coef', str(args.entropy_coef)]
    if args.entropy_coef_end is not None:
        cmd += ['--entropy-coef-end', str(args.entropy_coef_end)]
    if args.entropy_anneal_episodes is not None:
        cmd += ['--entropy-anneal-episodes', str(args.entropy_anneal_episodes)]
    if args.extra:
        cmd += args.extra.split()
    return cmd


def tail(path, n=12):
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            lines = fh.read().splitlines()
        return '\n'.join(lines[-n:])
    except OSError:
        return '(no console log)'


def main(argv=None):
    p = argparse.ArgumentParser(description='多随机种子训练启动器')
    p.add_argument('--n-seeds', type=int, default=5, help='训练几个种子（用 0..n-1）')
    p.add_argument('--seeds', default=None, help='显式指定种子，如 "0,1,2"（覆盖 --n-seeds）')
    p.add_argument('--episodes', type=int, default=500, help='每个种子的训练轮数')
    p.add_argument('--root', default='./models_v5', help='模型根目录（每个种子一个子目录）')
    p.add_argument('--log-root', default='./logs_v5', help='日志根目录')
    p.add_argument('--parallel', type=int, default=1, help='同时跑几个进程')
    p.add_argument('--twin-critics', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='是否启用 twin critic（默认启用；--no-twin-critics 关闭做 A/B）')
    p.add_argument('--actor-update-mode', default=None,
                   choices=['counterfactual', 'legacy'])
    p.add_argument('--target-action-mode', default=None, choices=['greedy', 'soft'])
    p.add_argument('--entropy-coef', type=float, default=None)
    p.add_argument('--entropy-coef-end', type=float, default=None)
    p.add_argument('--entropy-anneal-episodes', type=int, default=None)
    p.add_argument('--extra', default=None, help='额外参数原样追加（整串）')
    p.add_argument('--dry-run', action='store_true', help='只打印命令不执行')
    args = p.parse_args(argv)

    seeds = ([int(x) for x in args.seeds.split(',')] if args.seeds
             else list(range(args.n_seeds)))

    jobs = []
    for s in seeds:
        md = os.path.join(args.root, f'seed{s}')
        ld = os.path.join(args.log_root, f'seed{s}')
        jobs.append({'seed': s, 'model_dir': md, 'log_dir': ld,
                     'cmd': build_cmd(s, args, md, ld)})

    print('=' * 78)
    print(f'多随机种子训练：{len(seeds)} 个种子 x {args.episodes} 轮，并行 {args.parallel}')
    print('=' * 78)
    ok, missing, help_text = preflight()
    if not ok:
        print('\n[preflight] 参数校验未通过，已终止（不会启动任何训练）。')
        print('  原因：run_training.py 不接受这些开关 ' + str(missing))
        print('  说明：通常是 run_training.py 的 argparse 与 run_multiseed.py 不同步。')
        print('\n--- run_training.py --help ---')
        print(help_text)
        return 2
    print()
    for j in jobs:
        print(f"  seed {j['seed']}: {' '.join(j['cmd'])}")
    if args.dry_run:
        print('\n[dry-run] 未执行任何命令。')
        return 0

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)

    t0 = time.time()
    running, pending, done = [], list(jobs), []
    while pending or running:
        while pending and len(running) < max(args.parallel, 1):
            j = pending.pop(0)
            os.makedirs(j['model_dir'], exist_ok=True)
            os.makedirs(j['log_dir'], exist_ok=True)
            j['console'] = os.path.join(j['log_dir'], 'run_console.log')
            j['fh'] = open(j['console'], 'w', encoding='utf-8', errors='replace')
            print(f"[start ] seed {j['seed']}", flush=True)
            j['proc'] = subprocess.Popen(j['cmd'], cwd=HERE,
                                         stdout=j['fh'], stderr=subprocess.STDOUT)
            j['t0'] = time.time()
            running.append(j)
        time.sleep(2)
        for j in list(running):
            rc = j['proc'].poll()
            if rc is not None:
                j['returncode'] = rc
                j['seconds'] = round(time.time() - j['t0'], 1)
                try:
                    j['fh'].close()
                except OSError:
                    pass
                running.remove(j)
                done.append(j)
                flag = 'ok ' if rc == 0 else 'FAIL'
                print(f"[{flag} ] seed {j['seed']}  rc={rc}  {j['seconds']}s", flush=True)
                if rc != 0:
                    print(f"         console: {j['console']}")
                    print('         --- 最后几行 ---')
                    for line in tail(j['console']).splitlines():
                        print('         ' + line)
                    print('         ----------------')

    print('\n' + '=' * 78)
    print(f'全部结束，用时 {time.time() - t0:.0f}s')
    print('=' * 78)
    for j in done:
        print(f"  seed {j['seed']:<3} rc={j['returncode']:<3} {j['seconds']:>8}s   {j['model_dir']}")
    failed = [j for j in done if j['returncode'] != 0]
    if failed:
        print(f'\n⚠ {len(failed)} 个种子失败。上面已打印每个失败种子的控制台尾部；')
        print('  完整输出在各 seed 目录下的 run_console.log。')

    manifest = {'episodes': args.episodes, 'parallel': args.parallel,
                'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'jobs': [{k: v for k, v in j.items() if k not in ('proc', 'fh')}
                         for j in done]}
    mp = os.path.join(args.root, 'multiseed_manifest.json')
    with open(mp, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(f'\nmanifest -> {mp}')
    if not failed:
        print('\n下一步：筛查每个种子的检查点，挑出真正的 best：')
        print(f'  .venv\\Scripts\\python.exe tools\\screen_run.py --root "{args.root}" --seeds 24 '
              f'--reference models_matd3\\maddpg_best.pth --out analysis\\screen_v5.json '
              f'--copy-best "{args.root}\\BEST.pth"')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())

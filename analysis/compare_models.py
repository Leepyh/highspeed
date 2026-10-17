#!/usr/bin/env python
"""多个已训练检查点的横向对比（配对种子 + 确定性 argmax + 整回合平均）

与 `analysis/eval_paired.py` 同一套口径：所有模型跑在**完全相同的交通场景**上，
并额外汇总两个平凡锚点（AlwaysClose / AlwaysOpen），让数字有参照。

    # 常规用法（默认 160 个配对场景；单进程约 2.4 s/模型·场景，7 个模型约 45 分钟）
    python analysis/compare_models.py \
        --models "旧最优=models_matd3/maddpg_best.pth,新模型=models_v4/seed0/BEST.pth" \
        --seeds 160 --seed-start 9000 \
        --out analysis/model_compare.json --figure analysis/figures/fig6.png

    # 太慢就分片（按模型切分），并行跑完再合并
    python analysis/compare_models.py --models "A=...,B=..." --shard 0/2 --out analysis/cmp0.json
    python analysis/compare_models.py --models "A=...,B=..." --shard 1/2 --out analysis/cmp1.json
    python analysis/compare_models.py --merge-glob "analysis/cmp?.json" \
        --out analysis/model_compare.json --figure analysis/figures/fig6.png

统计口径：每个模型与**参考模型**逐场景配对求差（共享初始交通场景，场景方差被抵消），
报告配对差的均值、95% CI 与胜率。CI 不跨 0 才算显著。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_paired import (make_env, load_maddpg, MaddpgPolicy,          # noqa: E402
                         AlwaysOpen, AlwaysClose, rollout, aggregate)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def tcrit(n):
    """two-sided 95% t critical value (small table, no scipy needed)."""
    t = {1: 12.706, 2: 4.303, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042,
         40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980}
    df = max(n - 1, 1)
    ks = sorted(t)
    if df <= ks[0]:
        return t[ks[0]]
    if df >= ks[-1]:
        return 1.96
    lo = max(k for k in ks if k <= df)
    hi = min(k for k in ks if k >= df)
    if lo == hi:
        return t[lo]
    return t[lo] + (df - lo) / (hi - lo) * (t[hi] - t[lo])


def parse_models(spec):
    out = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        label, path = item.split('=', 1) if '=' in item else (os.path.basename(item), item)
        if not os.path.isabs(path):
            path = os.path.join(BASE, path)
        out.append((label.strip(), path.strip()))
    return out


def evaluate(models, seeds, with_anchors=True):
    results = {}
    for label, path in models:
        if not os.path.exists(path):
            print(f'[compare] SKIP (not found): {label} -> {path}')
            continue
        agent, meta = load_maddpg(path)
        agg = aggregate(rollout(make_env(), MaddpgPolicy(agent, 'argmax'), seeds))
        agg.update(meta=meta, path=path, label=label)
        results[label] = agg
        print(f"  {label:<26} ep={meta['episode']!s:>5} "
              f"reward={agg['mean_reward']:8.2f} (sem {agg['sem_reward']:5.2f})  "
              f"soc={agg['mean_social_speed']:.3f} em={agg['mean_emergency_speed']:.3f} "
              f"open={agg['mean_open_rate']:.3f} lane2={agg['mean_lane2_frac']:.4f}",
              flush=True)
    if with_anchors:
        for label, pol in (('AlwaysClose (锚点)', AlwaysClose()),
                           ('AlwaysOpen (锚点)', AlwaysOpen())):
            if label in results:
                continue
            agg = aggregate(rollout(make_env(), pol, seeds))
            agg['label'] = label
            results[label] = agg
            print(f"  {label:<26} {'':>9} reward={agg['mean_reward']:8.2f} "
                  f"(sem {agg['sem_reward']:5.2f})  soc={agg['mean_social_speed']:.3f}  "
                  f"em={agg['mean_emergency_speed']:.3f}")
    return results


def paired_vs_reference(results, ref_label):
    if ref_label not in results:
        return {}
    ref = {r['seed']: r['reward'] for r in results[ref_label]['_episodes']}
    out = {}
    print('\n' + '=' * 96)
    print(f'PAIRED DIFFERENCE vs "{ref_label}"   (positive = reference better)')
    print('=' * 96)
    print(f"{'model':<28}{'mean diff':>12}{'95% CI':>24}{'win rate':>11}{'sig':>6}")
    print('-' * 96)
    for lab, rec in results.items():
        if lab == ref_label:
            continue
        cur = {r['seed']: r['reward'] for r in rec['_episodes']}
        ks = sorted(set(ref) & set(cur))
        if len(ks) < 2:
            continue
        d = np.array([ref[k] - cur[k] for k in ks])
        sem = float(np.std(d, ddof=1) / np.sqrt(d.size))
        tc = tcrit(d.size)
        lo, hi = d.mean() - tc * sem, d.mean() + tc * sem
        sig = bool(lo > 0 or hi < 0)
        out[lab] = {'mean_diff': float(d.mean()), 'sem': sem,
                    'ci95': [float(lo), float(hi)],
                    'win_rate': float(np.mean(d > 0)), 'significant': sig,
                    'n_pairs': len(ks)}
        print(f"{lab:<28}{d.mean():>+12.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>24}"
              f"{np.mean(d > 0):>10.1%}{'YES' if sig else 'no':>6}")
    print('-' * 96)
    print('YES = 95% 置信区间不跨 0（差异显著）')
    return out


def draw(results, seeds, ref_label, figure):
    names = list(results)
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2))
    panels = [('mean_reward', '① 平均回合奖励', False, 'sem_reward'),
              ('mean_social_speed', '② 社会车辆平均速度', False, None),
              ('mean_emergency_speed', '③ 应急车辆平均速度', False, None),
              ('mean_lane2_frac', '④ 应急车道承载率', True, None)]
    for ax, (field, title, pct, errkey) in zip(axes, panels):
        vals = [results[n].get(field, 0.0) for n in names]
        errs = [results[n].get(errkey, 0.0) if errkey else 0.0 for n in names]
        if pct:
            vals = [v * 100 for v in vals]
        colors = ['#7f8c8d' if '锚点' in n else
                  ('#c0392b' if n == ref_label else '#2980b9') for n in names]
        x = np.arange(len(names))
        b = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.88,
                   edgecolor='black', linewidth=1.2,
                   error_kw=dict(ecolor='#2c3e50', lw=1.2, capsize=4))
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('=', '\n') for n in names], fontsize=8.5,
                           rotation=18, ha='right')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        hi = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, hi * 1.24)
        for bar, v in zip(b, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + hi * 0.02,
                    f'{v:.3f}' if not pct else f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.suptitle(f'模型横向对比（配对 {len(seeds)} 个完全相同场景，确定性 argmax 策略；'
                 f'误差棒为 ±1 SEM；红=参考模型）', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(figure)), exist_ok=True)
    plt.savefig(figure, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[compare] figure: {figure}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='模型横向对比（配对 + 确定性 argmax）')
    ap.add_argument('--models', default=None, help='"label=path,label=path"')
    ap.add_argument('--seeds', type=int, default=160, help='配对场景数（默认 160）')
    ap.add_argument('--seed-start', type=int, default=9000)
    ap.add_argument('--reference', default=None, help='参考模型 label（默认第一个）')
    ap.add_argument('--no-anchors', dest='anchors', action='store_false')
    ap.add_argument('--shard', type=lambda s: tuple(int(x) for x in s.split('/')),
                    default=None, help='按模型切分，如 0/2')
    ap.add_argument('--merge-glob', default=None, help='合并多个分片结果后出统计与图')
    ap.add_argument('--out', required=True)
    ap.add_argument('--figure', default=None)
    a = ap.parse_args(argv)

    if a.merge_glob:
        files = sorted(glob.glob(a.merge_glob))
        if not files:
            raise SystemExit(f'no files matched {a.merge_glob}')
        merged, seeds, ref = {}, [], a.reference
        for fp in files:
            with open(fp, encoding='utf-8') as fh:
                doc = json.load(fh)
            merged.update(doc['results'])
            if not seeds:
                seeds = doc.get('seeds') or []
            ref = ref or doc.get('reference')
        print(f'[compare] merged {len(files)} shard(s) -> {len(merged)} entries, '
              f'{len(seeds)} scenarios')
        results = merged
    else:
        if not a.models:
            raise SystemExit('--models is required (or use --merge-glob)')
        models = parse_models(a.models)
        if a.shard:
            k, of = a.shard
            models = models[k::of]
        seeds = list(range(a.seed_start, a.seed_start + a.seeds))
        print(f'[compare] {len(models)} model(s) x {len(seeds)} paired scenarios'
              + (f'  (shard {a.shard[0]}/{a.shard[1]})' if a.shard else ''))
        results = evaluate(models, seeds, a.anchors)

    labels = [n for n in results if '锚点' not in n]
    ref_label = a.reference or (labels[0] if labels else None)
    stats = paired_vs_reference(results, ref_label)

    payload = {'seeds': seeds, 'seed_start': a.seed_start, 'reference': ref_label,
               'results': results, 'paired_vs_reference': stats}
    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    print(f'\n[compare] written: {a.out}')

    if a.figure and len(results) > 1:
        draw(results, seeds, ref_label, a.figure)
    return 0


if __name__ == '__main__':
    sys.exit(main())

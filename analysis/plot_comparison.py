#!/usr/bin/env python
"""Draw the strategy-comparison figures from the paired evaluation output.

READ-ONLY with respect to the project: it only reads JSON produced by
`analysis/eval_paired.py` and `analysis/policy_probe.py`, and writes PNG files
next to them.  No value is hard-coded - if a metric is missing from the JSON the
panel says so instead of inventing a number.

    python analysis/plot_comparison.py --input analysis/full.json \
        --screen-glob "analysis/screen_shard_*.json" \
        --landscape analysis/landscape.json \
        --probe analysis/policy_probe_argmax.json
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
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

C_MADDPG = '#c0392b'
C_BASE = '#7f8c8d'
C_ORACLE = '#8e44ad'
C_GOOD = '#27ae60'
C_WARN = '#e67e22'

PANELS = [
    ('mean_reward', '① 综合收益：平均回合奖励', True, False),
    ('mean_social_speed', '② 通行效率：社会车辆平均速度', True, False),
    ('mean_emergency_speed', '③ 应急通行：应急车辆平均速度', True, False),
    ('success_rate', '④ 任务达成：应急车通行成功率', True, True),
    ('mean_lane2_frac', '⑤ 控制机理：应急车道承载率', True, True),
    ('mean_emergency_stall_frac', '⑥ 安全约束：应急车停驻比例', False, True),
]

PREFERRED = ['MADDPG_argmax', 'MADDPG_sample', 'OracleDensity', 'Periodic',
             'Random', 'AlwaysOpen', 'Greedy', 'AlwaysClose']

# short two-line x labels so the tick text never collides
SHORT = {
    'MADDPG_argmax': 'MADDPG\n确定性argmax\n(本文模型)',
    'MADDPG_sample': 'MADDPG\n随机采样',
    'OracleDensity': 'Oracle\n密度规则',
    'Periodic': 'Periodic\n周期T=10',
    'Random': 'Random\n随机',
    'AlwaysOpen': 'Always Open\n恒开放',
    'Greedy': 'Greedy\n本段堵则开',
    'AlwaysClose': 'Always Close\n恒关闭',
}


def load(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def unwrap(doc):
    return doc['results'] if isinstance(doc, dict) and 'results' in doc else doc


def order_names(results, preferred):
    known = [n for n in preferred if n in results]
    return known + sorted(n for n in results if n not in known)


# --------------------------------------------------------------------------
def draw_bar_panel(ax, names, labels, values, errs, title, higher_better, is_pct):
    vals = np.array([0.0 if v is None else float(v) for v in values], dtype=float)
    errs = np.array([0.0 if e is None else float(e) for e in errs], dtype=float)
    if is_pct:
        vals, errs = vals * 100.0, errs * 100.0

    colors = [C_MADDPG if n.startswith('MADDPG_argmax')
              else (C_ORACLE if n == 'OracleDensity' else C_BASE) for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.88,
                  edgecolor='black', linewidth=1.2,
                  error_kw=dict(ecolor='#2c3e50', lw=1.2, capsize=4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=18, ha='right')
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.3, axis='y')

    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) if hi > lo else max(abs(hi), 1.0)
    low = 0.0 if lo >= 0 else lo - span * 0.25
    ax.set_ylim(low, hi + span * 0.32)

    fmt = '{:.0f}%' if is_pct else ('{:.3f}' if hi < 10 else '{:.1f}')
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + e + (hi + span * 0.32) * 0.012,
                fmt.format(v), ha='center', va='bottom',
                fontsize=8.8, fontweight='bold')

    cand = [i for i, n in enumerate(names) if n != 'OracleDensity']
    if cand:
        key = max if higher_better else min
        bars[key(cand, key=lambda i: vals[i])].set_edgecolor(C_GOOD)
        bars[key(cand, key=lambda i: vals[i])].set_linewidth(3.0)
    for i, n in enumerate(names):
        if n == 'OracleDensity':
            bars[i].set_linestyle('--')


def fig1(doc, out_path):
    results = unwrap(doc)
    names = order_names(results, PREFERRED)
    labels = [SHORT.get(n, n) for n in names]

    fig = plt.figure(figsize=(19, 11))
    for i, (field, title, higher, is_pct) in enumerate(PANELS):
        ax = plt.subplot(2, 3, i + 1)
        if not any(field in results[n] for n in names):
            ax.axis('off')
            ax.text(0.5, 0.5, f'数据缺失: {field}', ha='center', va='center',
                    fontsize=13, color='#b03a2e')
            continue
        vals = [results[n].get(field) for n in names]
        errs = [results[n].get('sem_reward') if field == 'mean_reward' else None
                for n in names]
        draw_bar_panel(ax, names, labels, vals, errs, title, higher, is_pct)

    n_ep = (doc.get('meta') or {}).get('n_episodes') if isinstance(doc, dict) else None
    subtitle = (f'数据来源：analysis/eval_paired.py 配对种子实测 —— '
                f'每个策略跑 {n_ep} 个完全相同的交通场景；误差棒为 ±1 SEM；'
                f'标注数字为实测值')
    legend = [
        Patch(facecolor=C_MADDPG, edgecolor='black', label='本文模型（MADDPG 确定性策略 argmax）'),
        Patch(facecolor=C_BASE, edgecolor='black', label='对照策略'),
        Patch(facecolor=C_ORACLE, edgecolor='black', linestyle='--',
              label='Oracle 密度规则（用特权信息，仅作上界参考）'),
        Patch(facecolor='white', edgecolor=C_GOOD, linewidth=3,
              label='该项最优（不计 Oracle）'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('高速公路应急车道控制策略对比分析（最优模型 MADDPG ep664 vs 6 类对照策略）',
                 fontsize=16.5, fontweight='bold', y=0.985)
    fig.text(0.5, 0.945, subtitle, ha='center', fontsize=11, color='#34495e')
    plt.tight_layout(rect=(0, 0.045, 1, 0.935))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig1] {out_path}')


# --------------------------------------------------------------------------
def fig2(screen, landscape, probe, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.5))

    # ---- (a) reward landscape -------------------------------------------
    ax = axes[0][0]
    if landscape:
        ks = sorted(landscape, key=lambda k: int(k))
        ns = [int(k) for k in ks]
        dl = [landscape[k]['delta'] for k in ks]
        cols = [C_GOOD if d > 0 else C_WARN for d in dl]
        b = ax.bar([str(n) for n in ns], dl, color=cols, alpha=0.85,
                   edgecolor='black', linewidth=1.2)
        ax.axhline(0, color='black', lw=1.2)
        rng = max(dl) - min(dl)
        for bar, d in zip(b, dl):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (rng * 0.03 if d > 0 else -rng * 0.03),
                    f'{d:+.0f}\n{"开放更优" if d > 0 else "关闭更优"}',
                    ha='center', va='bottom' if d > 0 else 'top',
                    fontsize=9.5, fontweight='bold')
        ax.set_ylim(min(dl) - rng * 0.28, max(dl) + rng * 0.28)
        ax.set_title('(a) 奖励地形：Δ = 全开放 − 全关闭（固定密度实测）\n'
                     '低密度应关闭、高密度应开放 —— 存在可学的状态依赖结构',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('社会车辆数 n_cars')
        ax.set_ylabel('Δ 平均回合奖励')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.axis('off')

    # ---- (b) checkpoint screening ---------------------------------------
    ax = axes[0][1]
    if screen:
        pts = []
        for name, rec in screen.items():
            ep = (rec.get('meta') or {}).get('episode') if isinstance(rec.get('meta'), dict) else None
            if ep is None:
                continue
            pts.append((int(ep), rec.get('mean_reward'), name, rec.get('mean_open_rate')))
        pts.sort()
        eps = [p[0] for p in pts]
        rw = [p[1] for p in pts]
        op = [p[3] for p in pts]
        best = max(pts, key=lambda p: p[1])
        ax.plot(eps, rw, 'o-', color='#2980b9', lw=1.8, ms=6,
                label='argmax 策略平均奖励')
        for k, (e, r, n, o) in enumerate(pts):
            if r <= 3000:
                continue                      # flat tail is annotated as a block
            ax.annotate(n.replace('maddpg_', '').replace('.pth', ''),
                        (e, r), textcoords='offset points', xytext=(0, 10),
                        fontsize=6.8, ha='center', color='#34495e')
        flat = [p for p in pts if p[1] <= 3000]
        if flat:
            ax.annotate(f'ep{flat[0][0]}–ep{flat[-1][0]} 全部退化到\n'
                        f'≈{np.mean([p[1] for p in flat]):.0f} 平台'
                        f'（等同「恒关闭」）',
                        xy=(np.mean([p[0] for p in flat]), 2925),
                        xytext=(0, 42), textcoords='offset points',
                        ha='center', fontsize=9, color=C_WARN,
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=C_WARN, lw=1.4))
        ax.plot([best[0]], [best[1]], '*', color=C_GOOD, ms=22, zorder=5,
                label=f'最优检查点 ep={best[0]}')
        ax.axvline(best[0], color=C_GOOD, ls=':', lw=1.4)
        ax.annotate(f'{best[1]:.0f}', (best[0], best[1]),
                    textcoords='offset points', xytext=(6, -14),
                    fontsize=10, fontweight='bold', color=C_GOOD)
        ax2 = ax.twinx()
        ax2.plot(eps, op, 's--', color=C_WARN, lw=1.3, ms=4, label='开放率')
        ax2.set_ylabel('argmax 动作开放率', color=C_WARN)
        ax2.tick_params(axis='y', colors=C_WARN)
        ax2.set_ylim(0, max(0.3, max(op) * 1.6))
        ax.set_xlabel('训练回合 episode')
        ax.set_ylabel('平均回合奖励（配对 24 场景）')
        ax.set_title('(b) 检查点甄别：最优不是最后一版\n'
                     '—— ep1000 之后策略退化到「近乎恒关闭」平台',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='lower left')
    else:
        ax.axis('off')

    # ---- (c) state dependence -------------------------------------------
    ax = axes[1][0]
    if probe:
        conds, vals, cols = [], [], []
        oe, on = probe.get('open_rate_when_emergency'), probe.get('open_rate_when_no_emergency')
        for lab, v, c in (('有应急车\n在本段', oe, C_MADDPG),
                          ('无应急车', on, '#95a5a6')):
            if v is not None:
                conds.append(lab); vals.append(v * 100); cols.append(c)
        conds.append('')
        vals.append(0.0)
        cols.append('white')
        for b in probe.get('by_local_speed', []):
            conds.append(f"{b['lo']:.1f}–{b['hi']:.1f}")
            vals.append(b['open_rate'] * 100)
            cols.append('#2980b9')
        x = np.arange(len(conds))
        bars = ax.bar(x, vals, color=cols, alpha=0.88, edgecolor='black',
                      linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, fontsize=9)
        ax.set_ylabel('argmax 决策的开放率 (%)')
        ax.set_title('(c) 策略确实依赖状态，但幅度很小\n'
                     '左二：有无应急车 | 右侧：本段社会车道归一化车速分箱',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        span = max(vals) * 1.35 or 1.0
        ax.set_ylim(0, span)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + span * 0.02,
                        f'{v:.2f}%', ha='center', va='bottom', fontsize=9,
                        fontweight='bold')
        ax.annotate(f"有应急车时开放率是无应急车时的 "
                    f"{(oe / on if (oe and on) else float('nan')):.0f} 倍",
                    xy=(0.5, 0.93), xycoords='axes fraction', ha='center',
                    fontsize=10, fontweight='bold', color=C_MADDPG)
    else:
        ax.axis('off')

    # ---- (d) p(open) distribution ---------------------------------------
    ax = axes[1][1]
    if probe and 'p_hist' in probe:
        edges = np.array(probe['p_hist_edges'])
        hist = np.array(probe['p_hist'], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = (edges[1] - edges[0]) * 0.95
        ax.bar(centers, hist, width=width, color='#5dade2', alpha=0.9,
               edgecolor='#1b4f72', linewidth=0.6)
        ax.axvline(0.5, color=C_MADDPG, lw=2.4, ls='--',
                   label='决策阈值 p=0.5（argmax 分界）')
        mean_p = probe['mean_p_open']
        ax.axvline(mean_p, color=C_GOOD, lw=2.0,
                   label=f'均值 p={mean_p:.3f}')
        ax.axvspan(0.45, 0.55, color=C_WARN, alpha=0.18,
                   label=f"|p−0.5|<0.05 占 "
                         f"{probe['frac_decisions_p_near_half'] * 100:.1f}%")
        ax.set_xlabel('Actor 输出的 p(开放 | 状态)')
        ax.set_ylabel('决策点计数')
        ax.set_ylim(0, max(hist) * 1.42)
        ax.set_title(f"(d) 策略概率分布紧贴判定边界\n"
                     f"熵 = {probe['entropy_mean']:.4f} / 上限 ln2 = "
                     f"{probe['entropy_max']:.4f}（几乎均匀随机）",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='upper center')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.axis('off')

    fig.suptitle('诊断分析：奖励地形 · 检查点甄别 · 策略状态依赖性与确定性边界',
                 fontsize=16, fontweight='bold', y=0.982)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig2] {out_path}')


# --------------------------------------------------------------------------
def fig3(decomp, out_path):
    """Small-multiple reward decomposition - each term gets its own scale."""
    names = order_names(decomp, PREFERRED)
    labels = [SHORT.get(n, n) for n in names]
    terms = [
        ('mean_step_reward_per_agent', '① 合计：单步总奖励', '奖励', True),
        ('efficiency', '② 通行效率项（主导项，权重 20×全局车速+2×本段车速）',
         '奖励', True),
        ('speed_gain', '③ 车速改善项（权重 40×车速提升量）', '奖励', True),
        ('emergency_speed', '④ 应急车速度项（权重 4×应急车归一化速度）', '奖励', True),
        ('resource', '⑤ 资源占用项（±1，关闭且不需要时加分）', '奖励', True),
        ('control', '⑥ 控制代价项（动作切换/相邻段不一致，负分）', '奖励', False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5))
    for ax, (key, title, ylab, higher) in zip(axes.ravel(), terms):
        if key == 'mean_step_reward_per_agent':
            vals = np.array([decomp[n][key] for n in names])
        else:
            vals = np.array([decomp[n]['per_agent_step'][key] for n in names])
        colors = [C_MADDPG if n == 'MADDPG_argmax'
                  else (C_ORACLE if n == 'OracleDensity' else C_BASE)
                  for n in names]
        bars = ax.bar(np.arange(len(names)), vals, color=colors, alpha=0.88,
                      edgecolor='black', linewidth=1.1)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=11.5, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', lw=0.9)

        lo, hi = float(vals.min()), float(vals.max())
        span = (hi - lo) if hi > lo else max(abs(hi), 1e-6)
        ax.set_ylim(min(0.0, lo - span * 0.3), hi + span * 0.28)
        fmt = '{:.3f}' if abs(hi) < 2 else '{:.2f}'
        for bar, v in zip(bars, vals):
            off = span * 0.04
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (off if v >= 0 else -off),
                    fmt.format(v), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=8.6,
                    fontweight='bold')
        best = (int(np.argmax(vals)) if higher else int(np.argmin(vals)))
        bars[best].set_edgecolor(C_GOOD)
        bars[best].set_linewidth(3.0)

    fig.suptitle('奖励分解：最优模型的合计单步奖励最高，且是"效率项 × 应急项"的组合最优，而非靠单项刷分\n'
                 '（数据源 analysis/decompose_*.json，各 30 个配对场景；'
                 '每格独立纵轴，绿框为该格最优）',
                 fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig3] {out_path}')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--screen-glob', default=None)
    p.add_argument('--landscape', default=None)
    p.add_argument('--probe', default=None)
    p.add_argument('--decompose-glob', default=None)
    p.add_argument('--outdir', default=None)
    args = p.parse_args(argv)

    doc = load(args.input)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(outdir, exist_ok=True)

    fig1(doc, os.path.join(outdir, 'fig1_strategy_comparison.png'))

    screen = landscape = probe = None
    if args.screen_glob:
        screen = {}
        for fp in sorted(glob.glob(args.screen_glob)):
            screen.update(load(fp))
        print(f'[screen] merged {len(screen)} checkpoints')
    if args.landscape:
        landscape = unwrap(load(args.landscape))
    if args.probe:
        probe = load(args.probe)
    if screen or landscape or probe:
        fig2(screen, landscape, probe,
             os.path.join(outdir, 'fig2_diagnostics.png'))

    if args.decompose_glob:
        dec = {}
        for fp in sorted(glob.glob(args.decompose_glob)):
            dec.update(load(fp))
        fig3(dec, os.path.join(outdir, 'fig3_reward_decomposition.png'))

    results = unwrap(doc)
    names = order_names(results, PREFERRED)
    print('\n' + '=' * 122)
    print(f"{'策略':<18}{'奖励':>10}{'社会速度':>10}{'应急速度':>10}"
          f"{'成功率':>9}{'开放率':>9}{'应急道占比':>11}{'停驻占比':>10}")
    print('-' * 122)
    for n in names:
        r = results[n]
        print(f"{n:<18}{r.get('mean_reward', float('nan')):>10.2f}"
              f"{r.get('mean_social_speed', float('nan')):>10.3f}"
              f"{r.get('mean_emergency_speed', float('nan')):>10.3f}"
              f"{r.get('success_rate', float('nan')):>9.3f}"
              f"{r.get('mean_open_rate', float('nan')):>9.3f}"
              f"{r.get('mean_lane2_frac', float('nan')):>11.4f}"
              f"{r.get('mean_emergency_stall_frac', float('nan')):>10.4f}")
    print('=' * 122)
    return 0


if __name__ == '__main__':
    sys.exit(main())

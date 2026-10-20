#!/usr/bin/env python
"""
策略对比图（基于真实评估数据）
===============================

背景
----
原 strategy_comparison.py 的数据来自 generate_sample_results() 中写死的
常量，与 evaluate.py 的真实输出（evaluation_results.json）不一致，
连排序都不同。

本脚本**只读取**指定评估 JSON 出图，默认使用修复后的
evaluation_results_fixed.json，杜绝硬编码：
  - 数据缺失的指标明确标注"数据缺失"，不做任何补造；
  - 若 JSON 中缺少 MADDPG 行（未训练/未加载模型），如实反映。

运行
----
    python plot_strategy_comparison.py
    python plot_strategy_comparison.py --input evaluation_results_fixed.json \\
        --output strategy_comparison_fixed.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# 评估脚本中的策略名 -> 图中显示名（2026-10-08：缩短括号内文字，避免 7 个策略时标签互相挤压）
DISPLAY = {
    'MADDPG': 'MADDPG\n(本文模型)',
    'Always Open': 'Always Open\n(恒开放)',
    'Always Close': 'Always Close\n(恒关闭)',
    'Random': 'Random\n(随机)',
    'Periodic': 'Periodic\n(周期)',
    'Greedy': 'Greedy\n(贪心)',
    'Oracle Density': 'Oracle\n(密度规则)',
}
ORDER = ['MADDPG', 'Always Open', 'Always Close', 'Random', 'Periodic', 'Greedy',
         'Oracle Density']

# (JSON 字段, 子图标题, Y 轴标签, 数值格式, 是否越高越好, 是否乘 100 显示)
PANELS = [
    ('mean_reward', '策略对比：平均奖励\n(越高越好)', '平均奖励', '{:.0f}', True, False),
    ('std_reward', '策略对比：稳定性\n(越低越好)', '奖励标准差', '{:.1f}', False, False),
    ('mean_emergency_speed', '应急车辆速度对比\n(越高越好)', '平均速度', '{:.3f}', True, False),
    ('mean_social_speed', '社会车辆速度对比\n(越高越好)', '平均速度', '{:.3f}', True, False),
    ('mean_emergency_stall_steps', '应急车停驻比例对比\n(越低越好)', '停驻步数占比 (%)',
     '{:.1f}%', False, True),
    ('success_rate', '任务成功率对比\n(越高越好)', '成功率 (%)', '{:.0f}%', True, True),
]


def load_results(path):
    if not os.path.exists(path):
        raise SystemExit(f"数据文件不存在: {path}\n请先运行 evaluate.py 生成评估结果。")
    with open(path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    # evaluate.py（2026-10-01 修复版）输出 {"meta": ..., "results": {...}}；
    # 旧版直接输出 {策略名: {...}}。这里两种都接受。
    if isinstance(doc, dict) and 'results' in doc:
        return doc
    return doc


def pad_ylim(ax, values, top=0.15, bottom=0.08):
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo
    if span <= 0:
        span = abs(hi) if abs(hi) > 0 else 1.0
        lo = min(lo, hi - span * 0.5)
    ax.set_ylim(lo - span * bottom, hi + span * top)


def draw_panel(ax, names, values, field, title, ylabel, fmt, higher_better,
               missing_reason=None, scale=1.0):
    if missing_reason or values is None:
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linestyle('--'); sp.set_color('#b0b0b0')
        ax.text(0.5, 0.5, missing_reason or '数据缺失', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='#b03a2e', linespacing=1.7,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdecea',
                          edgecolor='#e74c3c', alpha=0.95))
        return

    values = [v * scale for v in values]
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    bars = ax.bar(names, values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # 7 个策略时标签会互相挤压：加大角度 + 右对齐 + 缩小字号
    ax.tick_params(axis='x', labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(22)
        lbl.set_ha('right')
        lbl.set_rotation_mode('anchor')

    lo, hi = min(values), max(values)
    top = hi + (hi - lo if hi > lo else max(abs(hi), 1.0)) * 0.18
    ax.set_ylim(min(0.0, lo), top)

    for bar, val in zip(bars, values):
        label = fmt.format(val)
        # 去掉 "100%" 这类整数百分比的多余小数位
        if label.endswith('.0%'):
            label = label[:-3] + '%'
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (top - min(0.0, lo)) * 0.02,
                label, ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # 标出最优策略
    best_idx = int(np.argmax(values)) if higher_better else int(np.argmin(values))
    bars[best_idx].set_edgecolor('#c0392b')
    bars[best_idx].set_linewidth(3)


def main() -> int:
    parser = argparse.ArgumentParser(description="基于真实评估数据绘制策略对比图")
    parser.add_argument('--input', default=os.path.join(BASE_DIR, 'evaluation_results_fixed.json'))
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'strategy_comparison_fixed.png'))
    args = parser.parse_args()

    doc = load_results(args.input)
    if isinstance(doc, dict) and 'results' in doc:
        meta = doc.get('meta') or {}
        results = doc['results']
    else:
        meta, results = {}, doc
    keys = [k for k in ORDER if k in results]
    if not keys:
        raise SystemExit(f"{args.input} 中没有任何已知策略: {list(results)}")

    names = [DISPLAY.get(k, k) for k in keys]

    fig = plt.figure(figsize=(20, 14))
    for i, (field, title, ylabel, fmt, higher, pct) in enumerate(PANELS):
        ax = plt.subplot(2, 3, i + 1)
        if not any(field in results[k] for k in keys):
            draw_panel(ax, names, None, field, title, ylabel, fmt, higher,
                       missing_reason=f"数据缺失\n\n{args.input} 中\n不含 {field} 字段。\n\n"
                                      f"需重新运行 evaluate.py\n采集该指标后才能绘制。")
            continue
        values = [results[k].get(field, 0.0) for k in keys]
        scale = 100.0 if pct else 1.0
        draw_panel(ax, names, values, field, title, ylabel, fmt, higher,
                   scale=scale)

    n_ep = results[keys[0]].get('n_episodes', meta.get('n_episodes', '?'))
    pairing = '配对场景' if meta.get('pairing') else '未配对'
    mode = meta.get('policy_mode', '')
    src = os.path.basename(args.input)
    fig.suptitle(f"策略对比（数据源：{src}，每个策略 {n_ep} 回合实测，{pairing}"
                 + (f"，MADDPG={mode}" if mode else "")
                 + "；红框为该项最优）",
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"策略对比图已保存: {args.output}")
    plt.close()

    # 汇总表
    print("\n" + "=" * 100)
    print("真实评估结果汇总")
    print("=" * 100)
    header = f"{'策略':<20}" + "".join(f"{f[:12]:>14}" for f, *_ in PANELS)
    print(header)
    print("-" * 100)
    for k in keys:
        row = f"{DISPLAY.get(k, k).replace(chr(10), ' '):<20}"
        for field, *_ in PANELS:
            v = results[k].get(field)
            row += f"{v:>14.4f}" if v is not None else f"{'N/A':>14}"
        print(row)
    print("=" * 100)

    missing = [f for f, *_ in PANELS if not any(f in results[k] for k in keys)]
    if missing:
        print(f"\n[注意] 以下指标在数据源中缺失，图中已标注为「数据缺失」而非补造: {missing}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

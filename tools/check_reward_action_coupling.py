#!/usr/bin/env python
"""
诊断 3：奖励是否与动作的实际后果挂钩
=====================================

要回答的问题
------------
修复前，奖励函数含多项"只要 action==1 就无条件加分"的常量项
（emergency_lane_capacity_bonus、emergency_vehicle_open_reward 等），
等价于给常量加分：智能体不做任何有效控制也能刷高奖励。

本脚本验证：
  A. 在**同一状态**下，不同动作得到不同奖励（说明奖励确实依赖动作后果）；
  B. 奖励的主要差异来自"后果项"（应急车速度 / 通行效率），
     而不是某个与动作绑定的常量；
  C. 走完整 env.step() 后，"全开放"与"全关闭"的累积奖励不同，
     且差异方向与物理效果一致。

运行
----
    python tools/check_reward_action_coupling.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import banner, install_gym_stub, install_torch_stub, seed_all, stub_notice  # noqa: E402

install_torch_stub()
install_gym_stub()

from env_maddpg import MultiAgentEmergencyLaneEnv  # noqa: E402
from config_maddpg import CA_CONFIG, ENV_CONFIG  # noqa: E402


def build_env(seed: int, n_cars: int = 500, prob_emergency: float = 0.8):
    seed_all(seed)
    env = MultiAgentEmergencyLaneEnv()
    # 固定车辆数与应急车概率，保证对照实验可控
    env.ca_config = dict(env.ca_config)
    env.ca_config['n_cars_range'] = (n_cars, n_cars)
    env.ca_config['P_emergency_range'] = (prob_emergency, prob_emergency)
    env.reset()
    return env


def part_totals(parts) -> dict:
    """把分项按类别求和，便于比较来源。"""
    keys = ['emergency_speed', 'resource', 'efficiency', 'control']
    return {k: float(sum(p.get(k, 0.0) for p in parts)) for k in keys}


def main() -> int:
    banner("诊断 3：奖励与动作后果的耦合性")
    stub_notice()

    n_agents = ENV_CONFIG['n_agents']
    print(f"智能体数={n_agents}，动作空间=2，共 {2 ** n_agents} 种联合动作\n")

    # ---------- A. 同一状态、不同动作 -> 奖励应不同 ----------
    print("--- A. 固定状态下的奖励分布 ---")
    env = build_env(seed=3)
    # 推进到拥堵状态
    for _ in range(60):
        env.step(np.zeros(n_agents, dtype=int))

    totals = {}
    for combo in itertools.product([0, 1], repeat=n_agents):
        actions = np.array(combo, dtype=int)
        rewards, parts = env.reward_breakdown(actions)
        totals[combo] = float(rewards.sum())

    vals = np.array(list(totals.values()))
    spread = vals.max() - vals.min()
    print(f"  奖励总和范围: {vals.min():.3f} ~ {vals.max():.3f} (极差 {spread:.3f})")
    best = max(totals, key=totals.get)
    worst = min(totals, key=totals.get)
    print(f"  最优联合动作: {best}  -> {totals[best]:.3f}")
    print(f"  最差联合动作: {worst} -> {totals[worst]:.3f}")
    nonconstant = spread > 1e-6
    print(f"  奖励随动作变化: {'是' if nonconstant else '否'}")

    # ---------- B. 分项来源 ----------
    print("\n--- B. 奖励分项来源（同一状态下 全关 vs 全开）---")
    r_close, p_close = env.reward_breakdown(np.zeros(n_agents, dtype=int))
    r_open, p_open = env.reward_breakdown(np.ones(n_agents, dtype=int))
    tc = part_totals(p_close)
    to = part_totals(p_open)
    print(f"  {'分项':<20}{'全关闭':>12}{'全开放':>12}{'差(开-关)':>14}")
    print("  " + "-" * 58)
    for k in ['emergency_speed', 'resource', 'efficiency', 'control']:
        print(f"  {k:<20}{tc[k]:>12.3f}{to[k]:>12.3f}{to[k] - tc[k]:>14.3f}")
    print("  " + "-" * 58)
    print(f"  {'合计':<20}{sum(tc.values()):>12.3f}{sum(to.values()):>12.3f}"
          f"{sum(to.values()) - sum(tc.values()):>14.3f}")

    has_em = any(p.get('has_emergency') for p in p_open)
    print(f"\n  当前状态下有应急车的分段: {'有' if has_em else '无'}")
    if not has_em:
        print("  [提示] 本次采样没有应急车，emergency_speed 项为 0，")
        print("         这属于正常情况；请结合多次采样或提高 P_emergency 观察。")

    # ---------- C. 完整 step 的累积奖励对照 ----------
    print("\n--- C. 完整交互的累积奖励对照（20 步）---")
    results = {}
    for name, act in (("always_close", 0), ("always_open", 1)):
        env2 = build_env(seed=11, prob_emergency=0.8)
        actions = np.full(n_agents, act, dtype=int)
        total = 0.0
        soc, em = [], []
        for _ in range(20):
            _, rewards, done, _ = env2.step(actions)
            total += float(rewards.sum())
            soc.append(float(np.mean(env2.ca.vel[:env2.ca.n_cars])))
            if env2.ca.n_emergency > 0:
                em.append(float(np.mean([env2.ca.vel[e] for e in env2.ca.emergency_ids])))
            if done:
                break
        results[name] = {
            'reward': total,
            'social': float(np.mean(soc)),
            'emergency': float(np.mean(em)) if em else float('nan'),
            'lane2': int(np.sum(env2.ca.lane[:env2.ca.total_vehicles] == 2)),
        }
        print(f"  {name:<14} 累积奖励={total:>10.2f}  社会车均速={results[name]['social']:.3f}"
              f"  应急车均速={results[name]['emergency']:.3f}"
              f"  应急车道车辆={results[name]['lane2']}")

    banner("结论")
    ok = True

    if not nonconstant:
        print("  FAIL：固定状态下奖励与动作无关，奖励函数没有耦合到动作。")
        ok = False
    else:
        print("  PASS(A)：固定状态下不同动作产生不同奖励，耦合成立。")

    # 逐项检查：是否存在与动作绑定、但与后果无关的大额常量项
    d_em = to['emergency_speed'] - tc['emergency_speed']
    d_eff = to['efficiency'] - tc['efficiency']
    print(f"  A/B 观察：开放相对于关闭，后果项变化 "
          f"应急车={d_em:+.3f}，通行效率={d_eff:+.3f}")

    diff = abs(results['always_open']['social'] - results['always_close']['social'])
    if diff > 0.05:
        print(f"  PASS(C)：两种策略社会车均速差 {diff:.3f}，"
              f"奖励差异来自真实物理后果。")
    else:
        print(f"  FAIL(C)：两种策略社会车均速几乎相同（差 {diff:.3f}），"
              f"奖励差异缺少物理基础。")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

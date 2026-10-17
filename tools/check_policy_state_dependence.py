#!/usr/bin/env python
"""
诊断 5：策略是否具有"状态依赖性"（最优动作随拥堵程度翻转）
=============================================================

要回答的问题
------------
一个只能"恒开"或"恒关"的问题不需要强化学习。
只有当最优动作**随状态变化**时，MADDPG 才有存在意义。

本脚本对比"全开放"与"全关闭"两条恒策略在不同车辆密度下的全场累积奖励：
  - 低密度：应急车道让给社会车辆收益有限，但会拖慢应急车 -> 期望 CLOSE 更优
  - 高密度：开放能显著疏解拥堵 -> 期望 OPEN 更优
若两者的最优动作在区间内发生翻转，说明奖励函数塑造出了
真实的状态依赖决策问题（PASS）。

修复前的对照：动作与仿真完全解耦，两种策略结果完全相同，
奖励差异只来自奖励函数里的常量项，不存在任何状态依赖性。

运行
----
    python tools/check_policy_state_dependence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import banner, install_gym_stub, install_torch_stub, seed_all, stub_notice  # noqa: E402

install_torch_stub()
install_gym_stub()

from env_maddpg import MultiAgentEmergencyLaneEnv  # noqa: E402
from config_maddpg import CA_CONFIG  # noqa: E402


def rollout(action_value: int, seed: int, n_cars: int, prob_emergency: float) -> dict:
    seed_all(seed)
    env = MultiAgentEmergencyLaneEnv()
    env.ca_config = dict(env.ca_config)
    env.ca_config['n_cars_range'] = (n_cars, n_cars)
    env.ca_config['P_emergency_range'] = (prob_emergency, prob_emergency)
    env.reset()

    acts = np.full(env.n_agents, action_value, dtype=int)
    total_reward = 0.0
    social, emergency = [], []

    while True:
        _, rewards, done, _ = env.step(acts)
        total_reward += float(np.sum(rewards))
        social.append(float(np.mean(env.ca.vel[:env.ca.n_cars])))
        if env.ca.n_emergency > 0:
            emergency.append(
                float(np.mean([env.ca.vel[e] for e in env.ca.emergency_ids])))
        if done:
            break

    return {
        'reward': total_reward,
        'social': float(np.mean(social)),
        'emergency': float(np.mean(emergency)) if emergency else float('nan'),
    }


def main() -> int:
    banner("诊断 5：最优动作是否随拥堵程度翻转")
    stub_notice()

    low = CA_CONFIG['n_cars_range'][0]
    high = CA_CONFIG['n_cars_range'][1]
    mid = (low + high) // 2
    prob_emergency = 0.4
    seeds = (1, 2, 3)

    print(f"车辆数取配置区间 {{low, mid, high}} = {{{low}, {mid}, {high}}}，"
          f"P_emergency={prob_emergency}，种子={seeds}\n")

    print(f"{'n_cars':>8}{'奖励 关闭':>12}{'奖励 开放':>12}"
          f"{'Δ(开-关)':>12}{'最优动作':>11}{'社会车Δ':>10}{'应急车Δ':>10}")
    print("-" * 78)

    flips = []
    for n_cars in (low, mid, high):
        closed = [rollout(0, s, n_cars, prob_emergency) for s in seeds]
        opened = [rollout(1, s, n_cars, prob_emergency) for s in seeds]

        rc = float(np.mean([r['reward'] for r in closed]))
        ro = float(np.mean([r['reward'] for r in opened]))
        dsoc = (float(np.mean([r['social'] for r in opened]))
                - float(np.mean([r['social'] for r in closed])))
        dem = (float(np.nanmean([r['emergency'] for r in opened]))
               - float(np.nanmean([r['emergency'] for r in closed])))
        delta = ro - rc
        best = 'OPEN' if delta > 0 else 'CLOSE'
        flips.append(best)

        print(f"{n_cars:>8}{rc:>12.0f}{ro:>12.0f}{delta:>+12.0f}{best:>11}"
              f"{dsoc:>+10.3f}{dem:>+10.3f}")

    banner("结论")
    distinct = set(flips)
    if len(distinct) > 1:
        print(f"  PASS：最优动作在密度区间内发生翻转 {flips}，")
        print("        说明奖励函数塑造出了状态依赖的决策问题，")
        print("        智能体必须根据拥堵程度决定是否开放应急车道——这正是 RL 的价值所在。")
        print()
        print("  物理解释：")
        print("    - 低密度：开放虽提升社会车速度，但应急车被拖慢的代价更大 -> 关闭")
        print("    - 高密度：严重拥堵时开放带来的疏解收益超过应急车代价 -> 开放")
        return 0

    print(f"  FAIL：三种密度下最优动作相同（{flips}），不存在状态依赖，")
    print("        说明奖励权重失衡，需要重新标定")
    print("        （重点调整 emergency_speed_weight 与 speed_improvement_weight 的比例）。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

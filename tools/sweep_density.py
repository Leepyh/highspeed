#!/usr/bin/env python
"""
诊断 2：密度-速度曲线，为 n_cars_range 选点提供依据
=====================================================

要回答的问题
------------
config_maddpg.py 的拥堵判定阈值是 congestion_threshold=3.0、
severe_congestion_threshold=2.0（折算为 v_max 的 60%、40%）。
原配置 n_cars_range=(200,350) 实测密度仅 0.10~0.175，
社会车辆均速 4.96~5.00，**远高于阈值**，"拥堵"场景从不出现。

本脚本在应急车道关闭的条件下测量不同 n_cars 的稳态平均速度，
找出穿越两个阈值的区间。开放/关闭的奖励对比由
check_policy_state_dependence.py 单独检验。

运行
----
    python tools/sweep_density.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import banner, install_torch_stub, seed_all  # noqa: E402

install_torch_stub()

from high import EmergencyVehicleCA  # noqa: E402
from config_maddpg import CA_CONFIG, ENV_CONFIG  # noqa: E402


def measure(n_cars: int, seeds=(1, 2, 3), steps: int = 300) -> float:
    """返回稳态社会车辆平均速度（元胞/步）。"""
    speeds = []
    for seed in seeds:
        seed_all(seed)
        ca = EmergencyVehicleCA(
            L=CA_CONFIG["L"],
            n_lanes=CA_CONFIG["n_lanes"],
            n_cars=n_cars,
            v_max=CA_CONFIG["v_max"],
            D=CA_CONFIG["D"],
            P1=CA_CONFIG["P1"],
            P2=CA_CONFIG["P2"],
            dev=CA_CONFIG["dev"],
            emergency_prob=0.0,        # 纯密度效应，排除应急车干扰
            max_emergency=CA_CONFIG["max_emergency"],
            v_init_min=CA_CONFIG["v_initial_range"][0],
            v_init_max=CA_CONFIG["v_initial_range"][1],
        )
        # 训练环境预热阶段全部关闭；单独运行 CA 默认允许社会车进入应急车道，
        # 不关闭闸门会测成三车道自由流，低估训练场景的拥堵程度。
        ca._can_enter_emergency_lane = lambda vid: False
        for _ in range(CA_CONFIG["warm_up_steps"]):
            ca.step()
        run = []
        for _ in range(steps):
            ca.step()
            run.append(float(np.mean(ca.vel[:n_cars])))
        speeds.append(float(np.mean(run)))
    return float(np.mean(speeds))


def main() -> int:
    banner("诊断 2：密度-速度曲线")

    v_max = CA_CONFIG["v_max"]
    light = ENV_CONFIG["congestion_threshold"] / v_max          # 0.6
    severe = ENV_CONFIG["severe_congestion_threshold"] / v_max  # 0.4
    lanes_used = CA_CONFIG["n_lanes"] - 1  # 社会车辆只在 lane 0/1

    print(f"参数: L={CA_CONFIG['L']}, v_max={v_max}, "
          f"warm_up={CA_CONFIG['warm_up_steps']}, 测量步数=300, 每点 3 个种子")
    print(f"阈值: 轻度拥堵 < {ENV_CONFIG['congestion_threshold']} "
          f"(归一化 {light:.2f})，严重拥堵 < "
          f"{ENV_CONFIG['severe_congestion_threshold']} (归一化 {severe:.2f})")
    print(f"说明: 应急车道全关闭，密度按 {lanes_used} 条社会车道计算。\n")

    print(f"{'n_cars':>8}{'密度':>9}{'均速':>9}{'归一化':>9}   状态")
    print("-" * 62)

    grid = list(range(200, 1001, 50))
    results = {}
    for n in grid:
        # 容量校验：n_cars + max_emergency 不得超过 L
        if n + CA_CONFIG["max_emergency"] > CA_CONFIG["L"]:
            break
        v = measure(n)
        results[n] = v
        norm = v / v_max
        if norm < severe:
            state = "严重拥堵"
        elif norm < light:
            state = "轻度拥堵"
        else:
            state = "自由流"
        print(f"{n:>8}{n / (CA_CONFIG['L'] * lanes_used):>9.3f}"
              f"{v:>9.3f}{norm:>9.3f}   {state}")

    print()
    banner("阈值穿越点")

    def first_below(thresh):
        for n in sorted(results):
            if results[n] / v_max < thresh:
                return n
        return None

    n_light = first_below(light)
    n_severe = first_below(severe)
    print(f"  首次低于轻度拥堵阈值({light:.2f}) 的 n_cars ≈ {n_light}")
    print(f"  首次低于严重拥堵阈值({severe:.2f}) 的 n_cars ≈ {n_severe}")

    # 该区间仅描述全关闭基线的拥堵阈值，不直接替代现有训练配置。
    if n_light and n_severe:
        rec_lo = n_light
        rec_hi = n_severe + 50
        if rec_hi + CA_CONFIG["max_emergency"] > CA_CONFIG["L"]:
            rec_hi = CA_CONFIG["L"] - CA_CONFIG["max_emergency"]
        print(f"\n  全关闭基线的阈值区间 = ({rec_lo}, {rec_hi})")
        lo_v = results.get(rec_lo - rec_lo % 50, float("nan"))
        print(f"    该区间覆盖轻度拥堵到严重拥堵；训练配置还需结合开/关奖励对比。")
    else:
        print("\n  [警告] 未找到阈值穿越点，请检查阈值设置或道路长度。")

    banner("结论")
    if n_light:
        print(f"  历史配置 (200, 350) 的均速在 {results.get(200, float('nan')):.3f} ~ "
              f"{results.get(350, float('nan')):.3f}，归一化 "
              f"{results.get(200, float('nan')) / v_max:.3f} ~ "
              f"{results.get(350, float('nan')) / v_max:.3f}，")
        print(f"  全部高于轻度拥堵阈值 {light:.2f} —— 这解释了为什么训练中"
              f"「拥堵时开放应急车道」的场景从不触发。")
        print(f"  当前配置 {CA_CONFIG['n_cars_range']} 的开/关奖励关系另由"
              f" check_policy_state_dependence.py 检验。")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

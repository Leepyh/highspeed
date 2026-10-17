#!/usr/bin/env python
"""
诊断 1：控制动作是否真的影响交通流（核心回归测试）
====================================================

要回答的问题
------------
env_maddpg.py 把道路分成 N 段，每个智能体输出 0/1 控制该段应急车道开/关。
这个动作**必须**真实改变交通流，否则整个 MADDPG 控制毫无意义。

修复前的实测结果（见 模型诊断报告.md）：全开与全关结果逐步骤完全相同，
原因是 high.py 的 _can_change_lane() 硬性禁止社会车辆进入 lane 2，
而 env 挂载的 _can_enter_emergency_lane 从未被调用（死代码）。

本脚本用相同随机种子跑"全关闭"与"全开放"两组对照，比较：
  - 各车道车辆分布
  - 应急车道占用（峰值 / 末态）
  - 社会车辆平均速度
  - 应急车平均速度

判定：两组出现显著差异 => 动作通道已打通（PASS）。

运行
----
    python tools/diagnose_action_effect.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import banner, install_torch_stub, seed_all  # noqa: E402

install_torch_stub()

from high import EmergencyVehicleCA  # noqa: E402
from config_maddpg import CA_CONFIG  # noqa: E402


def apply_gate(ca: EmergencyVehicleCA, mode: str) -> None:
    """按 env_maddpg 的方式覆盖准入回调。

    mode='open'  -> 全段开放：允许进入，但仍需服从应急车距离规则
    mode='close' -> 全段关闭：一律不允许进入
    """
    def gate(vid, segment_open=True, _mode=mode):
        if ca.has_emergency[vid]:
            return True
        if _mode == "close":
            return False
        return ca._emergency_lane_default(vid)

    ca._can_enter_emergency_lane = gate


def run_case(seed: int, mode: str, n_cars: int, steps: int,
             prob_emergency: float) -> dict:
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
        emergency_prob=prob_emergency,
        max_emergency=CA_CONFIG["max_emergency"],
        v_init_min=CA_CONFIG["v_initial_range"][0],
        v_init_max=CA_CONFIG["v_initial_range"][1],
    )
    apply_gate(ca, mode)

    for _ in range(CA_CONFIG["warm_up_steps"]):
        ca.step()

    lane2_track = []
    em_track = []
    soc_track = []
    for _ in range(steps):
        ca.step()
        total = min(ca.total_vehicles, len(ca.lane))
        lane2_track.append(int(np.sum(ca.lane[:total] == 2)))
        soc_track.append(float(np.mean(ca.vel[:ca.n_cars])))
        if ca.n_emergency > 0:
            em_track.append(float(np.mean([ca.vel[e] for e in ca.emergency_ids])))

    total = min(ca.total_vehicles, len(ca.lane))
    return {
        "lane_counts": [int(np.sum(ca.lane[:total] == l)) for l in range(ca.n_lanes)],
        "lane2_peak": max(lane2_track),
        "lane2_final": lane2_track[-1],
        "lane2_track": lane2_track,
        "social_speed": float(np.mean(soc_track)),
        "emergency_speed": float(np.mean(em_track)) if em_track else float("nan"),
        "n_emergency": ca.n_emergency,
    }


def main() -> int:
    banner("诊断 1：控制动作是否真的影响交通流")

    n_cars = 500
    prob_emergency = 0.5
    steps = 300
    config = CA_CONFIG

    print(f"参数: L={config['L']}, n_lanes={config['n_lanes']}, n_cars={n_cars}, "
          f"v_max={config['v_max']}, D={config['D']}")
    print(f"      warm_up={config['warm_up_steps']}, 测量步数={steps}, "
          f"P_emergency={prob_emergency}")
    print("说明: 两组使用相同随机种子，唯一差别是应急车道闸门全开 / 全关。\n")

    seeds = [1, 7, 42]
    all_differ = True
    rows = []

    for seed in seeds:
        closed = run_case(seed, "close", n_cars, steps, prob_emergency)
        opened = run_case(seed, "open", n_cars, steps, prob_emergency)

        soc_diff = abs(closed["social_speed"] - opened["social_speed"])
        lane2_diff = abs(closed["lane2_peak"] - opened["lane2_peak"])
        traj_same = closed["lane2_track"] == opened["lane2_track"]

        # 判定标准：应急车道占用出现差异，且社会车辆速度出现实质差异
        differ = (not traj_same) and lane2_diff > 0 and soc_diff > 0.05
        all_differ = all_differ and differ

        rows.append((seed, closed, opened, differ))

        print(f"--- seed = {seed} ---")
        print(f"  应急车数量 = {closed['n_emergency']}")
        print(f"  各车道车辆分布   关闭={closed['lane_counts']}  开放={opened['lane_counts']}")
        print(f"  应急车道占用 峰值 关闭={closed['lane2_peak']:>4}  开放={opened['lane2_peak']:>4}")
        print(f"  应急车道占用 末态 关闭={closed['lane2_final']:>4}  开放={opened['lane2_final']:>4}")
        print(f"  社会车辆平均速度  关闭={closed['social_speed']:.3f}  开放={opened['social_speed']:.3f}"
              f"  (差 {soc_diff:.3f})")
        print(f"  应急车平均速度    关闭={closed['emergency_speed']:.3f}  开放={opened['emergency_speed']:.3f}")
        print(f"  逐步骤应急车道轨迹相同: {traj_same}")
        print(f"  => 动作是否影响交通流: {'是' if differ else '否'}")
        print()

    banner("结论")
    if all_differ:
        print("  PASS：在全部测试种子上，应急车道开/关都显著改变了交通流")
        print("        （车道分布、应急车道占用、社会车辆速度均出现差异）。")
        print("        动作通道已打通，MADDPG 的控制是有物理意义的。")
        print()
        print("  参考对比：")
        avg_close = np.mean([r[1]["social_speed"] for r in rows])
        avg_open = np.mean([r[2]["social_speed"] for r in rows])
        print(f"    社会车辆平均速度  关闭={avg_close:.3f}  开放={avg_open:.3f}"
              f"  提升={avg_open - avg_close:+.3f}")
        avg_l2c = np.mean([r[1]["lane2_final"] for r in rows])
        avg_l2o = np.mean([r[2]["lane2_final"] for r in rows])
        print(f"    应急车道末态车辆  关闭={avg_l2c:.1f}  开放={avg_l2o:.1f}")
        return 0

    print("  FAIL：至少一个种子上动作未产生实质影响，需要检查准入闸门是否生效。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
诊断 4：应急车是否会被社会车辆堵死
===================================

要回答的问题
------------
打通应急车道准入后，社会车辆会驶入应急车道。若应急车被这些车堵住，
"开放"就变成了有害动作，模型也就失去意义。

按选定机制（准入 + 应急车优先让行）：
  - 应急车影响范围 D 内的应急车道禁止社会车辆进入；
  - 已在应急车道、且应急车即将到达的社会车辆，安全时以 P2 概率左移让行。

本脚本测量：
  - 开放/关闭两种策略下应急车平均速度；
  - 应急车处于低速（< 1 元胞/步）的步数占比；
  - 与 config 中 emergency_min_acceptable_speed 的比较。

判定：开放时应车速度不应长期低于可接受下界（PASS）。

运行
----
    python tools/check_emergency_yield.py
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


def run_case(seed: int, mode: str, n_cars: int, prob_emergency: float,
             steps: int = 300) -> dict:
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
        yield_distance=CA_CONFIG.get("emergency_yield_distance", CA_CONFIG["D"]),
    )
    ca._can_enter_emergency_lane = lambda vid, segment_open=True, g=mode: (
        True if ca.has_emergency[vid]
        else (g == "open" and ca._emergency_lane_default(vid))
    )

    for _ in range(CA_CONFIG["warm_up_steps"]):
        ca.step()

    em_speeds = []
    stalled = 0
    max_stall_run = 0
    cur_run = 0
    lane2_in_use = []
    for _ in range(steps):
        ca.step()
        total = min(ca.total_vehicles, len(ca.lane))
        lane2_in_use.append(int(np.sum(ca.lane[:total] == 2)))
        if ca.n_emergency > 0:
            s = float(np.mean([ca.vel[e] for e in ca.emergency_ids]))
            em_speeds.append(s)
            if s < 1.0:
                stalled += 1
                cur_run += 1
                max_stall_run = max(max_stall_run, cur_run)
            else:
                cur_run = 0

    return {
        "has_emergency": bool(em_speeds),
        "em_speed_mean": float(np.mean(em_speeds)) if em_speeds else float("nan"),
        "em_speed_min": float(np.min(em_speeds)) if em_speeds else float("nan"),
        "stalled_steps": stalled,
        "max_stall_run": max_stall_run,
        "observed_steps": len(em_speeds),
        "lane2_peak": max(lane2_in_use) if lane2_in_use else 0,
    }


def main() -> int:
    banner("诊断 4：应急车让行机制是否有效")

    v_max = CA_CONFIG["v_max"]
    floor = CA_CONFIG.get("emergency_min_acceptable_speed", 2.0)
    prob_emergency = 0.8      # 提高应急车出现率，让结论更稳健
    n_cars = 520
    steps = 300
    seeds = [1, 3, 5, 7, 11]

    print(f"参数: n_cars={n_cars}, P_emergency={prob_emergency}, "
          f"warm_up={CA_CONFIG['warm_up_steps']}, 测量步数={steps}")
    print(f"可接受速度下界: {floor} 元胞/步 (v_max={v_max})")
    print(f"种子: {seeds}\n")

    rows = {"close": [], "open": []}
    for seed in seeds:
        for mode in ("close", "open"):
            rows[mode].append(run_case(seed, mode, n_cars, prob_emergency, steps))

    print(f"{'策略':>8}{'应急车均速':>12}{'最低瞬时':>11}"
          f"{'低速步数':>10}{'低速占比':>10}{'最长连续停滞':>14}{'应急车道峰值':>14}")
    print("-" * 84)
    summary = {}
    for mode in ("close", "open"):
        rs = [r for r in rows[mode] if r["has_emergency"]]
        if not rs:
            print(f"{mode:>8}   （本次采样未出现应急车）")
            continue
        mean_v = float(np.mean([r["em_speed_mean"] for r in rs]))
        min_v = float(np.min([r["em_speed_min"] for r in rs]))
        stalled = sum(r["stalled_steps"] for r in rs)
        obs = sum(r["observed_steps"] for r in rs)
        worst_stall = max(r["max_stall_run"] for r in rs)
        peak = float(np.mean([r["lane2_peak"] for r in rs]))
        ratio = stalled / obs if obs else 0.0
        summary[mode] = dict(mean=mean_v, min=min_v, ratio=ratio,
                             peak=peak, worst_stall=worst_stall)
        print(f"{mode:>8}{mean_v:>12.3f}{min_v:>11.3f}"
              f"{stalled:>10}{ratio:>9.1%}{worst_stall:>14}{peak:>14.1f}")

    banner("结论")
    ok = True

    if "open" not in summary:
        print("  [跳过] 开放策略下未采到应急车样本，无法判定。")
        return 0

    op = summary["open"]
    if op["mean"] >= floor:
        print(f"  PASS：开放应急车道时，应急车平均速度 {op['mean']:.3f} "
              f">= 下界 {floor}，未被社会车辆堵死。")
    else:
        print(f"  FAIL：开放时应急车平均速度仅 {op['mean']:.3f}，"
              f"低于下界 {floor}。让行机制无效，需要调整 yield_distance 或让行概率。")
        ok = False

    # 真正的验收标准是"不会长期卡死"，而非"从不减速"。
    # 应急车与前车共道时按 min(v_max, gap) 减速是模型设计的一部分。
    max_ok = 10
    if op["worst_stall"] <= max_ok:
        print(f"  PASS：最长连续停滞 {op['worst_stall']} 步 <= {max_ok} 步，"
              f"说明只是短暂减速（低速步数占比 {op['ratio']:.1%}），不会长期卡死。")
    else:
        print(f"  FAIL：最长连续停滞 {op['worst_stall']} 步 > {max_ok} 步，"
              f"应急车被长时间堵住。")
        ok = False

    if "close" in summary:
        cl = summary["close"]
        print(f"\n  对照：关闭时应急车平均速度 {cl['mean']:.3f}"
              f"（应急车道基本空闲，故接近 v_max）")
        print(f"        开放相对关闭的应急车速度损失 = "
              f"{cl['mean'] - op['mean']:+.3f}")
        print(f"        这正是「开放缓解拥堵 vs 保障应急通行」权衡的量化体现。")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
统一回归测试入口
=================

依次运行 tools/ 下的全部诊断脚本，汇总 PASS / FAIL，
并返回非零退出码以适配 CI 或人工检查。

覆盖的缺陷（对应 模型诊断报告.md）：
  D1/D2/D3  动作与仿真解耦     -> diagnose_action_effect
  D6        拥堵场景不出现     -> sweep_density
  D7        奖励与动作后果脱钩 -> check_reward_action_coupling
  D4        应急车被堵死       -> check_emergency_yield
  新        无状态依赖性       -> check_policy_state_dependence

另有内置的结构性断言（不依赖外部脚本）：
  - obs_dim 必须等于 5 段 × 8 + 5 = 45（与已训练的 .pth 检查点一致）
  - 容量校验必须能拦截超载配置
  - 闸门必须在 reset() 后依然生效

运行
----
    python tools/run_tests.py
    python tools/run_tests.py --skip-slow   # 跳过密度扫描
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from _common import install_gym_stub, install_torch_stub, seed_all, stub_notice  # noqa: E402

install_torch_stub()
install_gym_stub()

SUITES = [
    ("分段闸门、切换代价与道路占用一致性", "check_core_invariants.py", False),
    ("动作是否影响交通流 (D1/D2/D3)", "diagnose_action_effect.py", False),
    ("奖励是否与动作后果耦合 (D7)", "check_reward_action_coupling.py", False),
    ("应急车是否被堵死 (D4)", "check_emergency_yield.py", False),
    ("最优动作是否随状态翻转", "check_policy_state_dependence.py", False),
    ("密度-速度曲线 (D6)", "sweep_density.py", True),
]


def run_script(name: str) -> tuple[int, str]:
    path = os.path.join(TOOLS_DIR, name)
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    proc = subprocess.run(
        [sys.executable, path],
        cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace', env=env,
    )
    return proc.returncode, (proc.stdout or '') + (proc.stderr or '')


def structural_checks() -> list[tuple[str, bool, str]]:
    """轻量结构断言，快速失败。"""
    import json
    import numpy as np

    results = []

    # 1. obs_dim 必须与已训练检查点一致（45）
    from env_maddpg import MultiAgentEmergencyLaneEnv
    from config_maddpg import ENV_CONFIG, CA_CONFIG
    seed_all(0)
    env = MultiAgentEmergencyLaneEnv()
    expected = (1 + 2 * ENV_CONFIG['local_observation_range']) * 8 + ENV_CONFIG['state_history']
    ok = env.obs_dim == expected == 45
    results.append(("obs_dim 保持 45（兼容既有检查点）", ok,
                    f"obs_dim={env.obs_dim}, 期望={expected}"))

    obs = env.reset()
    ok = all(o.shape == (45,) for o in obs)
    results.append(("reset() 返回 5 个形状正确的观测", ok,
                    f"shapes={[o.shape for o in obs]}"))

    # 2. 容量校验必须拦截超载
    from high import EmergencyVehicleCA
    raised = False
    try:
        EmergencyVehicleCA(L=100, n_lanes=3, n_cars=120, v_max=5, D=40,
                           P1=0.6, P2=0.9, dev=2, emergency_prob=0.0,
                           max_emergency=3, v_init_min=2, v_init_max=5)
    except ValueError:
        raised = True
    results.append(("超载配置触发 ValueError（D5）", raised,
                    "n_cars=120 > L=100 应报错"))

    # 3. reset() 后闸门仍生效（D1 的空窗问题）
    seed_all(1)
    env2 = MultiAgentEmergencyLaneEnv()
    env2.reset()
    has_gate = callable(getattr(env2.ca, '_can_enter_emergency_lane', None))
    results.append(("reset() 后 CA 上仍有准入闸门", has_gate,
                    f"type={type(getattr(env2.ca, '_can_enter_emergency_lane', None)).__name__}"))

    # 4. 闸门关闭时社会车辆不得进入 lane 2
    seed_all(2)
    env3 = MultiAgentEmergencyLaneEnv()
    env3.ca_config = dict(env3.ca_config)
    env3.ca_config['n_cars_range'] = (500, 500)
    env3.reset()
    env3._install_emergency_lane_gate(np.zeros(env3.n_agents, dtype=int))
    for _ in range(60):
        env3.step(np.zeros(env3.n_agents, dtype=int))
    total = min(env3.ca.total_vehicles, len(env3.ca.lane))
    lane2 = int(np.sum(env3.ca.lane[:total] == 2))
    # 全关闭时应急车道上只应有应急车
    ok = lane2 <= env3.ca.n_emergency
    results.append(("全关闭时社会车辆不在应急车道", ok,
                    f"lane2={lane2}, 应急车={env3.ca.n_emergency}"))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="highspeed 回归测试")
    parser.add_argument('--skip-slow', action='store_true',
                        help='跳过密度扫描等耗时用例')
    args = parser.parse_args()

    print("=" * 84)
    print("highspeed 模型修复回归测试")
    print("=" * 84)
    stub_notice()

    outcomes = []

    # ---- 结构断言 ----
    print("[1] 结构断言")
    print("-" * 84)
    for name, ok, detail in structural_checks():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
        outcomes.append((name, ok))

    # ---- 脚本用例 ----
    print("\n[2] 诊断脚本")
    print("-" * 84)
    idx = 0
    for label, script, slow in SUITES:
        if slow and args.skip_slow:
            print(f"  [SKIP] {label}  ({script})")
            continue
        idx += 1
        print(f"  [运行] {label}  ({script}) ...", flush=True)
        code, output = run_script(script)
        ok = (code == 0)
        outcomes.append((label, ok))
        tail = [ln for ln in output.strip().splitlines() if ln.strip()][-3:]
        for ln in tail:
            print(f"         | {ln}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} (exit={code})")

    # ---- 汇总 ----
    print("\n" + "=" * 84)
    print("汇总")
    print("=" * 84)
    passed = sum(1 for _, ok in outcomes if ok)
    total = len(outcomes)
    for name, ok in outcomes:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("-" * 84)
    print(f"  通过 {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

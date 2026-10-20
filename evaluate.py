#!/usr/bin/env python
"""
MADDPG 公平评估脚本（2026-10-01 修复版）

对比策略：
1. MADDPG（训练好的模型，默认用**确定性 argmax** 策略）
2. Always Open（始终开放）
3. Always Close（始终关闭）
4. Random（随机策略）
5. Periodic（周期性策略）
6. Greedy（贪心策略 - 哪段堵开哪段）
7. Oracle Density（密度规则，使用特权信息，仅作上界参考）

【本次修复的三个坑】
  坑 1：原实现 `select_action(evaluate=True)` **只关掉 ε 探索，动作仍按概率采样**，
        在熵接近 ln2 的策略下等于在评估一个随机策略。历史结论
        "MADDPG 排名第 3 / MADDPG ≈ Always Close" 全部由此而来。
        现在默认走 `deterministic=True`（argmax），并保留 --policy-mode sample
        以复现旧口径。
  坑 2：原实现每个策略各自 `env.reset()`，而每次 reset 都会重新随机采样
        n_cars 与 P_emergency —— **每个策略在完全不同的场景上被评分**，
        几十个回合的采样噪声足以打乱排名。现在默认**配对**：同一 seed 列表下，
        所有策略面对完全相同的交通场景。
  坑 3：原实现的 `_episode_metrics()` 在回合结束**之后**才被调用，
        读的是 t=500 那一刻的 env.ca —— 速度/停驻/成功率都是**单帧快照**。
        现在改为**整回合逐决策步平均**。

用法：
    python evaluate.py                                  # 160 回合、配对、确定性
    python evaluate.py 160 models_fixed_single/maddpg_best_avg.pth
    python evaluate.py 50 models_matd3/maddpg_best.pth --policy-mode sample
    python evaluate.py 50 ... --no-paired               # 复现旧的"各评各的"口径

输出：
    evaluation_results_fixed.json —— 与 plot_strategy_comparison.py 兼容
                                （顶层为 {"meta":..., "results":{...}}，
                                 plot_strategy_comparison.py 会自动拆包）
"""

import argparse
import json
import os
import random

import numpy as np

from env_maddpg import MultiAgentEmergencyLaneEnv
from config_maddpg import ENV_CONFIG, CA_CONFIG

# 说明：torch 与 maddpg_v2 改为**延迟导入**（见 create_maddpg_agent）。
# 这样在没有 torch 的环境下，依然可以评估基线策略；
# 只有真正需要 MADDPG 模型时才会要求安装 torch。

V_MAX = CA_CONFIG['v_max']
EM_FLOOR = CA_CONFIG.get('emergency_min_acceptable_speed', 2.0)


def create_env():
    """创建环境"""
    return MultiAgentEmergencyLaneEnv()


def create_maddpg_agent(env, model_path='./models_fixed_single/maddpg_best_avg.pth'):
    """创建并加载MADDPG智能体（需要 torch，故延迟导入）"""
    try:
        from maddpg_v2 import MADDPGAgent
    except ImportError as e:
        print(f"✗ 无法导入 MADDPG（缺少 torch？）: {e}")
        return None

    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = ENV_CONFIG['n_actions']

    # 这里的超参只影响网络结构与加载，推理阶段用不到学习率
    config = {
        'actor_hidden_dims': [128, 64],
        'critic_hidden_dims': [256, 128, 64],
        'actor_lr': 3e-5,
        'critic_lr': 1e-3,
        'gamma': 0.95,
        'tau': 0.005,
        'buffer_capacity': 100,
        'batch_size': 128,
        'policy_freq': 2,
        'max_grad_norm': 1.0,
    }

    agent = MADDPGAgent(n_agents, obs_dim, n_actions, config)

    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            agent.epsilon = 0  # 评估时关闭探索
            print(f"✓ MADDPG模型加载成功: {model_path}")
            return agent
        except Exception as e:
            print(f"✗ MADDPG模型加载失败: {e}")
            return None
    else:
        print(f"✗ MADDPG模型文件不存在: {model_path}")
        return None


# --------------------------------------------------------------------------
# 策略工厂
# --------------------------------------------------------------------------
def make_policy_always(value):
    return lambda env, obs, step, v=value: np.full(env.n_agents, v, dtype=int)


def make_policy_random(seed=None):
    rng = np.random.default_rng(seed)
    return lambda env, obs, step: rng.integers(0, 2, size=env.n_agents)


def make_policy_periodic(period=10):
    def policy(env, obs, step, p=period):
        cycle = (step // p) % 2
        return np.full(env.n_agents, cycle, dtype=int)
    return policy


def make_policy_greedy():
    """哪段堵开哪段：本段社会车道速度低于拥堵阈值则开放。"""
    def policy(env, obs, step):
        threshold = env.env_config['congestion_threshold'] / env.ca_config['v_max']
        actions = np.zeros(env.n_agents, dtype=int)
        for i in range(env.n_agents):
            if env._local_congestion_speed(i) < threshold:
                actions[i] = 1
        return actions
    return policy


def make_policy_oracle(threshold=500):
    """密度规则（使用特权信息 n_cars，仅作上界参考，不是公平对照）。"""
    def policy(env, obs, step, thr=threshold):
        n_cars = env.current_params['n_cars']
        return np.full(env.n_agents, 1 if n_cars >= thr else 0, dtype=int)
    return policy


def make_policy_maddpg(agent, deterministic=True):
    """MADDPG 策略。

    deterministic=True  -> argmax(pi)，即"学到的策略"（推荐，本次修复的默认）
    deterministic=False -> a ~ pi(.|s)，即旧 evaluate.py 的口径（仅用于复现）
    """
    if deterministic:
        return lambda env, obs, step: agent.select_action(
            obs, evaluate=True, deterministic=True)
    return lambda env, obs, step: agent.select_action(obs, evaluate=True)


# --------------------------------------------------------------------------
# 统一 rollout（配对 + 整回合平均指标）
# --------------------------------------------------------------------------
def run_strategy(env, policy, n_episodes=50, seeds=None, verbose=False):
    """用统一流程评估一个策略。

    参数:
        policy: callable(env, observations, step) -> actions[n_agents]
        seeds:  配对用的回合种子列表（长度须 >= n_episodes）。
                给出时每个回合在 reset 前 `random.seed(s); np.random.seed(s)`，
                保证不同策略在相同索引的回合面对**完全相同的交通场景**。
                策略调用期间会冻结两个全局随机流，避免策略自身的随机性
                污染环境（否则不同策略的场景会逐渐分叉）。

    指标全部为**整回合逐决策步平均**，不是回合末单帧快照。
    """
    rows = []
    for k in range(n_episodes):
        if seeds is not None:
            s = int(seeds[k])
            random.seed(s)
            np.random.seed(s)

        observations = env.reset()
        params = dict(env.current_params)
        done = False
        steps = 0
        episode_reward = 0.0
        soc, em, stall, lane2, openr = [], [], [], [], []

        while not done:
            rstate, nstate = random.getstate(), np.random.get_state()
            actions = np.asarray(policy(env, observations, steps),
                                 dtype=int).reshape(-1)
            random.setstate(rstate)
            np.random.set_state(nstate)

            observations, rewards, done, info = env.step(actions)
            ca = env.ca
            n = ca.n_cars

            episode_reward += float(np.mean(rewards))
            openr.append(float(np.mean(actions == 1)))
            soc.append(float(np.mean(ca.vel[:n])) if n else 0.0)
            # 社会车的应急道占比；应急车本来就在 lane 2，不能计入分子。
            lane2.append(float(np.sum(ca.lane[:n] == 2)) / max(n, 1))
            if ca.n_emergency > 0:
                sp = ca.vel[ca.emergency_ids]
                em.append(float(np.mean(sp)))
                stall.append(float(np.mean(sp <= 0)))
            steps += 1

        rows.append({
            'seed': int(seeds[k]) if seeds is not None else k,
            'n_cars': int(params['n_cars']),
            'P_emergency': float(params['P_emergency']),
            'reward': episode_reward,
            'length': steps,
            'social_speed': float(np.mean(soc)),
            'emergency_speed': float(np.mean(em)) if em else None,
            'emergency_stall_frac': float(np.mean(stall)) if stall else None,
            'lane2_frac': float(np.mean(lane2)),
            'open_rate': float(np.mean(openr)),
        })
        if verbose and (k + 1) % 10 == 0:
            print(f"      ... {k + 1}/{n_episodes}", flush=True)

    rew = np.array([r['reward'] for r in rows], dtype=float)
    social = np.array([r['social_speed'] for r in rows], dtype=float)
    lane2 = np.array([r['lane2_frac'] for r in rows], dtype=float)
    openr = np.array([r['open_rate'] for r in rows], dtype=float)

    # 仅统计有应急车的回合，避免"无应急车回合"污染应急车指标
    em_rows = [r for r in rows if r['emergency_speed'] is not None]
    em = np.array([r['emergency_speed'] for r in em_rows], dtype=float)
    st = np.array([r['emergency_stall_frac'] for r in em_rows], dtype=float)
    succ = np.array([1.0 if r['emergency_speed'] >= EM_FLOOR else 0.0
                     for r in em_rows], dtype=float)

    def m(a, d=0.0):
        return float(np.mean(a)) if a.size else d

    return {
        'mean_reward': float(np.mean(rew)),
        'std_reward': float(np.std(rew)),
        'sem_reward': (float(np.std(rew, ddof=1) / np.sqrt(len(rew)))
                       if len(rew) > 1 else 0.0),
        'mean_length': float(np.mean([r['length'] for r in rows])),
        'mean_emergency_speed': m(em),
        'mean_social_speed': m(social),
        # ---- 新增/修复指标 ----
        'min_emergency_speed': float(np.min(em)) if em.size else 0.0,
        'mean_emergency_stall_steps': m(st),      # 现在=停驻步数占比
        'success_rate': m(succ),
        'emergency_episodes': int(len(em_rows)),
        'n_episodes': int(n_episodes),
        # ---- 诊断字段（原脚本没有；用于判断"是不是真的在控制"）----
        'mean_open_rate': m(openr),
        'mean_lane2_frac': m(lane2),
        '_episodes': rows,
    }


def run_evaluation(n_episodes=160, model_path='./models_fixed_single/maddpg_best_avg.pth',
                   output='evaluation_results_fixed.json', seed=0,
                   seed_start=1000, paired=True, policy_mode='argmax',
                   include_oracle=True, oracle_threshold=500):
    """运行完整评估。

    所有策略走同一套 rollout（run_strategy）与**同一批场景**（配对），保证对比公平。
    """
    print("=" * 74)
    print("MADDPG Fair Evaluation (paired seeds + deterministic policy)")
    print("=" * 74)

    env = create_env()
    print(f"Environment: {env.n_agents} agents, {env.obs_dim} obs_dim, "
          f"{ENV_CONFIG['n_actions']} actions")
    print(f"Episodes per strategy: {n_episodes}")
    print("Pairing       : " + ("ON (identical scenarios for every strategy)"
                                if paired else "OFF (legacy behaviour)"))
    print(f"MADDPG policy : {policy_mode}")

    seeds = list(range(seed_start, seed_start + n_episodes)) if paired else None

    strategies = []

    print("\n[1/7] Evaluating MADDPG...")
    agent = create_maddpg_agent(env, model_path)
    if agent is not None:
        strategies.append(('MADDPG', make_policy_maddpg(
            agent, deterministic=(policy_mode == 'argmax'))))
    else:
        print("  [跳过] MADDPG 模型不可用")

    print("[2/7] Evaluating Always Open...")
    strategies.append(('Always Open', make_policy_always(1)))

    print("[3/7] Evaluating Always Close...")
    strategies.append(('Always Close', make_policy_always(0)))

    print("[4/7] Evaluating Random...")
    strategies.append(('Random', make_policy_random(seed)))

    print("[5/7] Evaluating Periodic...")
    strategies.append(('Periodic', make_policy_periodic(period=10)))

    print("[6/7] Evaluating Greedy...")
    strategies.append(('Greedy', make_policy_greedy()))

    if include_oracle:
        print("[7/7] Evaluating Oracle Density (privileged reference)...")
        strategies.append(('Oracle Density', make_policy_oracle(oracle_threshold)))

    results = {}
    for name, policy in strategies:
        print(f"  运行 {name} ...", flush=True)
        results[name] = run_strategy(env, policy, n_episodes,
                                     seeds=seeds, verbose=True)

    payload = {
        'meta': {
            'n_episodes': n_episodes,
            'pairing': bool(paired),
            'seed_start': seed_start,
            'policy_mode': policy_mode,
            'model': model_path,
        },
        'results': results,
    }
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output}")

    # 打印对比表
    print("\n" + "=" * 122)
    print("Evaluation Results Comparison")
    print("=" * 122)
    print(f"{'Strategy':<17}{'Reward':>11}{'SEM':>9}{'EmgSpd':>10}"
          f"{'SocSpd':>10}{'Stall%':>9}{'SuccRate':>10}{'OpenRate':>10}"
          f"{'Lane2%':>9}{'EmgEps':>8}")
    print("-" * 122)
    for name, res in results.items():
        print(f"{name:<17}{res['mean_reward']:>11.2f}{res['sem_reward']:>9.2f}"
              f"{res['mean_emergency_speed']:>10.3f}{res['mean_social_speed']:>10.3f}"
              f"{res['mean_emergency_stall_steps'] * 100:>8.1f}%"
              f"{res['success_rate']:>10.1%}{res['mean_open_rate']:>10.3f}"
              f"{res['mean_lane2_frac'] * 100:>8.1f}%{res['emergency_episodes']:>8}")
    print("=" * 122)
    print("列说明: EmgSpd=应急车均速  SocSpd=社会车均速  Stall%=应急车停驻步数占比")
    print("        SuccRate=应急车均速达到可接受下界的回合占比  OpenRate=动作开放率")
    print("        Lane2%=社会车应急道占比  EmgEps=含应急车的回合数")
    print("        （除 EmgEps 外，速度类指标均为整回合逐决策步平均，不是回合末快照）")

    if results:
        best = max(results.items(), key=lambda x: x[1]['mean_reward'])
        print(f"\n最佳策略(按平均奖励): {best[0]} ({best[1]['mean_reward']:.2f})")
        if 'MADDPG' in results:
            order = sorted(results.items(), key=lambda x: x[1]['mean_reward'],
                           reverse=True)
            rank = [k for k, _ in order].index('MADDPG') + 1
            print(f"MADDPG 排名: 第 {rank} / {len(order)}")

        # 配对显著性：MADDPG 与每个对手的逐回合差值
        if paired and 'MADDPG' in results:
            ref = {r['seed']: r['reward'] for r in results['MADDPG']['_episodes']}
            print("\n" + "-" * 74)
            print("PAIRED DIFFERENCE vs MADDPG  (positive = MADDPG better)")
            print("-" * 74)
            print(f"{'opponent':<17}{'mean diff':>12}{'95% CI':>24}{'win rate':>11}")
            try:
                from scipy import stats as _st
                tcrit = lambda n: float(_st.t.ppf(0.975, n - 1))     # noqa: E731
            except Exception:                                        # noqa: BLE001
                tcrit = lambda n: 2.0                                # noqa: E731
            for name, res in results.items():
                if name == 'MADDPG':
                    continue
                other = {r['seed']: r['reward'] for r in res['_episodes']}
                ks = sorted(set(ref) & set(other))
                d = np.array([ref[k] - other[k] for k in ks])
                if d.size < 2:
                    continue
                sem = float(np.std(d, ddof=1) / np.sqrt(d.size))
                tc = tcrit(d.size)
                lo, hi = d.mean() - tc * sem, d.mean() + tc * sem
                flag = 'YES' if (lo > 0 or hi < 0) else 'no'
                print(f"{name:<17}{d.mean():>+12.1f}"
                      f"{f'[{lo:+.1f}, {hi:+.1f}]':>24}"
                      f"{np.mean(d > 0):>10.1%}  {flag}")
            print("-" * 74)
            print("YES = 95% 置信区间不跨 0（差异显著）")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MADDPG 策略公平评估（配对 + 确定性）")
    parser.add_argument('n_episodes', nargs='?', type=int, default=160,
                        help='每个策略评估的回合数（默认 160；'
                             '实测 64 个场景测不出 +80 量级的差异，故提高默认值）')
    parser.add_argument('model_path', nargs='?',
                        default='./models_fixed_single/maddpg_best_avg.pth',
                        help='MADDPG 模型路径')
    parser.add_argument('--output', default='evaluation_results_fixed.json',
                        help='结果输出文件')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机策略的种子')
    parser.add_argument('--seed-start', type=int, default=1000,
                        help='配对场景的起始种子（默认 1000）')
    parser.add_argument('--policy-mode', choices=['argmax', 'sample'],
                        default='argmax',
                        help='MADDPG 动作模式：argmax=确定性(推荐)，sample=旧口径')
    parser.add_argument('--no-paired', dest='paired', action='store_false',
                        help='关闭配对（复现旧的"每个策略各自随机场景"口径）')
    parser.add_argument('--no-oracle', dest='include_oracle',
                        action='store_false', help='不评估 Oracle 参考策略')
    parser.add_argument('--oracle-threshold', type=int, default=500,
                        help='Oracle 密度规则的阈值（默认 500 辆）')
    args = parser.parse_args()

    run_evaluation(args.n_episodes, args.model_path,
                   output=args.output, seed=args.seed,
                   seed_start=args.seed_start, paired=args.paired,
                   policy_mode=args.policy_mode,
                   include_oracle=args.include_oracle,
                   oracle_threshold=args.oracle_threshold)

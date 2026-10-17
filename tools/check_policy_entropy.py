#!/usr/bin/env python
"""
诊断 6：检查训练后的策略熵与动作分布（2026-10-01 修复版）

要回答的问题
------------
训练出的策略到底"还在不在工作"？是不是已经退化成不做任何有效控制？

背景
----
早期版本训练到第 400 轮时，5 个智能体的策略熵全部降为 0
（上限 ln2 ≈ 0.693），退化成确定性策略"恒关闭"。这个问题修掉之后，
出现了一个**更隐蔽、原来这个工具完全看不到**的失效模式：

  - **采样策略**的熵全程正常（0.65~0.68 / 上限 0.693）
  - 但 **argmax 策略**早已退化：第 1000 轮之后 argmax 奖励从 ~3350
    掉到 ~2920，与 Always Close 完全持平
  - 更迷惑人的是：后期检查点的 argmax 开放率还有 11%~22%，
    看似在控制，但应急车道承载率只有 0.25%（和恒关闭一样）——
    "开了门但没车进得去"，属于**无效动作**

原因：旧实现只看 `select_action(evaluate=True)` 的采样开放率 —— 而那个函数
**只关掉 ε，仍然按概率采样**，在均匀策略下永远给出 ~42% 的开放率，
把 argmax 的退化完全掩盖了。

检查内容（三层，逐层加严）
--------------------------
  1) 策略熵            —— 会不会坍缩成确定性
  2) argmax 开放率      —— 确定性动作是不是恒定值
  3) **快速对照评估**   —— 用 argmax 策略与 AlwaysClose / AlwaysOpen
     在**同一批配对场景**上跑 N 个回合比较奖励。
     这是最有决定性的判据：一个"不做有效控制"的策略，奖励必然与
     恒关闭/恒开放持平。

判定
----
  argmax 奖励 <= 恒关闭奖励 + 50   -> FAIL 退化（策略不比"什么都不做"更好）
  argmax 奖励 <= 恒开放奖励 + 50   -> WARN 只能打平最简单的基线
  明显优于两者                     -> PASS

用法
----
    python tools/check_policy_entropy.py                              # 默认 best
    python tools/check_policy_entropy.py --model models_v3_a/a_maddpg_best_avg.pth
    python tools/check_policy_entropy.py --all --model-dir models_v3_a
    python tools/check_policy_entropy.py --model ... --quick-episodes 0   # 跳过第 3 层
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, ROOT)

LOG2 = float(np.log(2))
MARGIN = 50.0          # 与基线持平的判定余量（奖励单位）


def evaluate_model(model_path: str, n_episodes: int = 3,
                   quick_episodes: int = 10, seed_start: int = 5000) -> dict:
    """返回策略熵、采样/argmax 开放率，以及与两个平凡基线的配对对照。"""
    import torch
    from env_maddpg import MultiAgentEmergencyLaneEnv
    from maddpg_v2 import MADDPGAgent
    from config_maddpg import ENV_CONFIG, AGENT_CONFIG

    cfg = dict(
        actor_hidden_dims=AGENT_CONFIG['actor_hidden_dims'],
        critic_hidden_dims=AGENT_CONFIG['critic_hidden_dims'],
        actor_lr=AGENT_CONFIG['actor_lr'], critic_lr=AGENT_CONFIG['critic_lr'],
        gamma=AGENT_CONFIG['gamma'], tau=AGENT_CONFIG['tau'],
        buffer_capacity=100,           # 只做推理，不需要大缓冲
        batch_size=AGENT_CONFIG['batch_size'],
        policy_freq=AGENT_CONFIG['policy_freq'],
        train_every=AGENT_CONFIG.get('train_every', 1),
        max_grad_norm=AGENT_CONFIG['max_grad_norm'],
        entropy_coef=AGENT_CONFIG.get('entropy_coef', 0.0),
        epsilon_decay=AGENT_CONFIG.get('epsilon_decay', 1.0),
        epsilon_min=AGENT_CONFIG.get('epsilon_min', 0.0),
    )
    env = MultiAgentEmergencyLaneEnv()
    agent = MADDPGAgent(env.n_agents, env.obs_dim, ENV_CONFIG['n_actions'], cfg)
    agent.load(model_path)
    agent.epsilon = 0.0          # 只看策略本身，不看随机探索

    # ---- 第 1、2 层：熵 / argmax 开放率 ----
    ents, steps = [], 0
    open_sample, open_argmax, near_half = 0.0, 0.0, 0
    for ep in range(n_episodes):
        random.seed(ep); np.random.seed(ep)
        obs = env.reset()
        done = False
        while not done:
            probs = agent.action_probs(obs)                 # (A, 2)
            p1 = probs[:, 1]
            ents.append(float(np.mean(-(p1 * np.log(p1 + 1e-8)
                                        + (1 - p1) * np.log(1 - p1 + 1e-8)))))
            near_half += int(np.sum(np.abs(p1 - 0.5) < 0.05))
            open_argmax += float(np.sum(probs.argmax(axis=1) == 1))
            a = agent.select_action(obs, evaluate=True)     # 旧口径：采样
            open_sample += float(np.sum(a))
            steps += env.n_agents
            obs, _, done, _ = env.step(a)

    out = {
        'entropy': float(np.mean(ents)),
        'open_rate': open_sample / max(steps, 1),
        'argmax_open_rate': open_argmax / max(steps, 1),
        'near_half_frac': near_half / max(steps, 1),
        'n_decisions': steps,
    }

    # ---- 第 3 层：与平凡基线的配对对照 ----
    if quick_episodes > 0:
        from evaluate import run_strategy, make_policy_always, make_policy_maddpg
        seeds = list(range(seed_start, seed_start + quick_episodes))
        env2 = MultiAgentEmergencyLaneEnv()
        res_model = run_strategy(env2, make_policy_maddpg(agent, True),
                                 quick_episodes, seeds=seeds)
        res_close = run_strategy(env2, make_policy_always(0),
                                 quick_episodes, seeds=seeds)
        res_open = run_strategy(env2, make_policy_always(1),
                                quick_episodes, seeds=seeds)
        m = res_model['mean_reward']
        c = res_close['mean_reward']
        o = res_open['mean_reward']
        out.update({
            'quick_model_reward': m,
            'quick_close_reward': c,
            'quick_open_reward': o,
            'gain_vs_close': m - c,
            'gain_vs_open': m - o,
        })
    return out


def verdict(ent: float, argmax_open: float, r: dict):
    """三层判据；返回 (判定字符串, 是否通过)。

    注意：第 3 层只有几个回合，SEM 约 50~60，因此
      - 「vs 恒关闭」是稳健判据（退化时差值会掉到 0 附近，量级 500+ 的差距不会误判）
      - 「vs 恒开放」在小样本下噪声较大，只作为 WARN 提示
    """
    if ent <= 0.05:
        return 'FAIL  熵已坍缩（策略锁死）', False
    if 'gain_vs_close' in r:
        g_close, g_open = r['gain_vs_close'], r['gain_vs_open']
        if g_close <= MARGIN:
            return (f'FAIL  退化：比「恒关闭」只高 {g_close:+.0f}，等于不做有效控制',
                    False)
        if g_open <= 0.0:
            return (f'WARN  优于恒关闭 {g_close:+.0f}，但没跑赢「恒开放」'
                    f'({g_open:+.0f})；小样本下噪声较大，建议 --quick-episodes 加到 10+',
                    True)
        return (f'PASS  同时优于恒关闭({g_close:+.0f})与恒开放({g_open:+.0f})', True)
    if argmax_open <= 0.005:
        return 'FAIL  argmax 退化为「恒关闭」（旧工具看不到这个！）', False
    if argmax_open >= 0.995:
        return 'FAIL  argmax 退化为「恒开放」', False
    return 'PASS  （未做快速对照评估）', True


def main() -> int:
    parser = argparse.ArgumentParser(description="检查策略健康度：熵 / argmax / 对照评估")
    parser.add_argument('--model', default=os.path.join(ROOT, 'models_v2',
                                                        'maddpg_best_avg.pth'))
    parser.add_argument('--model-dir', default=os.path.join(ROOT, 'models_v2'),
                        help='--all 时要扫描的目录')
    parser.add_argument('--all', action='store_true', help='扫描目录下全部检查点')
    parser.add_argument('--episodes', type=int, default=3,
                        help='第 1、2 层的回合数')
    parser.add_argument('--quick-episodes', type=int, default=10,
                        help='第 3 层对照评估的回合数（默认 10；0 = 跳过，会快很多）')
    parser.add_argument('--seed-start', type=int, default=5000)
    args = parser.parse_args()

    print('=' * 104)
    print('策略健康度检查：熵 / argmax 开放率 / 与平凡基线的配对对照')
    print('=' * 104)
    print(f'熵上限 ln2 = {LOG2:.4f}')
    if args.quick_episodes > 0:
        print(f'对照评估：argmax 策略 vs 恒关闭/恒开放，'
              f'{args.quick_episodes} 个配对场景（奖励持平余量 {MARGIN:.0f}）')
    print()

    if args.all:
        paths = sorted(glob.glob(os.path.join(args.model_dir, '*.pth')))
        paths = [p for p in paths if 'resume' not in os.path.basename(p)]
        if not paths:
            print(f'未找到任何检查点: {args.model_dir}')
            return 1
    else:
        paths = [args.model]
        if not os.path.exists(args.model):
            print(f'模型不存在: {args.model}\n训练后重试，或用 --model 指定路径。')
            return 1

    print(f"{'检查点':<32}{'熵':>8}{'采样开率':>10}{'argmax开率':>12}"
          f"{'奖励':>10}{'vs关闭':>10}{'vs开放':>10}   判定")
    print('-' * 104)
    all_ok = True
    for p in paths:
        try:
            r = evaluate_model(p, args.episodes, args.quick_episodes,
                               args.seed_start)
        except Exception as e:  # noqa: BLE001
            print(f'{os.path.basename(p):<32}  加载失败: {e}')
            all_ok = False
            continue
        v, ok = verdict(r['entropy'], r['argmax_open_rate'], r)
        all_ok &= ok
        qm = r.get('quick_model_reward')
        gc = r.get('gain_vs_close')
        go = r.get('gain_vs_open')
        print(f"{os.path.basename(p):<32}{r['entropy']:>8.4f}"
              f"{r['open_rate']:>9.1%}{r['argmax_open_rate']:>11.1%}"
              f"{(f'{qm:.0f}' if qm is not None else 'N/A'):>10}"
              f"{(f'{gc:+.0f}' if gc is not None else 'N/A'):>10}"
              f"{(f'{go:+.0f}' if go is not None else 'N/A'):>10}   {v}")

    print('-' * 104)
    print()
    if all_ok:
        print('全部检查点通过。')
        return 0
    print('检测到问题。建议：')
    print('  - 熵坍缩        -> 提高 entropy_coef 起点 / 拉长 entropy_anneal_episodes')
    print('  - argmax 退化   -> 检查 actor_update_mode 是否为 counterfactual；')
    print('                     若已是，说明熵系数退火过头，把 entropy_coef_end 调大')
    print('  - 训练日志里每 10 轮会打印 ArgmaxOpen，可在线监控')
    return 1


if __name__ == '__main__':
    sys.exit(main())

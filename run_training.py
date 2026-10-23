#!/usr/bin/env python
"""
MADDPG 完整训练启动脚本（2026-10-01 修复版）

用法:
    python run_training.py                          # 按配置训练（写到 ./models_fixed_single）
    python run_training.py --episodes 200 --no-resume   # 小规模验证
    python run_training.py --episodes 1500 --seed 42 --tag v3a

输出位置（重要）:
    默认写 ./models_fixed_single 与 ./logs_fixed_single，**不会**覆盖旧目录里已有的
    maddpg_best.pth（那是当前已验证的最优模型）。
    如需写回旧目录，必须显式指定：
        --model-dir ./models_matd3 --log-dir ./logs_matd3
    （即使这样，本版本也不写 maddpg_best.pth，只写 best_avg / best_peak）

本次相对上一版的改动（详见 README.md 与 会话记录与交接文档.md）:
    1. Actor 更新改为「反事实期望 Q」：critic 的输入回到独热动作（分布内），
       解决"actor 拿不到有效 Q 信号、概率长期贴在 0.5"的根因
    2. 目标 Q 改为独热 argmax + 离散版目标策略平滑（原来把软概率喂给
       只在独热动作上训练过的 critic，属分布外求值）
    3. 熵系数线性退火（前期探索 / 后期允许确定），不再用固定常数
    4. ε 改为线性衰减，探索时长可控
    5. best 模型按**滑动平均**挑选（原来按单回合最高分，噪声极大）
    6. 训练日志新增 ArgmaxOpen / Entropy / EntCoef，退化当场可见
    7. 输出目录与修复前的模型隔离
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maddpg_v2 import train, _apply_cli_overrides          # noqa: E402
from config_maddpg import TRAIN_CONFIG, AGENT_CONFIG, CA_CONFIG, ENV_CONFIG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MADDPG 训练启动脚本")
    parser.add_argument('--episodes', type=int, default=None,
                        help='覆盖配置中的训练轮数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（用于复现；续训时忽略）')
    parser.add_argument('--no-resume', action='store_true',
                        help='忽略已有续训检查点，从头开始训练')
    parser.add_argument('--resume-from', default=None,
                        help='指定要恢复的检查点路径')
    parser.add_argument('--model-dir', default=None,
                        help='模型输出目录（默认 ./models_fixed_single）')
    parser.add_argument('--log-dir', default=None,
                        help='日志输出目录（默认 ./logs_fixed_single）')
    parser.add_argument('--tag', default=None,
                        help='文件名前缀，便于同一目录内放多组实验')
    parser.add_argument('--experiment-config', default=None,
                        help='JSON file with ca/env overrides for controlled experiments')
    parser.add_argument('--save-freq', type=int, default=None,
                        help='Periodic checkpoint interval in episodes')
    parser.add_argument('--entropy-coef', type=float, default=None,
                        help='熵系数退火起点（默认 1.0）')
    parser.add_argument('--entropy-coef-end', type=float, default=None,
                        help='熵系数退火终点（默认 0.05）')
    parser.add_argument('--entropy-anneal-episodes', type=int, default=None,
                        help='熵系数退火轮数（默认 800）')
    parser.add_argument('--target-action-mode', default=None,
                        choices=['greedy', 'soft'],
                        help='目标 Q 动作模式：greedy=独热(推荐)，soft=旧行为')
    parser.add_argument('--actor-update-mode', default=None,
                        choices=['counterfactual', 'legacy'],
                        help='Actor 更新：counterfactual=反事实期望Q(推荐)，legacy=旧行为(仅A/B)')
    parser.add_argument('--twin-critics', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='twin critic（TD3 clipped double-Q，抑制 Q 过估计）；'
                             '默认读 config_maddpg.AGENT_CONFIG["twin_critics"]')
    args = parser.parse_args()
    if args.save_freq is not None:
        if args.save_freq <= 0:
            parser.error('--save-freq must be positive')
        TRAIN_CONFIG['save_freq'] = args.save_freq

    if args.experiment_config:
        with open(args.experiment_config, encoding='utf-8') as handle:
            overrides = json.load(handle)
        if set(overrides) - {'ca', 'env'}:
            parser.error('experiment config may contain only ca and env')
        for section, target in (('ca', CA_CONFIG), ('env', ENV_CONFIG)):
            entries = overrides.get(section, {})
            unknown = set(entries) - set(target)
            if section == 'ca':
                unknown.discard('n_cars_choices')
            if unknown:
                parser.error(f'Unknown {section} keys: {unknown}')
            for key, value in entries.items():
                if key == 'reward_weights':
                    target[key] = {**target[key], **value}
                else:
                    target[key] = value
        if ENV_CONFIG['n_agents'] != ENV_CONFIG['n_segments']:
            parser.error('n_agents must equal n_segments')
        if CA_CONFIG['L'] % ENV_CONFIG['n_segments']:
            parser.error('Road length must be divisible by n_segments')

    _apply_cli_overrides(args)

    n_episodes = args.episodes if args.episodes else TRAIN_CONFIG['n_episodes']
    model_dir = args.model_dir or TRAIN_CONFIG['model_dir']
    log_dir = args.log_dir or TRAIN_CONFIG.get('log_dir') or model_dir
    prefix = f"{args.tag}_" if args.tag else ""
    resume_path = args.resume_from or os.path.join(model_dir,
                                                   f"{prefix}maddpg_resume.pth")
    has_resume = os.path.exists(resume_path) and not args.no_resume

    print("=" * 74)
    print(f"MADDPG Training - target {n_episodes} episodes")
    print("=" * 74)
    print("改进内容（本版本）:")
    print("  1. Actor 更新: 反事实期望 Q（critic 输入回到独热动作，分布内）")
    print("  2. 目标 Q   : 独热 argmax + 离散目标策略平滑（原为分布外软动作）")
    print("  3. 熵正则   : 固定 5.0 -> 线性退火 1.0→0.05")
    print("  4. 探索     : ε 线性衰减，探索时长可控")
    print("  5. best 挑选: 滑动平均窗口（原为单回合最高分，噪声极大）")
    print("  6. 日志     : 新增 ArgmaxOpen / Entropy / EntCoef 退化监控")
    print("  7. 输出目录 : models_fixed_single / logs_fixed_single")
    print("  8. 双 critic: twin critic + min(Q1,Q2)，抑制 Q 过估计")
    print("-" * 74)
    print(f"目标轮数  : {n_episodes}")
    print(f"模型目录  : {os.path.abspath(model_dir)}")
    print(f"日志目录  : {os.path.abspath(log_dir)}")
    print(f"续训检查点: {os.path.abspath(resume_path)}")
    print(f"Twin critic: {AGENT_CONFIG.get('twin_critics')}")
    print(f"熵退火    : {AGENT_CONFIG.get('entropy_coef')} -> "
          f"{AGENT_CONFIG.get('entropy_coef_end')} "
          f"over {AGENT_CONFIG.get('entropy_anneal_episodes')} eps")
    print(f"目标Q模式 : {AGENT_CONFIG.get('target_action_mode')} "
          f"(flip_p={AGENT_CONFIG.get('target_flip_p')})")
    if os.path.normpath(model_dir).replace('\\', '/').endswith('models_matd3'):
        print("\n[WARN] 你正在写入旧的 models_matd3 目录。本版本不会写 "
              "maddpg_best.pth，旧最优模型安全，但新旧产物会混在一起。")
    if has_resume:
        print("\n>>> 检测到续训检查点，将从上次中断处继续训练 <<<")
        print("    若要放弃并从头训练，加参数 --no-resume")
    else:
        print("\n>>> 未检测到续训检查点，将从头开始训练 <<<")
    print("=" * 74)

    train(n_episodes=n_episodes, seed=args.seed,
          resume=not args.no_resume, resume_path=args.resume_from,
          model_dir=args.model_dir, log_dir=args.log_dir, tag=args.tag)
    return 0


if __name__ == '__main__':
    sys.exit(main())

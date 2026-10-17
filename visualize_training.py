#!/usr/bin/env python
"""
MADDPG 训练过程和结果可视化
包含：训练曲线、策略对比、模型行为分析
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import re
import torch
from collections import defaultdict

from env_maddpg import MultiAgentEmergencyLaneEnv
from config_maddpg import ENV_CONFIG, TRAIN_CONFIG
from maddpg_v2 import Actor, MADDPGAgent


def parse_training_log(log_path=None):
    """解析训练日志，提取训练数据。

    日志路径优先级（2026-10-01 更新，加入多实验目录发现）：
      1) 显式传入的 log_path
      2) config_maddpg.TRAIN_CONFIG['log_file']（当前默认 ./logs_v2/...）
      3) 自动发现 ./logs*/training_output.log（如 logs_v3_a、logs_v3_b），
         有多个时取**最近修改**的那个
      4) 兼容旧的 ./training_output.log
    """
    if log_path is None:
        candidates = []
        try:
            from config_maddpg import TRAIN_CONFIG
            if TRAIN_CONFIG.get('log_file'):
                candidates.append(TRAIN_CONFIG['log_file'])
        except Exception:
            pass
        # 自动发现所有 logs*/training_output.log，按修改时间倒序
        found = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'logs*', 'training_output.log'))
        found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        candidates.extend(found)
        candidates.append('./training_output.log')
        existing = [p for p in candidates if os.path.exists(p)]
        if len(existing) > 1:
            print("[日志] 发现多个训练日志，使用最近修改的:")
            for p in existing[:4]:
                print(f"       {p}")
            print("       如需指定其它日志，请显式传入路径或调用 --log")
        log_path = existing[0] if existing else candidates[0]

    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        print("  提示：请先运行 python run_training.py 生成训练日志。")
        return None
    
    episodes = []
    rewards = []
    avg_rewards = []
    epsilons = []
    times = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配训练日志行
            match = re.search(
                r'Episode\s+(\d+)/(\d+)\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Avg:\s+([\d.]+)\s+\|\s+Epsilon:\s+([\d.]+)',
                line
            )
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(3)))
                avg_rewards.append(float(match.group(4)))
                epsilons.append(float(match.group(5)))
    
    if not episodes:
        print("未找到训练数据")
        return None
    
    return {
        'episodes': np.array(episodes),
        'rewards': np.array(rewards),
        'avg_rewards': np.array(avg_rewards),
        'epsilons': np.array(epsilons)
    }


def plot_training_curves(data, save_path='./visualization/training_curves.png'):
    """绘制训练曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = data['episodes']
    rewards = data['rewards']
    avg_rewards = data['avg_rewards']
    epsilons = data['epsilons']
    
    # 1. 原始奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.4, color='steelblue', linewidth=0.8, label='Episode Reward')
    ax1.plot(episodes, avg_rewards, color='darkred', linewidth=2, label='100-Episode Average')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Reward Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 平滑奖励曲线（不同窗口）
    ax2 = axes[0, 1]
    windows = [50, 100, 200]
    colors = ['green', 'orange', 'purple']
    
    for window, color in zip(windows, colors):
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax2.plot(episodes[window-1:], smoothed, color=color, linewidth=2, 
                    label=f'Window={window}')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Smoothed Reward', fontsize=12)
    ax2.set_title('Smoothed Reward Curves', fontsize=14, fontweight='bold')
    # 短训练（回合数少于最小窗口）时可能没有任何曲线，此时跳过图例避免告警
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon衰减曲线
    ax3 = axes[1, 0]
    ax3.plot(episodes, epsilons, color='teal', linewidth=2)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Epsilon', fontsize=12)
    ax3.set_title('Exploration Rate (Epsilon) Decay', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 奖励分布直方图
    ax4 = axes[1, 1]
    n_bins = 50
    ax4.hist(rewards, bins=n_bins, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(np.mean(rewards), color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rewards):.2f}')
    ax4.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(rewards):.2f}')
    ax4.set_xlabel('Reward', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")
    plt.close()


def plot_strategy_comparison(json_path='./evaluation_results.json', 
                              save_path='./visualization/strategy_comparison.png'):
    """绘制策略对比图"""
    if not os.path.exists(json_path):
        print(f"评估结果文件不存在: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    strategies = list(results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # 1. 平均奖励对比
    ax1 = axes[0, 0]
    mean_rewards = [results[s]['mean_reward'] for s in strategies]
    std_rewards = [results[s]['std_reward'] for s in strategies]
    
    bars = ax1.bar(strategies, mean_rewards, yerr=std_rewards, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Strategy Comparison: Mean Reward', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, val in zip(bars, mean_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 奖励标准差对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(strategies, std_rewards, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Reward Std Dev', fontsize=12)
    ax2.set_title('Strategy Comparison: Stability (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, std_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 应急车辆速度对比
    ax3 = axes[1, 0]
    emergency_speeds = [results[s]['mean_emergency_speed'] for s in strategies]
    bars3 = ax3.bar(strategies, emergency_speeds, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Mean Emergency Vehicle Speed', fontsize=12)
    ax3.set_title('Strategy Comparison: Emergency Vehicle Speed', fontsize=14, fontweight='bold')
    ax3.set_ylim([4.5, 5.2])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, emergency_speeds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 4. 社会车辆速度对比
    ax4 = axes[1, 1]
    social_speeds = [results[s]['mean_social_speed'] for s in strategies]
    bars4 = ax4.bar(strategies, social_speeds, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Mean Social Vehicle Speed', fontsize=12)
    ax4.set_title('Strategy Comparison: Social Vehicle Speed', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars4, social_speeds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"策略对比图已保存: {save_path}")
    plt.close()


def load_model_for_analysis(model_path='./models_matd3/maddpg_best.pth'):
    """加载训练好的模型用于分析"""
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 创建环境
    env = MultiAgentEmergencyLaneEnv()
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = ENV_CONFIG['n_actions']
    
    config = {
        'actor_hidden_dims': [128, 64],
        'critic_hidden_dims': [256, 128, 64],
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.95,
        'tau': 0.005,
        'buffer_capacity': 500000,
        'batch_size': 128,
        'policy_freq': 2,
        'max_grad_norm': 1.0,
    }
    
    agent = MADDPGAgent(n_agents, obs_dim, n_actions, config)
    agent.load(model_path)
    agent.epsilon = 0  # 评估时关闭探索
    
    return agent, env


def visualize_policy_behavior(agent, env, n_episodes=3, save_path='./visualization/policy_behavior.png'):
    """可视化策略行为"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_episode_data = []
    
    for ep in range(n_episodes):
        observations = env.reset()
        done = False
        step = 0
        
        episode_data = {
            'actions': [],
            'rewards': [],
            'emergency_speeds': [],
            'social_speeds': []
        }
        
        while not done and step < 200:
            actions = agent.select_action(observations, evaluate=True)
            next_observations, rewards, done, info = env.step(actions)
            
            episode_data['actions'].append(actions.copy())
            episode_data['rewards'].append(rewards.mean())
            episode_data['emergency_speeds'].append(info.get('emergency_speed', 0))
            episode_data['social_speeds'].append(info.get('social_speed', 0))
            
            observations = next_observations
            step += 1
        
        all_episode_data.append(episode_data)
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 使用第一个回合的数据进行详细展示
    data = all_episode_data[0]
    steps = range(len(data['actions']))
    actions_array = np.array(data['actions'])
    
    # 1. 智能体动作选择
    ax1 = axes[0, 0]
    for i in range(actions_array.shape[1]):
        ax1.plot(steps, actions_array[:, i], label=f'Agent {i+1}', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Action (0=Close, 1=Open)', fontsize=12)
    ax1.set_title('Agent Actions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.1, 1.1])
    
    # 2. 奖励变化
    ax2 = axes[0, 1]
    ax2.plot(steps, data['rewards'], color='green', linewidth=2)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 车辆速度
    ax3 = axes[1, 0]
    ax3.plot(steps, data['emergency_speeds'], label='Emergency Vehicle', 
             color='red', linewidth=2)
    ax3.plot(steps, data['social_speeds'], label='Social Vehicles', 
             color='blue', linewidth=2)
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Speed', fontsize=12)
    ax3.set_title('Vehicle Speeds Over Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 多回合奖励对比
    ax4 = axes[1, 1]
    for i, ep_data in enumerate(all_episode_data):
        ax4.plot(range(len(ep_data['rewards'])), ep_data['rewards'], 
                label=f'Episode {i+1}', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Reward', fontsize=12)
    ax4.set_title('Multi-Episode Reward Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"策略行为图已保存: {save_path}")
    plt.close()


def plot_training_statistics(data, save_path='./visualization/training_statistics.png'):
    """绘制训练统计信息"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = data['episodes']
    rewards = data['rewards']
    
    # 1. 累积平均奖励
    ax1 = axes[0, 0]
    cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    ax1.plot(episodes, cumulative_avg, color='navy', linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Cumulative Average Reward', fontsize=12)
    ax1.set_title('Cumulative Average Reward', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 奖励趋势（分段平均）
    ax2 = axes[0, 1]
    segment_size = 100
    n_segments = len(rewards) // segment_size
    segment_avgs = []
    segment_labels = []
    
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size
        segment_avgs.append(np.mean(rewards[start:end]))
        segment_labels.append(f'{start+1}-{end}')
    
    bars = ax2.bar(segment_labels, segment_avgs, color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Episode Range', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title(f'Reward by Episode Segments (size={segment_size})', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 奖励箱线图（按阶段）
    ax3 = axes[1, 0]
    phases = ['Early (1-500)', 'Middle (501-1000)', 'Late (1001+)']
    phase_data = [
        rewards[:500] if len(rewards) >= 500 else rewards,
        rewards[500:1000] if len(rewards) >= 1000 else [],
        rewards[1000:] if len(rewards) > 1000 else []
    ]
    phase_data = [d for d in phase_data if len(d) > 0]
    valid_phases = phases[:len(phase_data)]
    
    bp = ax3.boxplot(phase_data, tick_labels=valid_phases, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Reward Distribution by Training Phase', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 训练统计文本
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Training Statistics Summary
    ============================
    
    Total Episodes: {len(rewards)}
    
    Reward Statistics:
    • Mean: {np.mean(rewards):.2f}
    • Std Dev: {np.std(rewards):.2f}
    • Min: {np.min(rewards):.2f}
    • Max: {np.max(rewards):.2f}
    • Median: {np.median(rewards):.2f}
    
    Best Performance:
    • Best Episode: {np.argmax(rewards) + 1}
    • Best Reward: {np.max(rewards):.2f}
    
    Recent Performance:
    • Last 100 Avg: {np.mean(rewards[-100:]):.2f}
    • Last 50 Avg: {np.mean(rewards[-50:]):.2f}
    
    Training Progress:
    • First 100 Avg: {np.mean(rewards[:100]):.2f}
    • Improvement: {np.mean(rewards[-100:]) - np.mean(rewards[:100]):.2f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练统计图已保存: {save_path}")
    plt.close()


def generate_summary_report(data, save_path='./visualization/training_report.txt'):
    """生成训练总结报告"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rewards = data['rewards']
    
    report = f"""
{'='*70}
MADDPG Training Summary Report
{'='*70}

1. Training Overview
--------------------
Total Episodes: {len(rewards)}
Final Average Reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}
Best Episode Reward: {np.max(rewards):.2f} (Episode {np.argmax(rewards) + 1})

2. Reward Statistics
--------------------
Mean: {np.mean(rewards):.2f}
Standard Deviation: {np.std(rewards):.2f}
Minimum: {np.min(rewards):.2f}
Maximum: {np.max(rewards):.2f}
Median: {np.median(rewards):.2f}

3. Training Progress
--------------------
First 100 Episodes Average: {np.mean(rewards[:100]):.2f}
Last 100 Episodes Average: {np.mean(rewards[-100:]):.2f}
Overall Improvement: {np.mean(rewards[-100:]) - np.mean(rewards[:100]):.2f}

4. Performance by Phase
-----------------------
Early Phase (1-500): {np.mean(rewards[:500]):.2f}
"""
    
    if len(rewards) > 500:
        report += f"Middle Phase (501-1000): {np.mean(rewards[500:1000]):.2f}\n"
    if len(rewards) > 1000:
        report += f"Late Phase (1001-{len(rewards)}): {np.mean(rewards[1000:]):.2f}\n"
    
    report += f"""
5. Stability Analysis
---------------------
Coefficient of Variation: {np.std(rewards)/np.mean(rewards)*100:.2f}%
Reward Range: {np.max(rewards) - np.min(rewards):.2f}

6. Convergence Analysis
-----------------------
"""
    
    # 检测收敛
    window = 100
    if len(rewards) >= window * 2:
        recent_std = np.std(rewards[-window:])
        report += f"Recent {window} episodes std dev: {recent_std:.2f}\n"
        if recent_std < 50:
            report += "Status: Model appears to have converged (low variance in recent episodes)\n"
        else:
            report += "Status: Model may still be learning (high variance in recent episodes)\n"
    
    report += f"""
{'='*70}
End of Report
{'='*70}
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"训练报告已保存: {save_path}")
    return report


def main():
    """主函数"""
    print("="*70)
    print("MADDPG Training Visualization")
    print("="*70)
    
    # 创建可视化目录
    os.makedirs('./visualization', exist_ok=True)
    
    # 1. 解析训练日志
    print("\n[1/5] Parsing training log...")
    data = parse_training_log()
    
    if data is None:
        print("Failed to parse training log. Exiting.")
        return
    
    print(f"Successfully parsed {len(data['episodes'])} episodes")
    
    # 2. 绘制训练曲线
    print("\n[2/5] Plotting training curves...")
    plot_training_curves(data)
    
    # 3. 绘制训练统计
    print("\n[3/5] Plotting training statistics...")
    plot_training_statistics(data)
    
    # 4. 策略对比
    print("\n[4/5] Plotting strategy comparison...")
    plot_strategy_comparison()
    
    # 5. 策略行为可视化
    print("\n[5/5] Visualizing policy behavior...")
    model_path = './models_matd3/maddpg_best.pth'
    if os.path.exists(model_path):
        try:
            agent, env = load_model_for_analysis(model_path)
            visualize_policy_behavior(agent, env)
        except Exception as e:
            print(f"Warning: Could not visualize policy behavior: {e}")
            print("Skipping policy behavior visualization...")
    else:
        print(f"Model not found at {model_path}, skipping policy behavior visualization")
    
    # 6. 生成报告
    print("\n[Bonus] Generating summary report...")
    report = generate_summary_report(data)
    print(report)
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - visualization/training_curves.png")
    print("  - visualization/training_statistics.png")
    print("  - visualization/strategy_comparison.png")
    print("  - visualization/policy_behavior.png (if model available)")
    print("  - visualization/training_report.txt")
    print("="*70)


if __name__ == '__main__':
    main()

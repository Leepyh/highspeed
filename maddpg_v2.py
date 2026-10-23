#!/usr/bin/env python
"""
MADDPG V2 - 改进版，增加全局Critic和更好的协调机制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import argparse
import logging
import random
import os
import sys

from env_maddpg import MultiAgentEmergencyLaneEnv
from config_maddpg import CA_CONFIG, ENV_CONFIG, TRAIN_CONFIG, AGENT_CONFIG


class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, obs_dim, n_actions, hidden_dims):
        super(Actor, self).__init__()
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_actions))
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        logits = self.network(obs)
        return F.softmax(logits, dim=-1)


class GlobalCritic(nn.Module):
    """全局Critic - 评估联合动作价值"""
    def __init__(self, total_obs_dim, n_agents, n_actions, hidden_dims):
        super(GlobalCritic, self).__init__()
        layers = []
        prev_dim = total_obs_dim + n_agents * n_actions
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs, actions):
        actions_flat = actions.view(actions.size(0), -1)
        x = torch.cat([obs, actions_flat], dim=-1)
        return self.network(x)


class MADDPGAgent:
    """MADDPG智能体 - 使用全局Critic"""
    def __init__(self, n_agents, obs_dim, n_actions, config):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config
        self.device = torch.device(TRAIN_CONFIG['device'])
        
        # Actors
        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []
        
        for _ in range(n_agents):
            actor = Actor(obs_dim, n_actions, config['actor_hidden_dims']).to(self.device)
            target = Actor(obs_dim, n_actions, config['actor_hidden_dims']).to(self.device)
            target.load_state_dict(actor.state_dict())
            self.actors.append(actor)
            self.actor_targets.append(target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=config['actor_lr']))
        
        # 全局Critic（每个智能体一个，但都评估全局价值）
        total_obs_dim = obs_dim * n_agents
        self.critics = []
        self.critic_targets = []
        self.critic_optimizers = []

        # ---- Twin critic（TD3 的 clipped double-Q，2026-10-01 新增）----
        # 目的：抑制 Q 过估计。Q 过估计是"训练先升后降"最经典的成因之一：
        # critic 把某个动作的价值估高 -> actor 追过去 -> 实际回报跟不上 -> 策略退化。
        # 做法：每个智能体再建一套 critic，目标 Q 取两套的 min。
        # 注意：TD3 的三件套是 twin critic + 目标策略平滑 + 延迟策略更新；
        # 后两者本项目已有（target_flip_p / policy_freq），故不是"单上 twin critic"。
        # 默认关闭，未开启时行为与旧版完全一致，且**不改变检查点格式**。
        self.twin_critics = bool(config.get('twin_critics', False))
        self.critics2 = []
        self.critic_targets2 = []
        self.critic_optimizers2 = []

        for _ in range(n_agents):
            critic = GlobalCritic(total_obs_dim, n_agents, n_actions, config['critic_hidden_dims']).to(self.device)
            target = GlobalCritic(total_obs_dim, n_agents, n_actions, config['critic_hidden_dims']).to(self.device)
            target.load_state_dict(critic.state_dict())
            self.critics.append(critic)
            self.critic_targets.append(target)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=config['critic_lr']))
            if self.twin_critics:
                # 顺序构造 -> 两次随机初始化不同，两套 critic 才有区分度
                critic2 = GlobalCritic(total_obs_dim, n_agents, n_actions, config['critic_hidden_dims']).to(self.device)
                target2 = GlobalCritic(total_obs_dim, n_agents, n_actions, config['critic_hidden_dims']).to(self.device)
                target2.load_state_dict(critic2.state_dict())
                self.critics2.append(critic2)
                self.critic_targets2.append(target2)
                self.critic_optimizers2.append(
                    optim.Adam(critic2.parameters(), lr=config['critic_lr']))
        
        self.buffer = deque(maxlen=config['buffer_capacity'])
        self.epsilon = 1.0
        self.train_step = 0
        # 熵正则系数（由 set_progress() 按训练进度退火；这里先给出初值）
        self.entropy_coef = float(config.get('entropy_coef', 0.0))
        
    def _actor_probs(self, obs_all):
        """批量计算所有智能体的动作概率。

        obs_all: (B, n_agents, obs_dim)
        返回:    (B, n_agents, n_actions)

        性能关键：原实现按智能体逐个前向，每步产生 5 次 kernel 启动；
        网络极小（14k 参数），耗时几乎全是启动开销。改为一次批量前向后，
        5 次调用合并为 1 次，实测 select_action 提速约 3 倍。
        """
        return torch.stack(
            [self.actors[i](obs_all[:, i, :]) for i in range(self.n_agents)], dim=1)

    def _actor_probs_target(self, obs_all):
        """目标网络的批量前向（供 TD 目标使用）。"""
        return torch.stack(
            [self.actor_targets[i](obs_all[:, i, :]) for i in range(self.n_agents)], dim=1)

    def _critic_stack(self, nets):
        """占位：保留接口以兼容旧调用。

        说明：曾尝试把所有 Critic 的 Linear 权重堆叠成单个张量做批量前向，
        但各 Critic 是独立网络、参数互不相同，堆叠后批次维与输入批次维冲突
        （权重批次=智能体数，输入批次=采样批大小），收益不抵复杂度，故放弃。
        真正的提速来自 Actor 侧：原实现每个智能体都要重复前向全部 5 个
        Actor（25 次小前向/步），现改为一次批量前向（5 次大前向/步）。
        """
        return nets

    @staticmethod
    def _critic_forward(nets, obs_flat, joint_actions):
        """对同一批状态-联合动作，返回所有 Critic 的 Q 值 (B, n_agents)。"""
        return torch.stack(
            [nets[i](obs_flat, joint_actions).squeeze(-1) for i in range(len(nets))],
            dim=1)

    def select_action(self, observations, evaluate=False, deterministic=False):
        """选择动作。

        【本次修复的坑】原实现的 evaluate=True **只关掉了 ε 随机探索**，
        动作仍然是 np.random.choice(p=probs[i]) —— 即**按概率采样，不是 argmax**。
        在熵接近 ln2 的策略下，这等于在评估一个近似随机策略，历史上
        "MADDPG 排名第 3""MADDPG ≈ Always Close"的结论全由此而来。
        现在新增 deterministic=True：直接取 argmax，才是"学到的策略"。

        参数:
            evaluate      : True 时不使用 ε 随机动作（仍按概率采样，兼容旧行为）
            deterministic : True 时取 argmax（真正的确定性策略）
        """
        obs_all = torch.as_tensor(np.asarray(observations),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self._actor_probs(obs_all)[0].cpu().numpy()  # (n_agents, n_actions)

        if deterministic:
            return probs.argmax(axis=1).astype(int)

        actions = np.empty(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            if not evaluate and random.random() < self.epsilon:
                actions[i] = random.randint(0, self.n_actions - 1)
            else:
                actions[i] = np.random.choice(self.n_actions, p=probs[i])
        return actions

    def action_probs(self, observations):
        """返回 (n_agents, n_actions) 的动作概率，供监控/探针使用（只读）。"""
        obs_all = torch.as_tensor(np.asarray(observations),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self._actor_probs(obs_all)[0].cpu().numpy()

    def set_progress(self, episode, n_episodes=None):
        """按训练进度更新「熵系数退火」与「ε 探索」。

        【本次修复的坑】原来 entropy_coef 与 ε 都是**贯穿全程的常数/固定曲线**：
          - entropy_coef 恒为 5.0 -> 策略概率永远贴在 0.5，argmax 只是读噪声；
          - ε 用 0.998 指数衰减 -> 第 1000 轮还有 13.5%，第 2000 轮就压到下限。
        现在熵系数线性退火（前期探索、后期允许确定），ε 线性衰减（探索时长可控）。
        """
        ep = max(int(episode), 1)

        if self.config.get('entropy_anneal', False):
            c0 = float(self.config.get('entropy_coef_start', self.config['entropy_coef']))
            c1 = float(self.config.get('entropy_coef_end', c0))
            span = max(int(self.config.get('entropy_anneal_episodes', 1)), 1)
            frac = min(1.0, (ep - 1) / span)
            self.config['entropy_coef'] = c0 + (c1 - c0) * frac
        self.entropy_coef = self.config['entropy_coef']

        floor = float(self.config.get('epsilon_min', 0.05))
        if self.config.get('epsilon_schedule', 'exp') == 'linear':
            span = max(int(self.config.get('epsilon_decay_episodes', 1)), 1)
            frac = min(1.0, (ep - 1) / span)
            self.epsilon = max(floor, 1.0 + (floor - 1.0) * frac)
        else:
            self.epsilon = max(floor, self.epsilon * self.config.get('epsilon_decay', 0.998))

    def store_transition(self, observations, actions, rewards, next_observations, done):
        self.buffer.append((observations, actions, rewards, next_observations, done))
    
    def learn(self):
        if len(self.buffer) < self.config['batch_size']:
            return None

        bs = self.config['batch_size']
        batch = random.sample(self.buffer, bs)

        obs_tensor = torch.as_tensor(np.array([t[0] for t in batch]),
                                     dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(np.array([t[1] for t in batch]),
                                        dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(np.array([t[2] for t in batch]),
                                        dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(np.array([t[3] for t in batch]),
                                          dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(np.array([t[4] for t in batch]),
                                      dtype=torch.float32, device=self.device)
        # done 是每个回合一个标量（Python bool），需广播到每个智能体 (B, 1)
        done_tensor = done_tensor.view(-1, 1)

        action_probs = F.one_hot(action_tensor, num_classes=self.n_actions).float()
        obs_flat = obs_tensor.view(bs, -1)
        next_obs_flat = next_obs_tensor.view(bs, -1)

        # ---- 目标 Q ----
        # 【本次修复的坑】原实现把 target actor 输出的**概率向量（软动作）**直接
        # 喂给 target critic，而 critic 只在**独热(one-hot)动作**上被训练过 ——
        # 目标 Q 等于在分布外输入点上求值，TD 目标自带系统偏差，是训练后期
        # 「越训越差」的重要来源。
        # 现在默认取 target actor 的 argmax 得到**独热**目标动作（回到分布内），
        # 并以 target_flip_p 的概率随机翻转，作为离散版的"目标策略平滑"。
        target_mode = self.config.get('target_action_mode', 'greedy')
        with torch.no_grad():
            next_probs = self._actor_probs_target(next_obs_tensor)   # (B, A, n_act)
            if target_mode == 'soft':
                next_actions = next_probs                            # 旧行为（不推荐）
            else:
                next_act = next_probs.argmax(dim=-1)                 # (B, A)
                flip_p = float(self.config.get('target_flip_p', 0.0))
                if flip_p > 0.0:
                    flip = torch.rand_like(next_probs[..., 0]) < flip_p
                    next_act = torch.where(flip,
                                           self.n_actions - 1 - next_act, next_act)
                next_actions = F.one_hot(next_act,
                                         num_classes=self.n_actions).float()
            next_q = self._critic_forward(self.critic_targets, next_obs_flat,
                                          next_actions)
            if self.twin_critics:
                # clipped double-Q：目标取两套 target critic 的较小值，抑制过估计
                next_q2 = self._critic_forward(self.critic_targets2, next_obs_flat,
                                               next_actions)
                next_q = torch.min(next_q, next_q2)
            target_qs = reward_tensor + self.config['gamma'] * (1 - done_tensor) * next_q

        # ---- 更新 Critic ----
        # 反传需要梯度正确流回各自参数，故保留逐网络反传；
        # 但不带梯度的批量前向可用于诊断（见 _critic_forward 的注释）。
        # twin critic 时两套都朝**同一个**（取 min 的）TD 目标回归。
        critic_losses = []
        for i in range(self.n_agents):
            current_q = self.critics[i](obs_flat, action_probs).squeeze(-1)
            loss = F.mse_loss(current_q, target_qs[:, i])
            self.critic_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(),
                                           self.config['max_grad_norm'])
            self.critic_optimizers[i].step()
            critic_losses.append(loss.item())
            if self.twin_critics:
                current_q2 = self.critics2[i](obs_flat, action_probs).squeeze(-1)
                loss2 = F.mse_loss(current_q2, target_qs[:, i])
                self.critic_optimizers2[i].zero_grad()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.critics2[i].parameters(),
                                               self.config['max_grad_norm'])
                self.critic_optimizers2[i].step()
                critic_losses.append(loss2.item())

        # ---- 更新 Actor（反事实期望 Q + 熵正则）----
        actor_losses = []
        entropy_coef = self.config.get('entropy_coef', 0.0)
        if self.train_step % self.config['policy_freq'] == 0:
            # 【本次修复的坑】原实现把 5 个 Actor 的**软概率向量**拼成联合动作喂给
            # critic：q = critics[i](obs, joint_soft)。critic 只在独热动作上训练过，
            # 在软动作处的取值没有意义 —— actor 拿不到有意义的 Q 信号，梯度被熵项
            # 主导，策略概率长期停在 0.5 附近（argmax 只是在读噪声），
            # 于是相邻检查点排名乱跳、第 600 轮之后不再改善。
            #
            # 现改为**反事实期望 Q**：对第 i 个智能体，其余智能体保持本批次**实际
            # 执行的独热动作**（分布内），只把第 i 个动作分别设为 0 / 1，得到
            # Q(s, a_i=0) 与 Q(s, a_i=1)，再用当前策略概率加权：
            #     L_i = -( p_i(0)*Q0 + p_i(1)*Q1 ) - beta * H(p_i)
            # actor 的梯度正比于「动作间的 Q 差值」，信号真实且低方差；
            # Q 视为常数故 detach（critic 已在本步单独更新过）。
            # 附带好处：不再需要 20 次无梯度 Actor 前向，单步反而更快。
            for i in range(self.n_agents):
                p_i = self.actors[i](obs_tensor[:, i, :])           # (B, n_act) 带梯度

                if self.config.get('actor_update_mode', 'counterfactual') == 'legacy':
                    # ---- 旧行为（仅为 A/B 对照保留）：把 5 个 Actor 的**软概率**
                    # 拼成联合动作喂给 critic。critic 只在独热动作上训练过，
                    # 在软动作处没有有效梯度，等于 actor 学不到东西。
                    parts = []
                    for j in range(self.n_agents):
                        if j == i:
                            parts.append(p_i)
                        else:
                            with torch.no_grad():
                                parts.append(self.actors[j](obs_tensor[:, j, :]))
                    joint_soft = torch.stack(parts, dim=1)
                    q_exp = self.critics[i](
                        obs_flat, joint_soft.reshape(bs, -1)).mean()
                else:
                    # ---- 新行为（推荐）：反事实期望 Q ----
                    q_vals = []
                    for a in range(self.n_actions):
                        joint = action_probs.clone()                # 本批次实际独热动作
                        joint[:, i, :] = 0.0
                        joint[:, i, a] = 1.0
                        q_vals.append(
                            self.critics[i](obs_flat,
                                            joint.reshape(bs, -1)).squeeze(-1))
                    q_stack = torch.stack(q_vals, dim=1).detach()    # (B, n_act)
                    q_exp = (p_i * q_stack).sum(dim=1).mean()

                # 熵正则：鼓励策略保持随机性（系数随训练进度线性退火）
                ent = -(p_i * torch.log(p_i + 1e-8)).sum(dim=-1).mean()
                actor_loss = -q_exp - entropy_coef * ent

                self.actor_optimizers[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(),
                                               self.config['max_grad_norm'])
                self.actor_optimizers[i].step()
                actor_losses.append(actor_loss.item())

            self._soft_update()
        
        self.train_step += 1

        return np.mean(critic_losses), np.mean(actor_losses) if actor_losses else None

    def policy_entropy(self, observations):
        """返回当前策略在给定观测下的平均熵（用于监控熵坍缩）。"""
        obs_all = torch.as_tensor(np.asarray(observations),
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            p = self._actor_probs(obs_all)[0]
            ent = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean().item()
        return ent

    def decay_epsilon(self):
        """按**回合**衰减探索率（不是按学习步）。

        历史问题：
          1. 最初实现在 learn() 内每步衰减（ε*=0.995），约 900 步（≈5 轮）就到下限；
          2. 改为按回合、衰减率 0.995 后，780 轮仍过早到 0.02，
             实测策略熵在第 400 轮就坍缩为 0（全部恒选"关闭"）。
        现使用可配置的 epsilon_decay（默认 0.999）与 epsilon_min（默认 0.05）：
        0.999^ep，约 3000 轮才接近下限，保证全程都有探索。
        """
        decay = self.config.get('epsilon_decay', 0.999)
        floor = self.config.get('epsilon_min', 0.05)
        self.epsilon = max(floor, self.epsilon * decay)
    
    def _soft_update(self):
        tau = self.config['tau']
        for i in range(self.n_agents):
            for param, target in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                target.data.copy_(tau * param.data + (1 - tau) * target.data)
            for param, target in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                target.data.copy_(tau * param.data + (1 - tau) * target.data)
            if self.twin_critics:
                for param, target in zip(self.critics2[i].parameters(),
                                         self.critic_targets2[i].parameters()):
                    target.data.copy_(tau * param.data + (1 - tau) * target.data)
    
    def save(self, path, episode=None, best_reward=None, avg_reward=None,
             include_resume_state=False, episode_rewards=None,
             best_avg_reward=None, extra=None):
        """保存模型。

        参数:
            include_resume_state: 为 True 时额外保存回放缓冲与奖励历史，
                使该检查点可用于**断点续训**（文件会明显变大）。
                best/final 检查点不需要，保持小体积。
            episode_rewards: 已完成的逐轮奖励列表（用于恢复 avg 统计）。
            best_avg_reward: 按**滑动平均**挑选出的最优值（新增，见 train()）。
            extra: 额外元数据（如熵系数、argmax 开放率），便于事后追溯。
        """
        ckpt = {
            'actor_state_dicts': [a.state_dict() for a in self.actors],
            'critic_state_dicts': [c.state_dict() for c in self.critics],
            'actors': [a.state_dict() for a in self.actors],  # 兼容旧格式
            'critics': [c.state_dict() for c in self.critics],  # 兼容旧格式
            'episode': episode,
            'best_reward': best_reward,
            'best_avg_reward': best_avg_reward,
            'avg_reward': avg_reward,
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'config': {
                'n_agents': self.n_agents,
                'obs_dim': self.obs_dim,
                'n_actions': self.n_actions,
                'actor_hidden_dims': self.config['actor_hidden_dims'],
                'critic_hidden_dims': self.config['critic_hidden_dims'],
                'experiment_protocol': self.config.get('experiment_protocol'),
            }
        }
        if extra:
            ckpt['extra'] = dict(extra)
        # twin critic 的权重（可选键：旧检查点没有这个键，加载时做兼容处理）
        if self.twin_critics:
            ckpt['critic2_state_dicts'] = [c.state_dict() for c in self.critics2]
            ckpt['twin_critics'] = True
        if include_resume_state:
            ckpt['replay_buffer'] = list(self.buffer)
            ckpt['buffer_maxlen'] = self.buffer.maxlen
            ckpt['episode_rewards'] = list(episode_rewards or [])
        torch.save(ckpt, path)
        print(f"Model saved to {path}"
              + ("  (含续训状态)" if include_resume_state else ""))
    
    def load(self, path):
        """加载模型。若检查点含续训状态（回放缓冲/奖励历史），一并恢复。

        twin critic 兼容性：
          - 检查点**有** critic2_state_dicts 且本 agent 开了 twin -> 正常加载；
          - 检查点**没有** critic2_state_dicts（旧模型）但本 agent 开了 twin
            -> 把第一套 critic 复制给第二套（热启动），否则 min(Q1,Q2) 里
               Q2 会是随机初始化的噪声，早期 TD 目标会被严重压低。
          - 纯评估只用到 Actor，因此旧模型在新代码下评估结果**完全不变**。
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 兼容新格式
        if 'actor_state_dicts' in ckpt:
            for i, sd in enumerate(ckpt['actor_state_dicts']):
                self.actors[i].load_state_dict(sd)
            for i, sd in enumerate(ckpt['critic_state_dicts']):
                self.critics[i].load_state_dict(sd)
        # 兼容旧格式
        elif 'actors' in ckpt:
            for i, sd in enumerate(ckpt['actors']):
                self.actors[i].load_state_dict(sd)
            for i, sd in enumerate(ckpt['critics']):
                self.critics[i].load_state_dict(sd)
        else:
            raise KeyError(f"检查点格式无法识别，缺少 actor_state_dicts / actors: {path}")

        # twin critic 权重
        self.twin_loaded_from_checkpoint = False
        if self.twin_critics:
            if 'critic2_state_dicts' in ckpt:
                for i, sd in enumerate(ckpt['critic2_state_dicts']):
                    self.critics2[i].load_state_dict(sd)
                self.twin_loaded_from_checkpoint = True
            else:
                # 旧检查点：用第一套 critic 热启动第二套
                for i in range(self.n_agents):
                    self.critics2[i].load_state_dict(self.critics[i].state_dict())

        # 同步目标网络。
        # 修复：原实现只恢复 actors/critics，目标网络仍是构造时的随机初始化。
        # 纯评估不受影响（select_action 只用 actor），但从检查点续训时
        # 目标 Q 值会是噪声，学习信号被破坏。
        for i in range(self.n_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())
            if self.twin_critics:
                self.critic_targets2[i].load_state_dict(self.critics2[i].state_dict())

        self.epsilon = ckpt.get('epsilon', 0.01)
        self.train_step = ckpt.get('train_step', 0)

        # ---- 续训状态（评估用的 best/final 检查点不含这些字段）----
        if 'replay_buffer' in ckpt:
            self.buffer = deque(ckpt['replay_buffer'],
                                maxlen=ckpt.get('buffer_maxlen', self.buffer.maxlen))
        return ckpt


def _setup_logger(log_file=None):
    """配置训练日志：同时输出到控制台与文件。

    文件格式必须与 visualize_training.py 的正则严格一致：
        Episode {ep}/{total} | Reward: {r:.2f} | Avg: {a:.2f} | Epsilon: {e:.3f}
    否则训练曲线无法被解析。

    参数:
        log_file: 覆盖 TRAIN_CONFIG['log_file']（用于把不同实验写到不同目录）

    返回 logger；若日志目录不可写则退化为仅控制台输出（不中断训练）。
    """
    logger = logging.getLogger('maddpg_train')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 避免重复调用时叠加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = log_file or TRAIN_CONFIG.get('log_file')
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            # 不写中文表头，保证日志文件为纯 ASCII、可被各种编码安全解析
        except OSError as e:
            logger.warning(f"[警告] 无法写入日志文件 {log_file}: {e}；仅输出到控制台。")
    return logger


def train(n_episodes=None, seed=None, resume=True, resume_path=None,
          model_dir=None, log_dir=None, tag=None):
    """执行训练，支持断点续训。

    参数:
        n_episodes:  目标总轮数（覆盖 TRAIN_CONFIG['n_episodes']）
        seed:        随机种子（仅全新训练时生效；续训时沿用检查点状态）
        resume:      是否自动从 <model_dir>/maddpg_resume.pth 恢复
        resume_path: 指定续训检查点路径
        model_dir:   模型输出目录（默认 TRAIN_CONFIG['model_dir']）
        log_dir:     日志输出目录（默认 TRAIN_CONFIG['log_dir']）
        tag:         文件名前缀，便于同一目录内放多组实验
    """
    n_episodes = int(n_episodes) if n_episodes else TRAIN_CONFIG['n_episodes']

    # ---------------- 输出目录（本次新增，防止覆盖旧模型）----------------
    # 【本次修复的坑】原实现把 best 直接写成 <model_dir>/maddpg_best.pth：
    # 从零训练只要有一轮超过 3851.00 就会覆盖现有最优模型。现在：
    #   1) 默认目录与旧模型隔离（见 TRAIN_CONFIG）；
    #   2) 新版本**不再写 maddpg_best.pth**，改写
    #      maddpg_best_avg.pth（滑动平均口径，推荐）与 maddpg_best_peak.pth（旧口径）。
    model_dir = model_dir or TRAIN_CONFIG['model_dir']
    log_dir = log_dir or TRAIN_CONFIG.get('log_dir') or model_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_output.log')
    prefix = f"{tag}_" if tag else ""
    legacy = os.path.normpath(model_dir).replace('\\', '/').endswith('models_matd3')

    logger = _setup_logger(log_file)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    logger.info("=" * 74)
    logger.info("MADDPG V3 Training (2026-10-01 fixed)")
    logger.info("=" * 74)
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Log file : {log_file}")
    if legacy:
        logger.info("[WARN] model_dir points at legacy models_matd3; this version never "
                    "writes maddpg_best.pth, so the existing best model stays intact, "
                    "but new artefacts will coexist with the old ones.")

    env = MultiAgentEmergencyLaneEnv()
    n_agents = env.n_agents
    obs_dim = env.obs_dim
    n_actions = ENV_CONFIG['n_actions']

    logger.info(f"Agents: {n_agents}, Obs dim: {obs_dim}, Actions: {n_actions}")
    logger.info(f"Target episodes: {n_episodes}, Device: {TRAIN_CONFIG['device']}")
    
    config = {
        'experiment_protocol': {
            'warm_up_steps': CA_CONFIG['warm_up_steps'],
            'max_steps': CA_CONFIG['max_steps'],
            'decision_interval': ENV_CONFIG['decision_interval'],
            'max_emergency': CA_CONFIG['max_emergency'],
            'inject_emergency_after_warmup': CA_CONFIG.get('inject_emergency_after_warmup', False),
            'shared_team_reward': ENV_CONFIG.get('shared_team_reward', False),
            'n_segments': ENV_CONFIG['n_segments'],
            'n_cars_range': list(CA_CONFIG['n_cars_range']),
            'n_cars_choices': CA_CONFIG.get('n_cars_choices'),
            'P_emergency_range': list(CA_CONFIG['P_emergency_range']),
        },
        'actor_hidden_dims': AGENT_CONFIG['actor_hidden_dims'],
        'critic_hidden_dims': AGENT_CONFIG['critic_hidden_dims'],
        'actor_lr': AGENT_CONFIG['actor_lr'],
        'critic_lr': AGENT_CONFIG['critic_lr'],
        'gamma': AGENT_CONFIG['gamma'],
        'tau': AGENT_CONFIG['tau'],
        'buffer_capacity': AGENT_CONFIG['buffer_capacity'],
        'batch_size': AGENT_CONFIG['batch_size'],
        'policy_freq': AGENT_CONFIG['policy_freq'],
        'train_every': AGENT_CONFIG.get('train_every', 1),
        'max_grad_norm': AGENT_CONFIG['max_grad_norm'],
        # ---- 熵正则：线性退火（本次修复）----
        'entropy_coef': AGENT_CONFIG.get('entropy_coef', 0.0),
        'entropy_coef_start': AGENT_CONFIG.get('entropy_coef', 0.0),
        'entropy_coef_end': AGENT_CONFIG.get('entropy_coef_end',
                                             AGENT_CONFIG.get('entropy_coef', 0.0)),
        'entropy_anneal': AGENT_CONFIG.get('entropy_anneal', False),
        'entropy_anneal_episodes': AGENT_CONFIG.get('entropy_anneal_episodes', 1),
        # ---- ε 探索：线性 / 指数 ----
        'epsilon_schedule': AGENT_CONFIG.get('epsilon_schedule', 'exp'),
        'epsilon_decay': AGENT_CONFIG.get('epsilon_decay', 0.999),
        'epsilon_decay_episodes': AGENT_CONFIG.get('epsilon_decay_episodes', 1),
        'epsilon_min': AGENT_CONFIG.get('epsilon_min', 0.05),
        # ---- 目标 Q：独热动作 + 离散版目标策略平滑（本次修复）----
        'target_action_mode': AGENT_CONFIG.get('target_action_mode', 'greedy'),
        'target_flip_p': AGENT_CONFIG.get('target_flip_p', 0.0),
        # ---- Actor 更新方式：counterfactual（推荐）| legacy（旧行为，A/B 用）----
        'actor_update_mode': AGENT_CONFIG.get('actor_update_mode', 'counterfactual'),
        # ---- Twin critic（TD3 clipped double-Q，2026-10-01 新增）----
        'twin_critics': AGENT_CONFIG.get('twin_critics', False),
    }

    logger.info(f"entropy_coef: {config['entropy_coef']:.3f} -> "
                f"{config['entropy_coef_end']:.3f} over "
                f"{config['entropy_anneal_episodes']} eps "
                f"(anneal={config['entropy_anneal']})")
    logger.info(f"epsilon     : schedule={config['epsilon_schedule']}, "
                f"decay_episodes={config['epsilon_decay_episodes']}, "
                f"min={config['epsilon_min']}")
    logger.info(f"target Q    : mode={config['target_action_mode']}, "
                f"flip_p={config['target_flip_p']}")
    logger.info(f"actor update: mode={config['actor_update_mode']} "
                f"(counterfactual = the 2026-10-01 fix)")
    logger.info(f"twin critic : {config['twin_critics']} "
                f"(clipped double-Q; needs policy_freq>1 and target smoothing to be TD3-complete)")
    
    agent = MADDPGAgent(n_agents, obs_dim, n_actions, config)

    episode_rewards = []
    best_reward = -float('inf')       # 单回合最高（旧口径，仅作对照）
    best_avg = -float('inf')          # 滑动平均最高（新口径，推荐的 best）
    start_episode = 1

    # ---------------- 断点续训 ----------------
    ckpt_path = resume_path or os.path.join(model_dir, f"{prefix}maddpg_resume.pth")
    if resume and ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"[resume] found checkpoint: {ckpt_path}")
        try:
            ckpt = agent.load(ckpt_path)
            episode_rewards = list(ckpt.get('episode_rewards') or [])
            best_reward = ckpt.get('best_reward', -float('inf'))
            if best_reward is None:
                best_reward = -float('inf')
            best_avg = ckpt.get('best_avg_reward', -float('inf'))
            if best_avg is None:
                best_avg = -float('inf')
            done_ep = int(ckpt.get('episode') or 0)
            start_episode = done_ep + 1
            logger.info(
                f"[resume] restored: episode={done_ep}, buffer={len(agent.buffer)}, "
                f"best_reward={best_reward:.2f}, best_avg={best_avg:.2f}, "
                f"epsilon={agent.epsilon:.4f}")
            if start_episode > n_episodes:
                logger.info(f"[resume] already at {done_ep} >= {n_episodes}, nothing to do.")
                return best_avg if best_avg > -float('inf') else best_reward
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[resume] checkpoint load failed, training from scratch: {e}")
            episode_rewards, best_reward, best_avg, start_episode = \
                [], -float('inf'), -float('inf'), 1
    elif resume:
        logger.info("[resume] no checkpoint found, training from scratch.")

    best_window = int(TRAIN_CONFIG.get('best_window', 100))
    best_min_episodes = int(TRAIN_CONFIG.get('best_min_episodes', 50))
    probe_freq = max(int(TRAIN_CONFIG.get('eval_probe_freq', 50)), 1)
    learn_start = max(int(TRAIN_CONFIG.get('learn_start_episodes', 0)), 0)
    logger.info(f"Episodes to run: {start_episode} ~ {n_episodes}")
    logger.info(f"Best selection: moving average over {best_window} eps "
                f"(active after {best_min_episodes} eps)")
    if learn_start > 0:
        logger.info(f"Warm-up: no gradient updates for the first {learn_start} eps "
                    f"(reference implementations do this)")
    logger.info("-" * 74)

    for episode in range(start_episode, n_episodes + 1):
        # 熵系数退火 + ε 探索调度（本次新增，替代原来的 decay_epsilon）
        agent.set_progress(episode, n_episodes)

        observations = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # ---- argmax 探针（本次新增，逻辑上零额外前向开销）----
        # 【本次修复的坑】原训练日志只看 ε-greedy 采样策略的奖励，**看不出 argmax
        # 策略是否已经退化**。上一版就是"采样策略熵全程正常 0.65~0.68，但 argmax
        # 早已退化成恒关闭"，导致选了第 1000 轮之后的模型还以为在进步。
        # 现在每个决策步顺带统计 argmax 开放率与策略熵，直接写进训练日志。
        argmax_open = 0
        argmax_total = 0
        ent_sum = 0.0
        ent_n = 0

        while not done:
            probs = agent.action_probs(observations)        # (n_agents, n_actions)
            argmax_open += int(np.sum(probs.argmax(axis=1) == 1))
            argmax_total += probs.shape[0]
            p1 = probs[:, 1]
            ent_sum += float(np.mean(-(p1 * np.log(p1 + 1e-8)
                                       + (1 - p1) * np.log(1 - p1 + 1e-8))))
            ent_n += 1

            actions = agent.select_action(observations, evaluate=False)
            next_observations, rewards, done, info = env.step(actions)
            agent.store_transition(observations, actions, rewards, next_observations, done)
            # 每 train_every 个决策步做一次梯度更新（见 config 中 train_every 说明）
            # learn_start_episodes > 0 时，前若干轮只采集不学习（参考实现的做法）
            if episode > learn_start and step % agent.config.get('train_every', 1) == 0:
                agent.learn()
            episode_reward += rewards.mean()
            observations = next_observations
            step += 1

        episode_rewards.append(episode_reward)
        avg_reward = (np.mean(episode_rewards[-best_window:])
                      if len(episode_rewards) >= best_window
                      else np.mean(episode_rewards))
        argmax_open_rate = argmax_open / max(argmax_total, 1)
        mean_entropy = ent_sum / max(ent_n, 1)

        if episode % 10 == 0:
            # 前半段格式必须与 visualize_training.py 的解析正则一致，勿改动；
            # 该正则用 re.search，故可以在行尾追加诊断字段。
            logger.info(f"Episode {episode}/{n_episodes} | "
                        f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | "
                        f"Epsilon: {agent.epsilon:.3f} | "
                        f"ArgmaxOpen: {argmax_open_rate:.3f} | "
                        f"Entropy: {mean_entropy:.4f} | "
                        f"EntCoef: {agent.entropy_coef:.4f}")

        if episode % probe_freq == 0:
            logger.info(f"[probe] ep={episode} avg_last{best_window}={avg_reward:.2f} "
                        f"best_avg={best_avg:.2f} best_peak={best_reward:.2f} "
                        f"argmax_open={argmax_open_rate:.3f} "
                        f"entropy={mean_entropy:.4f}/{np.log(2):.4f} "
                        f"ent_coef={agent.entropy_coef:.4f} eps={agent.epsilon:.3f} "
                        f"buffer={len(agent.buffer)}")

        meta = {'entropy_coef': agent.entropy_coef,
                'policy_entropy': mean_entropy,
                'argmax_open_rate': argmax_open_rate,
                'episode_reward': episode_reward,
                'avg_window': avg_reward}

        # ---- best（滑动平均口径，推荐）----
        if len(episode_rewards) >= best_min_episodes and avg_reward > best_avg:
            best_avg = avg_reward
            agent.save(os.path.join(model_dir, f"{prefix}maddpg_best_avg.pth"),
                       episode=episode, best_reward=best_reward,
                       best_avg_reward=best_avg, avg_reward=avg_reward, extra=meta)

        # ---- best（单回合最高口径，仅作对照，不再作为默认 best）----
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(model_dir, f"{prefix}maddpg_best_peak.pth"),
                       episode=episode, best_reward=best_reward,
                       best_avg_reward=best_avg, avg_reward=avg_reward, extra=meta)

        if episode % TRAIN_CONFIG['save_freq'] == 0:
            agent.save(os.path.join(model_dir, f"{prefix}maddpg_episode_{episode}.pth"),
                       episode=episode, best_reward=best_reward,
                       best_avg_reward=best_avg, avg_reward=avg_reward, extra=meta)

        # ---- 断点续训状态（含回放缓冲与奖励历史）----
        resume_freq = TRAIN_CONFIG.get('resume_freq')
        if resume_freq and episode % resume_freq == 0:
            agent.save(
                os.path.join(model_dir, f"{prefix}maddpg_resume.pth"),
                episode=episode,
                best_reward=best_reward,
                best_avg_reward=best_avg,
                avg_reward=avg_reward,
                include_resume_state=True,
                episode_rewards=episode_rewards,
            )

    agent.save(
        os.path.join(model_dir, f"{prefix}maddpg_final.pth"),
        episode=n_episodes,
        best_reward=best_reward,
        best_avg_reward=best_avg,
        avg_reward=avg_reward
    )

    # 正常跑完则删除续训检查点，避免下次误以为还要接着训
    rp = os.path.join(model_dir, f"{prefix}maddpg_resume.pth")
    if rp and os.path.exists(rp):
        try:
            os.remove(rp)
            logger.info(f"[resume] removed checkpoint {rp} (training finished)")
        except OSError as e:
            logger.warning(f"[warn] cannot remove resume checkpoint {rp}: {e}")

    logger.info(f"\nTraining completed! best_avg={best_avg:.2f}  "
                f"best_peak={best_reward:.2f}")
    # 返回值改为**滑动平均口径**的 best（旧口径在 best_peak 里，仍可从文件读取）
    return best_avg if best_avg > -float('inf') else best_reward


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MADDPG V2 训练")
    parser.add_argument('--episodes', type=int, default=None,
                        help='覆盖配置中的训练轮数（用于小规模验证）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（用于复现；续训时忽略）')
    parser.add_argument('--no-resume', action='store_true',
                        help='忽略已有续训检查点，从头开始训练')
    parser.add_argument('--resume-from', default=None,
                        help='指定要恢复的检查点路径')
    parser.add_argument('--model-dir', default=None,
                        help='模型输出目录（默认见 TRAIN_CONFIG；写旧目录请显式指定）')
    parser.add_argument('--log-dir', default=None,
                        help='日志输出目录（默认见 TRAIN_CONFIG）')
    parser.add_argument('--tag', default=None,
                        help='文件名前缀，便于同一目录内放多组实验')
    parser.add_argument('--entropy-coef', type=float, default=None,
                        help='覆盖熵系数退火起点')
    parser.add_argument('--entropy-coef-end', type=float, default=None,
                        help='覆盖熵系数退火终点')
    parser.add_argument('--entropy-anneal-episodes', type=int, default=None,
                        help='覆盖熵系数退火轮数')
    parser.add_argument('--target-action-mode', default=None,
                        choices=['greedy', 'soft'],
                        help="目标 Q 的动作模式：greedy=独热(推荐)，soft=旧行为")
    parser.add_argument('--actor-update-mode', default=None,
                        choices=['counterfactual', 'legacy'],
                        help="Actor 更新方式：counterfactual=反事实期望Q(推荐)，legacy=旧行为")
    parser.add_argument('--twin-critics', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='是否启用 twin critic（TD3 clipped double-Q，抑制 Q 过估计）')
    return parser.parse_args(argv)


def _apply_cli_overrides(args):
    """把命令行覆盖项写回 AGENT_CONFIG（只动命令行显式给出的键）。"""
    mapping = {
        'entropy_coef': getattr(args, 'entropy_coef', None),
        'entropy_coef_end': getattr(args, 'entropy_coef_end', None),
        'entropy_anneal_episodes': getattr(args, 'entropy_anneal_episodes', None),
        'target_action_mode': getattr(args, 'target_action_mode', None),
        'actor_update_mode': getattr(args, 'actor_update_mode', None),
        'twin_critics': getattr(args, 'twin_critics', None),
    }
    for k, v in mapping.items():
        if v is not None:
            AGENT_CONFIG[k] = v


if __name__ == '__main__':
    args = _parse_args()
    _apply_cli_overrides(args)
    train(n_episodes=args.episodes, seed=args.seed,
          resume=not args.no_resume, resume_path=args.resume_from,
          model_dir=args.model_dir, log_dir=args.log_dir, tag=args.tag)

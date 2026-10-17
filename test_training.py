#!/usr/bin/env python
"""测试训练流程"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: Importing modules...")
from maddpg_v2 import MADDPGAgent, train
from env_maddpg import MultiAgentEmergencyLaneEnv
from config_maddpg import CA_CONFIG, ENV_CONFIG, TRAIN_CONFIG

print("Step 2: Creating environment...")
env = MultiAgentEmergencyLaneEnv()
print(f"Agents: {env.n_agents}, Obs dim: {env.obs_dim}, Actions: {env.n_actions}")

print("Step 3: Resetting environment...")
obs = env.reset()
print(f"Initial observation shape: {obs[0].shape}")

print("Step 4: Creating agent...")
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
agent = MADDPGAgent(env.n_agents, env.obs_dim, env.n_actions, config)
print("Agent created successfully!")

print("Step 5: Testing one episode...")
observations = env.reset()
episode_reward = 0
done = False
step = 0

while not done and step < 10:
    actions = agent.select_action(observations, evaluate=False)
    next_observations, rewards, done, info = env.step(actions)
    agent.store_transition(observations, actions, rewards, next_observations, done)
    result = agent.learn()
    episode_reward += rewards.mean()
    observations = next_observations
    step += 1
    print(f"Step {step}: Reward={rewards.mean():.2f}, Done={done}")

print(f"Test episode completed! Total reward: {episode_reward:.2f}")
print("All tests passed!")
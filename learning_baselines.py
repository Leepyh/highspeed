"""Discrete I-DQN, QMIX and MAPPO for the shared emergency-lane environment.

All methods consume the same per-agent observations, binary actions and mean
team reward. QMIX and MAPPO use the concatenated observations only during
centralized training; execution uses local observations.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, hidden=128):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def to_tensor(value, dtype=torch.float32):
    return torch.as_tensor(np.asarray(value), dtype=dtype)


class Replay:
    def __init__(self, capacity=100000):
        self.data = deque(maxlen=capacity)

    def add(self, obs, actions, reward, next_obs, done):
        self.data.append((np.asarray(obs, dtype=np.float32),
                          np.asarray(actions, dtype=np.int64), float(np.mean(reward)),
                          np.asarray(next_obs, dtype=np.float32), bool(done)))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.data), batch_size, replace=False)
        batches = list(zip(*(self.data[int(i)] for i in indices)))
        return (to_tensor(np.stack(batches[0])), to_tensor(np.stack(batches[1]), torch.long),
                to_tensor(batches[2]), to_tensor(np.stack(batches[3])),
                to_tensor(batches[4]))

    def __len__(self):
        return len(self.data)


class IDQN:
    name = 'I-DQN'

    def __init__(self, n_agents, obs_dim, n_actions=2, lr=3e-4, gamma=.95,
                 batch_size=128, target_interval=250, seed=0):
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.gamma, self.batch_size, self.target_interval = gamma, batch_size, target_interval
        self.q = nn.ModuleList([mlp(obs_dim, n_actions) for _ in range(n_agents)])
        self.target = nn.ModuleList([mlp(obs_dim, n_actions) for _ in range(n_agents)])
        self.target.load_state_dict(self.q.state_dict())
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=lr) for net in self.q]
        self.replay = Replay()
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs, deterministic=True, epsilon=0.0):
        obs = to_tensor(obs)
        actions = np.array([int(net(obs[i]).argmax()) for i, net in enumerate(self.q)])
        if not deterministic:
            mask = self.rng.random(self.n_agents) < epsilon
            actions[mask] = self.rng.integers(self.n_actions, size=int(mask.sum()))
        return actions

    def store(self, obs, actions, rewards, next_obs, done):
        self.replay.add(obs, actions, rewards, next_obs, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        obs, actions, reward, next_obs, done = self.replay.sample(self.batch_size, self.rng)
        losses = []
        for i, (online, target, opt) in enumerate(zip(self.q, self.target, self.optimizers)):
            current = online(obs[:, i]).gather(1, actions[:, i, None]).squeeze(1)
            with torch.no_grad():
                best = online(next_obs[:, i]).argmax(1, keepdim=True)
                future = target(next_obs[:, i]).gather(1, best).squeeze(1)
                expected = reward + self.gamma * (1 - done) * future
            loss = F.smooth_l1_loss(current, expected)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10)
            opt.step()
            losses.append(float(loss.detach()))
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(np.mean(losses))

    def state_dict(self):
        return {'q': self.q.state_dict(), 'target': self.target.state_dict(),
                'optimizers': [o.state_dict() for o in self.optimizers], 'updates': self.updates}

    def load_state_dict(self, state):
        self.q.load_state_dict(state['q'])
        self.target.load_state_dict(state.get('target', state['q']))
        if 'optimizers' in state:
            for opt, saved in zip(self.optimizers, state['optimizers']):
                opt.load_state_dict(saved)
        self.updates = state.get('updates', 0)


class Mixer(nn.Module):
    """Monotonic QMIX mixer: nonnegative hypernetwork weights ensure dQtot/dQi >= 0."""

    def __init__(self, state_dim, n_agents, hidden=32):
        super().__init__()
        self.n_agents, self.hidden = n_agents, hidden
        self.w1 = mlp(state_dim, n_agents * hidden, hidden)
        self.b1 = nn.Linear(state_dim, hidden)
        self.w2 = mlp(state_dim, hidden, hidden)
        self.b2 = mlp(state_dim, 1, hidden)

    def forward(self, agent_q, state):
        batch = agent_q.shape[0]
        w1 = self.w1(state).abs().view(batch, self.n_agents, self.hidden)
        hidden = F.elu(torch.bmm(agent_q.unsqueeze(1), w1).squeeze(1) + self.b1(state))
        w2 = self.w2(state).abs().view(batch, self.hidden, 1)
        return torch.bmm(hidden.unsqueeze(1), w2).squeeze(1).squeeze(1) + self.b2(state).squeeze(1)


class QMIX:
    name = 'QMIX'

    def __init__(self, n_agents, obs_dim, n_actions=2, lr=3e-4, gamma=.95,
                 batch_size=128, target_interval=250, seed=0):
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.gamma, self.batch_size, self.target_interval = gamma, batch_size, target_interval
        self.q = mlp(obs_dim + n_agents, n_actions)
        self.target_q = mlp(obs_dim + n_agents, n_actions)
        self.mixer = Mixer(n_agents * obs_dim, n_agents)
        self.target_mixer = Mixer(n_agents * obs_dim, n_agents)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(list(self.q.parameters()) + list(self.mixer.parameters()), lr=lr)
        self.replay = Replay()
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def _inputs(self, obs):
        batch, agents, _ = obs.shape
        eye = torch.eye(agents, device=obs.device).expand(batch, -1, -1)
        return torch.cat((obs, eye), dim=-1)

    @torch.no_grad()
    def act(self, obs, deterministic=True, epsilon=0.0):
        q = self.q(self._inputs(to_tensor(obs)[None]))[0]
        actions = q.argmax(-1).numpy()
        if not deterministic:
            mask = self.rng.random(self.n_agents) < epsilon
            actions[mask] = self.rng.integers(self.n_actions, size=int(mask.sum()))
        return actions

    def store(self, obs, actions, rewards, next_obs, done):
        self.replay.add(obs, actions, rewards, next_obs, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        obs, actions, reward, next_obs, done = self.replay.sample(self.batch_size, self.rng)
        local = self.q(self._inputs(obs)).gather(-1, actions[..., None]).squeeze(-1)
        current = self.mixer(local, obs.flatten(1))
        with torch.no_grad():
            best = self.q(self._inputs(next_obs)).argmax(-1, keepdim=True)
            future_local = self.target_q(self._inputs(next_obs)).gather(-1, best).squeeze(-1)
            future = self.target_mixer(future_local, next_obs.flatten(1))
            expected = reward + self.gamma * (1 - done) * future
        loss = F.smooth_l1_loss(current, expected)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(self.q.parameters()) + list(self.mixer.parameters()), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(loss.detach())

    def state_dict(self):
        return {'q': self.q.state_dict(), 'target_q': self.target_q.state_dict(),
                'mixer': self.mixer.state_dict(), 'target_mixer': self.target_mixer.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'updates': self.updates}

    def load_state_dict(self, state):
        self.q.load_state_dict(state['q']); self.target_q.load_state_dict(state.get('target_q', state['q']))
        self.mixer.load_state_dict(state['mixer'])
        self.target_mixer.load_state_dict(state.get('target_mixer', state['mixer']))
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.updates = state.get('updates', 0)


class MAPPO:
    name = 'MAPPO'

    def __init__(self, n_agents, obs_dim, n_actions=2, lr=3e-4, gamma=.95,
                 gae_lambda=.95, clip=.2, epochs=4, minibatch=64, seed=0):
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.gamma, self.gae_lambda, self.clip = gamma, gae_lambda, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.actor = mlp(obs_dim + n_agents, n_actions)
        self.critic = mlp(n_agents * obs_dim, 1)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.trajectory = []

    def _inputs(self, obs):
        batch, agents, _ = obs.shape
        eye = torch.eye(agents, device=obs.device).expand(batch, -1, -1)
        return torch.cat((obs, eye), dim=-1)

    def _dist(self, obs):
        return torch.distributions.Categorical(logits=self.actor(self._inputs(obs)))

    @torch.no_grad()
    def act(self, obs, deterministic=True, epsilon=0.0):
        dist = self._dist(to_tensor(obs)[None])
        return (dist.probs.argmax(-1) if deterministic else dist.sample())[0].numpy()

    def store(self, obs, actions, rewards, next_obs, done):
        obs_t = to_tensor(obs)[None]
        actions_t = to_tensor(actions, torch.long)[None]
        with torch.no_grad():
            logp = self._dist(obs_t).log_prob(actions_t).sum(-1).item()
            value = self.critic(obs_t.flatten(1)).item()
            next_value = self.critic(to_tensor(next_obs)[None].flatten(1)).item()
        self.trajectory.append((np.asarray(obs, np.float32), np.asarray(actions, np.int64),
                                float(np.mean(rewards)), bool(done), logp, value, next_value))

    def update(self):
        if not self.trajectory:
            return None
        obs, actions, rewards, dones, old_logp, values, next_values = zip(*self.trajectory)
        advantage = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantage[t] = gae
        returns = advantage + np.asarray(values, dtype=np.float32)
        obs = to_tensor(np.stack(obs)); actions = to_tensor(np.stack(actions), torch.long)
        old_logp = to_tensor(old_logp); returns = to_tensor(returns)
        advantage = to_tensor(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        losses = []
        for _ in range(self.epochs):
            for indices in np.array_split(self.rng.permutation(len(obs)),
                                          max(1, int(np.ceil(len(obs) / self.minibatch)))):
                dist = self._dist(obs[indices])
                logp = dist.log_prob(actions[indices]).sum(-1)
                ratio = (logp - old_logp[indices]).exp()
                a = advantage[indices]
                policy_loss = -torch.minimum(ratio * a, ratio.clamp(1-self.clip, 1+self.clip) * a).mean()
                value = self.critic(obs[indices].flatten(1)).squeeze(-1)
                value_loss = F.mse_loss(value, returns[indices])
                loss = policy_loss + .5 * value_loss - .01 * dist.entropy().mean()
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 10)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        self.trajectory.clear()
        return float(np.mean(losses))

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor']); self.critic.load_state_dict(state['critic'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])


ALGORITHMS = {'idqn': IDQN, 'qmix': QMIX, 'mappo': MAPPO}


def save_checkpoint(path, algorithm, agent, env, episode, seed):
    torch.save({'algorithm': algorithm, 'agent': agent.state_dict(), 'episode': episode,
                'seed': seed, 'n_agents': env.n_agents, 'obs_dim': env.obs_dim,
                'protocol': {'ca': env.ca_config, 'env': env.env_config}}, path)


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    kind = data['algorithm']
    agent = ALGORITHMS[kind](data['n_agents'], data['obs_dim'])
    agent.load_state_dict(data['agent'])
    return agent, data

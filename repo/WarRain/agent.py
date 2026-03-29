import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
import copy

class NoisyLinear(nn.Module):
    """Noisy Linear layer with factorized Gaussian noise."""
    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("eps_in", torch.zeros(1, in_features))
        self.register_buffer("eps_out", torch.zeros(out_features, 1))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_bias", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)
        self.sigma_weight.data.fill_(self.sigma_0 / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_0 / math.sqrt(self.out_features))

    def sample_noise(self):
        self.eps_in.copy_(self._scale_noise(self.in_features))
        self.eps_out.copy_(self._scale_noise(self.out_features).unsqueeze(-1))
        self.eps_bias.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.mu_weight.device)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * torch.matmul(self.eps_out, self.eps_in)
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(x, weight, bias)

class DuelingQNetwork(nn.Module):
    """Dueling distributional Q-network with noisy linear layers."""
    def __init__(self, in_channels: int = 4, n_actions: int = 6, n_atoms: int = 51):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.shared = NoisyLinear(7 * 7 * 64, 512)
        self.value_stream = NoisyLinear(512, n_atoms)
        self.adv_stream = NoisyLinear(512, n_actions * n_atoms)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        feat = F.relu(self.shared(x))
        val = self.value_stream(feat)
        adv = self.adv_stream(feat).view(x.size(0), self.n_actions, self.n_atoms)
        q_logits = val.unsqueeze(1) + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(q_logits, dim=-1)

    def sample_noise(self):
        self.shared.sample_noise()
        self.value_stream.sample_noise()
        self.adv_stream.sample_noise()

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha
        self.beta = beta
    def __len__(self):
        return len(self.buffer)
    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = (max_prio + 1e-5) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:N]

        probs = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(N, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in idxs]

        total = len(self.buffer)
        weights = (total * probs[idxs]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = np.stack([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.stack([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])

        return states, actions, rewards, next_states, dones, idxs, weights

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        prios = (np.abs(td_errors) + 1e-5) ** self.alpha
        for idx, prio in zip(idxs, prios):
            self.priorities[idx] = prio

class RainbowDQN:
    """Rainbow DQN agent with all components: distributional C51, double DQN, n-step, PER, noisy nets, dueling."""
    def __init__(self, obs_shape: tuple, n_actions: int):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DuelingQNetwork(obs_shape[2], n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=2.5e-4)

        self.replay_buffer = PrioritizedReplayBuffer(1_000_000, alpha=0.6, beta=0.4)
        self.n_atoms = 51
        self.vmin = -10.0
        self.vmax = 10.0
        self.support = torch.linspace(self.vmin, self.vmax, self.n_atoms, device=self.device).view(1, self.n_atoms)
        self.delta_z = (self.vmax - self.vmin) / (self.n_atoms - 1)
        self.gamma = 0.99
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_steps = 1_000_000
        self.batch_size = 32
        self.beta_start = 0.4
        self.steps = 0

    def act(self, state, training=True):
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon - (1.0 - self.epsilon_min) / self.epsilon_decay_steps)

        if training:
            self.q_net.train()
            if random.random() < self.epsilon:
                return random.randint(0, self.n_actions - 1)
        else:
            self.q_net.eval()

        state_t = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if training:
                self.q_net.sample_noise()
            dist = self.q_net(state_t)[0]
            q_vals = (dist * self.support).sum(dim=1)
        return q_vals.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step or done:
            rs = [t[2] for t in self.n_step_buffer]
            R = sum((self.gamma ** i) * rs[i] for i in range(len(rs)))
            first = self.n_step_buffer[0]
            last = self.n_step_buffer[-1]
            self.replay_buffer.push(first[0], first[1], R, last[3], last[4])
            self.n_step_buffer.clear()

    def update(self):
        if len(self.replay_buffer.buffer) < self.batch_size * 50:
            return

        # Anneal beta
        self.replay_buffer.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.steps / self.epsilon_decay_steps))

        states, actions, rewards, next_states, dones_np, idxs, weights_np = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        dones = torch.BoolTensor(dones_np).to(self.device)
        weights = torch.FloatTensor(weights_np).to(self.device)

        self.q_net.sample_noise()
        current_dist = self.q_net(states)
        action_dist = current_dist[torch.arange(self.batch_size), actions]  # (batch_size, n_atoms)

        with torch.no_grad():
            next_dist_target = self.target_net(next_states)
            next_q_vals = self.q_net(next_states)
            next_actions = next_q_vals.mean(dim=2).max(dim=1)[1]
            next_action_dist = next_dist_target[torch.arange(self.batch_size), next_actions]

            gamma_n = self.gamma ** self.n_step
            Tz = rewards.view(-1, 1) + gamma_n * (~dones).float().view(-1, 1) * self.support
            Tz = Tz.clamp(self.vmin, self.vmax)

            # Projection
            b = torch.arange(self.batch_size, device=self.device).unsqueeze(1)
            dist_span = (Tz - self.vmin) / self.delta_z

            l = dist_span.floor().long().clamp(0, self.n_atoms - 2)
            u = l + 1

            dist_l = 1.0 - (Tz - (self.vmin + l.float() * self.delta_z)) / self.delta_z
            dist_u = 1.0 - dist_l

            target_dist = torch.zeros((self.batch_size, self.n_atoms), device=self.device)
            target_dist.scatter_add_(1, l, next_action_dist * dist_l)
            target_dist.scatter_add_(1, u, next_action_dist * dist_u)

        # Loss
        log_p = F.log_softmax(action_dist, dim=1)
        loss = -(weights * (target_dist * log_p).sum(dim=1)).mean()

        # TD errors for PER (scalarized)
        mean_online = (action_dist * self.support).sum(dim=1)
        mean_target = (target_dist * self.support).sum(dim=1)
        td_errors = (mean_online - mean_target).detach().cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        self.replay_buffer.update_priorities(idxs, td_errors)

    def target_update(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
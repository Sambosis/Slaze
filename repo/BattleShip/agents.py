import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    """
    PyTorch MLP for DQN Q-value approximation.
    Architecture: 200 -> 256 -> 128 -> 100 (ReLU, dropout 0.1).
    """
    def __init__(self, state_dim: int = 200, action_dim: int = 100):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    """
    PyTorch MLP for Policy Gradient policy (logits -> log_softmax).
    Same architecture as QNetwork.
    """
    def __init__(self, state_dim: int = 200, action_dim: int = 100):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc3(x), dim=-1)


class ReplayBuffer:
    """
    Experience replay buffer with deque storage and recent-biased sampling.
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = list(self.buffer)[-batch_size * 10:]
        random.shuffle(batch)
        batch = batch[:batch_size]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))


class DQNAgent:
    """
    DQN Agent A with epsilon-greedy exploration, experience replay, and target network.
    """
    def __init__(self,
                 state_dim: int = 200,
                 action_dim: int = 100,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 eps_start: float = 1.0,
                 eps_decay: float = 0.995,
                 eps_min: float = 0.01,
                 replay_size: int = 100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_size)
        self.steps = 0
        self.losses = []

    def act(self, state: np.ndarray, training: bool = True) -> int:
        state = torch.FloatTensor(state).unsqueeze(0)
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        qvals = self.policy_net(state)
        return qvals.argmax(1).item()

    def update(self) -> None:
        if len(self.replay.buffer) < 32:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(32)
        qvals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_qvals = self.target_net(next_states).max(1)[0].detach()
        targets = rewards + self.gamma * next_qvals * (1 - dones)
        loss = F.mse_loss(qvals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


class PolicyGradientAgent:
    """
    REINFORCE Policy Gradient Agent B with entropy regularization.
    Episode-based updates using full trajectories.
    """
    def __init__(self,
                 state_dim: int = 200,
                 action_dim: int = 100,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 entropy_beta: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.entropy_beta = entropy_beta
        self.episode_losses = []

    def act(self, state: np.ndarray, training: bool = True) -> int:
        state = torch.FloatTensor(state).unsqueeze(0)
        log_probs = self.policy_net(state)
        action = torch.multinomial(torch.exp(log_probs), 1).item()
        return action

    def update(self, trajectory: list[dict]) -> None:
        if not trajectory:
            return
        # Compute discounted returns
        returns = []
        R = 0.0
        for t in reversed(trajectory):
            R = t['reward'] + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        # Compute log probs
        log_probs = []
        for t in trajectory:
            st = torch.FloatTensor(t['state']).unsqueeze(0)
            lps = self.policy_net(st)[0]  # [action_dim]
            lp = lps[t['action']]
            log_probs.append(lp)
        log_probs = torch.stack(log_probs)
        # Losses
        policy_loss = -(log_probs * returns).sum()
        entropy = -(log_probs.exp() * log_probs).sum()
        loss = policy_loss - self.entropy_beta * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.episode_losses.append(loss.item())
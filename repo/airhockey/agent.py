import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Optional, List, Tuple, Dict, Any

# ── Device auto-detection ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """
    Dueling DQN network.
    Splits the final layers into separate Value V(s) and Advantage A(s,a)
    streams, then combines: Q(s,a) = V(s) + A(s,a) - mean(A(s,a)).
    This lets the network learn state values independently of action advantages,
    leading to more stable and faster learning.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()

        # Shared feature backbone
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Value stream  V(s) -> scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream  A(s,a) -> one value per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns Q(s,a) = V(s) + A(s,a) - mean(A)."""
        features = self.feature(x)
        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, action_dim)
        # Combine using the mean-centering trick for identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ── Numpy-backed replay buffer ────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size replay buffer using pre-allocated numpy arrays for speed."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.pos = 0          # write cursor
        self.size = 0         # current occupancy

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions      = np.zeros(capacity, dtype=np.int64)
        self.rewards      = np.zeros(capacity, dtype=np.float32)
        self.next_states  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones        = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        idx = self.pos
        self.states[idx]     = state
        self.actions[idx]    = action
        self.rewards[idx]    = reward
        self.next_states[idx]= next_state
        self.dones[idx]      = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


class DQNAgent:
    """
    Double DQN agent with soft target updates (Polyak averaging),
    experience replay, and epsilon-greedy exploration.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000,
                 batch_size: int = 64, target_update_freq: int = 100,
                 learn_every: int = 4, tau: float = 0.005,
                 lr_min: float = 1e-6, lr_T0: int = 1000,
                 lr_T_mult: int = 2):
        """
        Initialize the Double DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Starting value for epsilon in epsilon-greedy policy
            epsilon_end: Minimum value for epsilon
            epsilon_decay: Multiplicative factor for epsilon decay
            memory_size: Maximum size of replay buffer
            batch_size: Batch size for training
            target_update_freq: (unused, kept for API compat) 
            learn_every: Only run a gradient update every N calls to learn()
            tau: Soft update interpolation factor for Polyak averaging
            lr_min: Minimum learning rate (eta_min for cosine annealing)
            lr_T0: Number of episodes in the first cosine annealing cycle
            lr_T_mult: Multiplier for cycle length after each restart
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_every = learn_every
        self.tau = tau
        
        # Numpy-backed replay buffer (faster sampling than deque-of-tuples)
        self.memory = ReplayBuffer(memory_size, state_dim)
        
        # Initialize main and target networks – on GPU when available
        self.model = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_model = QNetwork(state_dim, action_dim).to(DEVICE)
        self.update_target_network(hard=True)  # Hard copy to start
        
        # Initialize optimizer and loss function
        # Huber loss is standard for DQN; more stable than MSE with large TD errors
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Learning rate scheduler — cosine annealing with warm restarts
        # Smoothly decays LR then restarts, helping escape local optima in self-play
        self.lr_min = lr_min
        self.lr_T0 = lr_T0
        self.lr_T_mult = lr_T_mult
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=lr_T0, T_mult=lr_T_mult, eta_min=lr_min
        )
        
        # Training step counter
        self.train_step_counter = 0
        # Internal counter for learn_every gating
        self._learn_call_counter = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation as numpy array
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Selected action index (integer)
        """
        if explore and random.random() < self.epsilon:
            # Explore: select random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: select best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> float:
        """
        Sample a batch from memory and perform a Double DQN learning step.
        Only actually runs the gradient update every `learn_every` calls.
        
        Returns:
            Loss value for the current training step (0.0 when skipped)
        """
        self._learn_call_counter += 1

        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples to learn

        # Skip gradient update on most calls for throughput
        if self._learn_call_counter % self.learn_every != 0:
            return 0.0
        
        # Sample from numpy replay buffer
        states_np, actions_np, rewards_np, next_states_np, dones_np = \
            self.memory.sample(self.batch_size)

        # Convert to tensors on the correct device
        states_tensor      = torch.from_numpy(states_np).to(DEVICE)
        actions_tensor     = torch.from_numpy(actions_np).unsqueeze(1).to(DEVICE)
        rewards_tensor     = torch.from_numpy(rewards_np).unsqueeze(1).to(DEVICE)
        next_states_tensor = torch.from_numpy(next_states_np).to(DEVICE)
        dones_tensor       = torch.from_numpy(dones_np).unsqueeze(1).to(DEVICE)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.model(states_tensor).gather(1, actions_tensor)
        
        # ── Double DQN ──
        # Main network selects the best action, target network evaluates it.
        # This reduces Q-value overestimation compared to vanilla DQN.
        with torch.no_grad():
            best_actions = self.model(next_states_tensor).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states_tensor).gather(1, best_actions)
        
        # Compute target Q-values: r + γ * Q_target(s', argmax_a Q_main(s', a)) * (1 - done)
        target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
        
        # Compute loss (Huber / SmoothL1)
        loss = self.criterion(current_q_values, target_q_values)
        
        # Perform gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Soft-update target network (Polyak averaging) every learning step
        self.train_step_counter += 1
        self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self, hard: bool = False):
        """
        Update target network weights.
        
        Args:
            hard: If True, perform a full copy. Otherwise use soft Polyak averaging.
        """
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            for tp, mp in zip(self.target_model.parameters(), self.model.parameters()):
                tp.data.copy_(self.tau * mp.data + (1.0 - self.tau) * tp.data)

    def update_epsilon(self):
        """
        Decay epsilon at the end of an episode (not every step).
        Call this once per episode.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def step_lr_scheduler(self):
        """Advance the learning rate scheduler by one step (call once per episode)."""
        self.scheduler.step()

    def get_current_lr(self) -> float:
        """Return the current learning rate from the scheduler."""
        return self.scheduler.get_last_lr()[0]
    
    def save_model(self, filepath: str):
        """
        Save the main network weights to a file.
        
        Args:
            filepath: Path to save the model weights
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'train_step_counter': self.train_step_counter
        }, filepath)
    
    def load_model(self, filepath: str, override_lr: Optional[float] = None, override_epsilon: Optional[float] = None):
        """
        Load network weights from a file.
        
        Args:
            filepath: Path to load the model weights from
            override_lr: Optional learning rate to override the saved optimizer schedule
            override_epsilon: Optional epsilon to override the saved starting epsilon
        """
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        
        if override_lr is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=override_lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.lr_T0, T_mult=self.lr_T_mult,
                eta_min=self.lr_min
            )
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if override_epsilon is not None:
            self.epsilon = override_epsilon
        else:
            self.epsilon = checkpoint['epsilon']
            
        self.train_step_counter = checkpoint['train_step_counter']
    
    def get_stats(self) -> dict:
        """
        Get current training statistics.
        
        Returns:
            Dictionary containing agent statistics
        """
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'train_steps': self.train_step_counter,
            'learning_rate': self.get_current_lr()
        }
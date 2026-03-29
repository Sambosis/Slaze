"""
agent.py - Deep Q-Network (DQN) agent implementation for the solitaire RL project.

This module implements the DQNAgent class which uses a neural network to learn
to play solitaire through reinforcement learning. The agent includes methods for
state representation, action selection, training, and experience replay.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional

from rl.memory import ReplayMemory
import config

class DQN(nn.Module):
    """
    Neural network architecture for the DQN agent.

    This network takes the game state as input and outputs Q-values for each
    possible action. The architecture consists of fully connected layers with
    ReLU activation functions.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the DQN network.

        Args:
            input_size: The size of the input state vector.
            output_size: The number of possible actions (output Q-values).
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, config.HIDDEN_LAYER_1_SIZE)
        self.fc2 = nn.Linear(config.HIDDEN_LAYER_1_SIZE, config.HIDDEN_LAYER_2_SIZE)
        self.fc3 = nn.Linear(config.HIDDEN_LAYER_2_SIZE, output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the network weights using Xavier initialization.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor representing the game state.

        Returns:
            Tensor containing Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    Deep Q-Network agent for learning to play solitaire.

    This class implements the DQN algorithm with experience replay and target
    network for stable training. The agent can select actions using an
    epsilon-greedy policy and learn from experiences stored in memory.
    """

    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        """
        Initialize the DQN agent.

        Args:
            state_size: The size of the state representation vector.
            action_size: The number of possible actions.
            device: The device to run training on ('cpu' or 'cuda').
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_start = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.steps = 0

        # Initialize networks
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self._update_target_network()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)

        # Initialize replay memory
        self.memory = ReplayMemory(config.MEMORY_SIZE)

    def _update_target_network(self):
        """
        Update the target network with weights from the main network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, float]:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: The current game state as a numpy array.
            valid_actions: List of indices representing valid actions.

        Returns:
            A tuple containing:
            - The selected action index
            - The exploration rate (epsilon) used for this action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action from valid actions
            action_idx = random.choice(valid_actions)
            return action_idx, self.epsilon

        # Get Q-values from the model
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Get Q-values for valid actions only
        valid_q_values = q_values[0][valid_actions]

        # Select action with highest Q-value
        action_idx = valid_actions[torch.argmax(valid_q_values).item()]

        return action_idx, self.epsilon

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Store an experience in the replay memory.

        Args:
            state: The current game state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting game state.
            done: Whether the episode ended.
        """
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size: int) -> Optional[float]:
        """
        Train the agent using experiences from the replay memory.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            The average loss for the batch, or None if not enough experiences.
        """
        if len(self.memory) < batch_size:
            return None

        # Sample a batch of experiences
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q-values for the actions taken
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Get maximum Q-values for next states from target network
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calculate loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % config.TARGET_NETWORK_UPDATE_FREQ == 0:
            self._update_target_network()

        return loss.item()

    def decay_epsilon(self):
        """
        Decay the exploration rate according to the decay factor.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        """
        Save the model weights to a file.

        Args:
            path: The file path to save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load_model(self, path: str):
        """
        Load model weights from a file.

        Args:
            path: The file path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in a given state.

        Args:
            state: The game state as a numpy array.

        Returns:
            Array of Q-values for each action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
        return q_values.cpu().numpy()[0]

    def reset_epsilon(self):
        """Reset epsilon to its starting value."""
        self.epsilon = self.epsilon_start

    def get_stats(self) -> dict:
        """
        Get training statistics for the agent.

        Returns:
            Dictionary containing current epsilon, steps taken, and memory size.
        """
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'memory_size': len(self.memory)
        }
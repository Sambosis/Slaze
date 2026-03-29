"""
memory.py - Experience replay buffer for the DQN agent.

This module implements the ReplayMemory class, which stores and manages experiences
for the reinforcement learning agent. The experience replay buffer allows the agent
to learn from past experiences by sampling random batches, which helps stabilize
training and improve learning efficiency.
"""

import random
from collections import deque
from typing import Tuple, List, Any
import numpy as np

class ReplayMemory:
    """
    Experience replay buffer for storing and sampling experiences.

    The replay memory stores tuples of (state, action, reward, next_state, done)
    and provides methods for adding new experiences and sampling random batches
    for training the DQN agent.

    Attributes:
        capacity (int): Maximum number of experiences to store.
        memory (deque): Collection of stored experiences.
        batch_size (int): Number of experiences to sample in each batch.
    """

    def __init__(self, capacity: int = 10000, batch_size: int = 64):
        """
        Initialize the ReplayMemory with specified capacity and batch size.

        Args:
            capacity: Maximum number of experiences to store.
            batch_size: Number of experiences to sample in each batch.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state: np.ndarray, action: Any, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a new experience to the memory.

        Args:
            state: The current game state.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The resulting game state after the action.
            done: Whether the episode ended after this action.
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size: int = None) -> Tuple[List[np.ndarray], List[Any], List[float],
                             List[np.ndarray], List[bool]]:
        """
        Sample a random batch of experiences from memory.

        Args:
            batch_size: Number of experiences to sample. If None, uses the default batch_size.

        Returns:
            A tuple containing:
            - List of states
            - List of actions
            - List of rewards
            - List of next states
            - List of done flags

        Raises:
            ValueError: If the memory doesn't contain enough experiences to sample.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            raise ValueError(f"Not enough experiences in memory. "
                           f"Current size: {len(self.memory)}, "
                           f"Required: {batch_size}")

        # Randomly sample batch_size experiences
        batch = random.sample(self.memory, batch_size)

        # Unzip the batch into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self) -> int:
        """
        Return the number of experiences currently stored in memory.

        Returns:
            The current size of the memory.
        """
        return len(self.memory)

    def is_ready(self, batch_size: int = None) -> bool:
        """
        Check if the memory contains enough experiences for sampling.

        Args:
            batch_size: The number of samples needed. If None, uses the default batch_size.

        Returns:
            True if the memory has at least batch_size experiences, False otherwise.
        """
        if batch_size is None:
            batch_size = self.batch_size
        return len(self.memory) >= batch_size

    def clear(self) -> None:
        """
        Clear all experiences from the memory.
        """
        self.memory.clear()

    def get_memory_stats(self) -> dict:
        """
        Get statistics about the current state of the memory.

        Returns:
            A dictionary with memory statistics including:
            - current_size: Current number of experiences
            - capacity: Maximum capacity
            - usage_percentage: Percentage of capacity used
        """
        return {
            'current_size': len(self.memory),
            'capacity': self.capacity,
            'usage_percentage': (len(self.memory) / self.capacity * 100) if self.capacity > 0 else 0
        }
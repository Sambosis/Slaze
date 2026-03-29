"""AI Utilities for Reinforcement Learning.

Contains the ReplayBuffer class for experience replay and utility functions
for tensor conversions and curiosity reward calculation.
"""

import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional, Union
from config import Config


class ReplayBuffer:
    """
    Experience replay buffer using deque for efficient storage and sampling.
    
    Stores experience tuples: (state, action, reward, next_state, done)
    
    Args:
        config: Global configuration object
        buffer_size: Maximum number of experiences to store (default: 50,000)
    """
    
    def __init__(self, config: Config, buffer_size: int = None):
        self.config = config
        self.buffer_size = buffer_size or config.REPLAY_BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE
        
        # Buffer stores tuples: (state, action, reward, next_state, done)
        self.buffer = deque(maxlen=self.buffer_size)
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add a single experience tuple to the buffer.
        
        Args:
            state: Current state tensor (channels, H, W)
            action: Action index (int)
            reward: Reward received (float)
            next_state: Next state tensor (channels, H, W)
            done: Episode completion flag (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                             torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
            - states: (batch_size, channels, H, W)
            - actions: (batch_size,)
            - rewards: (batch_size,)
            - next_states: (batch_size, channels, H, W)
            - dones: (batch_size,)
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and stack
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


def calculate_curiosity(icm: 'ICMModel', state: torch.Tensor, 
                       next_state: torch.Tensor, action: torch.Tensor, 
                       config: Config) -> torch.Tensor:
    """
    Compute intrinsic (curiosity) reward using the ICM forward model prediction error.
    
    Args:
        icm: Trained ICM model
        state: Current state (batch_size, channels, H, W)
        next_state: Next state (batch_size, channels, H, W)
        action: Action indices (batch_size,) or (batch_size, 1)
        config: Global configuration
    
    Returns:
        intrinsic_reward: (batch_size,) scaled prediction errors
    """
    # Forward pass through ICM
    icm_output = icm(state, next_state, action)
    
    # L2 prediction error between predicted and actual next features
    pred_next_feat = icm_output['pred_next_feat']
    next_state_feat = icm_output['next_state_feat']
    
    # Mean squared error across feature dimensions
    prediction_error = F.mse_loss(pred_next_feat, next_state_feat, 
                                reduction='none').mean(dim=1)
    
    # Scale by ICM eta parameter
    intrinsic_reward = config.ICM_ETA * prediction_error
    
    return intrinsic_reward


def state_to_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert numpy state array to PyTorch tensor on specified device.
    
    Args:
        state: Numpy array (channels, H, W) or (H, W)
        device: Target torch device
    
    Returns:
        tensor: (channels, H, W) on specified device
    """
    if len(state.shape) == 3:  # Already (C, H, W)
        tensor = torch.FloatTensor(state)
    else:  # Convert (H, W) to single channel
        tensor = torch.FloatTensor(state).unsqueeze(0)
    
    return tensor.to(device)


def actions_to_dict(unit_ids: list, actions: torch.Tensor) -> dict:
    """
    Convert batched action tensor to dictionary mapping unit_id -> action.
    
    Args:
        unit_ids: List of unit IDs corresponding to actions
        actions: Tensor of action indices (batch_size,)
    
    Returns:
        actions_dict: {unit_id: action_index}
    """
    return {unit_id: int(action.item()) for unit_id, action in zip(unit_ids, actions)}


def compute_team_reward(rewards_dict: dict, team: int) -> float:
    """
    Compute average reward for a specific team from rewards dictionary.
    
    Args:
        rewards_dict: {unit_id: reward} mapping
        team: Team ID (0 for Red, 1 for Blue)
    
    Returns:
        avg_reward: Mean reward for alive units of the team
    """
    from game.environment import BattleEnv  # Lazy import for unit access
    
    # Note: This assumes global access to env.units - in practice, pass env
    team_rewards = [r for uid, r in rewards_dict.items() 
                   if any(u.id == uid and u.team == team and u.is_alive() 
                         for u in BattleEnv.units)]
    
    return np.mean(team_rewards) if team_rewards else 0.0
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from pathlib import Path

import config
from src.environment import WarGameEnv
from src.model import Agent
from src.utils import save_model, load_model, HallOfFame
from src.visualizer import render_game

# Global Configuration
CFG = config.CFG

class Memory:
    """
    Experience Replay Buffer for PPO.
    Stores transition data: (state, action, log_prob, reward, value, done).
    """
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def store(self, state: torch.Tensor, action: torch.Tensor, 
              log_prob: torch.Tensor, reward: float, value: torch.Tensor, done: bool):
        """Store a single transition tuple."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Clear all stored transitions."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert stored transitions to batched tensors on target device."""
        if not self.states:
            raise ValueError("Memory is empty")
        
        return {
            'states': torch.stack(self.states).to(device),
            'actions': torch.stack(self.actions).to(device),
            'old_log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32).to(device),
            'values': torch.stack(self.values).to(device),
            'dones': torch.tensor(self.dones, dtype=torch.float32).to(device),
        }

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, 
                next_value: torch.Tensor, gamma: float, gae_lambda: float) -> torch.Tensor:
    """
    Computes Generalized Advantage Estimation (GAE).
    """
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(1, device=rewards.device)
    
    # Ensure values are [T] not [T, 1]
    values = values.squeeze(-1)
    next_value = next_value.squeeze(-1)
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
    
    return advantages

def ppo_update(agent: Agent, optimizer: torch.optim.Optimizer, memory: Memory) -> Dict[str, float]:
    """
    Performs PPO update on collected experience using Clipped Surrogate Objective.
    """
    agent.train()
    device = next(agent.parameters()).device
    batch = memory.get_batch(device)
    
    # Hyperparameters
    clip_coef = CFG.training.CLIP_COEF
    value_coef = CFG.training.VALUE_COEF
    entropy_coef = CFG.training.ENTROPY_COEF
    max_grad_norm = CFG.training.MAX_GRAD_NORM
    gamma = CFG.training.GAMMA
    gae_lambda = CFG.training.GAE_LAMBDA
    
    # Compute Advantages using GAE
    with torch.no_grad():
        # Bootstrap value is 0 because we typically finish episodes or handle truncation elsewhere
        # For strictly correct bootstrapping on truncation, next_value should be estimated from next_state
        # Here we assume episode completion or 0 value for simplicity in this architecture
        next_value = torch.zeros(1, device=device)
        
        advantages = compute_gae(
            batch['rewards'], 
            batch['values'], 
            batch['dones'], 
            next_value,
            gamma, 
            gae_lambda
        )
        returns = advantages + batch['values'].squeeze(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Training loop
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    entropy_loss_sum = 0
    
    batch_size = len(batch['states'])
    num_minibatches = CFG.training.NUM_MINIBATCHES
    minibatch_size = batch_size // num_minibatches
    
    # Indices for shuffling
    indices = torch.arange(batch_size, device=device)
    
    for _ in range(CFG.training.UPDATE_EPOCHS):
        # Shuffle batch
        indices = torch.randperm(batch_size, device=device)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            # Get minibatch data
            mb_states = batch['states'][mb_indices]
            mb_actions = batch['actions'][mb_indices]
            mb_old_log_probs = batch['old_log_probs'][mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            
            # Evaluate current policy
            new_log_probs, new_entropy, new_values = agent.evaluate_actions(mb_states, mb_actions)
            
            # PPO Ratio
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()
            
            # Surrogate Loss
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss (Clipped or MSE)
            value_loss = 0.5 * ((new_values.squeeze(-1) - mb_returns) ** 2).mean()
            
            # Entropy Loss
            entropy_loss = -new_entropy.mean()
            
            # Total Loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            
            # Tracking
            total_loss +=
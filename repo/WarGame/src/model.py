"""
Neural Network Architecture - Actor-Critic Agent
================================================
PyTorch implementation of the Actor-Critic network for the WarGame environment.
Features a shared CNN encoder for spatial processing, followed by separate Actor
(multi-discrete policy for unit control) and Critic (state value estimation) heads.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Tuple, Optional
import config


class Agent(nn.Module):
    """
    Actor-Critic Neural Network for the WarGame environment.
    
    Implements a shared Convolutional Encoder for spatial feature extraction,
    followed by separate Actor and Critic heads. The Actor head utilizes a 
    Multi-Discrete action space to control multiple units simultaneously 
    assuming conditional independence given the state.
    """

    def __init__(self):
        super(Agent, self).__init__()
        
        self.cfg = config.CFG
        self.device_type = self.cfg.get_device()
        
        # Game constraints
        self.num_units = self.cfg.game.TANK_COUNT + self.cfg.game.ARTILLERY_COUNT
        self.action_dim = self.cfg.game.ACTION_SPACE_SIZE
        self.obs_channels = self.cfg.game.OBSERVATION_CHANNELS
        self.grid_h = self.cfg.game.GRID_HEIGHT
        self.grid_w = self.cfg.game.GRID_WIDTH

        # --- 1. Convolutional Encoder ---
        # Dynamically build CNN layers based on config
        cnn_layers = []
        in_channels = self.obs_channels
        
        for out_channels in self.cfg.model.CNN_CHANNELS:
            cnn_layers.append(self._layer_init(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=self.cfg.model.CNN_KERNEL_SIZE,
                    stride=self.cfg.model.CNN_STRIDE,
                    padding=self.cfg.model.CNN_PADDING
                )
            ))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
            
        self.encoder = nn.Sequential(*cnn_layers)

        # Calculate output size of CNN to feed into Linear layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.obs_channels, self.grid_h, self.grid_w)
            cnn_output = self.encoder(dummy_input)
            self.flat_size = cnn_output.view(1, -1).shape[1]

        # --- 2. Shared Embedding ---
        self.embedding = nn.Sequential(
            self._layer_init(nn.Linear(self.flat_size, self.cfg.model.EMBEDDING_SIZE)),
            nn.ReLU()
        )

        # --- 3. Critic Head (Value Function) ---
        critic_layers = []
        curr_size = self.cfg.model.EMBEDDING_SIZE
        for hidden_size in self.cfg.model.CRITIC_HIDDEN_SIZES:
            critic_layers.append(self._layer_init(nn.Linear(curr_size, hidden_size)))
            critic_layers.append(nn.Tanh()) # Tanh is often preferred in PPO over ReLU for heads
            curr_size = hidden_size
        
        # Output is a single scalar value
        critic_layers.append(self._layer_init(nn.Linear(curr_size, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # --- 4. Actor Head (Policy Function) ---
        # The actor outputs logits for ALL units flattened.
        # Shape: [Batch, Num_Units * Action_Dim]
        actor_layers = []
        curr_size = self.cfg.model.EMBEDDING_SIZE
        for hidden_size in self.cfg.model.ACTOR_HIDDEN_SIZES:
            actor_layers.append(self._layer_init(nn.Linear(curr_size, hidden_size)))
            actor_layers.append(nn.Tanh())
            curr_size = hidden_size
        
        # Final layer outputs logits for all units' actions
        actor_layers.append(self._layer_init(
            nn.Linear(curr_size, self.num_units * self.action_dim),
            std=0.01
        ))
        self.actor = nn.Sequential(*actor_layers)

        self.to(self.device_type)

    def _layer_init(self, layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        """
        Orthogonal or Kaiming initialization for layers based on config.
        """
        if self.cfg.model.ORTHOGONAL_INIT:
            # Orthogonal initialization (common in RL)
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, bias_const)
            elif isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, bias_const)
        else:
            # Kaiming initialization as fallback
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, bias_const)
        return layer

    def calculate_embedding(self, state_grid: torch.Tensor) -> torch.Tensor:
        """
        CNN layer sequence that converts the visual grid state into a feature vector.
        
        Args:
            state_grid: [B, C, H, W] observation tensor
            
        Returns:
            torch.Tensor: [B, embedding_size] feature vector
        """
        # Encode spatial features
        encoded = self.encoder(state_grid)
        # Global average pool + flatten
        encoded = encoded.mean(dim=(-2, -1)).squeeze(-1).squeeze(-1)
        # Project to embedding space
        embedding = self.embedding(encoded)
        return embedding

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Actor-Critic network.
        
        Args:
            state: [B, C, H, W] batched observation tensor
            
        Returns:
            action: [B] selected action index (flattened across all units)
            log_prob: [B] log probability of selected action
            entropy: [B] entropy of action distribution
            value: [B, 1] state value estimate
        """
        # Ensure state is on correct device and has batch dim
        if state.dim() == 3:  # [C, H, W]
            state = state.unsqueeze(0)
        
        state = state.to(self.device_type)
        
        # Shared embedding
        embedding = self.calculate_embedding(state)
        
        # Actor forward pass - get logits for all units
        logits = self.actor(embedding)  # [B, num_units * action_dim]
        
        # Reshape to [B, num_units, action_dim] for per-unit distributions
        logits = logits.view(-1, self.num_units, self.action_dim)  # [B, num_units, action_dim]
        
        # Sample one action per unit independently
        dist = Categorical(logits=logits)
        actions = dist.sample()  # [B, num_units]
        
        # Flatten back to single action index for storage (unit_id * action_dim + action)
        flat_actions = (actions * self.action_dim).long().squeeze(-1)  # [B]
        
        log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum log probs across units
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic forward pass
        values = self.critic(embedding)
        
        return flat_actions, log_probs, entropy, values

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only (no action sampling).
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        state = state.to(self.device_type)
        embedding = self.calculate_embedding(state)
        return self.critic(embedding)

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under current policy (used in PPO).
        
        Args:
            state: [B, C, H, W]
            actions: [B] flat action indices
            
        Returns:
            log_probs: [B] log probability of actions
            entropy: [B] policy entropy
            values: [B, 1] state values
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        state = state.to(self.device_type)
        embedding = self.calculate_embedding(state)
        
        # Actor
        logits = self.actor(embedding)
        logits = logits.view(-1, self.num_units, self.action_dim)
        
        dist = Categorical(logits=logits)
        
        # Convert flat actions back to per-unit actions
        unit_actions = (actions // self.action_dim).long()  # Unit action indices
        action_indices = (actions % self.action_dim).long()  # Action within unit
        
        log_probs = dist.log_prob(torch.stack([unit_actions, action_indices], dim=-1)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic
        values = self.critic(embedding)
        
        return log_probs, entropy, values
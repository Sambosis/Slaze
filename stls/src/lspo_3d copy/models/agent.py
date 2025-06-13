"""
Defines the PPO agent's policy and value networks for the 3D-LSPO project.

This module contains the `AgentPolicy` class, which is an actor-critic model
implemented using PyTorch. This model is central to the reinforcement learning
agent, as it learns to select optimal sequences of design motifs to generate
3D models.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class AgentPolicy(nn.Module):
    """
    Implements the PPO agent's combined policy (actor) and value (critic) networks.

    This network takes the current environment state as input and outputs:
    1. A probability distribution over the discrete action space (design motifs).
    2. An estimate of the value of the current state.

    The actor and critic share a common set of feature extraction layers to improve
    efficiency and learning performance. The network is designed to be used
    within a PPO training loop.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the layers of the actor-critic network.

        Args:
            state_dim (int): The dimensionality of the input state space. This corresponds
                             to the size of the embedding for the design prompt and the
                             current sequence of selected motifs.
            action_dim (int): The number of possible discrete actions (design motifs).
            hidden_dim (int): The size of the hidden layers in the network.
        """
        super(AgentPolicy, self).__init__()

        # Shared network layers for feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: Outputs logits for the action probability distribution
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head: Outputs a single value representing the state's value
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Performs the forward pass through the network.

        This method is not typically called directly, but rather through helper methods
        like `get_action_and_value` or `evaluate_actions`.

        Args:
            state (torch.Tensor): A tensor representing the current state of the
                                  environment. Shape: (batch_size, state_dim).

        Returns:
            Tuple[Categorical, torch.Tensor]:
                - A Categorical distribution over the actions based on the policy logits.
                - A tensor representing the estimated value of the state.
                  Shape: (batch_size, 1).
        """
        # 1. Pass state through shared layers to get shared features.
        shared_features = self.shared_net(state)
        
        # 2. Pass shared features through the actor head to get action logits.
        action_logits = self.actor_head(shared_features)
        
        # 3. Create a Categorical distribution from the logits.
        dist = Categorical(logits=action_logits)
        
        # 4. Pass shared features through the critic head to get the state value.
        value = self.critic_head(shared_features)
        
        # 5. Return the distribution and the value.
        return dist, value

    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets an action, its log probability, and the state value for a given state.

        This method is typically used during the environment interaction phase
        (rollout collection). It operates in evaluation mode and without tracking
        gradients to speed up inference.

        Args:
            state (torch.Tensor): A tensor representing the current state.
                                  Expected shape: (1, state_dim) or (state_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - action (torch.Tensor): The sampled action (motif ID).
                - log_prob (torch.Tensor): The log probability of the sampled action.
                - value (torch.Tensor): The estimated value of the state.
        """
        # 1. Ensure state tensor is correctly shaped for a single batch item.
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # 2. Use torch.no_grad() to disable gradient calculations for inference.
        with torch.no_grad():
            # 3. Pass the state through the shared network and heads to get distribution and value.
            dist, value = self.forward(state)
            
            # 4. Sample an action from the Categorical distribution.
            action = dist.sample()
            
            # 5. Calculate the log probability of the sampled action.
            log_prob = dist.log_prob(action)

        # 6. Return the sampled action, its log probability, and the state value.
        return action, log_prob, value

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates a batch of states and actions to compute their log probabilities,
        the policy's entropy, and the state values.

        This method is used during the PPO update/learning phase to compute the
        components of the loss function. It requires gradient tracking.

        Args:
            state (torch.Tensor): A batch of states. Shape: (batch_size, state_dim).
            action (torch.Tensor): A batch of actions taken in those states.
                                   Shape: (batch_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - log_probs (torch.Tensor): The log probability of each action in the
                                            batch under the current policy.
                - entropy (torch.Tensor): The entropy of the action distribution for
                                          each state in the batch.
                - values (torch.Tensor): The estimated value of each state in the batch.
        """
        # 1. Pass the batch of states through the shared network and heads to get
        #    the action distributions and state values.
        dist, values = self.forward(state)

        # 2. Use the distributions to calculate the log probability of the provided
        #    actions (`dist.log_prob(action)`).
        log_probs = dist.log_prob(action)
        
        # 3. Calculate the entropy of the distributions (`dist.entropy()`).
        entropy = dist.entropy()
        
        # 4. Return the calculated log probabilities, entropy, and state values.
        # .squeeze(-1) ensures values tensor has shape (batch_size,)
        return log_probs, entropy, values.squeeze(-1)
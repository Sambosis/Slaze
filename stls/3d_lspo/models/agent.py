import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

class AgentPolicy(nn.Module):
    """
    An Actor-Critic network for the PPO agent.

    This class defines the neural network architecture for the policy (actor) and
    the value function (critic). It uses a shared backbone to process the state
    and then splits into two heads: one for action selection and one for state
    valuation. This is a common and efficient design for PPO.

    Attributes:
        shared_network (nn.Sequential): A sequence of layers that serves as a
            common feature extractor for both the actor and the critic.
        actor_head (nn.Linear): The final layer of the policy network, which
            outputs logits for the action distribution.
        critic_head (nn.Linear): The final layer of the value network, which
            outputs a single value estimating the return from the state.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the Actor-Critic network layers.

        Args:
            state_dim (int): The dimensionality of the input state space.
                This corresponds to the size of the encoded representation of the
                design prompt and the current sequence of motifs.
            action_dim (int): The dimensionality of the action space. This is the
                total number of unique design motifs the agent can choose from.
            hidden_dim (int, optional): The number of neurons in the hidden
                layers. Defaults to 256.
        """
        super(AgentPolicy, self).__init__()

        # A simple MLP is a good starting point for the shared backbone.
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # The actor head outputs logits for each possible action (motif).
        # A softmax will be applied to these logits by the Categorical distribution.
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # The critic head outputs a single scalar value for the given state.
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Performs a forward pass through the shared network and both heads.

        This method is not typically called directly during training but is
        used by helper methods like `get_action_and_value`.

        Args:
            state (torch.Tensor): A tensor representing the current state of the
                environment. Shape: (batch_size, state_dim).

        Returns:
            Tuple[Categorical, torch.Tensor]: A tuple containing:
                - The action distribution (Categorical) from the actor head.
                - The estimated state value (torch.Tensor) from the critic head.
        """
        # Pass the state through the shared feature extractor.
        shared_features = self.shared_network(state)

        # Calculate action logits and create a distribution for the actor.
        action_logits = self.actor_head(shared_features)
        action_dist = Categorical(logits=action_logits)

        # Calculate the state value for the critic.
        state_value = self.critic_head(shared_features)

        return action_dist, state_value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes an action, its log probability, and the state value.

        This is the primary method used during the PPO training loop. It gets the
        action distribution from the actor, samples an action if one is not
        provided, and calculates the log probability of that action. It also
        returns the critic's valuation of the state.

        Args:
            state (torch.Tensor): A tensor representing the current state(s).
                Shape: (batch_size, state_dim).
            action (torch.Tensor, optional): A tensor representing a specific
                action taken. If provided, its log probability will be
                calculated. If None, a new action is sampled from the policy.
                Shape: (batch_size,). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - action (torch.Tensor): The action sampled or provided.
                - log_prob (torch.Tensor): The log probability of the action.
                - value (torch.Tensor): The critic's valuation of the state,
                  squeezed to match log_prob's shape.
        """
        # 1. Pass state through the shared network.
        shared_features = self.shared_network(state)

        # 2. Get action logits from the actor head.
        action_logits = self.actor_head(shared_features)

        # 3. Create a Categorical distribution.
        dist = Categorical(logits=action_logits)

        # 4. Get the value from the critic head.
        value = self.critic_head(shared_features)

        # 5. If action is None, sample from the distribution.
        if action is None:
            action = dist.sample()

        # 6. Calculate log_prob of the action using the distribution.
        log_prob = dist.log_prob(action)

        # The value tensor is squeezed to remove the last dimension (from 1 to None)
        # to match the shape of the log_prob tensor for easier calculations in PPO.
        return action, log_prob, value.squeeze(-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the state value using only the critic network.

        This is a convenience method used during the PPO update to get the
        value of the next state in the gathered trajectories.

        Args:
            state (torch.Tensor): A tensor representing the state(s) to be
                evaluated. Shape: (batch_size, state_dim).

        Returns:
            torch.Tensor: The estimated value of the state(s).
                Shape: (batch_size, 1).
        """
        # 1. Pass state through the shared network.
        shared_features = self.shared_network(state)
        
        # 2. Pass the result through the critic head.
        value = self.critic_head(shared_features)
        
        # 3. Return the value.
        return value
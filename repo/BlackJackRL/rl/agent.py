"""
Reinforcement Learning Agent for Blackjack Bet Sizing.

This module implements a Q-learning agent that optimizes bet sizing in blackjack
based on the current count, bankroll percentage, and previous bet size. The agent
uses a discretized state space and implements exploration vs. exploitation through
epsilon-greedy policy.

Key Features:
- Q-learning implementation for bet sizing optimization
- Discretized state representation (count, bankroll %, previous bet)
- Epsilon-greedy exploration policy
- Model persistence support
- Integration with blackjack game engine

The agent's state representation includes:
1. Current true count (discretized)
2. Bankroll percentage (discretized)
3. Previous bet size as percentage of min bet (discretized)

Actions correspond to bet sizes as multiples of the minimum bet.
"""

import random
import math
import pickle
import numpy as np
from typing import Dict, Tuple, List, Optional
import config
import os

class RLAgent:
    """
    Reinforcement Learning Agent for optimizing bet sizing in blackjack.

    This agent uses Q-learning to determine optimal bet sizes based on game state.
    The state includes the current count, bankroll percentage, and previous bet size.

    Attributes:
        q_table (dict): Dictionary mapping states to action values
        learning_rate (float): Learning rate for Q-learning updates
        discount_factor (float): Discount factor for future rewards
        exploration_rate (float): Current exploration rate (epsilon)
        min_exploration_rate (float): Minimum exploration rate
        exploration_decay (float): Decay rate for exploration
        state_discretization (dict): Configuration for state discretization
        reward_scaling (float): Scaling factor for rewards
        action_space (list): Possible bet sizes as multiples of min bet
        count_bins (list): Bins for discretizing count values
        bankroll_bins (list): Bins for discretizing bankroll percentages
        bet_size_bins (list): Bins for discretizing previous bet sizes
    """

    def __init__(self,
                 learning_rate: float = None,
                 discount_factor: float = None,
                 initial_exploration_rate: float = None,
                 min_exploration_rate: float = None,
                 exploration_decay: float = None,
                 state_discretization: dict = None,
                 reward_scaling: float = None):
        """
        Initialize the RLAgent with configuration parameters.

        Args:
            learning_rate: Learning rate for Q-learning (default from config)
            discount_factor: Discount factor for future rewards (default from config)
            initial_exploration_rate: Initial exploration rate (default from config)
            min_exploration_rate: Minimum exploration rate (default from config)
            exploration_decay: Decay rate for exploration (default from config)
            state_discretization: Configuration for state discretization (default from config)
            reward_scaling: Scaling factor for rewards (default from config)
        """
        # Load configuration with defaults from config.py
        rl_config = config.RL_CONFIG
        self.learning_rate = learning_rate or rl_config["learning_rate"]
        self.discount_factor = discount_factor or rl_config["discount_factor"]
        self.exploration_rate = initial_exploration_rate or rl_config["initial_exploration_rate"]
        self.min_exploration_rate = min_exploration_rate or rl_config["min_exploration_rate"]
        self.exploration_decay = exploration_decay or rl_config["exploration_decay"]
        self.state_discretization = state_discretization or rl_config["state_discretization"]
        self.reward_scaling = reward_scaling or rl_config["reward_scaling"]

        # Initialize Q-table
        self.q_table = {}

        # Define action space (bet sizes as multiples of min bet)
        self.action_space = list(range(1, 13))  # 1x to 12x min bet

        # Create discretization bins
        self._initialize_discretization_bins()

        # Validate configuration
        self._validate_config()

    def _initialize_discretization_bins(self) -> None:
        """Initialize bins for state discretization."""
        # Count bins (typically -10 to +10)
        count_range = (-10, 10)
        self.count_bins = self._create_bins(
            count_range[0], count_range[1],
            self.state_discretization["count_bins"]
        )

        # Bankroll percentage bins (0% to 100%)
        bankroll_range = (0, 100)
        self.bankroll_bins = self._create_bins(
            bankroll_range[0], bankroll_range[1],
            self.state_discretization["bankroll_bins"]
        )

        # Bet size bins (1x to 12x min bet)
        bet_size_range = (1, 12)
        self.bet_size_bins = self._create_bins(
            bet_size_range[0], bet_size_range[1],
            self.state_discretization["bet_size_bins"]
        )

    def _create_bins(self, min_val: float, max_val: float, num_bins: int) -> List[float]:
        """
        Create discretization bins for a given range using numpy.

        Args:
            min_val: Minimum value of the range
            max_val: Maximum value of the range
            num_bins: Number of bins to create

        Returns:
            List of bin edges
        """
        return np.linspace(min_val, max_val, num_bins).tolist()

    def _validate_config(self) -> None:
        """Validate agent configuration parameters."""
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1")

        if not 0 <= self.discount_factor <= 1:
            raise ValueError("Discount factor must be between 0 and 1")

        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("Exploration rate must be between 0 and 1")

        if not 0 <= self.min_exploration_rate <= 1:
            raise ValueError("Minimum exploration rate must be between 0 and 1")

        if not 0 <= self.exploration_decay <= 1:
            raise ValueError("Exploration decay must be between 0 and 1")

        if self.exploration_rate < self.min_exploration_rate:
            raise ValueError("Initial exploration rate must be >= minimum exploration rate")

    def discretize_state(self, count: float, bankroll: float, min_bet: float, prev_bet: float = None) -> Tuple[int, int, int]:
        """
        Discretize continuous state variables into bins.

        Args:
            count: Current true count
            bankroll: Current bankroll amount
            min_bet: Minimum bet size
            prev_bet: Previous bet size (optional)

        Returns:
            Tuple of discretized state indices (count_bin, bankroll_bin, bet_size_bin)
        """
        # Discretize count
        count_bin = self._discretize_value(count, self.count_bins)

        # Discretize bankroll percentage (relative to min bet)
        bankroll_pct = (bankroll / min_bet) if min_bet > 0 else 0
        bankroll_bin = self._discretize_value(bankroll_pct, self.bankroll_bins)

        # Discretize previous bet size (as multiple of min bet)
        if prev_bet is None:
            bet_size_bin = 0  # Special case for first bet
        else:
            bet_size_multiple = prev_bet / min_bet if min_bet > 0 else 0
            bet_size_bin = self._discretize_value(bet_size_multiple, self.bet_size_bins)

        return (count_bin, bankroll_bin, bet_size_bin)

    def _discretize_value(self, value: float, bins: List[float]) -> int:
        """
        Discretize a continuous value into a bin index using numpy.

        Args:
            value: Value to discretize
            bins: List of bin edges

        Returns:
            Index of the bin containing the value
        """
        # Handle edge cases
        if not bins or len(bins) == 0:
            return 0

        if value <= bins[0]:
            return 0
        if value >= bins[-1]:
            return len(bins) - 1

        # Use numpy's digitize for efficient binning
        return int(np.digitize(value, bins)) - 1

    def get_state_key(self, state: Tuple[int, int, int]) -> str:
        """
        Convert state tuple to a string key for Q-table lookup.

        Args:
            state: State tuple (count_bin, bankroll_bin, bet_size_bin)

        Returns:
            String representation of the state
        """
        return f"{state[0]}_{state[1]}_{state[2]}"

    def get_bet_size(self, count: float, bankroll: float, min_bet: float, prev_bet: float = None) -> float:
        """
        Determine bet size based on current state and RL policy.

        Args:
            count: Current true count
            bankroll: Current bankroll amount
            min_bet: Minimum bet size
            prev_bet: Previous bet size (optional)

        Returns:
            Bet size as a float
        """
        # Discretize state
        state = self.discretize_state(count, bankroll, min_bet, prev_bet)
        state_key = self.get_state_key(state)

        # Initialize Q-table entry if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.action_space}

        # Epsilon-greedy policy
        if random.random() < self.exploration_rate:
            # Exploration: random action
            action = random.choice(self.action_space)
        else:
            # Exploitation: best known action
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)

        # Convert action (multiple of min bet) to actual bet size
        bet_size = min_bet * action

        # Ensure bet size doesn't exceed bankroll or max bet
        game_config = config.GAME_CONFIG
        max_possible_bet = min(game_config["max_bet"], bankroll)
        bet_size = min(bet_size, max_possible_bet)

        return max(min_bet, bet_size)  # Ensure at least minimum bet

    def update_q_table(self, state: Tuple[int, int, int], action: int, reward: float, new_state: Tuple[int, int, int]) -> None:
        """
        Update Q-table based on observed reward and new state.

        Args:
            state: Original state before taking action
            action: Action taken (bet size multiple)
            reward: Reward received
            new_state: New state after taking action
        """
        state_key = self.get_state_key(state)
        new_state_key = self.get_state_key(new_state)

        # Initialize Q-table entries if they don't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.action_space}
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = {a: 0.0 for a in self.action_space}

        # Get current and max future Q-values
        current_q = self.q_table[state_key][action]
        max_future_q = max(self.q_table[new_state_key].values()) if self.q_table[new_state_key] else 0.0

        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state_key][action] = new_q

    def decay_exploration(self) -> None:
        """Decay the exploration rate according to the decay factor."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

    def calculate_reward(self, bankroll_change: float, count: float) -> float:
        """
        Calculate reward based on bankroll change and current count.

        Args:
            bankroll_change: Change in bankroll from the hand
            count: Current true count

        Returns:
            Calculated reward value
        """
        # Base reward is scaled bankroll change
        reward = bankroll_change * self.reward_scaling

        # Add bonus for winning with high count
        if bankroll_change > 0 and count > 2:
            reward *= 1.2

        # Penalty for losing with high count
        if bankroll_change < 0 and count > 2:
            reward *= 0.8

        # Small penalty for inaction when count is favorable
        if abs(bankroll_change) < 0.01 and count > 2:
            reward -= 0.1

        return reward

    def save_model(self, filepath: str) -> None:
        """
        Save the agent's Q-table and parameters to a file using pickle.

        Args:
            filepath: Path to save the model
        """
        model_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'min_exploration_rate': self.min_exploration_rate,
            'exploration_decay': self.exploration_decay,
            'state_discretization': self.state_discretization,
            'reward_scaling': self.reward_scaling,
            'action_space': self.action_space,
            'count_bins': self.count_bins,
            'bankroll_bins': self.bankroll_bins,
            'bet_size_bins': self.bet_size_bins
        }

        # Create directories if needed
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                raise IOError(f"Failed to create directory {directory}: {e}")

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Failed to save model to {filepath}: {e}")

    def load_model(self, filepath: str) -> None:
        """
        Load the agent's Q-table and parameters from a file.

        Args:
            filepath: Path to load the model from
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load model from {filepath}: {e}")

        # Restore model parameters
        self.q_table = model_data.get('q_table', {})
        self.learning_rate = model_data.get('learning_rate', config.RL_CONFIG["learning_rate"])
        self.discount_factor = model_data.get('discount_factor', config.RL_CONFIG["discount_factor"])
        self.exploration_rate = model_data.get('exploration_rate', config.RL_CONFIG["initial_exploration_rate"])
        self.min_exploration_rate = model_data.get('min_exploration_rate', config.RL_CONFIG["min_exploration_rate"])
        self.exploration_decay = model_data.get('exploration_decay', config.RL_CONFIG["exploration_decay"])
        self.state_discretization = model_data.get('state_discretization', config.RL_CONFIG["state_discretization"])
        self.reward_scaling = model_data.get('reward_scaling', config.RL_CONFIG["reward_scaling"])
        self.action_space = model_data.get('action_space', list(range(1, 13)))

        # Restore discretization bins
        self.count_bins = model_data.get('count_bins', self.count_bins)
        self.bankroll_bins = model_data.get('bankroll_bins', self.bankroll_bins)
        self.bet_size_bins = model_data.get('bet_size_bins', self.bet_size_bins)

        # Re-initialize bins if not loaded properly (fallback)
        if not hasattr(self, 'count_bins') or len(self.count_bins) == 0:
            self._initialize_discretization_bins()

    def get_q_value(self, state: Tuple[int, int, int], action: int) -> float:
        """
        Get the Q-value for a specific state-action pair.

        Args:
            state: State tuple
            action: Action (bet size multiple)

        Returns:
            Q-value for the state-action pair
        """
        state_key = self.get_state_key(state)
        return self.q_table.get(state_key, {}).get(action, 0.0)

    def get_best_action(self, state: Tuple[int, int, int]) -> int:
        """
        Get the best action for a given state according to current Q-values.

        Args:
            state: State tuple

        Returns:
            Best action (bet size multiple)
        """
        state_key = self.get_state_key(state)
        actions = self.q_table.get(state_key, {})
        if not actions:
            return self.action_space[len(self.action_space) // 2]  # Default to middle action
        return max(actions.items(), key=lambda x: x[1])[0]

    def reset_exploration(self, initial_rate: float = None) -> None:
        """
        Reset the exploration rate to its initial value.

        Args:
            initial_rate: Optional new initial exploration rate
        """
        if initial_rate is None:
            initial_rate = config.RL_CONFIG["initial_exploration_rate"]
        self.exploration_rate = initial_rate

    def get_exploration_rate(self) -> float:
        """
        Get the current exploration rate.

        Returns:
            Current exploration rate
        """
        return self.exploration_rate

    def get_state_statistics(self) -> Dict:
        """
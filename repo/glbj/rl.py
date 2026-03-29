"""
Q-Learning agent module for BlackjackBetOptimizer.
Implements tabular Q-learning for optimal bet sizing using card counting signals.
State: (bankroll_fraction_bin, true_count_bin, recent_bet_avg_bin)
Actions: discrete bet multipliers.
Supports epsilon-greedy exploration, parameter decay, save/load.
"""

import pickle
import random
from typing import Dict, Tuple, List, Any

class QLearner:
    """
    QLearner class implementing off-policy Q-learning for bet multiplier optimization.

    State space (discretized):
    - bankroll_fraction: current / initial (0.0-2.0), 41 bins (int(frac * 20), clamped 0-40)
    - true_count_bin: floor(tc) (-10 to +15), 26 bins (int(tc + 10), clamped 0-25)
    - recent_bet_avg_units: avg last 10 bets / unit (0-20), 21 bins (int(avg), clamped 0-20)

    Action space: list of 10 bet multipliers e.g. [1.0, 1.5, ..., 12.0]

    Q-table: dict[tuple[int,int,int]: dict[float, float]]

    Learning: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    Exploration: epsilon-greedy (decays per call to decay_params)
    Post-training: greedy via training=False in get_action

    Attributes:
        q_table: Learned Q-values {state: {action: q_value}}.
        state_bins: (n_bankroll, n_true_count, n_recent_bet) = (41,26,21)
        actions: List of possible bet multipliers.
        alpha: Learning rate (decays).
        gamma: Discount factor.
        epsilon: Exploration rate (decays to epsilon_min).
        epsilon_min: Minimum epsilon.
        epsilon_decay: Multiplier for epsilon decay.
        alpha_decay: Multiplier for alpha decay (to min 0.01).
    """

    def __init__(
        self,
        state_bins: Tuple[int, int, int] = (41, 26, 21),
        actions: List[float] = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        alpha_decay: float = 0.999,
    ):
        """
        Initialize QLearner with hyperparameters.
        Q-table starts empty, initialized lazily.
        """
        self.q_table: Dict[Tuple[int, int, int], Dict[float, float]] = {}
        self.state_bins = state_bins
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay

    def get_state(
        self, bankroll_frac: float, true_count: float, recent_bet_avg_units: float
    ) -> Tuple[int, int, int]:
        """
        Discretize continuous state variables into bin indices.

        Args:
            bankroll_frac: current_bankroll / initial_bankroll [0.0, 2.0]
            true_count: Hi-Lo true count (float, typically -10 to +15)
            recent_bet_avg_units: mean(last 10 bets) / unit_size [0.0, 20.0+]

        Returns:
            Tuple (b_bin, tc_bin, rb_bin) clamped to state_bins-1.
        """
        b_bin = min(
            self.state_bins[0] - 1, max(0, int(bankroll_frac * 20))
        )  # 0.05 steps to 2.0
        tc_bin = min(
            self.state_bins[1] - 1, max(0, int(true_count + 10))
        )  # floor equiv via int(tc+10)
        rb_bin = min(
            self.state_bins[2] - 1, max(0, int(recent_bet_avg_units))
        )  # 0-20 units
        return (b_bin, tc_bin, rb_bin)

    def get_action(self, state: Tuple[int, int, int], training: bool = True) -> float:
        """
        Select action via epsilon-greedy.

        Args:
            state: Discretized state tuple.
            training: If True, use epsilon-greedy; else pure greedy.

        Returns:
            Selected bet multiplier (float from self.actions).
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        if training and random.random() < self.epsilon:
            return random.choice(self.actions)

        q_vals = self.q_table[state]
        return max(q_vals, key=q_vals.get)

    def update(
        self, state: Tuple[int, int, int], action: float, reward: float, next_state: Tuple[int, int, int]
    ) -> None:
        """
        Perform Q-learning update (TD(0)).

        Args:
            state: Current state.
            action: Taken action (multiplier).
            reward: Observed reward (delta bankroll).
            next_state: Resulting state.
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        current_q = self.q_table[state][action]
        next_q_dict = self.q_table.get(
            next_state, {a: 0.0 for a in self.actions}
        )
        max_next_q = max(next_q_dict.values())
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[state][action] += self.alpha * td_error

    def decay_params(self) -> None:
        """
        Decay epsilon and alpha (call every N hands, e.g., 1000).
        Epsilon -> max(epsilon_min, epsilon * epsilon_decay)
        Alpha -> max(0.01, alpha * alpha_decay)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha = max(0.01, self.alpha * self.alpha_decay)

    def save(self, filename: str = "q_table.pkl") -> None:
        """
        Save Q-table to pickle file.

        Args:
            filename: Path to save (default 'q_table.pkl').
        """
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename: str = "q_table.pkl") -> None:
        """
        Load Q-table from pickle file (ignores if not found).

        Args:
            filename: Path to load (default 'q_table.pkl').
        """
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}
            # Silently continue with empty table
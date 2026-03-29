"""
BettingAgent module for BlackjackBetOptimizer.
Implements tabular Q-learning for optimal bet sizing based on true count and bankroll fraction.
"""

import random
from typing import Tuple, Dict, Optional

class BettingAgent:
    """
    RL Q-learning agent for selecting optimal bet multiplier based on count and bankroll.
    
    State: Tuple of (true_count_bin: int from -10 to +10, bankroll_fraction_bin: int from 0 to 20)
    Actions: Discrete multipliers [1x, 2x, 4x, 6x, 8x, 10x, 12x] filtered to <= max_spread (indices 0 to num_actions-1)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the betting agent with configuration parameters.
        
        Args:
            config: Configuration dictionary with keys like 'min_bet', 'max_spread', etc.
        """
        self.q_table: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.min_bet = config['min_bet']
        self.spread = config['max_spread']
        self.initial_epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon = self.initial_epsilon
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.total_hands = 0
        self.num_training_hands = config['num_training_hands']
        
        base_multipliers = [1, 2, 4, 6, 8, 10, 12]
        self.action_multipliers = [m for m in base_multipliers if m <= self.spread]
        self.num_actions = len(self.action_multipliers)
    
    def get_state(self, true_count: float, bankroll_frac: float) -> Tuple[int, int]:
        """
        Bin true count and bankroll fraction into discrete state space.
        
        Args:
            true_count: Current true count (float)
            bankroll_frac: Bankroll fraction (0.0 to 1.0)
        
        Returns:
            Tuple of binned true_count (-10 to 10) and bankroll_frac (0 to 20)
        """
        tc_bin = max(-10, min(10, round(true_count)))
        frac_bin = max(0, min(20, int(bankroll_frac * 20)))
        return (tc_bin, frac_bin)
    
    def _ensure_state_init(self, state: Tuple[int, int]) -> None:
        """Ensure state exists in Q-table with initialized action values to 0."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.num_actions)}
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current binned state tuple
            
        Returns:
            Selected action index (0 to num_actions-1)
        """
        self._ensure_state_init(state)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)
    
    def _compute_bet(self, action: int, current_bankroll: float) -> float:
        """
        Compute bet size for a given action index, applying clamps.
        
        Args:
            action: Selected action index
            current_bankroll: Current bankroll
            
        Returns:
            Clamped bet size (>= min_bet or 0.0)
        """
        multi = self.action_multipliers[action]
        proposed = multi * self.min_bet
        max_bet_spread = self.min_bet * self.spread
        max_bet_kelly = 0.2 * current_bankroll
        bet = min(proposed, max_bet_spread, max_bet_kelly)
        return bet if bet >= self.min_bet else 0.0
    
    def get_bet_and_action(self, initial_bankroll: float, current_bankroll: float, true_count: float) -> Tuple[float, Optional[int]]:
        """
        Get recommended bet and the action index chosen (for training/updates).
        
        Args:
            initial_bankroll: Starting bankroll for fraction calculation
            current_bankroll: Current bankroll
            true_count: Current true count
            
        Returns:
            Tuple of (bet_size: float, action: int or None if no bet)
        """
        if current_bankroll < self.min_bet:
            return 0.0, None
        
        bankroll_frac = min(1.0, current_bankroll / initial_bankroll)
        state = self.get_state(true_count, bankroll_frac)
        action = self.choose_action(state)
        bet = self._compute_bet(action, current_bankroll)
        return bet, action
    
    def get_bet(self, initial_bankroll: float, current_bankroll: float, true_count: float) -> float:
        """
        Get recommended bet size (for visualization/inference).
        
        Args:
            initial_bankroll: Starting bankroll for fraction calculation
            current_bankroll: Current bankroll
            true_count: Current true count
            
        Returns:
            Recommended bet size (0.0 if ruined or clamped too low)
        """
        return self.get_bet_and_action(initial_bankroll, current_bankroll, true_count)[0]
    
    def update(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]) -> None:
        """
        Update Q-table using Q-learning update rule.
        
        Q(s,a) <- Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Previous state tuple
            action: Action taken (index)
            reward: Reward received (e.g., payout)
            next_state: Next state tuple
        """
        self._ensure_state_init(state)
        self._ensure_state_init(next_state)
        
        current_q = self.q_table[state][action]
        next_q_max = max(self.q_table[next_state].values())
        target = reward + self.gamma * next_q_max
        self.q_table[state][action] += self.alpha * (target - current_q)
        
        # Decay epsilon linearly over training hands
        self.total_hands += 1
        if self.num_training_hands > 0:
            progress = min(1.0, self.total_hands / self.num_training_hands)
            self.epsilon = self.epsilon_end + (self.initial_epsilon - self.epsilon_end) * (1 - progress)
    
    def reset_epsilon(self) -> None:
        """Reset epsilon to initial value and hands counter (for new training)."""
        self.epsilon = self.initial_epsilon
        self.total_hands = 0
    
    def get_q_table_size(self) -> int:
        """Get total number of states in Q-table."""
        return len(self.q_table)
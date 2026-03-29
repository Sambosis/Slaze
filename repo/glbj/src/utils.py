import collections
import numpy as np
from typing import List, Dict, Union, Optional

class StatsManager:
    """
    Tracks game performance metrics including wins, losses, pushes, bankroll history,
    and derived statistics like Win/Loss ratio and Drawdown.
    Designed for efficient updates during fast training loops.
    """

    def __init__(self, initial_bankroll: float = 0.0):
        """
        Initialize the statistics manager.

        Args:
            initial_bankroll (float): The starting bankroll amount.
        """
        self.initial_bankroll = initial_bankroll
        self.reset()

    def reset(self) -> None:
        """Reset all statistics for a new session."""
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.hands_played = 0
        
        # History tracking
        self.bankroll_history: List[float] = [self.initial_bankroll]
        
        # Performance metrics
        self.peak_bankroll = self.initial_bankroll
        self.max_drawdown_pct = 0.0
        
        # Rolling windows for trend analysis (last 1000 hands)
        self.recent_results = collections.deque(maxlen=1000)  # 1 for win, 0 for loss
        self.recent_returns = collections.deque(maxlen=1000)

    def update(self, result: str, current_bankroll: float) -> None:
        """
        Update statistics after a hand completes.

        Args:
            result (str): Outcome of the hand ('win', 'loss', 'push', or 'blackjack').
            current_bankroll (float): The player's bankroll after the hand.
        """
        self.hands_played += 1
        
        # Update Counters and Recent History
        if result in ('win', 'blackjack'):
            self.wins += 1
            self.recent_results.append(1)
        elif result == 'loss':
            self.losses += 1
            self.recent_results.append(0)
        elif result == 'push':
            self.pushes += 1
            # Pushes are not added to recent_results (win/loss ratio)
        
        # Calculate return for this specific hand
        prev_bankroll = self.bankroll_history[-1] if self.bankroll_history else self.initial_bankroll
        hand_return = current_bankroll - prev_bankroll
        self.recent_returns.append(hand_return)

        # Update Bankroll History
        self.bankroll_history.append(current_bankroll)

        # Incremental Max Drawdown Calculation
        if current_bankroll > self.peak_bankroll:
            self.peak_bankroll = current_bankroll
        else:
            if self.peak_bankroll > 0:
                current_dd = (self.peak_bankroll - current_bankroll) / self.peak_bankroll
                if current_dd > self.max_drawdown_pct:
                    self.max_drawdown_pct = current_dd

    @property
    def current_bankroll(self) -> float:
        """Get the most recent bankroll value."""
        return self.bankroll_history[-1] if self.bankroll_history else self.initial_bankroll

    @property
    def win_rate(self) -> float:
        """
        Calculate global win rate (Wins / (Wins + Losses)).
        Excludes pushes to reflect decisive game performance.
        """
        decisive_hands = self.wins + self.losses
        return self.wins / decisive_hands if decisive_hands > 0 else 0.0

    @property
    def recent_win_rate(self) -> float:
        """Calculate win rate over the recent history window (e.g., last 1000 hands)."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def avg_return(self) -> float:
        """Calculate average profit/loss per hand over the recent history window."""
        if not self.recent_returns:
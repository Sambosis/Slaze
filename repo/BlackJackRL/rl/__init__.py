"""
Reinforcement Learning Package for Blackjack Bet Sizing Agent.

This package provides the RLAgent class and related functionality for implementing
reinforcement learning in blackjack bet sizing optimization. The agent uses Q-learning
to determine optimal bet sizes based on game state including running count and bankroll.

Key Features:
- Q-learning implementation for bet sizing
- State representation including count, bankroll, and bet history
- Model persistence support
- Integration with blackjack game engine

Example:
    >>> from rl import RLAgent
    >>> agent = RLAgent(learning_rate=0.1, discount_factor=0.9)
    >>> bet_size = agent.get_bet_size(count=2.5, bankroll=1000)
"""

from .agent import RLAgent, get_bet_size, update_q_table
from .persistence import save_model, load_model

__all__ = [
    'RLAgent',
    'get_bet_size',
    'update_q_table',
    'save_model',
    'load_model'
]

__version__ = '1.0.0'
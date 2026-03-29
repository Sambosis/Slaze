"""
Blackjack Reinforcement Learning (RL) Simulation Package.

This package implements a sophisticated Blackjack simulation designed to train
a Q-Learning agent to determine optimal bet sizes based on card counting
(True Count), bankroll management, and match history.

The application comprises three distinct layers:
1.  **Game Engine**: Handles deck management, rules, and Basic Strategy enforcement.
2.  **RL Agent**: Implements Q-Learning with state discretization.
3.  **Visualization**: Provides a Pygame-based GUI for live feedback and plotting.
"""

__version__ = "1.0.0"
__title__ = "Blackjack RL Agent"
__author__ = "AI Developer"

from .config import Config
from .game_logic import BlackjackGame, Card, Deck
from .agent import QLearningAgent
from .visualization import GuiManager

__all__ = [
    "Config",
    "BlackjackGame",
    "Card",
    "Deck",
    "QLearningAgent",
    "GuiManager",
]
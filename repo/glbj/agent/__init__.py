"""
Blackjack Reinforcement Learning (RL) Bet Sizer.

This package initializes the Blackjack RL environment, exposing core components
for the Game Engine, Q-Learning Agent, and Visualization layers. It supports
training an agent to optimize bet sizing based on card counting and bankroll
management strategies.
"""

__version__ = "1.0.0"
__title__ = "Blackjack RL Bet Sizer"
__author__ = "Trailblazer Labs"

from .config import Config
from .game_logic import (
    Card,
    Deck,
    BlackjackGame,
    get_basic_strategy_move
)
from .agent import QLearningAgent
from .visualization import GuiManager
from .utils import calculate_drawdown, GameStats

__all__ = [
    "Config",
    "Card",
    "Deck",
    "BlackjackGame",
    "get_basic_strategy_move",
    "QLearningAgent",
    "GuiManager",
    "calculate_drawdown",
    "GameStats",
]
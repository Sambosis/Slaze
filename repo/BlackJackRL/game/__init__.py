"""
Package initialization for the game module.

This module contains the core game logic for the Blackjack Reinforcement Learning Agent,
including the BlackjackGame, Card, Deck, and BasicStrategy classes.
"""

from .blackjack import BlackjackGame, Card, Deck
from .strategy import BasicStrategy

__version__ = "1.0.0"
__description__ = "Blackjack game engine with reinforcement learning capabilities"

__all__ = [
    "BlackjackGame",
    "Card",
    "Deck",
    "BasicStrategy",
    "__version__",
    "__description__"
]
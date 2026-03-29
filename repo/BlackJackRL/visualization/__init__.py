"""
Visualization package initialization for the Blackjack Reinforcement Learning Agent.

This package provides Pygame-based rendering functionality for the blackjack game,
including game state visualization, statistics display, and training progress tracking.
The primary interface is the render_game function which handles all rendering operations.
"""

from .renderer import render_game

__all__ = ['render_game']
__version__ = "1.0.0"
__author__ = "Blackjack RL Project Team"
__description__ = "Visualization components for the Blackjack Reinforcement Learning Agent"
"""
Package initialization for the utils module.

This module provides utility functions for the Blackjack Reinforcement Learning project,
including model persistence functionality for saving and loading RL agent states.
"""

from .persistence import save_model, load_model

__all__ = ['save_model', 'load_model']
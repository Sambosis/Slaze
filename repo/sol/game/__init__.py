"""
Game module for the PyGame Solitaire project.

This module contains the core game logic and rendering components for the
solitaire card game, including the main SolitaireGame class that manages
the game state and rules.
"""

from .solitaire import SolitaireGame

__all__ = ['SolitaireGame']

# Allow direct import from the game package
# Example: from game import SolitaireGame
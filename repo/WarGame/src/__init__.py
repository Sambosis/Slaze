"""
Operation Neural Storm - Source Package
=======================================

This package contains the core logic for the competitive Reinforcement Learning project,
implementing a history-based competitive self-play environment with auto-curriculum.

Modules:
    - environment: Game logic, physics, and state transitions (WarGameEnv).
    - model: Neural network architecture (Agent).
    - trainer: Training loop, PPO implementation, and memory management.
    - visualizer: Pygame-based rendering engine.
    - utils: Utilities for model persistence and Hall of Fame management.
"""

from .environment import WarGameEnv, Unit
from .model import Agent
from .trainer import train_loop, ppo_update, Memory
from .visualizer import render_game
from .utils import save_model, load_model, HallOfFame

__all__ = [
    "WarGameEnv",
    "Unit",
    "Agent",
    "train_loop",
    "ppo_update",
    "Memory",
    "render_game",
    "save_model",
    "load_model",
    "HallOfFame",
]

__version__ = "0.1.0"
__author__ = "Operation Neural Storm Team"
"""
Configuration module for BlackjackBetOptimizer.

Defines the global CONFIG dictionary containing all tunable parameters for the application,
including game settings, table limits, training hyperparameters, and RL agent parameters.

Import as: from config import CONFIG
"""

from typing import Any

CONFIG: dict[str, Any] = {
    "initial_bankroll": 10000,
    "table_min_bet": 10,
    "table_max_bet": 1000,
    "num_decks": 6,
    "training_hands": 1000000,
    "rl_alpha": 0.1,
    "rl_gamma": 0.95,
    "rl_epsilon_start": 1.0,
    "rl_epsilon_min": 0.01,
    "rl_epsilon_decay": 0.999,
    "unit_size": 10,
    "penetration": 0.75,
}
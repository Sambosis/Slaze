"""
Configuration parameters for the Blackjack Reinforcement Learning Agent.

This module contains default settings for the game, including:
- Game parameters (number of decks, initial bankroll, bet sizes)
- Reinforcement learning hyperparameters
- Visualization settings
- Basic strategy configuration
- Counting system configuration
"""

# Game Configuration
GAME_CONFIG = {
    "num_decks": 6,
    "initial_bankroll": 1000.0,
    "min_bet": 10.0,
    "max_bet": 120.0,
    "penetration": 0.75,
    "blackjack_payout": 1.5,
    "insurance_payout": 2.0,
    "dealer_stands_on_soft_17": True,
    "double_after_split": True,
    "resplit_aces": False,
    "late_surrender": False,
}

# Reinforcement Learning Hyperparameters
RL_CONFIG = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "initial_exploration_rate": 0.5,
    "exploration_decay": 0.9995,
    "min_exploration_rate": 0.01,
    "state_discretization": {
        "count_bins": 21,
        "bankroll_bins": 10,
        "bet_size_bins": 12,
    },
    "reward_scaling": 0.1,
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "screen_width": 1200,
    "screen_height": 800,
    "card_width": 160,
    "card_height": 232,
    "card_spacing": 30,
    "fps": 60,
    "stats_update_interval": 10,
    "full_render_interval": 50,
    "font_sizes": {
        "title": 36,
        "stats": 18,
        "ui": 24,
    },
    "colors": {
        "background": (34, 139, 34),
        "text": (255, 255, 255),
        "button": (0, 0, 255),
        "highlight": (255, 255, 0),
        "info_box": (0, 0, 0, 200),
    },
}

# Counting System Configuration (Hi-Lo)
COUNTING_CONFIG = {
    "card_values": {
        "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
        "7": 0, "8": 0, "9": 0,
        "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
    },
    "betting_spread": {
        "min_bet": 1,
        "max_bet": 12,
        "spread_thresholds": {
            1: 1,
            2: 2,
            3: 4,
            4: 6,
            5: 8,
            6: 10,
            7: 12,
        },
    },
}

# Training Configuration
TRAINING_CONFIG = {
    "num_episodes": 10000,
    "episode_length": 100,
    "save_interval": 100,
    "model_save_path": "models/blackjack_agent.pkl",
    "log_file": "training_log.csv",
    "stats_update_frequency": 10,
    "visualization_frequency": 50,
}

# File Paths
PATHS = {
    "assets": "assets/",
    "card_images": "assets/cards/",
    "models": "models/",
    "logs": "logs/",
    "stats": "stats/",
}

# Debug Configuration
DEBUG_CONFIG = {
    "log_level": "INFO",
    "console_log": True,
    "file_log": True,
    "log_file": "logs/debug.log",
    "show_counts": True,
    "show_q_values": False,
}

def validate_config():
    """Validates configuration parameters for logical consistency."""
    validation_errors = []

    # Validate bet sizing
    if GAME_CONFIG["max_bet"] < GAME_CONFIG["min_bet"]:
        validation_errors.append("MAX_BET must be >= MIN_BET")

    if GAME_CONFIG["min_bet"] <= 0:
        validation_errors.append("MIN_BET must be positive")

    # Validate RL parameters
    if not 0 <= RL_CONFIG["learning_rate"] <= 1:
        validation_errors.append("LEARNING_RATE must be in [0, 1]")

    if not 0 <= RL_CONFIG["discount_factor"] <= 1:
        validation_errors.append("DISCOUNT_FACTOR must be in [0, 1]")

    if not 0 <= RL_CONFIG["initial_exploration_rate"] <= 1:
        validation_errors.append("INITIAL_EXPLORATION_RATE must be in [0, 1]")

    # Validate game parameters
    if GAME_CONFIG["initial_bankroll"] < GAME_CONFIG["min_bet"]:
        validation_errors.append("INITIAL_BANKROLL must be >= MIN_BET")

    if not 0 < GAME_CONFIG["penetration"] <= 1:
        validation_errors.append("PENETRATION must be in (0, 1]")

    if GAME_CONFIG["num_decks"] < 1:
        validation_errors.append("NUM_DECKS must be at least 1")

    if validation_errors:
        raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")

    return True

# Auto-validate configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"Warning: Configuration validation failed: {e}")
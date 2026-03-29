"""
Configuration file for the Solitaire RL project.
Stores game settings, rendering options, RL hyperparameters, and file paths.
"""

import os
from pathlib import Path


# Project Base Directory
# Uses the directory where this config file is located as the base path.
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# 1. Game Configuration
# ==============================================================================

# Screen dimensions for the PyGame window
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# Card dimensions
CARD_WIDTH = 80
CARD_HEIGHT = 120

# Spacing between card piles
PILE_SPACING = 10

# Vertical offset for cards stacked in a tableau pile
TABLEAU_OFFSET = 25


# ==============================================================================
# 2. Rendering Configuration (Colors)
# ==============================================================================

# Color palette for the game
GAME_COLORS = {
    'TABLE': (34, 139, 34),     # Forest Green for the background "felt"
    'CARD_FACE': (255, 255, 255), # White for card faces
    'CARD_BACK': (0, 100, 0),     # Dark Green for card backs
    'TEXT': (255, 255, 255),      # White for text
    'HIGHLIGHT': (255, 215, 0),   # Gold for highlighting valid moves
}

# Text/Font settings for UI elements
FONT_NAME = 'Arial'
FONT_SIZE = 18
FONT_SIZE_LARGE = 24


# ==============================================================================
# 3. Positioning Configuration
# ==============================================================================

# Positions are defined as (x, y) coordinates.
# The top-left corner of the screen is (0, 0).

# Top area: Foundations and Stock/Waste
TOP_AREA_Y = 50
FOUNDATIONS_START_X = 400
STOCK_POS = (50, TOP_AREA_Y)
WASTE_POS = (170, TOP_AREA_Y)

# Tableau area: where the main 7 piles are located
TABLEAU_START_X = 50
TABLEAU_START_Y = 250
TABLEAU_SPACING = CARD_WIDTH + PILE_SPACING * 2


# ==============================================================================
# 4. Reinforcement Learning (RL) Hyperparameters
# ==============================================================================

# DQN Model Architecture
# State representation dimensions: (num_tableau_piles * max_cards_per_pile * card_features)
# A typical flattened state vector might be large.
# For simplicity, we define a hidden layer size.
HIDDEN_LAYER_1_SIZE = 512
HIDDEN_LAYER_2_SIZE = 256

# Training Hyperparameters
GAMMA = 0.99          # Discount factor for future rewards
LR = 0.0001            # Learning rate for the optimizer
BATCH_SIZE = 64        # Number of experiences sampled from replay memory for training
TARGET_NETWORK_UPDATE_FREQ = 1000  # Steps after which the target network is synced with the policy network

# Epsilon-greedy exploration
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Decay rate per episode

# Replay Memory
MEMORY_SIZE = 20000    # Maximum number of experiences to store
MEMORY_WARMUP_SIZE = 1000 # Minimum experiences in memory before training starts

# Rewards
REWARD_WIN = 100
REWARD_FOUNDATION_MOVE = 10
REWARD_TABLEAU_MOVE = 5
REWARD_INVALID_MOVE = -1

# Training Loop
RENDER_EVERY_N_EPISODES = 10  # Render the game visually every Nth training episode
MAX_STEPS_PER_EPISODE = 200   # Maximum number of moves the agent can make in a single game


# ==============================================================================
# 5. Paths and Assets
# ==============================================================================

# Directory for card images
ASSETS_DIR = BASE_DIR / 'assets'
CARDS_IMAGE_DIR = ASSETS_DIR / 'cards'
# Fallback: If specific card images aren't found, the renderer will use programmatic shapes.
BACKGROUND_IMAGE_PATH = ASSETS_DIR / 'background.png'

# Directory for saving training artifacts
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the directory exists

# Paths for saving models and logs
MODEL_SAVE_PATH = OUTPUT_DIR / 'dqn_model.pth'
TARGET_MODEL_SAVE_PATH = OUTPUT_DIR / 'dqn_target_model.pth'
TRAINING_LOG_PATH = OUTPUT_DIR / 'training_log.csv'


# ==============================================================================
# 6. Miscellaneous Settings
# ==============================================================================

# Game play settings
AUTOCOMPLETE = False # If True, automatically move cards to foundations when possible (standard rule)

# Controls whether the main loop should keep the PyGame window open for debugging
# during a headless training run. For full training, this should be False.
DEBUG_RENDERING = False
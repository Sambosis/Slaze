"""
Main entry point for the Blackjack Reinforcement Learning Agent.

This module implements the main game loop, connects all components (game engine,
RL agent, visualization, and persistence), and manages the training process.
It handles initialization, training loop execution, visualization intervals,
and model persistence.
"""

import pygame
import sys
import os
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Any

import config
from game.blackjack import BlackjackGame, Card, Deck
from game.strategy import BasicStrategy
from rl.agent import RLAgent
from visualization.renderer import render_game
from utils.persistence import save_model, load_model, log_training_stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_environment() -> bool:
    """
    Validate that the required directories and files exist.

    Returns:
        bool: True if environment is valid, False otherwise
    """
    try:
        # Check if directories exist, create if necessary
        directories = [
            'logs',
            'models',
            'stats'
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

        return True
    except Exception as e:
        logger.error(f"Failed to validate environment: {e}")
        return False

class BlackjackRLApp:
    """
    Main application class for the Blackjack Reinforcement Learning Agent.

    This class manages the complete application lifecycle, including:
    - Game initialization and configuration
    - Pygame setup and main event loop
    - Integration between game engine, RL agent, and visualization
    - Training loop and model persistence
    - Statistics tracking and display
    """

    def __init__(self):
        """Initialize the application with configuration and components."""
        logger.info("Initializing Blackjack RL Agent Application...")

        # Validate environment first
        if not validate_environment():
            raise RuntimeError("Environment validation failed. Application cannot start.")

        # Initialize Pygame
        try:
            pygame.init()
            pygame.mixer.quit()  # Disable audio to avoid delays
            logger.info("Pygame initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pygame: {e}")
            raise

        # Set up display
        try:
            screen_width = config.VISUALIZATION_CONFIG["screen_width"]
            screen_height = config.VISUALIZATION_CONFIG["screen_height"]
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Blackjack RL Agent - Training")
            logger.info(f"Display initialized ({screen_width}x{screen_height})")
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            raise

        # Set up clock
        self.clock = pygame.time.Clock()

        # Validate configuration
        try:
            config.validate_config()
            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            pygame.quit()
            raise

        # Initialize game components
        self.game = self._initialize_game()
        self.strategy = BasicStrategy()
        self.agent = self._initialize_rl_agent()

        # Application state
        self.running = True
        self.show_full_game = False
        self.current_episode = 0
        self.current_hand = 0
        self.prev_bet = None  # Track previous bet for RL state
        self.paused = False

        # Statistics tracking
        self.game_stats = {
            "total_profit": 0.0,
            "total_hands": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "blackjacks": 0,
            "last_update_time": time.time(),
            "episode_rewards": [],
            "bankroll_history": [],
            "win_rate_history": []
        }

        # Load existing model if available
        self._load_existing_model()

        # Set random seeds for reproducibility
        random.seed(42)
        logger.info("Application initialized successfully")

    def _initialize_game(self) -> BlackjackGame:
        """Initialize the blackjack game with configuration parameters."""
        game_config = config.GAME_CONFIG
        logger.debug(f"Initializing game: {game_config['num_decks']} decks, ${game_config['initial_bankroll']} bankroll")
        return BlackjackGame(
            num_decks=game_config["num_decks"],
            initial_bankroll=game_config["initial_bankroll"],
            min_bet=game_config["min_bet"],
            max_bet=game_config["max_bet"],
            penetration=game_config["penetration"]
        )

    def _initialize_rl_agent(self) -> RLAgent:
        """Initialize the reinforcement learning agent."""
        rl_config = config.RL_CONFIG
        logger.debug(f"Initializing RL agent with learning_rate={rl_config['learning_rate']}, "
                    f"discount_factor={rl_config['discount_factor']}, "
                    f"exploration_rate={rl_config['initial_exploration_rate']}")
        return RLAgent(
            learning_rate=rl_config["learning_rate"],
            discount_factor=rl_config["discount_factor"],
            initial_exploration_rate=rl_config["initial_exploration_rate"],
            min_exploration_rate=rl_config["min_exploration_rate"],
            exploration_decay=rl_config["exploration_decay"],
            state_discretization=rl_config["state_discretization"],
            reward_scaling=rl_config["reward_scaling"]
        )

    def _load_existing_model(self) -> None:
        """Load existing RL agent model if it exists."""
        model_path = config.TRAINING_CONFIG["model_save_path"]
        if os.path.exists(model_path):
            try:
                self.agent.load_model(model_path)
                logger.info(f"Loaded existing model from {model_path}")
                # If model loaded, might want to reset exploration rate for continued learning
                self.agent.exploration_rate = max(
                    self.agent.min_exploration_rate,
                    self.agent.exploration_rate * 0.9  # Slightly reduce exploration
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Starting with a new agent")
        else:
            logger.info(f"No existing model found at {model_path}, starting fresh")

    def run(self) -> None:
        """Run the main application loop."""
        logger.info("Starting Blackjack RL Agent Training...")
        logger.info(f"Configuration: {config.GAME_CONFIG['num_decks']} decks, "
                   f"${config.GAME_CONFIG['initial_bankroll']} bankroll, "
                   f"${config.GAME_CONFIG['min_bet']}-${config.GAME_CONFIG['max_bet']} bet range")

        # Main training loop
        num_episodes = config.TRAINING_CONFIG["num_episodes"]
        episode_length = config.TRAINING_CONFIG["episode_length"]
        stats_update_freq = config.TRAINING_CONFIG["stats_update_frequency"]
        visualization_freq = config.TRAINING_CONFIG["visualization_frequency"]
        save_interval = config.TRAINING_CONFIG["save_interval"]

        while self.running and self.current_episode < num_episodes:
            self._handle_events()

            if not self.paused:
                # Check if episode complete
                if self.current_hand > 0 and self.current_hand % episode_length == 0:
                    self._start_new_episode()

                # Run a single hand
                hand_result = self._run_hand()
                self.current_hand += 1

                # Update statistics
                self._update_statistics(hand_result)

                # Show statistics periodically
                if self.current_hand % stats_update_freq == 0:
                    self._print_statistics()

                # Log training stats periodically
                if self.current_hand % stats_update_freq == 0:
                    self._log_training_stats()

                # Save model periodically
                if self.current_hand % save_interval == 0:
                    self._save_model()

            # Update display at regular intervals
            if self.current_hand % visualization_freq == 0 or self.show_full_game:
                self._render_game_state()

            pygame.display.flip()
            self.clock.tick(config.VISUALIZATION_CONFIG["fps"])

        # Training complete - show final results
        self._show_final_results()
        pygame.quit()
        sys.exit()

    def _handle_events(self) -> None:
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                logger.info("User requested quit (window close)")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    logger.info("User requested quit (ESC key)")
                elif event.key == pygame.K_SPACE:
                    # Toggle full game visualization
                    self.show_full_game = not self.show_full_game
                    logger.info(f"Full game visualization: {'ON' if self.show_full_game else 'OFF'}")
                elif event.key == pygame.K_s:
                    # Save model
                    self._save_model()
                elif event.key == pygame.K_l:
                    # Load model
                    self._load_existing_model()
                elif event.key == pygame.K_p:
                    # Pause/unpause training
                    self.paused = not self.paused
                    logger.info(f"Training {'paused' if self.paused else 'resumed'}")

    def _start_new_episode(self) -> None:
        """Start a new training episode."""
        self.current_episode += 1

        # Calculate episode metrics
        episode_length = config.TRAINING_CONFIG["episode_length"]
        start_idx = max(0, len(self.game_stats["episode_rewards"]) - episode_length)
        recent_rewards = self.game_stats["episode_rewards"][start_idx:]

        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        total_reward = sum(recent_rewards)

        # Log episode summary
        logger.info(f"Episode {self.current_episode} complete. "
                   f"Total reward: {total_reward:.2f}, Avg reward: {avg_reward:.2f}")

        # Decay exploration for next episode
        self.agent.decay_exploration()

    def _run_hand(self) -> float:
        """
        Run a single blackjack hand with RL agent bet sizing.

        Returns:
            float: Net profit/loss from the hand
        """
        try:
            # Get current game state for RL decision
            game_state = self.game.get_game_state()

            # Agent determines bet size based on current state
            bet_size = self.agent.get_bet_size(
                count=game_state["true_count"],
                bankroll=self.game.bankroll,
                min_bet=self.game.min_bet,
                prev_bet=self.prev_bet
            )
            logger.debug(f"Agent chose bet size: {bet_size:.2f} (true count: {game_state['true_count']:.2f})")

            # Store state before taking action for RL update
            prev_state = self.agent.discretize_state(
                count=game_state["true_count"],
                bankroll=self.game.bankroll,
                min_bet=self.game.min_bet,
                prev_bet=self.prev_bet
            )

            # Place bet
            if not self.game.place_bet(bet_size):
                logger.warning(f"Failed to place bet of ${bet_size:.2f} (bankroll: ${self.game.bankroll:.2f})")
                return 0.0

            self.prev_bet = bet_size

            # Play the hand using basic strategy
            result = self.game.play_hand(self.strategy)

            # Store result for episode tracking
            self.game_stats["episode_rewards"].append(result)

            # Calculate reward for RL agent
            reward = self.agent.calculate_reward(result, game_state["true_count"])
            logger.debug(f"Hand result: {result:.2f}, RL reward: {reward:.4f}")

            # Get new state after hand resolution
            new_game_state = self.game.get_game_state()
            new_state = self.agent.discretize_state(
                count=new_game_state["true_count"],
                bankroll=self.game.bankroll,
                min_bet=self.game.min_bet,
                prev_bet=bet_size
            )

            # Update Q-table with observed state-action-reward-new_state
            action = max(1, int(bet_size / self.game.min_bet))  # Convert to bet multiple
            self.agent.update_q_table(prev_state, action, reward, new_state)

            return result

        except Exception as e:
            logger.error(f"Error running hand: {e}", exc_info=True)
            return 0.0

    def _update_statistics(self, hand_result: float) -> None:
        """
        Update game statistics based on hand result.

        Args:
            hand_result: Net profit/loss from the hand
        """
        self.game_stats["total_hands"] += 1
        self.game_stats["total_profit"] += hand_result

        # Track wins, losses, and pushes
        if hand_result > 0:
            self.game_stats["wins"] += 1
        elif hand_result < 0:
            self.game_stats["losses"] += 1
        else:
            self.game_stats["pushes"] += 1

        # Track bankroll history for visualization
        self.game_stats["bankroll_history"].append(self.game.bankroll)
        if len(self.game_stats["bankroll_history"]) > 1000:  # Keep last 1000 points
            self.game_stats["bankroll_history"] = self.game_stats["bankroll_history"][-1000:]

        # Track win rate history
        win_rate = self.game_stats["wins"] / max(1, self.game_stats["total_hands"])
        self.game_stats["win_rate_history"].append(win_rate)
        if len(self.game_stats["win_rate_history"]) > 1000:
            self.game_stats["win_rate_history"] = self.game_stats["win_rate_history"][-1000:]

    def _print_statistics(self) -> None:
        """Print current training statistics to console."""
        stats = self.game_stats
        win_rate = stats["wins"] / max(1, stats["total_hands"]) * 100
        avg_profit_per_hand = stats["total_profit"] / max(1, stats["total_hands"])

        print(f"\n--- Training Progress ---")
        print(f"Episode: {self.current_episode + 1}")
        print(f"Hand: {self.current_hand}")
        print(f"Bankroll: ${self.game.bankroll:.2f}")
        print(f"Total Profit: ${stats['total_profit']:.2f}")
        print(f"Win Rate: {win_rate:.1f}% ({stats['wins']}/{stats['total_hands']})")
        print(f"Average Profit/Hand: ${avg_profit_per_hand:.4f}")
        print(f"Exploration Rate: {self.agent.get_exploration_rate():.4f}")
        agent_stats = self.agent.get_state_statistics()
        print(f"States Visited: {agent_stats['num_states']}")
        print(f"------------------------")

    def _log_training_stats(self) -> None:
        """Log training statistics to CSV file."""
        stats = self.game_stats
        win_rate = stats["wins"] / max(1, stats["total_hands"])

        log_data = {
            "bankroll": self.game.bankroll,
            "win_rate": win_rate,
            "total_profit": stats["total_profit"],
            "avg_profit_per_hand": stats["total_profit"] / max(1, stats["total_hands"]),
            "exploration_rate": self.agent.get_exploration_rate(),
            "states_visited": self.agent.get_state_statistics()["num_states"],
            "true_count": self.game.true_count,
            "running_count": self.game.deck_count,
            "episode": self.current_episode
        }

        try:
            log_file = config.TRAINING_CONFIG["log_file"]
            log_training_stats(
                filepath=log_file,
                episode=self.current_hand,
                stats=log_data
            )
        except Exception as e:
            logger.error(f"Failed to log training stats: {e}")

    def _render_game_state(self) -> None:
        """Render the current game state."""
        # Get current game state
        game_state = self.game.get_game_state()

        try:
            # Convert Card objects to dictionaries for rendering
            player_hand = [{"rank": card.rank, "suit": card.suit, "value": card.value}
                          for card in game_state["player_hand"]]
            dealer_hand = [{"rank": card.rank, "suit": card.suit, "value": card.value}
                          for card in game_state["dealer_hand"]]

            # Prepare complete game state for visualization
            display_state = {
                "player_hand": player_hand,
                "dealer_hand": dealer_hand,
                "player_value": game_state["player_value"],
                "dealer_value": game_state["dealer_value"],
                "bankroll": self.game.bankroll,
                "current_bet": self.game.current_bet,
                "running_count": self.game.deck_count,
                "true_count": self.game.true_count,
                "decks_remaining": self.game.decks[0].decks_remaining(),
                "game_over": game_state["game_over"],
                "player_bust": game_state["player_bust"],
                "dealer_bust": game_state["dealer_bust"],
                "player_blackjack": game_state["player_blackjack"],
                "dealer_blackjack": game_state["dealer_blackjack"],
                "hands_played": self.game_stats["total_hands"],
                "win_rate": self.game_stats["wins"] / max(1, self.game_stats["total_hands"]),
                "exploration_rate": self.agent.get_expl
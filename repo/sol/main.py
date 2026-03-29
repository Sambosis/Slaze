"""
main.py - Main entry point for the PyGame Solitaire with Reinforcement Learning project.

This module initializes the game environment, sets up the RL agent, and manages the
training loop. It handles rendering, model saving, and performance logging for the
solitaire RL agent.
"""

import pygame
import torch
import numpy as np
import csv
import time
import os
import sys
import random
from typing import List, Tuple, Optional

# Internal imports
from game.solitaire import SolitaireGame
from game.renderer import GameRenderer
from rl.agent import DQNAgent
from utils.state_encoder import get_state_representation, get_state_size
from utils.action_utils import get_valid_actions, apply_action
import config

def initialize_training():
    """Initialize training components and return them."""
    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Solitaire with RL Training")

    # Initialize renderer
    renderer = GameRenderer(screen)

    # Initialize game and agent
    game = SolitaireGame()
    state_size = get_state_size()

    # Detect CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = DQNAgent(state_size, action_size=1000, device=device)  # Large action space

    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Initialize training log
    if not os.path.exists(config.TRAINING_LOG_PATH):
        with open(config.TRAINING_LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Steps', 'Score', 'Reward', 'Epsilon', 'Loss', 'Win'])

    return screen, renderer, game, agent

def log_performance(episode, steps, score, total_reward, epsilon, loss=None, win=False):
    """Log performance metrics to CSV file."""
    with open(config.TRAINING_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if loss is None:
            writer.writerow([episode, steps, score, total_reward, epsilon, 0, win])
        else:
            writer.writerow([episode, steps, score, total_reward, epsilon, loss, win])

def render_training_screen(renderer, game, episode, epsilon, average_reward):
    """Render the training screen with game state and training info."""
    renderer.render(game, highlight_valid_moves=True)
    renderer.render_training_info(episode, epsilon, average_reward)
    pygame.display.flip()

def train_agent(episodes: int, render_every: int = 10) -> None:
    """
    Main training loop for the RL agent.

    Args:
        episodes: Number of training episodes to run.
        render_every: Render the game every N episodes for visual monitoring.
    """
    # Initialize training components
    screen, renderer, game, agent = initialize_training()

    # Training statistics
    episode_rewards = []
    episode_losses = []
    total_steps = 0
    last_save_time = time.time()
    win_count = 0

    print(f"Starting training for {episodes} episodes...")

    try:
        for episode in range(1, episodes + 1):
            # Reset game state and episode statistics
            game.reset()
            state = get_state_representation(game)
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_loss = 0

            print(f"\n=== Episode {episode}/{episodes} ===")

            while not done and episode_steps < config.MAX_STEPS_PER_EPISODE:
                # Handle PyGame events to keep window responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Get valid actions
                valid_actions = get_valid_actions(game)

                # Get action from agent
                action_idx, epsilon = agent.get_action(state, list(range(len(valid_actions))))

                # Map action index to actual action
                if action_idx < len(valid_actions):
                    action = valid_actions[action_idx]
                else:
                    # Fallback to a random valid action
                    action = random.choice(valid_actions)

                # Apply action and get reward
                reward = apply_action(game, action)
                episode_reward += reward

                # Get next state
                next_state = get_state_representation(game)

                # Check if episode is done
                done = game._check_win() or episode_steps >= config.MAX_STEPS_PER_EPISODE

                # Store experience in replay memory
                agent.remember(state, action_idx, reward, next_state, done)

                # Train the agent
                loss = agent.replay(config.BATCH_SIZE)
                if loss is not None:
                    episode_loss = loss

                # Update state and increment step counter
                state = next_state
                episode_steps += 1
                total_steps += 1

                # Decay epsilon
                agent.decay_epsilon()

            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)
            average_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)

            # Check if game was won
            win = game._check_win()
            if win:
                win_count += 1

            # Print episode statistics
            print(f"Steps: {episode_steps}, Score: {game.score}, Reward: {episode_reward:.2f}, "
                  f"Epsilon: {epsilon:.4f}, Avg Reward: {average_reward:.2f}, Win: {win}")

            # Render game state periodically
            if episode % render_every == 0:
                print(f"Rendering game state for episode {episode}...")
                render_training_screen(renderer, game, episode, epsilon, average_reward)
                # Wait for user input to continue
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False

            # Log performance
            log_performance(episode, episode_steps, game.score, episode_reward, epsilon, episode_loss, win)

            # Save model periodically
            current_time = time.time()
            if current_time - last_save_time > 300:  # Save every 5 minutes
                print("Saving model...")
                agent.save_model(config.MODEL_SAVE_PATH)
                last_save_time = current_time

            # Save model at the end of training
            if episode == episodes:
                print("Saving final model...")
                agent.save_model(config.MODEL_SAVE_PATH)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        agent.save_model(config.MODEL_SAVE_PATH)

    finally:
        # Clean up
        pygame.quit()
        print("Training completed.")

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    episodes = 1000
    render_every = 10

    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print("Error: Number of episodes must be an integer")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            render_every = int(sys.argv[2])
        except ValueError:
            print("Error: Render interval must be an integer")
            sys.exit(1)

    # Start training
    train_agent(episodes, render_every)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main entry point for the Air Hockey RL training application.
Sets up hyperparameters and initiates the training loop.
"""

import argparse
from train import train


def main() -> None:
    """
Main entry point for the air hockey training script.
Parses command line arguments, sets hyperparameters, and starts training.
"""
    import json
    import os
    
    # Load configuration from config.json if it exists
    config = {}
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            print("Loaded configuration from config.json")
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    
    # Merge config with defaults
    training_config = config.get("training", {})
    dqn_config = config.get("dqn", {})
    env_config = config.get("environment", {})
    reward_config = config.get("rewards", {})
    
    parser = argparse.ArgumentParser(
        description="Train two DQN agents to play air hockey against each other.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=training_config.get("num_episodes", 10000),
        help="Total number of training episodes"
    )
    parser.add_argument(
        "--visualize_every", 
        type=int, 
        nargs="?",
        const=training_config.get("visualize_every", 250),
        default=training_config.get("visualize_every", 250),
        help="Visualize every nth episode using Pygame"
    )
    
    # DQN hyperparameters
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=dqn_config.get("learning_rate", 0.001),
        help="Learning rate for DQN agents"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=dqn_config.get("gamma", 0.99),
        help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--epsilon_start", 
        type=float, 
        default=dqn_config.get("epsilon_start", 1.0),
        help="Starting epsilon for exploration"
    )
    parser.add_argument(
        "--epsilon_end", 
        type=float, 
        default=dqn_config.get("epsilon_end", 0.01),
        help="Minimum epsilon value"
    )
    parser.add_argument(
        "--epsilon_decay", 
        type=float, 
        default=dqn_config.get("epsilon_decay", 0.995),
        help="Epsilon decay rate per episode"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=dqn_config.get("batch_size", 64),
        help="Batch size for training"
    )
    parser.add_argument(
        "--memory_size", 
        type=int, 
        default=dqn_config.get("memory_size", 10000),
        help="Replay memory buffer size"
    )
    parser.add_argument(
        "--target_update_freq", 
        type=int, 
        default=dqn_config.get("target_update_freq", 16),
        help="Update target network every n episodes"
    )
    
    # Environment parameters
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=training_config.get("max_steps", 1000),
        help="Maximum steps per episode before auto-reset"
    )
    
    parser.add_argument(
        "--learn_every", 
        type=int, 
        default=training_config.get("learn_every", 24),
        help="Number of steps between agent learning updates"
    )
    
    # Soft update / LR scheduler parameters
    parser.add_argument(
        "--tau",
        type=float,
        default=dqn_config.get("tau", 0.005),
        help="Polyak averaging factor for soft target updates"
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=dqn_config.get("lr_step_size", 2000),
        help="Decay LR every N episodes"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=dqn_config.get("lr_decay", 0.5),
        help="Multiplicative LR decay factor"
    )
    
    # Resume training from checkpoint
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from saved checkpoint files (agent1_final.pth, agent2_final.pth)"
    )
    
    parser.add_argument(
        "--override_lr",
        type=float,
        default=None,
        help="Override the learning rate when resuming training from a checkpoint"
    )
    
    parser.add_argument(
        "--override_epsilon",
        type=float,
        default=None,
        help="Override the starting epsilon when resuming training from a checkpoint"
    )
    
    args = parser.parse_args()
    
    # Print training configuration
    print("=" * 50)
    print("Air Hockey RL Training Configuration:")
    print("=" * 50)
    print(f"Training Episodes: {args.num_episodes}")
    print(f"Visualization Frequency: Every {args.visualize_every} episodes")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Discount Factor (gamma): {args.gamma}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print(f"Batch Size: {args.batch_size}")
    print(f"Memory Size: {args.memory_size}")
    print(f"Target Update Frequency: Every {args.target_update_freq} episodes")
    print(f"Max Steps per Episode: {args.max_steps}")
    print(f"Learn Every N Steps: {args.learn_every}")
    print(f"Tau (soft update): {args.tau}")
    print(f"LR Step Size: {args.lr_step_size}")
    print(f"LR Decay: {args.lr_decay}")
    print(f"Resume Training: {args.resume}")
    if args.override_lr is not None:
        print(f"Override LR: {args.override_lr}")
    if args.override_epsilon is not None:
        print(f"Override Epsilon: {args.override_epsilon}")
    print("=" * 50)
    
    # Start training
    try:
        train(
            num_episodes=args.num_episodes,
            visualize_every=args.visualize_every,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            target_update_freq=args.target_update_freq,
            max_steps=args.max_steps,
            learn_every=args.learn_every,
            resume=args.resume,
            env_config=env_config,
            reward_config=reward_config,
            tau=args.tau,
            lr_step_size=args.lr_step_size,
            lr_decay=args.lr_decay,
            override_lr=args.override_lr,
            override_epsilon=args.override_epsilon
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
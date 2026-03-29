#!/usr/bin/env python3
"""
main.py - Entry point for the WarThink turn-based strategy war game AI training application.

Initializes the Trainer and launches the infinite self-play PPO training loop.

Usage:
    python main.py                  # Start new training
    python main.py --load <path>    # Continue training from a saved model

- Periodic visualization occurs every 100 episodes.
- Checkpoints are saved every 500 episodes to 'models/'.
- Close the evaluation window or press ESC to continue training.
- Press Ctrl+C in the terminal to stop and save a final checkpoint.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

def main(load_path: Optional[str]) -> None:
    """
    Application entry point: ensures setup, initializes Trainer, and starts training loop.
    
    Args:
        load_path: Optional path to a saved model to continue training from.
    """
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    
    print("WarThink: Starting AI self-play PPO training...")
    if load_path:
        print(f"Attempting to load model from: {load_path}")
    print("Periodic evaluation renders every 100 episodes (close window/ESC to resume).")
    print("Checkpoints saved every 500 episodes to 'models/'. Ctrl+C to stop.\n")
    
    try:
        from trainer import Trainer
        trainer = Trainer(load_model_path=load_path)
        trainer.train()  # Infinite loop until interrupted
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        print("Ensure all project files are present: game.py, env.py, rl.py, trainer.py, renderer.py.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WarThink AI model.")
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Path to a saved model checkpoint to continue training from (e.g., models/checkpoint_500.zip).'
    )
    args = parser.parse_args()
    main(load_path=args.load)
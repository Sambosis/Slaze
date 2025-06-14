# -*- coding: utf-8 -*-
"""
Main executable script for training the Reinforcement Learning agent.

This script initializes the design environment and the PPO agent, then runs the
main RL training loop. It handles the orchestration of training, including
periodic model saving and the potential for Direct Preference Optimization (DPO)
fine-tuning of the generator model based on high-reward trajectories.

This script is intended to be run from the command line.
Example:
    python -m lspo_3d.train_agent \\
        --total-timesteps 1000000 \\
        --log-dir ./logs/ \\
        --motif-path ./artifacts/motifs/motif_centroids.pt \\
        --generator-path ./artifacts/generator/ \\
        --num-motifs 50

"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
# Check if stable_baselines3 and its dependencies are installed
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    raise ImportError(
        "stable_baselines3 is not installed. "
        "Please install it with: pip install stable-baselines3[extra]"
    )

from src.lspo_3d.config import PRUSA_SLICER_PATH
from lspo_3d.environment import DesignEnvironment
from lspo_3d.models.generator import CadQueryGenerator


def refine_generator_with_dpo(
    generator: CadQueryGenerator,
    high_reward_trajectories: List[Dict[str, Any]],
    dpo_config: Dict[str, Any]
) -> None:
    """
    Fine-tunes the CadQueryGenerator using Direct Preference Optimization (DPO).

    This function takes a collection of high-reward episodes (trajectories) and
    uses them to refine the generator. The assumption is that the generated
    `cadquery` scripts from these successful episodes are "preferred" outcomes.

    This is a placeholder for the actual DPO implementation, which would
    typically leverage a library like `trl`.

    Args:
        generator (CadQueryGenerator): The generator model instance to be
            fine-tuned.
        high_reward_trajectories (List[Dict[str, Any]]): A list of dictionaries,
            where each dictionary represents a successful episode containing
            the motif sequence and the corresponding generated script.
        dpo_config (Dict[str, Any]): A dictionary containing hyperparameters
            for the DPO training process (e.g., learning rate, epochs).
    """
    # TODO: Implement the DPO fine-tuning loop. This will involve:
    # 1. Formatting the trajectories into preference pairs (chosen vs. rejected).
    #    Since we only have positive examples, a "rejected" set might need to
    #    be synthetically generated or sampled from lower-reward trajectories.
    # 2. Using a DPO-specific trainer (e.g., from the `trl` library) to update
    #    the weights of the generator model.
    # 3. Saving the updated generator model back to disk.
    print("\nStarting DPO refinement for the generator model...")
    if not high_reward_trajectories:
        print("No high-reward trajectories provided for DPO. Skipping.")
        return

    print(f"Refining with {len(high_reward_trajectories)} successful trajectories.")
    # The actual DPO training call would go here.
    # generator.fine_tune_with_dpo(dataset, dpo_config)
    print("DPO refinement is currently a placeholder. No training was performed.")
    # generator.save_model(dpo_config['output_dir'])
    # print("DPO refinement complete. Saved updated generator.")


def train_agent(config: argparse.Namespace) -> None:
    """
    Sets up and runs the PPO agent training loop.

    Args:
        config (argparse.Namespace): An object containing the script's
            command-line arguments and configurations, including learning rates,
            timesteps, and paths for models and logs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # --- Set up directories ---
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env_output_dir = log_dir / "env_artifacts"
    env_output_dir.mkdir(parents=True, exist_ok=True)
    agent_save_path = log_dir / "agent_models"
    agent_save_path.mkdir(parents=True, exist_ok=True)

    # --- Initialize Generator and Environment ---
    print(f"Loading CadQueryGenerator from: {config.generator_path}")
    generator = CadQueryGenerator.load_model(config.generator_path)

    # Example reward weights. In a real scenario, these would be carefully tuned.
    reward_weights = {
        'success_bonus': 100.0,
        'failure_penalty': -200.0,
        'support_material_penalty': -1.0,  # Per mm^3
        'print_time_penalty': -0.01,       # Per second
        'filament_penalty': -0.5,          # Per mm^3
    }

    slicer_config = {
        "slicer_path": PRUSA_SLICER_PATH,
        "config_path": None,
    }

    physics_config = {
        "duration_steps": 2400,
        "load_config": {},
    }
    
    print("Initializing DesignEnvironment...")
    env = DesignEnvironment(
        generator=generator,
        slicer_config=slicer_config,
        physics_config=physics_config,
        num_motifs=config.num_motifs,
        design_prompt="Design a vertical stand for a standard smartphone.",
        max_steps=config.max_episode_steps,
        reward_weights=reward_weights,
        output_dir=str(env_output_dir)
    )

    # --- Setup Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(agent_save_path),
        name_prefix="ppo_agent",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # --- Define PPO Policy and Hyperparameters ---
    # Use SB3's built-in MlpPolicy with a simple network architecture.
    # This avoids incompatibilities with the earlier custom AgentPolicy class.
    policy_kwargs = {
        "net_arch": [dict(pi=[config.hidden_dim], vf=[config.hidden_dim])],
    }

    print("Instantiating PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard_logs"),
        device=device,
    )

    # --- Start Training ---
    print("\n" + "="*50)
    print("Starting agent training...")
    print("="*50)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=checkpoint_callback
    )

    # --- Save Final Model ---
    final_model_path = agent_save_path / "ppo_agent_final.zip"
    model.save(final_model_path)
    print(f"\nTraining complete. Final agent model saved to {final_model_path}")

    # --- Optional DPO Refinement ---
    if config.run_dpo:
        # In a real implementation, high-reward trajectories would be collected during
        # training, possibly via a custom callback that logs episodes where the
        # final reward exceeds a certain threshold.
        print("\nGathering high-reward trajectories for DPO...")
        high_reward_trajectories = []  # Placeholder for collected data

        if high_reward_trajectories:
            dpo_config = {
                "learning_rate": config.dpo_learning_rate,
                "output_dir": log_dir / "generator_dpo_tuned"
                # ... other DPO hyperparameters ...
            }
            refine_generator_with_dpo(generator, high_reward_trajectories, dpo_config)
        else:
            print("No high-reward trajectories collected. Skipping DPO.")


def main() -> None:
    """
    Main function to parse arguments and launch the training process.
    """
    parser = argparse.ArgumentParser(
        description="Train the 3D-LSPO PPO Agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Environment and Model Paths ---
    grp_paths = parser.add_argument_group("Paths and Directories")
    grp_paths.add_argument(
        "--log-dir", type=str, default="./3d_lspo_logs",
        help="Directory to save training logs and models."
    )
    grp_paths.add_argument(
        "--motif-path", type=str, required=True,
        help="Path to the file containing motif data (e.g., motif_centroids.pt)."
    )
    grp_paths.add_argument(
        "--generator-path", type=str, required=True,
        help="Path to the pre-trained CadQueryGenerator model directory."
    )
    grp_paths.add_argument(
        "--save-freq", type=int, default=50000,
        help="Frequency (in timesteps) to save a checkpoint of the agent."
    )

    # --- Training Hyperparameters ---
    grp_train = parser.add_argument_group("Training Hyperparameters")
    grp_train.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total number of timesteps for the training."
    )
    grp_train.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="The learning rate for the PPO optimizer."
    )
    grp_train.add_argument(
        "--n-steps", type=int, default=2048,
        help="Number of steps to run for each environment per update (rollout buffer size)."
    )
    grp_train.add_argument(
        "--batch-size", type=int, default=64,
        help="The mini-batch size for PPO updates."
    )
    grp_train.add_argument(
        "--n-epochs", type=int, default=10,
        help="Number of epochs when optimizing the surrogate loss."
    )
    grp_train.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor."
    )
    grp_train.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="Factor for trade-off of bias vs variance for GAE."
    )
    grp_train.add_argument(
        "--clip-range", type=float, default=0.2, help="Clipping parameter for PPO."
    )
    grp_train.add_argument(
        "--ent-coef", type=float, default=0.0, help="Entropy coefficient."
    )
    grp_train.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient."
    )
    grp_train.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="The maximum value for the gradient clipping."
    )
    grp_train.add_argument(
        "--device", type=str, default="cuda", choices=['cuda', 'cpu'],
        help="The device to use for training."
    )


    # --- Architecture and Environment Config ---
    grp_arch = parser.add_argument_group("Architecture and Environment")
    grp_arch.add_argument(
        "--num-motifs", type=int, required=True,
        help="Total number of available design motifs (action space size)."
    )
    grp_arch.add_argument(
        "--max-episode-steps", type=int, default=10,
        help="Maximum number of motifs to select in one episode."
    )
    grp_arch.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Size of the hidden layers in the agent's policy/value network."
    )

    # --- DPO Configuration ---
    grp_dpo = parser.add_argument_group("DPO Configuration")
    grp_dpo.add_argument(
        "--run-dpo", action="store_true",
        help="If set, run DPO refinement on the generator after agent training."
    )
    grp_dpo.add_argument(
        "--dpo-learning-rate", type=float, default=1e-5,
        help="Learning rate for the DPO fine-tuning stage."
    )

    args = parser.parse_args()

    # Simple check for motif file existence
    if not Path(args.motif_path).exists():
        parser.error(f"Motif path does not exist: {args.motif_path}")
    if not Path(args.generator_path).exists():
        parser.error(f"Generator path does not exist: {args.generator_path}")

    train_agent(args)


if __name__ == "__main__":
    main()
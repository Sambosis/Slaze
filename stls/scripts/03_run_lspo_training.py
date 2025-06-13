import os
import logging
from typing import Any, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO as sb3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Internal project imports
from src.lspo_3d import config
from src.lspo_3d import utils
# Import the experimental CSGGenerator from the src package. The
# top-level `lspo_3d` package only provides the CadQuery generator.
from src.lspo_3d.models.generator import CSGGenerator
from src.lspo_3d.oracles.slicer import SlicerOracle
from src.lspo_3d.oracles.physics import PhysicsOracle
from src.lspo_3d.rl.environment import DesignEnvironment
from src.lspo_3d.rl.agent import PPOAgent  # Potentially holds custom policies/logic for SB3


def fine_tune_generator(
    generator: CSGGenerator,
    optimizer: torch.optim.Optimizer,
    high_reward_data: List[Dict[str, Any]],
    device: torch.device
) -> None:
    """
    Fine-tunes the CSG Generator on a dataset of high-reward sequences.

    This function takes the successful (motif_sequence, scad_script) pairs
    generated during the RL exploration phase and performs supervised
    fine-tuning on the generator model to improve its ability to translate
    motifs into high-quality scripts.

    Args:
        generator (CSGGenerator): The generator model to be fine-tuned.
        optimizer (torch.optim.Optimizer): The optimizer for the generator.
        high_reward_data (List[Dict[str, Any]]): A list of dictionaries, where
            each dictionary represents a high-reward sample and should contain
            keys like 'motif_sequence' and 'tokenized_scad_script'. Both are
            expected to be torch.Tensors.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to perform
            training on.
    """
    logging.info(f"Starting generator fine-tuning with {len(high_reward_data)} samples.")
    if not high_reward_data:
        logging.warning("No data provided for fine-tuning. Skipping.")
        return

    # Prepare data for DataLoader
    try:
        # Assuming motif_sequence and tokenized_scad_script are already tensors
        motif_sequences = torch.stack([d['motif_sequence'] for d in high_reward_data])
        scad_scripts = torch.stack([d['tokenized_scad_script'] for d in high_reward_data])
    except Exception as e:
        logging.error(f"Error processing high-reward data for DataLoader: {e}")
        return

    dataset = TensorDataset(motif_sequences, scad_scripts)
    dataloader = DataLoader(
        dataset,
        batch_size=config.GENERATOR_FINETUNE_BATCH_SIZE,
        shuffle=True
    )

    # Loss function - ignores padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.TOKENIZER_PAD_ID)

    generator.train()
    for epoch in range(config.GENERATOR_FINETUNE_EPOCHS):
        total_loss = 0
        for i, (batch_motifs, batch_scripts) in enumerate(dataloader):
            batch_motifs = batch_motifs.to(device)
            batch_scripts = batch_scripts.to(device)

            # The generator's input is the motif sequence.
            # a target sequence for generation is the SCAD script tokens.
            # We want to predict the next token given the previous ones,
            # so the input to the decoder part should be the script shifted by one.
            # The motif sequence acts as the initial memory/context.
            target = batch_scripts
            
            # Forward pass
            optimizer.zero_grad()
            output_logits = generator(
                motif_sequence=batch_motifs,
                tgt_token_ids=target[:, :-1] # Use all but the last token as input
            )
            
            # Reshape for loss calculation
            # output_logits: [batch_size, seq_len-1, vocab_size]
            # target: [batch_size, seq_len] -> [batch_size, seq_len-1] for actual targets
            loss = loss_fn(
                output_logits.reshape(-1, output_logits.size(-1)),
                target[:, 1:].reshape(-1) # Target is the sequence shifted left
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Generator Fine-Tuning Epoch {epoch + 1}/{config.GENERATOR_FINETUNE_EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    logging.info("Generator fine-tuning complete.")
    generator.eval() # Set back to evaluation mode


def run_lspo_loop(
    agent: sb3_PPO,
    generator: CSGGenerator,
    env: DummyVecEnv,
    generator_optimizer: torch.optim.Optimizer,
    device: torch.device
) -> None:
    """
    Executes the main Latent Strategy Policy Optimization (LSPO) training loop.

    This loop alternates between two phases:
    1. RL Exploration: The PPO agent explores the design space by selecting
       sequences of motifs to maximize the reward from the environment's oracles.
    2. Generator Fine-Tuning: The CSG Generator is fine-tuned on the best
       designs discovered during the exploration phase.

    Args:
        agent (sb3_PPO): The Stable-Baselines3 PPO agent.
        generator (CSGGenerator): The CSG generator model.
        env (DummyVecEnv): The vectorized gym-style design environment.
        generator_optimizer (torch.optim.Optimizer): The optimizer for the generator.
        device (torch.device): The device to use for training.
    """
    logging.info("Starting LSPO training loop...")

    for lspo_cycle in range(config.LSPO_CYCLES):
        logging.info(f"--- Starting LSPO Cycle {lspo_cycle + 1}/{config.LSPO_CYCLES} ---")

        # --- Phase 1: RL Exploration ---
        logging.info("Phase 1: Running RL agent for exploration...")
        # The environment collects high-reward trajectories internally during agent.learn()
        agent.learn(
            total_timesteps=config.RL_EXPLORATION_STEPS_PER_CYCLE,
            reset_num_timesteps=False,
            tb_log_name="PPO_LSPO"
        )
        logging.info("RL exploration phase complete.")

        # --- Phase 2: Generator Fine-Tuning ---
        logging.info("Phase 2: Fine-tuning the CSG Generator...")

        # Retrieve the collected data from the environment.
        # Note: env is a DummyVecEnv, so we access the underlying environment with .envs[0]
        high_reward_data = env.envs[0].get_high_reward_trajectories()

        if high_reward_data:
            fine_tune_generator(
                generator=generator,
                optimizer=generator_optimizer,
                high_reward_data=high_reward_data,
                device=device
            )
            # Clear the environment's buffer after training to prepare for next cycle
            env.envs[0].clear_trajectories_buffer()
        else:
            logging.warning(
                f"No new high-reward trajectories found in cycle {lspo_cycle + 1}. "
                "Skipping generator fine-tuning."
            )

        # --- Save Models Periodically ---
        if (lspo_cycle + 1) % config.SAVE_FREQUENCY_CYCLES == 0:
            cycle_num = lspo_cycle + 1
            logging.info(f"Saving models at cycle {cycle_num}...")
            
            save_dir = config.MODEL_SAVE_DIR
            os.makedirs(save_dir, exist_ok=True)
            
            agent_path = os.path.join(save_dir, f"ppo_agent_cycle_{cycle_num}.zip")
            generator_path = os.path.join(save_dir, f"generator_cycle_{cycle_num}.pt")

            agent.save(agent_path)
            torch.save(generator.state_dict(), generator_path)
            
            logging.info(f"PPO agent saved to {agent_path}")
            logging.info(f"Generator model saved to {generator_path}")

    logging.info("LSPO training completed.")


def main() -> None:
    """
    Initializes all components and starts the LSPO training process.

    This function sets up the environment, loads the pretrained models,
    initializes the RL agent, and kicks off the main training loop.
    """
    utils.setup_project_logging()
    logging.info("Initializing 3D-LSPO training script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Initialize Oracles from config
    logging.info("Initializing Slicer and Physics oracles...")
    slicer_oracle = SlicerOracle(
        slicer_path=config.PRUSA_SLICER_PATH,
        config_path=config.PRUSA_SLICER_CONFIG
    )
    physics_oracle = PhysicsOracle(
        simulation_steps=config.PYBULLET_SIMULATION_STEPS
    )

    # 2. Load pre-trained Generator model
    logging.info("Loading pre-trained CSG Generator...")
    generator = CSGGenerator(
        vocab_size=config.TOKENIZER_VOCAB_SIZE,
        d_model=config.MODEL_DIM,
        num_heads=config.MODEL_NUM_HEADS,
        num_encoder_layers=config.GENERATOR_NUM_ENCODER_LAYERS,
        num_decoder_layers=config.GENERATOR_NUM_DECODER_LAYERS,
        dim_feedforward=config.MODEL_DIM_FEEDFORWARD,
        dropout=config.MODEL_DROPOUT
    )
    try:
        generator.load_state_dict(torch.load(config.PRETRAINED_GENERATOR_PATH, map_location=device))
        logging.info(f"Successfully loaded generator weights from {config.PRETRAINED_GENERATOR_PATH}")
    except FileNotFoundError:
        logging.error(f"Generator weights file not found at {config.PRETRAINED_GENERATOR_PATH}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading generator weights: {e}. Exiting.")
        return
        
    generator.to(device)
    generator.eval() # Start in evaluation mode
    
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.GENERATOR_FINETUNE_LR
    )

    # 3. Initialize the RL Environment
    logging.info("Initializing the design environment...")
    def make_env():
        return DesignEnvironment(
            generator=generator,
            slicer_oracle=slicer_oracle,
            physics_oracle=physics_oracle,
            motif_centroids_path=config.MOTIF_CENTROIDS_PATH,
            prompt=config.TRAINING_PROMPT,
            device=device,
            reward_threshold=config.RL_REWARD_THRESHOLD,
            max_trajectory_length=config.MAX_MOTIF_SEQUENCE_LENGTH,
            tokenizer_path=config.TOKENIZER_PATH
        )
    env = DummyVecEnv([make_env])

    # 4. Initialize the PPO Agent using Stable Baselines 3
    logging.info("Initializing PPO agent...")
    agent = sb3_PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.PPO_LEARNING_RATE,
        n_steps=config.PPO_N_STEPS,
        batch_size=config.PPO_BATCH_SIZE,
        n_epochs=config.PPO_N_EPOCHS,
        gamma=config.PPO_GAMMA,
        gae_lambda=config.PPO_GAE_LAMBDA,
        clip_range=config.PPO_CLIP_RANGE,
        ent_coef=config.PPO_ENT_COEF,
        vf_coef=config.PPO_VF_COEF,
        verbose=1,
        tensorboard_log=config.TENSORBOARD_LOG_DIR,
        device=device
    )

    # 5. Run the main LSPO training loop
    run_lspo_loop(
        agent=agent,
        generator=generator,
        env=env,
        generator_optimizer=generator_optimizer,
        device=device
    )


if __name__ == "__main__":
    main()
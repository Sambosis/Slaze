# /3d_lspo/train_agent.py

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import TrainingArguments

# Internal project imports
from lspo_3d.environment import DesignEnvironment
from lspo_3d.models.agent import AgentPolicy
from lspo_3d.models.generator import CadQueryGenerator, train_generator_with_dpo


class DPODatacollectorCallback(BaseCallback):
    """
    A custom Stable-Baselines3 callback to collect high-reward trajectories.

    This callback inspects the environment at the end of each episode. If the
    episode's cumulative reward exceeds a predefined threshold, it stores the
    trajectory (e.g., the sequence of motifs and the generated script) for
    later use in fine-tuning the generator model with Direct Preference
    Optimization (DPO).
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        """
        Initializes the callback.

        Args:
            reward_threshold (float): The minimum cumulative reward an episode
                must achieve for its trajectory to be collected.
            verbose (int): Verbosity level, 0 for no output, 1 for info.
        """
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.successful_trajectories: List[Tuple[str, List[int], str]] = []
        if self.verbose > 0:
            print(f"DPODatacollectorCallback initialized with reward_threshold={self.reward_threshold}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        It checks for the 'done' flag. If an episode is finished and its reward
        is high enough, it extracts the trajectory data from the environment's
        info dictionary and saves it.

        Returns:
            bool: False to stop training, True to continue.
        """
        # In a vectorized environment, 'dones' is a list/array
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][idx]
                # We assume the environment provides 'episode_reward', 'motif_sequence',
                # and 'cadquery_script' in the info dict upon episode termination.
                episode_reward = info.get("episode_reward")
                if episode_reward is not None and episode_reward > self.reward_threshold:
                    prompt = info.get("design_prompt")
                    motif_sequence = info.get("motif_sequence")
                    cadquery_script = info.get("cadquery_script")

                    if all([prompt, motif_sequence, cadquery_script]):
                        if self.verbose > 0:
                            print(f"\n[DPODatacollectorCallback] Collected high-reward trajectory with reward {episode_reward:.2f}")
                        self.successful_trajectories.append((prompt, motif_sequence, cadquery_script))
                    elif self.verbose > 0:
                        print(f"\n[DPODatacollectorCallback] High reward episode, but trajectory data missing from info dict.")

        return True

def setup_environment(config: Dict[str, Any]) -> DummyVecEnv:
    """
    Initializes and wraps the custom design environment.

    This function sets up the DesignEnvironment with the necessary components,
    such as the motif library and the code generator model. It also validates
    the environment using the stable-baselines3 checker and wraps it for
    compatibility.

    Args:
        config (Dict[str, Any]): A dictionary containing configuration for the
            environment, including paths to motifs and the generator model.

    Returns:
        DummyVecEnv: The wrapped, ready-to-use learning environment.
    """
    print("Setting up the design environment...")

    try:
        # Load motif data (assuming it's a file with a list of motif IDs)
        with open(config['motif_path'], 'r') as f:
            motif_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(motif_ids)} motif IDs from {config['motif_path']}")
    except Exception as e:
        print(f"Error loading motif IDs from {config['motif_path']}: {e}")
        motif_ids = list(range(config.get("n_motifs", 100)))
        print(f"Using placeholder motif IDs: {len(motif_ids)}")

    # Load the pre-trained generator model
    print(f"Loading generator model from {config['generator_path']}...")
    generator_model = CadQueryGenerator(
        model_name_or_path=config['generator_path'],
        device=config.get('device', 'cpu')
    )

    env_output_dir = Path(config['log_dir']) / "env_output"

    env = DesignEnvironment(
        design_prompt=config['design_prompt'],
        motif_ids=motif_ids,
        generator_model=generator_model,
        output_dir=str(env_output_dir),
        max_steps=config.get("max_episode_steps", 15)
    )

    print("Checking environment compatibility...")
    check_env(env)

    # Wrap the environment for the SB3 agent
    vec_env = DummyVecEnv([lambda: env])
    print("Environment setup complete.")
    return vec_env

def train_ppo_agent(env: DummyVecEnv, config: Dict[str, Any]) -> PPO:
    """
    Sets up and trains the PPO agent.

    This function configures the PPO algorithm from stable-baselines3. It sets
    up callbacks for saving model checkpoints and for collecting data for DPO,
    then starts the training loop.

    Args:
        env (DummyVecEnv): The vectorized training environment.
        config (Dict[str, Any]): A dictionary containing training hyperparameters
            and configuration, such as total timesteps, learning rate, and
            save paths.

    Returns:
        PPO: The trained PPO model.
    """
    print("Configuring PPO agent...")

    # Callback to save model checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path=config['log_dir'],
        name_prefix="ppo_agent"
    )

    # Callback to collect high-quality data for DPO
    dpo_collector_callback = DPODatacollectorCallback(
        reward_threshold=config['dpo_reward_threshold'],
        verbose=1
    )
    
    # The skeleton for this file suggests using `AgentPolicy` as a features extractor within `policy_kwargs`.
    # However, `AgentPolicy` is defined as a full actor-critic network, making it incompatible with that role.
    # The most robust interpretation is to use SB3's standard `MlpPolicy` and customize its network architecture,
    # which aligns with the goal of using a custom network without requiring a rewrite of `AgentPolicy`.
    policy_kwargs = {
        "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
    }

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        verbose=1,
        device=config.get('device', 'auto'),
        tensorboard_log=config['log_dir'],
    )

    print(f"Starting PPO training for {config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[checkpoint_callback, dpo_collector_callback],
        tb_log_name="PPO_run"
    )
    print("PPO training finished.")

    print("\n--- Starting DPO fine-tuning of the generator model ---")
    finetune_generator_with_dpo(
        generator=env.envs[0].generator,
        trajectories=dpo_collector_callback.successful_trajectories,
        config=config
    )

    return model

def finetune_generator_with_dpo(
    generator: "CadQueryGenerator",
    trajectories: List[Tuple[str, List[int], str]],
    config: Dict[str, Any]
) -> None:
    """
    Fine-tunes the generator model using Direct Preference Optimization (DPO).

    This function takes the successful trajectories collected during PPO
    training and uses them to refine the generator. It formats the data into a
    preference dataset (chosen vs. rejected) and then calls the DPO training
    function from the generator module.

    Args:
        generator (CadQueryGenerator): The generator model instance to be trained.
        trajectories (List[Tuple[str, List[int], str]]): A list of
            (prompt, motif_sequence, generated_script) tuples from high-reward episodes.
        config (Dict[str, Any]): The main training configuration dictionary.
    """
    if not trajectories:
        print("No high-reward trajectories collected for DPO. Skipping fine-tuning.")
        return

    print(f"Preparing {len(trajectories)} successful examples for DPO fine-tuning.")
    dpo_dataset = []
    for prompt, motif_seq, chosen_script in trajectories:
        input_text = f"prompt: {prompt} motifs: {' '.join(map(str, motif_seq))}"
        
        # Generate a "rejected" sample by using the model with sampling enabled
        # to produce a plausible but likely less optimal alternative.
        rejected_script = generator.generate(
            prompt,
            motif_seq,
            temperature=1.2,
            do_sample=True,
            top_k=50,
            max_length=512
        )

        if rejected_script != chosen_script and rejected_script:
            dpo_dataset.append({
                "prompt": input_text,
                "chosen": chosen_script,
                "rejected": rejected_script
            })

    if not dpo_dataset:
        print("Could not generate unique rejected samples. Skipping DPO fine-tuning.")
        return

    print(f"Created {len(dpo_dataset)} preference pairs for DPO.")

    dpo_output_dir = Path(config['log_dir']) / "dpo_finetuned_generator"
    dpo_output_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(dpo_output_dir),
        per_device_train_batch_size=config.get('dpo_batch_size', 2),
        num_train_epochs=config.get('dpo_epochs', 1),
        learning_rate=config.get('dpo_lr', 1e-5),
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=4
    )

    # This function is defined in `lspo_3d/models/generator.py`
    train_generator_with_dpo(
        generator=generator,
        dataset=dpo_dataset,
        output_dir=str(dpo_output_dir),
        training_args=training_args
    )

    print(f"DPO fine-tuning complete. Fine-tuned generator saved to {dpo_output_dir}")

def main() -> None:
    """
    Main execution function for the agent training script.

    Parses command-line arguments, sets up the environment and agent,
    runs the training process, and saves the final artifacts.
    """
    parser = argparse.ArgumentParser(description="Train the 3D-LSPO PPO Agent.")
    parser.add_argument("--motif-path", type=str, required=True, help="Path to the file containing motif IDs, one per line.")
    parser.add_argument("--generator-path", type=str, required=True, help="Path to the initial generator model directory.")
    parser.add_argument("--log-dir", type=str, default="./logs/agent_training", help="Directory to save logs and models.")
    parser.add_argument("--design-prompt", type=str, default="A functional and stable stand for a modern smartphone.", help="The default design prompt for the environment.")

    # PPO Hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps for PPO training.")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Frequency to save model checkpoints.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the PPO optimizer.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size.")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of optimization epochs per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance for GAE.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clipping parameter for PPO.")

    # DPO Hyperparameters
    parser.add_argument("--dpo-reward-threshold", type=float, default=50.0, help="Reward threshold to collect trajectories for DPO.")

    # System
    parser.add_argument("--device", type=str, default="auto", help="Device for training ('auto', 'cpu', 'cuda').")

    args = parser.parse_args()
    config = vars(args)

    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    # 1. Setup Environment
    env = setup_environment(config)

    # 2. Train Agent (PPO + DPO)
    final_model = train_ppo_agent(env, config)

    # 3. Save Final Model
    final_model_path = os.path.join(config['log_dir'], "ppo_agent_final.zip")
    final_model.save(final_model_path)
    print(f"\nFinal agent model saved to {final_model_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
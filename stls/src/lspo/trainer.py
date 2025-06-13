import os
import time
import copy
import itertools
from typing import List, Dict, Any, Tuple, Optional  
import matplotlib
matplotlib.use('agg')

import torch
import torch.nn.functional as F
import numpy as np
import gym
from transformers import T5ForConditionalGeneration, AutoTokenizer, AdamW
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from lspo import config
from lspo import models
from lspo import oracle


class LSPOTrainer:
    """
    Orchestrates the main training loop for the 3D-LSPO project.

    This class coordinates the Reinforcement Learning (RL) agent, the generative
    model, and the evaluation oracle. It manages the iterative process of
    generating a design strategy, translating it into a 3D model, evaluating
    the model's quality, and updating the agent and generator based on the
    feedback.
    """

    def __init__(self, cfg: config.Config, device: Optional[torch.device] = None):
        """
        Initializes the trainer, loading models, oracle, and configurations.

        Args:
            cfg (config.Config): The configuration object containing all
                hyperparameters and settings for the training run.
            device (Optional[torch.device]): The PyTorch device (CPU or GPU) to run on.
                                              If None, auto-detects.
        """
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize core components
        print("Initializing trainer components...")
        self.oracle = oracle.Oracle(self.cfg)
        self.embedder = models.SequenceEmbedder(model_name=self.cfg.embedding.model_name, device=self.device)
        self.generator_model, self.generator_tokenizer = self._initialize_generator()
        self.dpo_optimizer = AdamW(self.generator_model.parameters(), lr=self.cfg.generator.dpo_learning_rate)
        
        self.rl_agent = self._initialize_rl_agent()

        # Data structures for storing experiences for model updates
        self.rl_rollout_buffer = RolloutBuffer(
            buffer_size=self.cfg.trainer.num_steps_per_update,
            observation_space=self.rl_agent.observation_space,
            action_space=self.rl_agent.action_space,
            device=self.device,
            gamma=self.cfg.agent.gamma,
            gae_lambda=self.cfg.agent.gae_lambda,
            n_envs=1  # We are running a single environment loop
        )
        self.dpo_preference_buffer: List[Dict[str, Any]] = []
        self.ref_model: Optional[T5ForConditionalGeneration] = None
        print("Trainer initialization complete.")

    def _initialize_generator(self) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
        """
        Loads and prepares the T5 generator model and its tokenizer.

        Returns:
            Tuple[T5ForConditionalGeneration, AutoTokenizer]: The initialized generator
            model and its tokenizer.
        """
        print(f"Loading generator model: {self.cfg.generator.model_name}")
        model = T5ForConditionalGeneration.from_pretrained(self.cfg.generator.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.generator.model_name)
        model.to(self.device)
        return model, tokenizer

    def _initialize_rl_agent(self) -> PPO:
        """
        Initializes the PPO agent from Stable-Baselines3.

        Returns:
            PPO: The initialized Stable-Baselines3 PPO agent.
        """
        print("Initializing PPO agent...")
        embedding_dim = self.embedder.model.get_sentence_embedding_dimension()
        
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Discrete(self.cfg.agent.action_space_size)

        # SB3 PPO requires a VecEnv for initialization, even if we run a custom loop.
        # We create a dummy environment with the correct spaces.
        def dummy_env_creator():
            env = gym.Env()
            env.observation_space = observation_space
            env.action_space = action_space
            return env
        
        dummy_vec_env = DummyVecEnv([dummy_env_creator])

        policy_kwargs = dict(
            net_arch=dict(pi=self.cfg.agent.actor_hidden_dims, vf=self.cfg.agent.critic_hidden_dims)
        )

        agent = PPO(
            policy="MlpPolicy",
            env=dummy_vec_env,
            learning_rate=self.cfg.agent.learning_rate,
            n_steps=self.cfg.trainer.num_steps_per_update,
            batch_size=self.cfg.agent.mini_batch_size,
            n_epochs=self.cfg.agent.ppo_epochs,
            gamma=self.cfg.agent.gamma,
            gae_lambda=self.cfg.agent.gae_lambda,
            clip_range=self.cfg.agent.clip_param,
            tensorboard_log=str(self.cfg.paths.logs_dir),
            device=self.device,
            policy_kwargs=policy_kwargs,
            seed=self.cfg.seed,
            verbose=0,
        )
        return agent

    def train(self) -> None:
        """
        Executes the main training loop for a specified number of total timesteps.
        """
        print("--- Starting LSPO Training ---")
        start_time = time.time()
        
        # A sample prompt for the design task
        text_prompt = "a sturdy stand for a mobile phone"
        
        obs = self.embedder.embed([text_prompt]).to(self.device)

        for timestep in range(1, self.cfg.trainer.total_timesteps + 1):
            episode_data = self._run_episode(obs, text_prompt, timestep)
            
            # Store experience for RL update
            self.rl_rollout_buffer.add(
                obs=episode_data["observation"],
                action=episode_data["action"],
                reward=np.array([episode_data["reward"]]),
                done=np.array([True]),
                value=episode_data["value"],
                log_prob=episode_data["log_prob"],
            )

            # Store experience for DPO update
            self.dpo_preference_buffer.append(episode_data)
            
            # --- Model Updates ---
            if self.rl_rollout_buffer.full:
                self._update_rl_agent(obs)
            
            if len(self.dpo_preference_buffer) >= self.cfg.generator.dpo_batch_size * 2:
                self._update_generator_with_dpo()

            # --- Logging and Checkpointing ---
            if timestep % self.cfg.trainer.log_interval == 0:
                fps = int(timestep / (time.time() - start_time + 1e-8))
                print(f"Time: {time.time()-start_time:.2f}s | Timesteps: {timestep}/{self.cfg.trainer.total_timesteps} | FPS: {fps}")
                print(f"Last Reward: {episode_data['reward']:.2f} | Motif: {episode_data['action'].item()}")

            if timestep % self.cfg.trainer.save_interval == 0:
                self._save_checkpoint(timestep)
        
        print("--- Training Completed ---")

    def _run_episode(self, observation: torch.Tensor, text_prompt: str, timestep: int) -> Dict[str, Any]:
        """
        Executes a single design-generation-evaluation episode.

        Args:
            observation (torch.Tensor): The embedded representation of the text prompt.
            text_prompt (str): The high-level design goal for the episode.
            timestep (int): Current global timestep, for unique IDs.

        Returns:
            Dict[str, Any]: A dictionary containing episode results.
        """
        with torch.no_grad():
            action, value, log_prob = self.rl_agent.policy.predict(observation, deterministic=False)

        motif_id = action.item()
        generator_prompt = f"Design a 3D model based on this description: '{text_prompt}'. Use design motif number {motif_id}."
        
        input_ids = self.generator_tokenizer(generator_prompt, return_tensors='pt').input_ids.to(self.device)
        
        # Generate script using the T5 model
        output_ids = self.generator_model.generate(
            input_ids,
            max_length=self.cfg.generator.max_seq_length,
            num_beams=4,
            early_stopping=True
        )
        generated_script = self.generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Evaluate with the Oracle
        reward, info_dict = self.oracle.evaluate_design(generated_script, f"t{timestep}")
        
        return {
            "observation": observation.cpu().numpy(),
            "action": action.cpu().numpy(),
            "value": value,
            "log_prob": log_prob,
            "reward": reward,
            "text_prompt": text_prompt,
            "generated_script": generated_script,
            "info": info_dict
        }

    def _update_rl_agent(self, last_obs: torch.Tensor) -> None:
        """
        Updates the PPO agent's policy using collected experiences.
        """
        print("Updating RL agent...")
        with torch.no_grad():
            # The next state is the same as the current state in this bandit-like setting.
            # We compute the value of the last observation to bootstrap.
            last_value = self.rl_agent.policy.predict(last_obs, deterministic=True)[1]

        self.rl_rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=np.array([True]))
        self.rl_agent.train()
        self.rl_rollout_buffer.reset()

    def _get_sequence_log_probs(self, model: T5ForConditionalGeneration, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Helper to compute log probabilities of a sequence."""
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities of the actual label tokens
        sequence_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens (where label is -100 or tokenizer.pad_token_id)
        mask = (labels != self.generator_tokenizer.pad_token_id) & (labels != -100)
        sequence_log_probs = sequence_log_probs * mask
        
        # Sum log probs to get total log prob for each sequence in the batch
        return sequence_log_probs.sum(dim=1)

    def _update_generator_with_dpo(self) -> None:
        """
        Fine-tunes the generator model using Direct Preference Optimization (DPO).
        """
        print("Updating generator with DPO...")
        if self.ref_model is None:
            self.ref_model = copy.deepcopy(self.generator_model).eval()

        # Sort buffer by reward to create preference pairs
        self.dpo_preference_buffer.sort(key=lambda x: x["reward"], reverse=True)
        
        batch = []
        # Create pairs of (best, worst) from the buffer
        for chosen_item, rejected_item in itertools.zip_longest(
            self.dpo_preference_buffer[:len(self.dpo_preference_buffer)//2],
            self.dpo_preference_buffer[len(self.dpo_preference_buffer)//2:]
        ):
            if rejected_item is None: continue # Uneven number of items
            if chosen_item["reward"] > rejected_item["reward"]:
                batch.append({
                    "prompt": chosen_item["text_prompt"],
                    "chosen": chosen_item["generated_script"],
                    "rejected": rejected_item["generated_script"]
                })

        if not batch: 
            self.dpo_preference_buffer.clear()
            return
            
        # DPO training step
        self.generator_model.train()
        total_loss = 0.0
        for data in batch:
            self.dpo_optimizer.zero_grad()

            # Tokenize inputs
            prompt_tokenized = self.generator_tokenizer(data["prompt"], return_tensors="pt").to(self.device)
            chosen_tokenized = self.generator_tokenizer(data["chosen"], return_tensors="pt").to(self.device)
            rejected_tokenized = self.generator_tokenizer(data["rejected"], return_tensors="pt").to(self.device)
            
            # Get log probabilities from policy and reference models
            with torch.no_grad():
                ref_logp_chosen = self._get_sequence_log_probs(self.ref_model, prompt_tokenized.input_ids, chosen_tokenized.input_ids)
                ref_logp_rejected = self._get_sequence_log_probs(self.ref_model, prompt_tokenized.input_ids, rejected_tokenized.input_ids)
            
            policy_logp_chosen = self._get_sequence_log_probs(self.generator_model, prompt_tokenized.input_ids, chosen_tokenized.input_ids)
            policy_logp_rejected = self._get_sequence_log_probs(self.generator_model, prompt_tokenized.input_ids, rejected_tokenized.input_ids)

            # DPO Loss calculation
            pi_logratios = policy_logp_chosen - policy_logp_rejected
            ref_logratios = ref_logp_chosen - ref_logp_rejected
            
            loss = -F.logsigmoid(self.cfg.generator.dpo_beta * (pi_logratios - ref_logratios)).mean()
            
            loss.backward()
            self.dpo_optimizer.step()
            total_loss += loss.item()

        print(f"DPO update complete. Average loss: {total_loss / len(batch):.4f}")
        self.dpo_preference_buffer.clear()
        
        # Update the reference model periodically
        self.ref_model.load_state_dict(self.generator_model.state_dict())
        self.generator_model.train()


    def _save_checkpoint(self, timestep: int) -> None:
        """
        Saves the current state of the RL agent and the generator model.

        Args:
            timestep (int): The current training timestep for naming the checkpoint.
        """
        print(f"Saving checkpoint at timestep {timestep}...")
        checkpoint_dir = self.cfg.paths.checkpoints_dir
        
        # Save RL Agent
        rl_path = checkpoint_dir / f"ppo_agent_{timestep}.zip"
        self.rl_agent.save(rl_path)

        # Save Generator Model and Tokenizer
        generator_path = checkpoint_dir / f"generator_{timestep}"
        self.generator_model.save_pretrained(generator_path)
        self.generator_tokenizer.save_pretrained(generator_path)
        
        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Loads model states from a specified checkpoint path.

        Args:
            checkpoint_dir (str): The directory path containing the saved models.
        """
        print(f"Loading checkpoint from: {checkpoint_dir}")
        # This assumes separate files for PPO and generator.
        # Logic to find the latest or specific files would be needed.
        ppo_checkpoint = os.path.join(checkpoint_dir, "ppo_agent.zip") # Example name
        generator_checkpoint = os.path.join(checkpoint_dir, "generator") # Example name

        if os.path.exists(ppo_checkpoint):
           self.rl_agent = PPO.load(ppo_checkpoint, device=self.device)
           print("Loaded PPO agent checkpoint.")
        
        if os.path.isdir(generator_checkpoint):
            self.generator_model = T5ForConditionalGeneration.from_pretrained(generator_checkpoint).to(self.device)
            self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_checkpoint)
            # Re-initialize the optimizer with the new model's parameters
            self.dpo_optimizer = AdamW(self.generator_model.parameters(), lr=self.cfg.generator.dpo_learning_rate)
            print("Loaded Generator checkpoint.")
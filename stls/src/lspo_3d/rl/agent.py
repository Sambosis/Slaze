# C:\Users\Machine81\Slazy\repo\stls\src\lspo_3d\rl\agent.py
"""
Implements the PPO agent, including its policy and value networks.

This module defines the PPOAgent class, which serves as a high-level wrapper
around the Stable Baselines3 PPO implementation. It facilitates the
initialization, training, and use of the RL agent within the project's
custom design environment.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback

if typing.TYPE_CHECKING:
    from src.lspo_3d import config
    from lspo_3d.rl.environment import DesignEnvironment


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    A custom policy network for the PPO agent.

    This class extends the standard ActorCriticPolicy from Stable Baselines3,
    allowing for a custom architecture for the policy (actor) and value (critic)
    networks. This is crucial for tailoring the network to the specific
    observation and action spaces of the 3D design task.

    Attributes:
        features_extractor (nn.Module): The network that processes observations.
        pi_net (nn.Module): The policy network head that outputs action logits.
        vf_net (nn.Module): The value network head that outputs a state value.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[list] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        """
        Initializes the custom policy and value networks.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            lr_schedule: The learning rate schedule.
            net_arch (Optional[list]): The architecture of the policy and value
                networks. If None, a default architecture is used.
            activation_fn (Type[nn.Module]): The activation function to use.
            *args: Additional arguments passed to the parent constructor.
            **kwargs: Additional keyword arguments passed to the parent constructor.
        """
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )
        # The actual network construction is handled by the parent class,
        # specifically in `_build_mlp_extractor`. We don't need to add
        # anything here unless we are overriding the entire `__init__` logic.

    def _build_mlp_extractor(self) -> None:
        """
        Build the Multi-Layer Perceptron (MLP) extractor for the policy and
        value networks. This is where custom layers would be defined.

        For this implementation, we rely on the default MLP extractor provided
        by Stable Baselines3, which creates two separate MLPs for the policy
        and value functions based on the `net_arch` argument.
        """
        super()._build_mlp_extractor()


class PPOAgent:
    """
    A wrapper class for the Stable Baselines3 PPO algorithm.

    This class encapsulates the PPO model, providing a simplified interface for
    training, predicting actions, saving, and loading within the context of the
    LSPO project.
    """

    def __init__(
        self,
        env: "DesignEnvironment",
        agent_config: Type["config.PPOConfig"],
        log_path: Optional[Path] = None,
        device: str = "auto",
    ):
        """
        Initializes the PPO agent.

        Args:
            env (DesignEnvironment): The gym-style environment for the agent to
                interact with.
            agent_config (Type[config.PPOConfig]): A configuration class
                containing all hyperparameters for the PPO agent.
            log_path (Optional[Path]): The path to a directory for saving
                TensorBoard logs. Defaults to None.
            device (str): The device to use for training ('cpu', 'cuda', 'auto').
                Defaults to "auto".
        """
        self.env = env
        self.config = agent_config
        self.log_path = log_path
        self.device = device

        policy_kwargs: Dict[str, Any] = {
            "net_arch": self.config.POLICY_NET_ARCH,
            "activation_fn": torch.nn.ReLU,
        }

        self.model = PPO(
            policy=CustomActorCriticPolicy,
            env=self.env,
            learning_rate=self.config.LEARNING_RATE,
            n_steps=self.config.N_STEPS,
            batch_size=self.config.BATCH_SIZE,
            n_epochs=self.config.N_EPOCHS,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            clip_range=self.config.CLIP_RANGE,
            ent_coef=self.config.ENT_COEF,
            vf_coef=self.config.VF_COEF,
            max_grad_norm=self.config.MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.log_path) if self.log_path else None,
            device=self.device,
            verbose=1,
        )

    def learn(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """
        Trains the PPO agent for a given number of timesteps.

        Args:
            total_timesteps (int): The total number of samples (env steps) to
                train on.
            callback (Optional[BaseCallback]): A Stable Baselines3 callback to use during
                training. Can be used for evaluation, saving checkpoints, etc.
                Defaults to None.
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(
        self, observation: Any, deterministic: bool = True
    ) -> Tuple[Any, Optional[Any]]:
        """
        Predicts the action to take for a given observation.

        Args:
            observation (Any): The current observation from the environment.
            deterministic (bool): Whether to sample from the action probability
                distribution (False) or to take the most likely action (True).
                Defaults to True.

        Returns:
            Tuple[Any, Optional[Any]]: A tuple containing the predicted action
                and the model's recurrent state (if applicable, typically None
                for PPO).
        """
        # No pre-processing is needed as the environment already returns
        # the observation in the format expected by the model.
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    def save(self, save_path: Path) -> None:
        """
        Saves the trained agent model to a file.

        The model is saved as a .zip file containing all necessary components.
        This method will create parent directories if they do not exist.

        Args:
            save_path (Path): The file path where the model should be saved.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        print(f"Agent model saved to {save_path}")

    @classmethod
    def load(
        cls,
        load_path: Path,
        env: "DesignEnvironment",
        agent_config: Type["config.PPOConfig"],
        device: str = "auto"
    ) -> "PPOAgent":
        """
        Loads a pre-trained agent model from a file.

        Args:
            load_path (Path): The file path of the model to load.
            env (DesignEnvironment): The environment for the loaded agent.
            agent_config (Type[config.PPOConfig]): The configuration class used
                to initialize the agent structure.
            device (str): The device to load the model onto ('cpu', 'cuda', 'auto').
                Defaults to "auto".

        Returns:
            PPOAgent: An instance of the PPOAgent with the loaded model.

        Raises:
            FileNotFoundError: If the specified `load_path` does not exist.
        """
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at: {load_path}")
            
        # Recreate the agent instance wrapper.
        # The configuration is needed to define the agent's structure, even
        # though the weights will be overwritten.
        loaded_agent = cls(env=env, agent_config=agent_config, device=device)
        
        # Load the trained model parameters from the specified path into the
        # stable_baselines3 PPO object.
        loaded_agent.model = PPO.load(load_path, env=env, device=device)

        print(f"Agent model loaded from {load_path}")
        return loaded_agent
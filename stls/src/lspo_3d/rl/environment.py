# src/lspo_3d/rl/environment.py

from typing import Tuple, List, Dict, Any, Optional

import gymnasium as gym
import numpy as np
from gym import spaces

from src.lspo_3d import config
from lspo_3d.models.generator import CSGGenerator
from src.lspo_3d.oracles.reward import calculate_reward


class DesignEnvironment(gym.Env):
    """
    Custom Gym environment for 3D model generation.

    The RL agent interacts with this environment to learn a policy for selecting
    sequences of design motifs. The environment orchestrates the generation of
    a 3D model from the motif sequence and calculates a reward based on its
    printability, efficiency, and stability.

    This class adheres to the OpenAI Gym interface.

    Attributes:
        action_space (gym.spaces.Discrete): The space of possible actions,
            corresponding to the available design motifs.
        observation_space (gym.spaces.Box): The observation space, representing
            the current sequence of chosen motifs, padded to a fixed length.
        generator (CSGGenerator): The model used to generate an OpenSCAD script
            from a sequence of motif IDs.
        prompt (str): The high-level textual prompt guiding the design.
        motif_sequence (List[int]): The current sequence of motifs chosen in
            the episode.
        current_step (int): The number of steps taken in the current episode.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, generator: CSGGenerator, prompt: str):
        """
        Initializes the DesignEnvironment.

        Args:
            generator (CSGGenerator): An initialized, pre-trained generator model
                capable of translating motif sequences into OpenSCAD scripts.
            prompt (str): The textual design prompt. While not directly used in
                this basic environment state, it is passed to the reward oracle
                and is crucial for the overall generation process.
        """
        super().__init__()

        self.generator: CSGGenerator = generator
        self.prompt: str = prompt

        # The action space is the set of all possible abstract design motifs.
        # The size is the number of clusters (motifs) from the latent space.
        # An action is an integer from 0 to RL_NUM_MOTIFS - 1.
        self.action_space: spaces.Discrete = spaces.Discrete(config.RL_NUM_MOTIFS)

        # The observation is the sequence of motifs chosen so far.
        # We represent it as a fixed-size vector padded with a special value (0).
        # The values are the motif IDs (1-based), and the padding token is 0.
        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=config.RL_NUM_MOTIFS,
            shape=(config.RL_MAX_SEQUENCE_LENGTH,),
            dtype=np.int32
        )

        self.motif_sequence: List[int] = []
        self.current_step: int = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        The agent selects a motif (action), which is appended to the current sequence.
        If the sequence is complete (reaches max length), it is passed to the
        generator, and the resulting model is evaluated by the reward oracle.

        Args:
            action (int): The ID of the motif chosen by the agent. Note that
                          motif IDs are 1-based, so the action from the agent
                          (0 to N-1) is mapped to motif IDs 1 to N.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A tuple containing:
                - observation (np.ndarray): The new state of the environment.
                - reward (float): The reward obtained from the action.
                - done (bool): Whether the episode has ended.
                - info (Dict[str, Any]): A dictionary for diagnostic information,
                                         e.g., partial rewards.
        """
        # 1. Validate and append the action (motif_id) to self.motif_sequence.
        assert self.action_space.contains(action), f"Invalid action: {action}"
        # Map agent action (0-indexed) to motif ID (1-indexed)
        motif_id = action + 1
        self.motif_sequence.append(motif_id)

        # 2. Increment self.current_step.
        self.current_step += 1

        # 3. Check for termination condition.
        done = self.current_step >= config.RL_MAX_SEQUENCE_LENGTH
        reward: float = 0.0
        info: Dict[str, Any] = {}

        # 4. If done, generate model and calculate final reward.
        if done:
            try:
                # a. Generate the .scad script using the generator.
                scad_script = self.generator.generate(motif_sequence=self.motif_sequence)
                # b. Call `calculate_reward` to get the final reward and info dict.
                # This function orchestrates the entire evaluation pipeline.
                reward, info = calculate_reward(scad_script=scad_script, prompt=self.prompt)
            except Exception as e:
                # Handle potential errors during generation or evaluation
                print(f"An error occurred during step evaluation: {e}")
                # Assign a large penalty for complete failure
                reward = config.FAILURE_PENALTY
                info = {"error": str(e), "motif_sequence": self.motif_sequence}
        # 5. If not done, apply a step penalty.
        else:
            reward = config.STEP_PENALTY
            info = {}

        # 6. Construct the next_observation by padding self.motif_sequence.
        next_observation = np.zeros((config.RL_MAX_SEQUENCE_LENGTH,), dtype=np.int32)
        next_observation[:self.current_step] = self.motif_sequence

        # 7. Return the results for this step.
        return next_observation, reward, done, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        """
        Resets the environment to an initial state for a new episode.

        This method clears the current motif sequence and step count. It must be
        called at the beginning of each new episode.

        Args:
            seed (Optional[int]): The seed for the random number generator,
                used for reproducibility.
            options (Optional[Dict]): Additional options for resetting the
                environment. Not used in this implementation.

        Returns:
            np.ndarray: The initial observation of the environment, which is a
                        zero-padded array representing an empty sequence.
        """
        super().reset(seed=seed)

        self.motif_sequence = []
        self.current_step = 0

        # Initial observation is a zero-padded array, representing an empty sequence.
        initial_observation = np.zeros(
            (config.RL_MAX_SEQUENCE_LENGTH,),
            dtype=np.int32
        )
        return initial_observation

    def render(self, mode: str = 'human') -> None:
        """
        Renders the environment's state.

        In 'human' mode, it prints the current step number and the chosen
        motif sequence to the console.

        Args:
            mode (str): The mode to render with. Currently only 'human' is supported.
        """
        if mode == 'human':
            print(f"Step: {self.current_step}, Sequence: {self.motif_sequence}")
        else:
            # For compatibility with gym.Env, though other modes aren't implemented.
            super().render(mode=mode)

    def close(self) -> None:
        """
        Performs any necessary cleanup operations.

        This method is called when the environment is no longer needed. In this
        case, no specific cleanup is required.
        """
        pass
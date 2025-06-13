# C:\Users\Machine81\Slazy\repo\stls\3d_lspo\environment.py

"""
Defines the main reinforcement learning environment for the 3D-LSPO project.

This module contains the DesignEnvironment class, which adheres to the OpenAI Gym
API. It manages the agent's state (the design prompt and sequence of motifs),
the action space (the set of available design motifs), and calculates the reward
by orchestrating the generator model and the various verification oracles
(slicer, physics).
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import gymnasium as gym
import numpy as np

# --- Internal Imports ---
from lspo_3d.models.generator import CadQueryGenerator
from src.lspo_3d.oracles.csg_executor import execute_cad_script
from src.lspo_3d.oracles.physics_verifier import verify_stability
from src.lspo_3d.oracles.slicer_verifier import get_slicer_metrics, SlicerMetrics

# Set up a logger for this module
logger = logging.getLogger(__name__)


class DesignEnvironment(gym.Env):
    """
    A gym-compliant environment for learning to generate 3D models.

    The environment orchestrates the entire design generation and verification
    pipeline. An agent interacts with this environment by selecting a sequence
    of high-level "design motifs". At the end of an episode (when a full
    sequence is selected), the environment uses a generator model to translate
    the motif sequence into a concrete CadQuery script, executes it to create
    an STL file, and then uses slicer and physics oracles to evaluate the
    design's quality and functionality. A composite reward is then calculated
    based on these evaluations.

    Attributes:
        generator (CadQueryGenerator): The model used to translate motif
            sequences into CadQuery scripts.
        num_motifs (int): The total number of unique design motifs available,
            defining the size of the discrete action space.
        design_prompt (str): The high-level design goal for the current session.
        max_steps (int): The maximum number of motifs allowed in a single design
            sequence, defining the episode length.
        reward_weights (Dict[str, float]): A dictionary of weights for
            calculating the composite reward from oracle metrics.
        output_dir (Path): Path to the directory for saving generated scripts
            and STL files.
        slicer_config (Dict[str, Any]): Configuration for the slicer oracle.
        physics_config (Dict[str, Any]): Configuration for the physics oracle.
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Dict): The space of possible observations.
        current_step (int): The number of steps taken in the current episode.
        motif_sequence (List[int]): The sequence of motif IDs selected by the
            agent in the current episode.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 generator: CadQueryGenerator,
                 num_motifs: int,
                 design_prompt: str,
                 max_steps: int,
                 reward_weights: Dict[str, float],
                 output_dir: str,
                 slicer_config: Dict[str, Any],
                 physics_config: Dict[str, Any]):
        """
        Initializes the DesignEnvironment.

        Args:
            generator (CadQueryGenerator): An instantiated generator model.
            num_motifs (int): The total number of available design motifs (the
                size of the action space).
            design_prompt (str): A text description of the design goal.
            max_steps (int): The fixed length of a motif sequence for a design.
            reward_weights (Dict[str, float]): Weights for reward components, e.g.,
                {'stability': 100.0, 'print_time_penalty': -0.1}.
            output_dir (str): Directory path to store intermediate files.
            slicer_config (Dict[str, Any]): Configuration for the slicer verifier,
                e.g., {'slicer_path': '...', 'config_path': '...'}.
            physics_config (Dict[str, Any]): Configuration for the physics verifier,
                e.g., {'duration_steps': 2400, 'load_config': {...}}.
        """
        super().__init__()
        self.generator = generator
        self.num_motifs = num_motifs
        self.design_prompt = design_prompt
        self.max_steps = max_steps
        self.reward_weights = reward_weights
        self.output_dir = Path(output_dir)
        self.slicer_config = slicer_config
        self.physics_config = physics_config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_step: int = 0
        self.motif_sequence: List[int] = []

        # Define the action space: A discrete set of motif IDs.
        self.action_space = gym.spaces.Discrete(self.num_motifs)

        # Define the observation space.
        # This is a simplified representation. A real implementation would use a
        # proper text encoder for the prompt.
        prompt_emb_dim = getattr(self.generator.model.config, 'hidden_size', 768)
        self.observation_space = gym.spaces.Dict({
            "prompt_embedding": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(prompt_emb_dim,), dtype=np.float32
            ),
            "sequence": gym.spaces.Box(
                low=-1, high=self.num_motifs, shape=(self.max_steps,), dtype=np.int32
            )
        })

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the environment to an initial state for a new episode.

        Args:
            seed (Optional[int]): The seed for the random number generator.
            options (Optional[Dict[str, Any]]): E.g., {'design_prompt': 'new prompt'}.

        Returns:
            A tuple containing the initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)

        if options and 'design_prompt' in options:
            self.design_prompt = options['design_prompt']
            logger.info(f"New design prompt set: '{self.design_prompt}'")

        self.current_step = 0
        self.motif_sequence = []

        logger.debug(f"Environment reset for prompt: '{self.design_prompt}'")
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        The agent's action is appended to the sequence. If the sequence reaches
        max length, the generation/verification pipeline is triggered to get a
        final reward. Conforms to the modern gym API (gym > 0.26).

        Args:
            action (int): The motif ID selected by the agent.

        Returns:
            - observation (Dict): The observation after the step.
            - reward (float): The reward received from this step.
            - terminated (bool): Whether the episode ended naturally.
            - truncated (bool): Whether the episode was externally truncated.
            - info (Dict): Auxiliary diagnostic information.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. It is not in {self.action_space}")

        self.motif_sequence.append(action)
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        reward = 0.0
        info = {}

        if terminated:
            logger.debug(f"Episode terminated. Sequence: {self.motif_sequence}")
            reward, info = self._run_pipeline()

        observation = self._get_observation()
        # For compatibility, done = terminated or truncated
        return observation, reward, terminated, False, info

    def _run_pipeline(self) -> Tuple[float, Dict[str, Any]]:
        """Runs the full generation and verification pipeline."""
        session_id = str(uuid.uuid4())
        
        # 1. Generate CadQuery script
        cq_script = self.generator.generate(
            design_prompt=self.design_prompt,
            motif_ids=self.motif_sequence
        )
        
        # 2. Execute script to get STL file
        output_filename = session_id
        stl_path_str = str(self.output_dir / f"{output_filename}.stl")

        generated_stl_path = execute_cad_script(
            script_string=cq_script,
            output_dir=str(self.output_dir),
            output_filename=output_filename
        )

        if generated_stl_path is None:
            logger.warning(f"CSG execution failed for sequence {self.motif_sequence}")
            reward = self.reward_weights.get('failure_penalty', -500.0)
            info = {'reason': 'CSG script execution failed', 'cq_script': cq_script}
            return reward, info

        # 3. Verify with oracles
        slicer_metrics = get_slicer_metrics(stl_file_path=stl_path_str, **self.slicer_config)
        physics_results = verify_stability(stl_file_path=Path(stl_path_str), simulation_config=self.physics_config)

        # 4. Calculate composite reward
        reward = self._calculate_reward(slicer_metrics, physics_results)
        info = {
            'slicer_metrics': slicer_metrics,
            'physics_results': physics_results,
            'reward': reward,
            'cq_script': cq_script,
            'stl_path': stl_path_str
        }
        logger.debug(f"Evaluation results: {info}")
        return reward, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Constructs the current observation from the environment's state."""
        # Pad the motif sequence with -1.
        padded_sequence = np.full((self.max_steps,), -1, dtype=np.int32)
        if self.motif_sequence:
            padded_sequence[:len(self.motif_sequence)] = self.motif_sequence

        # Get prompt embedding (placeholder).
        prompt_emb_dim = self.observation_space['prompt_embedding'].shape[0]
        prompt_embedding = np.zeros(prompt_emb_dim, dtype=np.float32)

        return {
            "prompt_embedding": prompt_embedding,
            "sequence": padded_sequence
        }

    def _calculate_reward(self,
                          slicer_metrics: SlicerMetrics,
                          physics_results: Dict[str, Any]) -> float:
        """Calculates a composite reward based on oracle outputs."""
        if not slicer_metrics['slicing_successful'] or not physics_results.get('passed', False):
            return self.reward_weights.get('failure_penalty', -500.0)

        reward = self.reward_weights.get('success_bonus', 100.0)

        # Penalty for print time
        reward -= self.reward_weights.get('print_time_penalty', 0.01) * \
                  slicer_metrics.get('estimated_print_time_s', 0.0)

        # Penalty for support material
        reward -= self.reward_weights.get('support_material_penalty', 1.0) * \
                  slicer_metrics.get('support_material_volume_mm3', 0.0)
        
        # Penalty for total filament
        reward -= self.reward_weights.get('total_filament_penalty', 0.5) * \
                  slicer_metrics.get('total_filament_volume_mm3', 0.0)

        return float(reward)

    def render(self, mode: str = 'human') -> None:
        """Renders the environment's state to the console."""
        if mode == 'human':
            print("=" * 40)
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Design Prompt: {self.design_prompt}")
            print(f"Motif Sequence: {self.motif_sequence}")
            print("=" * 40)
        else:
            super().render(mode=mode)

    def close(self) -> None:
        """Performs any necessary cleanup."""
        logger.info("DesignEnvironment closed.")
        pass
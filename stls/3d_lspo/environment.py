# C:\Users\Machine81\Slazy\repo\stls\3d_lspo\environment.py

"""
Defines the main reinforcement learning environment for 3D model generation.

This module contains the `DesignEnvironment` class, which conforms to the
OpenAI Gym API. It orchestrates the process of generating a 3D model by
sequentially selecting "design motifs". Each step involves translating the
current motif sequence into a CAD script, executing it, and then using
slicer and physics oracles to verify the output and calculate a reward.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import uuid

import gym
import numpy as np

# Internal project imports - assuming `lspo_3d` is the package name.
# The prompt provides `3d_lspo` but project structure and other files suggest `lspo_3d`.
# Using `lspo_3d` as it seems more consistent with other provided skeletons.
try:
    from lspo_3d.models.generator import CadQueryGenerator
    from lspo_3d.oracles.csg_executor import execute_cad_script, CSGExecutionError
    from lspo_3d.oracles.slicer_verifier import get_slicer_metrics, SlicerMetrics
    from lspo_3d.oracles.physics_verifier import verify_stability
except ImportError:
    # Fallback to the provided import style for compatibility
    from 3d_lspo.models.generator import CadQueryGenerator
    from 3d_lspo.oracles.csg_executor import execute_cad_script, CSGExecutionError
    from 3d_lspo.oracles.slicer_verifier import get_slicer_metrics, SlicerMetrics
    from 3d_lspo.oracles.physics_verifier import verify_stability


DEFAULT_REWARD_WEIGHTS = {
    "success": 100.0,
    "csg_fail": -100.0,
    "slice_fail": -80.0,
    "physics_fail": -60.0,
    "penalty_support": -1.0,  # per mm^3
    "penalty_filament": -0.1, # per mm^3
    "penalty_time": -0.01,    # per second
}


class DesignEnvironment(gym.Env):
    """
    An OpenAI Gym-compliant environment for learning to generate 3D models.

    The environment manages the state, which consists of the design prompt and
    the sequence of design motifs selected so far. The action space is the
    set of available design motifs. At each step, the agent selects a motif,
    and the environment generates a corresponding 3D model, verifies it, and
    calculates a reward.

    Attributes:
        action_space (gym.spaces.Discrete): The space of all possible actions.
            Each action corresponds to selecting a design motif.
        observation_space (gym.spaces.Dict): The observation space, containing
            the design prompt embedding and the current sequence of motifs.
        generator (CadQueryGenerator): The model used to translate motif
            sequences into CadQuery scripts.
        output_dir (Path): The directory to save temporary files like generated
            scripts and .stl models.
        max_steps (int): The maximum number of motifs that can be in a sequence.
        design_prompt (str): The high-level design goal for the current episode.
        current_step (int): The current step number within the episode.
        motif_sequence (List[int]): The sequence of motif IDs selected so far
            in the current episode.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 design_prompt: str,
                 motif_ids: List[int],
                 generator_model: CadQueryGenerator,
                 slicer_executable_path: str,
                 physics_config: Dict[str, Any],
                 output_dir: str = "C:\\Users\\Machine81\\Slazy\\repo\\stls\\output\\episodes",
                 max_steps: int = 15,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initializes the DesignEnvironment.

        Args:
            design_prompt (str): A natural language description of the design
                goal (e.g., "Design a vertical stand for an iPhone 15").
            motif_ids (List[int]): A list of all unique motif IDs that the agent
                can select from.
            generator_model (CadQueryGenerator): An instantiated generator model
                for translating motifs to CadQuery code.
            slicer_executable_path (str): Path to the slicer command-line executable.
            physics_config (Dict[str, Any]): Configuration dictionary for the
                physics simulation oracle.
            output_dir (str): Path to the directory for storing intermediate
                and final model files.
            max_steps (int): The maximum number of steps (motifs) per episode.
            reward_weights (Optional[Dict[str, float]]): Dictionary to configure
                reward calculation.
        """
        super(DesignEnvironment, self).__init__()

        self.design_prompt = design_prompt
        self.generator = generator_model
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.all_motif_ids = motif_ids
        self.slicer_executable_path = Path(slicer_executable_path)
        self.physics_config = physics_config
        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS

        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(self.all_motif_ids))

        # The observation space. The prompt is pre-processed into a fixed-size
        # embedding. The motif sequence is a variable-length list of integers.
        # NOTE: The shape for 'prompt_embedding' (768) is a common size for
        # BERT-like models. Ensure the agent's preprocessor matches this.
        self.observation_space = gym.spaces.Dict({
            "prompt_embedding": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
            "motif_sequence": gym.spaces.Sequence(gym.spaces.Discrete(len(self.all_motif_ids) + 1)) # +1 to allow for a large vocab
        })

        # State variables for the current episode
        self.current_step: int = 0
        self.motif_sequence: List[int] = []
        self.episode_id: str = ""

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.

        This involves:
        1. Appending the chosen action (motif ID) to the current sequence.
        2. Calling the generator to create a CadQuery script.
        3. Executing the script to generate an STL file.
        4. Running slicer and physics verifications on the STL file.
        5. Calculating a composite reward based on the verification results.
        6. Determining if the episode is finished.

        Args:
            action (int): The index corresponding to the chosen motif ID.

        Returns:
            A tuple containing:
            - observation (Dict[str, Any]): The next state of the environment.
            - reward (float): The reward for the taken action.
            - done (bool): Whether the episode has terminated.
            - info (Dict[str, Any]): A dictionary with auxiliary diagnostic
              information from the oracles.
        """
        self.current_step += 1
        chosen_motif_id = self.all_motif_ids[action]
        self.motif_sequence.append(chosen_motif_id)

        info: Dict[str, Any] = {}
        done = False

        # --- Generation & Verification Pipeline ---
        # 1. Generate CadQuery script
        cad_script = self.generator.generate(self.design_prompt, self.motif_sequence)
        info['cad_script'] = cad_script

        # 2. Execute script to get STL
        stl_path = self.output_dir / f"{self.episode_id}_step_{self.current_step}.stl"
        csg_success, csg_error = execute_cad_script(cad_script, stl_path)
        info['csg_success'] = csg_success
        info['csg_error'] = csg_error

        slicer_metrics: Optional[SlicerMetrics] = None
        physics_results: Optional[Dict[str, Any]] = None

        if csg_success:
            # 3. Get slicer metrics
            slicer_metrics = get_slicer_metrics(stl_path, self.slicer_executable_path)
            info['slicer_metrics'] = slicer_metrics

            if slicer_metrics.slicing_successful:
                # 4. Get physics verification results
                physics_results = verify_stability(stl_path, self.physics_config)
                info['physics_results'] = physics_results

        # --- Reward Calculation & Termination ---
        physics_passed = physics_results.get('passed_stability_check', False) if physics_results else False
        reward = self._calculate_reward(csg_success, slicer_metrics, physics_passed)
        
        # Check termination conditions
        if not csg_success or (slicer_metrics and not slicer_metrics.slicing_successful):
            done = True
        if self.current_step >= self.max_steps:
            done = True
            # Final reward might depend on final state being stable
            if not physics_passed:
                 reward = self.reward_weights['physics_fail']

        observation = self._get_obs()

        return observation, reward, done, info

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment to its initial state for a new episode.

        Clears the current motif sequence and resets the step counter.

        Returns:
            Dict[str, Any]: The initial observation of the new episode.
        """
        self.current_step = 0
        self.motif_sequence = []
        self.episode_id = str(uuid.uuid4())[:8] # Unique ID for this episode's files
        return self._get_obs()

    def render(self, mode: str = 'human') -> None:
        """
        Renders the environment.

        Args:
            mode (str): The mode to render with. Currently unused.
        """
        print(f"--- Episode {self.episode_id}, Step: {self.current_step} ---")
        print(f"Motif Sequence: {self.motif_sequence}")
        latest_stl = self.output_dir / f"{self.episode_id}_step_{self.current_step}.stl"
        if latest_stl.exists():
            print(f"Last generated model: {latest_stl}")
        else:
            print("No model generated in this step.")
        print("-" * (len(self.episode_id) + 20))

    def close(self) -> None:
        """
        Performs any necessary cleanup.
        """
        print("Closing DesignEnvironment.")
        pass

    def _get_obs(self) -> Dict[str, Any]:
        """
        Constructs the observation dictionary from the current state.

        Returns:
            Dict[str, Any]: The current observation.
        """
        # In a real implementation, a text encoder (e.g., Sentence-BERT) would
        # produce this embedding from self.design_prompt. For this environment,
        # we provide a zero vector as a placeholder. The agent's wrapper or
        # policy network is expected to handle the actual encoding.
        prompt_shape = self.observation_space["prompt_embedding"].shape
        prompt_embedding = np.zeros(prompt_shape, dtype=np.float32)

        # The observation must match the defined observation_space structure.
        obs = {
            "prompt_embedding": prompt_embedding,
            "motif_sequence": tuple(self.motif_sequence) # Sequence space expects a tuple
        }
        return obs

    def _calculate_reward(self,
                          csg_success: bool,
                          slicer_metrics: Optional[SlicerMetrics],
                          physics_passed: Optional[bool]) -> float:
        """
        Calculates a composite reward from the outputs of the oracles.

        A large positive reward is given for a successful slice and passing
        the physics test. Penalties are applied for failures at any stage,
        and for inefficient designs (e.g., high support material).

        Args:
            csg_success (bool): Whether the CadQuery script executed without errors.
            slicer_metrics (Optional[SlicerMetrics]): The output from the
                slicer verifier.
            physics_passed (Optional[bool]): The result of the physics simulation.

        Returns:
            float: The calculated composite reward.
        """
        if not csg_success:
            return self.reward_weights['csg_fail']

        if slicer_metrics is None or not slicer_metrics.slicing_successful:
            return self.reward_weights['slice_fail']

        # If execution reaches here, the model is at least manufacturable.
        # Now evaluate its quality.
        base_reward = self.reward_weights['success']

        # Penalize for inefficiency based on slicer metrics
        support_penalty = self.reward_weights['penalty_support'] * (slicer_metrics.support_material_volume_mm3 or 0.0)
        filament_penalty = self.reward_weights['penalty_filament'] * (slicer_metrics.filament_volume_mm3 or 0.0)
        time_penalty = self.reward_weights['penalty_time'] * (slicer_metrics.print_time_seconds or 0.0)
        
        total_reward = base_reward + support_penalty + filament_penalty + time_penalty

        # If the model is not physically functional, apply a large penalty.
        # This is checked last because a nonsensical shape might still slice.
        if physics_passed is None or not physics_passed:
            total_reward = self.reward_weights['physics_fail']

        return total_reward
"""
config.py

Centralizes all hyperparameters and configuration settings for the 3D-LSPO project.

This module uses dataclasses to define a hierarchical and type-safe configuration
structure. All parts of the application should import their required settings from
the `get_config()` instance defined at the end of this file. This approach
avoids magic numbers and scattered constants, making the system easier to
understand, modify, and tune.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Base directory of the project
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass(frozen=True)
class PathsConfig:
    """
    Configuration for all project-related file and directory paths.
    """
    # Base directory for all data
    data_dir: Path = PROJECT_ROOT / "data"
    # Directory for raw, unprocessed model data (e.g., JSONs of cadquery commands)
    raw_models_dir: Path = data_dir / "raw_models"
    # Directory for processed and tokenized sequences ready for training
    processed_sequences_dir: Path = data_dir / "processed_sequences"

    # Base directory for all generated outputs
    output_dir: Path = PROJECT_ROOT / "output"
    # Directory to save generated STL files during training and inference
    generated_stls_dir: Path = output_dir / "generated_stls"
    # Directory for logs (training, evaluation, etc.)
    logs_dir: Path = output_dir / "logs"
    # Directory to save model checkpoints
    checkpoints_dir: Path = output_dir / "checkpoints"

    # Full path to the PrusaSlicer command-line executable.
    # Note: This must be set by the user to a valid path.
    # Example: "C:/Program Files/Prusa3D/PrusaSlicer/prusa-slicer-console.exe"
    prusa_slicer_path: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for the design motif embedding model and clustering.
    """
    # Pre-trained sentence transformer model for embedding cadquery sequences
    model_name: str = "all-MiniLM-L6-v2"
    # The number of clusters (design motifs) to create using k-means
    num_clusters: int = 50
    # Batch size for generating embeddings
    batch_size: int = 32


@dataclass(frozen=True)
class GeneratorConfig:
    """
    Configuration for the T5 Generator model.
    """
    # Pre-trained T5 model to be fine-tuned as the generator
    model_name: str = "t5-small"
    # Maximum sequence length for the generator's input and output
    max_seq_length: int = 512
    # Learning rate for Direct Preference Optimization (DPO) fine-tuning
    dpo_learning_rate: float = 1e-5
    # Batch size for DPO fine-tuning
    dpo_batch_size: int = 4
    # Beta parameter for DPO loss
    dpo_beta: float = 0.1


@dataclass(frozen=True)
class AgentConfig:
    """
    Configuration for the PPO Reinforcement Learning agent.
    """
    # Learning rate for the Adam optimizer
    learning_rate: float = 3e-4
    # Discount factor for future rewards
    gamma: float = 0.99
    # Lambda for Generalized Advantage Estimation (GAE)
    gae_lambda: float = 0.95
    # Clipping parameter for the PPO policy loss
    clip_param: float = 0.2
    # Number of training epochs per policy update
    ppo_epochs: int = 10
    # Number of samples in a mini-batch for PPO updates
    mini_batch_size: int = 64
    # Dimensions of the hidden layers in the actor (policy) network
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    # Dimensions of the hidden layers in the critic (value) network
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    # The number of discrete actions (corresponds to num_clusters in EmbeddingConfig)
    action_space_size: int = 50


@dataclass(frozen=True)
class RewardWeights:
    """
    Weights for calculating the composite reward score from the Oracle.
    """
    # Reward for a successfully generated manifold mesh
    printability_success: float = 10.0
    # Penalty for a failed (non-manifold) mesh generation
    printability_failure: float = -10.0
    # Weight for the structural stability score from the physics simulation
    stability_weight: float = 1.0
    # Negative weight (penalty) for filament usage (in mm^3)
    # A smaller value encourages exploration of larger but potentially more stable designs.
    filament_usage_weight: float = -0.001
    # Negative weight (penalty) for support material usage (in mm^3)
    # A larger penalty to discourage designs that are hard to print.
    support_material_weight: float = -0.005


@dataclass(frozen=True)
class PhysicsSimConfig:
    """
    Configuration for the PyBullet physics simulation.
    """
    # Force to apply to the model in the simulation (in Newtons)
    applied_load: float = 1.0  # e.g., representing a ~100g phone's weight
    # Location on the model where the load is applied (x, y, z) offset from center
    load_position_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 10.0])
    # Number of simulation steps to run to check for stability
    sim_steps: int = 240 * 5  # 5 seconds at the default 240Hz physics rate


@dataclass(frozen=True)
class SlicerConfig:
    """
    Configuration for the PrusaSlicer CLI tool.
    """
    # Path to a specific PrusaSlicer configuration file (.ini).
    # If None, default slicer settings are used.
    slicer_profile: Optional[str] = None
    # Timeout in seconds for the slicer process
    timeout: int = 120


@dataclass(frozen=True)
class OracleConfig:
    """
    Configuration for the evaluation Oracle.
    """
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    physics_sim: PhysicsSimConfig = field(default_factory=PhysicsSimConfig)
    slicer: SlicerConfig = field(default_factory=SlicerConfig)


@dataclass(frozen=True)
class TrainerConfig:
    """
    Configuration for the main training loop.
    """
    # Total number of agent-environment interaction steps for training
    total_timesteps: int = 1_000_000
    # The number of steps to run for each environment per update (PPO rollout buffer size)
    num_steps_per_update: int = 2048
    # Log training progress every N updates
    log_interval: int = 1
    # Save a model checkpoint every N updates
    save_interval: int = 10


@dataclass(frozen=True)
class Config:
    """
    Root configuration class for the 3D-LSPO project.
    """
    project_name: str = "3D-LSPO"
    # Seed for random number generators to ensure reproducibility
    seed: int = 42

    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self):
        """
        Perform validation and create directories after initialization.
        """
        # Create all necessary directories if they don't exist
        os.makedirs(self.paths.raw_models_dir, exist_ok=True)
        os.makedirs(self.paths.processed_sequences_dir, exist_ok=True)
        os.makedirs(self.paths.generated_stls_dir, exist_ok=True)
        os.makedirs(self.paths.logs_dir, exist_ok=True)
        os.makedirs(self.paths.checkpoints_dir, exist_ok=True)


_config_instance: Optional[Config] = None

def get_config() -> Config:
    """
    Returns a singleton instance of the project configuration.

    This function ensures that the configuration is instantiated only once
    and can be accessed globally from any part of the application.

    Returns:
        Config: The frozen, singleton configuration object.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
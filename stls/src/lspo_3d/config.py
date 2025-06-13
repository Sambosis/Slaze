# -*- coding: utf-8 -*-
"""
Centralized configuration for the 3D-LSPO project.

This file contains all hyperparameters, file paths, model dimensions, and
reward weights used throughout the project. It is designed to be a "flattened"
configuration, meaning all parameters are defined as top-level constants for
easy import and access from any module.

Example Usage:
    from src.lspo_3d import config
    learning_rate = config.PPO_POLICY_LR
    data_path = config.SCAD_CORPUS_DIR
"""

from pathlib import Path
from typing import Final, Literal

# -----------------------------------------------------------------------------
# General Project & Path Configuration
# -----------------------------------------------------------------------------

#: The root directory of the project. Used to build absolute paths.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

#: Directory to store all generated artifacts (models, logs, intermediate files).
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

#: Directory containing the corpus of source .scad files.
# Users should place their .scad files in a 'data/scad_corpus' subdirectory.
SCAD_CORPUS_DIR: Final[Path] = PROJECT_ROOT / "data" / "scad_corpus"

#: Path to the main log file for the project.
LOG_FILE: Final[Path] = ARTIFACTS_DIR / "lspo_3d.log"

PAD_TOKEN: Final[str] = "[PAD]"
UNK_TOKEN: Final[str] = "[UNK]"
BOS_TOKEN: Final[str] = "[BOS]"
TOKENIZER_VOCAB_SIZE: Final[int] = 5000
TOKENIZER_MIN_FREQUENCY: Final[int] = 2

EOS_TOKEN: Final[str] = "[EOS]"

#: Full path to the OpenSCAD executable.
# IMPORTANT: This path must be changed to match your local installation.
# Windows example: "C:/Program Files/OpenSCAD/openscad.exe"
# macOS example: "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
OPENSCAD_PATH: Final[str] = "/usr/bin/openscad"  # Example for Linux

#: Full path to the PrusaSlicer console app executable.
# IMPORTANT: This path must be changed to match your local installation.
# Windows example: "C:/Program Files/Prusa3D/PrusaSlicer/prusa-slicer-console.exe"
# macOS example: "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"
PRUSA_SLICER_PATH: Final[str] = "/usr/bin/prusa-slicer-console" # Example for Linux


# -----------------------------------------------------------------------------
# Data & Tokenizer Configuration
# -----------------------------------------------------------------------------

#: The maximum vocabulary size for the CSGTokenizer.
VOCAB_SIZE: Final[int] = 5000

#: The maximum number of tokens in a sequence for model processing.
MAX_SEQUENCE_LENGTH: Final[int] = 1024

#: Path to save the trained tokenizer's vocabulary file.
TOKENIZER_PATH: Final[Path] = ARTIFACTS_DIR / "csg_tokenizer.json"


# -----------------------------------------------------------------------------
# Encoder Model Hyperparameters (Transformer Encoder)
# -----------------------------------------------------------------------------

#: Path to save/load the trained CSGEncoder model weights.
ENCODER_MODEL_PATH: Final[Path] = ARTIFACTS_DIR / "csg_encoder.pth"

#: The dimensionality of the token embeddings and the model's hidden states.
ENCODER_EMBEDDING_DIM: Final[int] = 512

#: Number of attention heads in the multi-head attention mechanism.
ENCODER_NUM_HEADS: Final[int] = 8

#: Number of Transformer encoder layers.
ENCODER_NUM_LAYERS: Final[int] = 6

#: Dimensionality of the feed-forward layer within the Transformer encoder.
ENCODER_FF_DIM: Final[int] = 2048

#: Dropout probability for regularization in the encoder.
ENCODER_DROPOUT: Final[float] = 0.1


# -----------------------------------------------------------------------------
# Latent Space (K-Means) Configuration
# -----------------------------------------------------------------------------

#: The number of clusters for K-means, representing the number of "design motifs".
NUM_MOTIFS: Final[int] = 128

#: Path to save the numpy array of cluster centroids (the motifs).
MOTIF_CENTROIDS_PATH: Final[Path] = ARTIFACTS_DIR / "motif_centroids.npy"

#: Path to cache the generated embeddings for the entire dataset to speed up clustering.
EMBEDDINGS_CACHE_PATH: Final[Path] = ARTIFACTS_DIR / "scad_embeddings.npy"


# -----------------------------------------------------------------------------
# Generator Model Hyperparameters (GPT-2 Style Decoder)
# -----------------------------------------------------------------------------

#: Path to save/load the trained CSGGenerator model weights.
GENERATOR_MODEL_PATH: Final[Path] = ARTIFACTS_DIR / "csg_generator.pth"

#: The dimensionality of the token embeddings for the generator.
GENERATOR_EMBEDDING_DIM: Final[int] = 512

#: Number of attention heads in the generator's multi-head attention mechanism.
GENERATOR_NUM_HEADS: Final[int] = 8

#: Number of Transformer decoder layers in the generator.
GENERATOR_NUM_LAYERS: Final[int] = 6

#: Dimensionality of the feed-forward layer within the Transformer decoder.
GENERATOR_FF_DIM: Final[int] = 2048

#: Dropout probability for regularization in the generator.
GENERATOR_DROPOUT: Final[float] = 0.1


# -----------------------------------------------------------------------------
# Reward Oracle Configuration
# -----------------------------------------------------------------------------

#: Directory to store temporary .stl files generated for evaluation.
TEMP_STL_DIR: Final[Path] = ARTIFACTS_DIR / "temp_stl"
TEMP_STL_DIR.mkdir(parents=True, exist_ok=True)

#: Directory to store temporary G-code output from the slicer.
TEMP_GCODE_DIR: Final[Path] = ARTIFACTS_DIR / "temp_gcode"
TEMP_GCODE_DIR.mkdir(parents=True, exist_ok=True)

#: Reward weight for successful slicing (binary reward).
REWARD_WEIGHT_PRINTABILITY: Final[float] = 10.0

#: Reward weight (penalty) for the amount of required support material.
REWARD_WEIGHT_SUPPORT: Final[float] = -1.5

#: Reward weight for material efficiency (inversely proportional to filament volume).
REWARD_WEIGHT_MATERIAL: Final[float] = 0.5

#: Reward weight for the physics simulation stability score.
REWARD_WEIGHT_STABILITY: Final[float] = 20.0


# -----------------------------------------------------------------------------
# RL Agent (PPO) Hyperparameters
# -----------------------------------------------------------------------------

#: Path to save/load the trained PPO agent models (policy and value networks).
PPO_AGENT_PATH: Final[Path] = ARTIFACTS_DIR / "ppo_agent.pth"

#: The maximum length of a motif sequence the agent can generate.
MAX_MOTIF_SEQUENCE_LENGTH: Final[int] = 10

#: Learning rate for the PPO policy network optimizer.
PPO_POLICY_LR: Final[float] = 3e-4

#: Learning rate for the PPO value network optimizer.
PPO_VALUE_LR: Final[float] = 1e-3

#: Discount factor for future rewards.
PPO_GAMMA: Final[float] = 0.99

#: Clipping parameter for the PPO policy update.
PPO_EPSILON_CLIP: Final[float] = 0.2

#: Number of update epochs to run on a collected batch of experience.
PPO_UPDATE_EPOCHS: Final[int] = 10


# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------

#: Global batch size for training all neural networks.
BATCH_SIZE: Final[int] = 32

#: Learning rate for the encoder and generator pre-training.
LEARNING_RATE: Final[float] = 1e-4

#: Number of epochs for training the CSG Encoder (script 01).
NUM_ENCODER_TRAIN_EPOCHS: Final[int] = 5

#: Number of epochs for pre-training the CSG Generator (script 02).
NUM_GENERATOR_PRETRAIN_EPOCHS: Final[int] = 10

#: Total number of training iterations in the main LSPO loop (script 03).
NUM_LSPO_ITERATIONS: Final[int] = 1000

#: Number of environment steps to collect for each PPO update.
PPO_STEPS_PER_BATCH: Final[int] = 2048

#: Device to use for training ('cuda' or 'cpu'). Will auto-detect torch.cuda.
DEVICE: Final[Literal["cuda", "cpu"]] = "cuda"
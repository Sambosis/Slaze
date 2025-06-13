"""

3D-LSPO Package Initialization.

This file marks the 'lspo_3d' directory as a Python package and exposes key
components from its submodules for easier access throughout the project. By
importing the main classes and functions here, users can access them directly
from the `lspo_3d` namespace instead of needing to know the detailed internal
structure.

For example, instead of `from lspo_3d.data.tokenizer import CSGTokenizer`, one can
simply use `from lspo_3d import CSGTokenizer`.

Attributes:
    __version__ (str): The current version of the lspo_3d package.
"""

# Standard library imports
import logging

# Define package version
__version__ = "0.1.0"

# Import key components from submodules to expose them at the package level
from .data.dataset import CSGDataset
from .data.tokenizer import CSGTokenizer
from .models.encoder import CSGEncoder
from .models.generator import CSGGenerator
from .oracles.physics import PhysicsOracle
from .oracles.reward import calculate_reward
from .oracles.slicer import SlicerOracle
from .rl.agent import PPOAgent
from .rl.environment import DesignEnvironment

# Define the public API of the package
__all__ = [
    "CSGDataset",
    "CSGTokenizer",
    "CSGEncoder",
    "CSGGenerator",
    "PhysicsOracle",
    "SlicerOracle",
    "calculate_reward",
    "PPOAgent",
    "DesignEnvironment",
]

# Configure a basic logger for the package
# This allows other modules to get a logger instance with `logging.getLogger(__name__)`
# and have it configured consistently.
logging.getLogger(__name__).addHandler(logging.NullHandler())
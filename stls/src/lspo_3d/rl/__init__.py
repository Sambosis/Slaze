# src/lspo_3d/rl/__init__.py
"""
This package contains modules for Reinforcement Learning within the 3D-LSPO project.

It provides the PPO agent for policy learning and the custom gym-style environment
that facilitates the interaction between the agent and the design generation/evaluation
pipeline.

This __init__.py file exposes the key classes from the submodules for convenient
access, making them available directly under the `lspo_3d.rl` namespace.
"""

from typing import List

from .agent import PPOAgent
from .environment import DesignEnvironment

__all__: List[str] = [
    "PPOAgent",
    "DesignEnvironment",
]
"""
AI Package
==========

This package contains the artificial intelligence and reinforcement learning components 
for the "Strategy War Game with Curiosity-Driven Self-Play" project.

It implements the Double Deep Q-Network (Double DQN) architecture augmented with an 
Intrinsic Curiosity Module (ICM) to facilitate self-play training and exploration.

Exposed Classes & Functions:
    - Agent: The main RL agent handling action selection and model updates.
    - DQN: The Deep Q-Network policy model.
    - ICMModel: The Intrinsic Curiosity Module for exploration rewards.
    - ReplayBuffer: Experience replay memory for training.
    - calculate_curiosity: Utility to compute intrinsic rewards from prediction error.
"""

from .models import DQN, ICMModel
from .agent import Agent
from .utils import ReplayBuffer, calculate_curiosity

__all__ = [
    "DQN",
    "ICMModel",
    "Agent",
    "ReplayBuffer",
    "calculate_curiosity",
]
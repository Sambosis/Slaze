# -*- coding: utf-8 -*-
"""
3D-LSPO: Latent Space Policy Optimization for 3D Generative Design.

This package provides the core components for the 3D-LSPO project, an AI system
that generates novel, functional, and optimized 3D-printable models by
learning from existing designs.

The top-level package namespace aggregates the most important classes and functions
from the sub-packages for convenient access.
"""

__author__ = "3D-LSPO Project Team"
__version__ = "0.1.0"
__license__ = "MIT"


# --- Aggregate key components from sub-packages ---

# From the data processing module
from .data.processor import process_raw_models

# From the models sub-package
from .models.motif_encoder import MotifEncoder
from .models.generator import CadQueryGenerator
from .models.agent import AgentPolicy

# From the oracles sub-package
from .oracles.csg_executor import execute_cad_script
from .oracles.slicer_verifier import get_slicer_metrics
from .oracles.physics_verifier import verify_stability

# From the RL environment module
from .environment import DesignEnvironment


# --- Define the public API for the package ---

__all__ = [
    # Data Processing
    'process_raw_models',

    # Models
    'MotifEncoder',
    'CadQueryGenerator',
    'AgentPolicy',

    # Oracles
    'execute_cad_script',
    'get_slicer_metrics',
    'verify_stability',

    # Environment
    'DesignEnvironment',
]
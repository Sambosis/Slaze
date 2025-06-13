# /3d_lspo/oracles/__init__.py
"""
Initializes the Oracles sub-package for the 3D-LSPO project.

This file makes the 'oracles' directory a Python sub-package and exposes its
key functionalities for easy access from other parts of the application. The
oracles are responsible for verifying the manufacturability and functional
viability of the generated 3D designs.

The exposed components are:
- execute_cad_script: Executes a CadQuery script to generate a 3D model.
- verify_stability: Uses a physics engine to test the functional stability of a model.
- get_slicer_metrics: Uses a 3D slicer to check for printability and extract metrics.
"""

from .csg_executor import execute_cad_script
from .physics_verifier import verify_stability
from .slicer_verifier import get_slicer_metrics

__all__ = [
    "execute_cad_script",
    "verify_stability",
    "get_slicer_metrics",
]
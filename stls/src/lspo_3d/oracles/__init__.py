# src/lspo_3d/oracles/__init__.py
"""
Initializes the 'oracles' subpackage.

This file makes the 'src/lspo_3d/oracles' directory a Python package, allowing
for the organized import of its modules. The oracles are responsible for
evaluating the quality of generated 3D designs based on various criteria like
printability and physical stability.

This __init__.py also facilitates easier imports of the key components from
the sibling modules within this package. For example, instead of:
`from lspo_3d.oracles.physics import PhysicsOracle`
one can use:
`from lspo_3d.oracles import PhysicsOracle`

"""

from .physics import PhysicsOracle
from .slicer import SlicerOracle
from .reward import calculate_reward

__all__ = [
    "PhysicsOracle",
    "SlicerOracle",
    "calculate_reward",
]
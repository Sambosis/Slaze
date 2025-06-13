"""
3D-LSPO Package Initialization.

This file marks the 'lspo' directory as a Python package, allowing for modular
imports of its components. It may also expose key classes or functions for
easier access from outside the package.
"""

# Define package-level metadata
__version__ = "0.1.0"
__author__ = "3D-LSPO Development Team"
__email__ = "contact@example.com"


# The presence of this file is sufficient to make 'lspo' a package.
# No other code is strictly necessary for this file's primary purpose.
# Future additions could include selectively importing key components
# from submodules to provide a simplified public API for the package, e.g.:
#
# from .trainer import LSPOTrainer
# from .oracle import Oracle
#
# For now, it will be kept minimal.
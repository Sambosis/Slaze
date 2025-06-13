"""
3D-LSPO: 3D Latent Space Policy Optimization Project.

This __init__.py file serves two primary purposes:
1. It marks the '3d_lspo' directory as a Python package, allowing its modules
   to be imported from other parts of the project or by external applications.
2. It can be used to define package-level variables, such as the version number,
   or to expose key components from sub-packages for convenient access.

Example:
    import lspo_3d
    print(lspo_3d.__version__)
"""

# Package-level metadata
__version__ = "0.1.0"
__author__ = "The 3D-LSPO Development Team"
__email__ = "contact@example.com"  # Replace with a real contact email

# By leaving the rest of this file empty, we encourage explicit imports
# from submodules, e.g., `from 3d_lspo.data.processor import ...`.
# This can help avoid circular dependency issues and keeps the namespace clean.

pass
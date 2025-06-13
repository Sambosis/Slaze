"""
Makes the 'models' directory a Python package and exposes its public
interfaces for convenient importing.

This allows users to import model classes directly from the 'lspo_3d.models'
namespace instead of from the specific module files.

Example Usage:
    from lspo_3d.models import CSGEncoder, CSGGenerator
"""

from .encoder import CSGEncoder
from .generator import CSGGenerator

__all__ = [
    "CSGEncoder",
    "CSGGenerator",
]
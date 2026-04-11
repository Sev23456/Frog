"""
BioFrog v2.0 - Биологически достоверный агент лягушки
Переписан с 0 на основе ARCHITECTURE_COMPLETE.md
"""

__version__ = "2.0.0"
__author__ = "BioFrog Team"

from .bio_frog_agent import BioFrogAgent, BioFrogBrain
from .simulation import run_simulation

__all__ = [
    'BioFrogAgent',
    'BioFrogBrain', 
    'run_simulation'
]

"""
BioFrog package for the toy bio-inspired frog agent.

The package exposes the reusable simulation pieces without printing banners
on import, which keeps tests and CLI tools quiet and predictable.
"""

from .bio_frog_agent import BioFrogAgent, BioFrogBrain
from .simulation import BioFlyCatchingSimulation, Fly

from .core.biological_neuron import (
    LIFNeuron,
    PyramidalNeuron,
    FastSpikingInterneuron,
)
from .core.synapse_models import BiologicalSynapse, DynamicSynapse
from .core.glial_cells import Astrocyte, GlialNetwork
from .core.neurotransmitter_diffusion import (
    NeurotransmitterDiffusion,
    MultiNeurotransmitterSystem,
)

from .architecture.visual_system import CenterSurroundFilter, RetinalProcessing
from .architecture.tectum import TectalColumn, Tectum
from .architecture.motor_hierarchy import MotorHierarchy

from .metabolism.systemic_metabolism import NeuronMetabolism, SystemicMetabolism

from .plasticity.functional_plasticity import FunctionalPlasticityManager
from .plasticity.structural_plasticity import StructuralPlasticityManager

__version__ = "2.0"
__author__ = "BioFrog Team"
__all__ = [
    "BioFrogAgent",
    "BioFrogBrain",
    "BioFlyCatchingSimulation",
    "Fly",
    "LIFNeuron",
    "PyramidalNeuron",
    "FastSpikingInterneuron",
    "BiologicalSynapse",
    "DynamicSynapse",
    "Astrocyte",
    "GlialNetwork",
    "NeurotransmitterDiffusion",
    "MultiNeurotransmitterSystem",
    "CenterSurroundFilter",
    "RetinalProcessing",
    "TectalColumn",
    "Tectum",
    "MotorHierarchy",
    "NeuronMetabolism",
    "SystemicMetabolism",
    "FunctionalPlasticityManager",
    "StructuralPlasticityManager",
]

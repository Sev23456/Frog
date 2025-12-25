# -*- coding: utf-8 -*-
"""
BioFrog v2.0 - Биологически достоверная симуляция нейросети лягушки

Полностью переработанная версия с реалистичными биологическими компонентами:
- Диффузионные модели нейромодуляторов
- Глиальные клетки и их влияние на синаптическую передачу
- Разнообразие типов нейронов (LIF, пирамидальные, быстро спайкающие)
- Метаболическая модель энергопотребления
- Анатомически точная архитектура (тектум, ретинальная обработка)
- Структурная пластичность (создание/удаление синапсов и нейронов)
- Рецептивные поля с центр-периферической организацией
"""

__version__ = "2.0.0"
__author__ = "BioFrog Development Team"

from .core.biological_neuron import LIFNeuron, PyramidalNeuron, FastSpikingInterneuron
from .core.synapse_models import BiologicalSynapse
from .core.glial_cells import Astrocyte, GlialNetwork
from .core.neurotransmitter_diffusion import NeurotransmitterDiffusion

from .architecture.visual_system import RetinalProcessing
from .architecture.tectum import Tectum, TectalColumn
from .architecture.motor_hierarchy import MotorHierarchy

from .metabolism import NeuronMetabolism
from .metabolism.systemic_metabolism import SystemicMetabolism

from .plasticity.structural_plasticity import StructuralPlasticityManager
from .plasticity.functional_plasticity import FunctionalPlasticityManager

from .bio_frog_agent import BioFrogAgent
from .simulation import BioFlyCatchingSimulation

__all__ = [
    'LIFNeuron', 'PyramidalNeuron', 'FastSpikingInterneuron',
    'BiologicalSynapse',
    'Astrocyte', 'GlialNetwork',
    'NeurotransmitterDiffusion',
    'RetinalProcessing',
    'Tectum', 'TectalColumn',
    'MotorHierarchy',
    'NeuronMetabolism',
    'SystemicMetabolism',
    'StructuralPlasticityManager',
    'FunctionalPlasticityManager',
    'BioFrogAgent',
    'BioFlyCatchingSimulation',
]

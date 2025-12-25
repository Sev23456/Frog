"""
╔════════════════════════════════════════════════════════════════════════════╗
║                 BioFrog v2.0 - ПАКЕТ БИОЛОГИЧЕСКОЙ НЕЙРОСЕТИ             ║
║   Полностью интегрированная система биологически достоверной симуляции     ║
╚════════════════════════════════════════════════════════════════════════════╝

Основные компоненты:
  • bio_frog_agent.py - Главный класс интегрированного агента
  • simulation.py - Симуляция и визуализация

Вспомогательные модули:
  • core/ - Биологические нейроны, синапсы, глия
  • architecture/ - Визуальная система, тектум, моторика
  • metabolism/ - Энергетика и циркадные ритмы
  • plasticity/ - Пластичность синапсов
"""

from .bio_frog_agent import BioFrogAgent, BioFrogBrain
from .simulation import BioFlyCatchingSimulation, Fly

# Импорт основных компонентов из подмодулей
from .core.biological_neuron import (
    LIFNeuron, 
    PyramidalNeuron, 
    FastSpikingInterneuron
)
from .core.synapse_models import BiologicalSynapse, DynamicSynapse
from .core.glial_cells import Astrocyte, GlialNetwork
from .core.neurotransmitter_diffusion import (
    NeurotransmitterDiffusion, 
    MultiNeurotransmitterSystem
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
    # Интеграция
    'BioFrogAgent',
    'BioFrogBrain',
    'BioFlyCatchingSimulation',
    'Fly',
    
    # Core
    'LIFNeuron',
    'PyramidalNeuron',
    'FastSpikingInterneuron',
    'BiologicalSynapse',
    'DynamicSynapse',
    'Astrocyte',
    'GlialNetwork',
    'NeurotransmitterDiffusion',
    'MultiNeurotransmitterSystem',
    
    # Architecture
    'CenterSurroundFilter',
    'RetinalProcessing',
    'TectalColumn',
    'Tectum',
    'MotorHierarchy',
    
    # Metabolism
    'NeuronMetabolism',
    'SystemicMetabolism',
    
    # Plasticity
    'FunctionalPlasticityManager',
    'StructuralPlasticityManager',
]

try:
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   BioFrog v2.0 инициализирован                            ║
║          Все 11 биологических компонентов загружены и готовы              ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
except UnicodeEncodeError:
    # Fallback для системы с ограниченной поддержкой Unicode
    pass

# -*- coding: utf-8 -*-
"""
Структурная и функциональная пластичность

Включает:
- StructuralPlasticityManager: управление созданием/удалением синапсов
- FunctionalPlasticityManager: функциональные изменения в сети
"""

import numpy as np
from typing import List, Optional


class StructuralPlasticityManager:
    """Управление структурными изменениями в нейронной сети"""
    
    def __init__(self):
        self.synapse_creation_threshold = 0.1
        self.synapse_elimination_threshold = 0.01
        self.synapse_creation_rate = 0.0001
        self.synapse_elimination_rate = 0.00005
        
        self.created_synapses_count = 0
        self.eliminated_synapses_count = 0
    
    def update_structure(self, synapses: List, neural_activity: np.ndarray, dt: float):
        """Обновить структуру синапсов на основе активности"""
        for synapse in synapses:
            # Синапсы с высокой активностью становятся сильнее и более стабильными
            if synapse.weight > self.synapse_creation_threshold:
                # Усиление синапса через структурные изменения
                growth = self.synapse_creation_rate * synapse.weight * dt
                synapse.weight = min(synapse.weight + growth, synapse.max_weight)
                self.created_synapses_count += 1
            
            # Синапсы с низкой активностью могут быть элиминированы
            elif synapse.weight < self.synapse_elimination_threshold:
                # Вероятность исчезновения пропорциональна инактивности
                elimination_prob = self.synapse_elimination_rate * dt
                if np.random.random() < elimination_prob:
                    synapse.weight = 0.0
                    self.eliminated_synapses_count += 1


class FunctionalPlasticityManager:
    """Управление функциональными изменениями в сети"""
    
    def __init__(self):
        self.learning_rate = 0.01
        self.homeostatic_target = 0.3  # Целевая средняя активность нейрона
        self.homeostatic_learning_rate = 0.0001
        
        self.synaptic_scaling_enabled = True
        self.intrinsic_plasticity_enabled = True
    
    def apply_homeostatic_scaling(self, neurons: List, dt: float):
        """Применить гомеостатическое масштабирование синаптических весов"""
        if not self.synaptic_scaling_enabled:
            return
        
        for neuron in neurons:
            # Вычислить среднюю активность
            if len(neuron.membrane_potential_history) > 0:
                recent_activity = np.mean([1 if v > neuron.threshold else 0 
                                          for v in neuron.membrane_potential_history[-100:]])
            else:
                recent_activity = 0.0
            
            # Если активность слишком высокая, ослабить синапсы
            if recent_activity > self.homeostatic_target * 1.5:
                scaling_factor = 1.0 - self.homeostatic_learning_rate * dt
            # Если активность слишком низкая, усилить синапсы
            elif recent_activity < self.homeostatic_target * 0.5:
                scaling_factor = 1.0 + self.homeostatic_learning_rate * dt
            else:
                scaling_factor = 1.0
    
    def apply_intrinsic_plasticity(self, neurons: List, dt: float):
        """Применить внутреннюю пластичность (изменение возбудимости)"""
        if not self.intrinsic_plasticity_enabled:
            return
        
        for neuron in neurons:
            # Адаптивное изменение порога спайкования
            if len(neuron.membrane_potential_history) > 0:
                recent_activity = np.mean([1 if v > neuron.threshold else 0 
                                          for v in neuron.membrane_potential_history[-100:]])
                
                # Если нейрон слишком активный, поднять порог
                if recent_activity > self.homeostatic_target * 1.5:
                    neuron.threshold += self.homeostatic_learning_rate * dt
                # Если нейрон малоактивный, понизить порог
                elif recent_activity < self.homeostatic_target * 0.5:
                    neuron.threshold -= self.homeostatic_learning_rate * dt

# -*- coding: utf-8 -*-
"""
Модели глиальных клеток

Включает:
- Astrocyte: астроциты как модуляторы синаптической передачи
- GlialNetwork: сеть глиальных клеток для модуляции нейронной активности
"""

import numpy as np
from typing import List, Dict, Tuple


class Astrocyte:
    """Модель астроцита с кальциевой динамикой и выбросом глиотрансмиттеров"""
    
    def __init__(self, position: Tuple[float, float], influence_radius: float = 50.0):
        self.position = np.array(position, dtype=float)
        self.influence_radius = influence_radius
        
        # Кальциевая динамика
        self.calcium_level = 0.0
        self.calcium_resting = 0.1
        self.calcium_peak = 2.0
        self.calcium_decay_tau = 500.0  # мс
        
        # Глиотрансмиттеры
        self.gliotransmitter_release = 0.0
        self.gliotransmitter_threshold = 0.3
        self.glutamate_level = 0.0
        self.atp_level = 0.0
        
        # История активности
        self.neural_activity_history = []
        self.activation_threshold = 0.1
    
    def respond_to_neural_activity(self, neural_activity_map: np.ndarray, 
                                   neural_positions: np.ndarray, dt: float):
        """Реагировать на нейронную активность в радиусе влияния"""
        if len(neural_positions) == 0:
            return
        
        # Вычислить среднюю активность в радиусе влияния
        distances = np.linalg.norm(neural_positions - self.position, axis=1)
        nearby_indices = distances < self.influence_radius
        
        if np.any(nearby_indices):
            nearby_activity = neural_activity_map[nearby_indices]
            local_activity = np.mean(nearby_activity)
        else:
            local_activity = 0.0
        
        # Кальциевая динамика (экспоненциальный рост и спад)
        if local_activity > self.activation_threshold:
            self.calcium_level += (self.calcium_peak - self.calcium_level) * 0.1
        else:
            decay = np.exp(-dt / self.calcium_decay_tau)
            self.calcium_level = self.calcium_resting + (self.calcium_level - self.calcium_resting) * decay
        
        # Выброс глиотрансмиттеров зависит от кальция
        if self.calcium_level > self.gliotransmitter_threshold:
            self.gliotransmitter_release = (self.calcium_level - self.gliotransmitter_threshold) / \
                                          (self.calcium_peak - self.gliotransmitter_threshold)
            self.glutamate_level = 0.3 * self.gliotransmitter_release
            self.atp_level = 0.5 * self.gliotransmitter_release
        else:
            self.gliotransmitter_release = 0.0
            self.glutamate_level *= np.exp(-dt / 100.0)
            self.atp_level *= np.exp(-dt / 150.0)
        
        self.neural_activity_history.append(local_activity)
        if len(self.neural_activity_history) > 1000:
            self.neural_activity_history = self.neural_activity_history[-1000:]
    
    def modulate_synapses(self, synapses: List, weight_modifier: float = 0.05):
        """Модулировать эффективность синапсов в радиусе действия"""
        if self.gliotransmitter_release > 0:
            # ATP из астроцитов может усиливать синаптическую передачу
            weight_change = self.atp_level * weight_modifier
            for synapse in synapses:
                synapse.current_efficacy = min(2.0, synapse.current_efficacy + weight_change)


class GlialNetwork:
    """Сеть глиальных клеток для масштабной модуляции мозга"""
    
    def __init__(self, num_astrocytes: int = 24, brain_size: Tuple[float, float] = (400.0, 400.0)):
        self.num_astrocytes = num_astrocytes
        self.brain_size = brain_size
        
        # Создаём сетку астроцитов
        grid_size = int(np.sqrt(num_astrocytes))
        self.astrocytes: List[Astrocyte] = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) * brain_size[0] / grid_size
                y = (j + 0.5) * brain_size[1] / grid_size
                astrocyte = Astrocyte((x, y), influence_radius=brain_size[0] / (2 * grid_size))
                self.astrocytes.append(astrocyte)
        
        self.average_gliotransmitter = 0.0
        self.brain_state = "resting"  # "resting", "active", "excited"
    
    def update(self, neural_activity_map: np.ndarray, 
               neural_positions: np.ndarray, dt: float):
        """Обновить состояние всей глиальной сети"""
        # Обновить каждый астроцит
        for astrocyte in self.astrocytes:
            astrocyte.respond_to_neural_activity(neural_activity_map, neural_positions, dt)
        
        # Вычислить среднее состояние сети
        self.average_gliotransmitter = np.mean([a.gliotransmitter_release for a in self.astrocytes])
        
        # Определить состояние мозга
        if self.average_gliotransmitter > 0.5:
            self.brain_state = "excited"
        elif self.average_gliotransmitter > 0.2:
            self.brain_state = "active"
        else:
            self.brain_state = "resting"
    
    def get_local_modulation(self, position: np.ndarray) -> Dict[str, float]:
        """Получить локальную модуляцию в конкретной позиции"""
        modulation = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'acetylcholine': 0.3
        }
        
        # Астроциты могут локально влиять на нейромодуляторы
        for astrocyte in self.astrocytes:
            distance = np.linalg.norm(position - astrocyte.position)
            if distance < astrocyte.influence_radius:
                influence = (1.0 - distance / astrocyte.influence_radius) * astrocyte.gliotransmitter_release
                modulation['acetylcholine'] += influence * 0.1
        
        return modulation
    
    def reset(self):
        """Сброс всей сети"""
        for astrocyte in self.astrocytes:
            astrocyte.calcium_level = astrocyte.calcium_resting
            astrocyte.gliotransmitter_release = 0.0

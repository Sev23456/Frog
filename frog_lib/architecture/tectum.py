# -*- coding: utf-8 -*-
"""
Тектум (визуальный центр обработки движения)

Включает:
- TectalColumn: колонка в тектуме для обработки направления движения
- Tectum: полная архитектура тектума
"""

import numpy as np
from typing import List, Tuple
from ..core.biological_neuron import PyramidalNeuron, FastSpikingInterneuron
from ..core.synapse_models import BiologicalSynapse


class TectalColumn:
    """Колонка в тектуме, специализированная на обработке направленного движения"""
    
    def __init__(self, position: Tuple[float, float], preferred_direction: float):
        self.position = np.array(position, dtype=float)
        self.preferred_direction = preferred_direction  # в радианах
        
        # Нейроны в колонке
        self.pyramidal_neurons = [PyramidalNeuron() for _ in range(8)]
        self.interneurons = [FastSpikingInterneuron() for _ in range(4)]
        self.output_neurons = [PyramidalNeuron() for _ in range(2)]
        
        # Синапсы между слоями
        self.pyramidal_to_interneurons = [[BiologicalSynapse() for _ in range(4)] 
                                         for _ in range(8)]
        self.interneurons_to_output = [[BiologicalSynapse() for _ in range(2)] 
                                      for _ in range(4)]
        
        self.output = 0.0
        self.direction_selectivity = 1.0
    
    def process_visual_input(self, visual_input: float, motion_vector: np.ndarray) -> float:
        """Обработать визуальный ввод с учётом направления движения"""
        # Вычислить селективность к направлению
        if np.linalg.norm(motion_vector) > 0.01:
            motion_direction = np.arctan2(motion_vector[1], motion_vector[0])
            direction_difference = np.abs(motion_direction - self.preferred_direction)
            
            # Циклическое расстояние
            direction_difference = np.min([direction_difference, 2 * np.pi - direction_difference])
            
            # Гауссова селективность
            self.direction_selectivity = np.exp(-(direction_difference ** 2) / (2 * 0.5 ** 2))
        else:
            self.direction_selectivity = 0.5
        
        # Активация пирамидальных нейронов
        input_current = visual_input * 10.0 * self.direction_selectivity
        for pyr in self.pyramidal_neurons:
            pyr.integrate(0.01, input_current)
        
        # Интернейроны опосредуют торможение
        inter_input = np.mean([pyr.spike_output for pyr in self.pyramidal_neurons]) * 15.0
        for inter in self.interneurons:
            inter.integrate(0.01, inter_input)
        
        # Выходные нейроны интегрируют возбуждающий и тормозящий вводы
        exc_input = np.mean([pyr.spike_output for pyr in self.pyramidal_neurons]) * 10.0
        inh_input = np.mean([inter.spike_output for inter in self.interneurons]) * 5.0
        
        for out in self.output_neurons:
            out.integrate(0.01, exc_input - inh_input)
        
        self.output = np.mean([out.spike_output for out in self.output_neurons])
        return self.output
    
    def reset(self):
        """Сброс колонки"""
        for pyr in self.pyramidal_neurons:
            pyr.reset()
        for inter in self.interneurons:
            inter.reset()
        for out in self.output_neurons:
            out.reset()


class Tectum:
    """Полная архитектура тектума с сетью колонок"""
    
    def __init__(self, columns: int = 16, neurons_per_column: int = 12):
        self.num_columns = columns
        self.neurons_per_column = neurons_per_column
        
        # Создание колонок с предпочтительными направлениями
        self.columns: List[TectalColumn] = []
        for i in range(columns):
            x = (i % 4) * 100.0
            y = (i // 4) * 100.0
            preferred_dir = (i / columns) * 2 * np.pi
            column = TectalColumn((x, y), preferred_dir)
            self.columns.append(column)
        
        self.tectal_output = np.zeros(columns)
        self.dominant_direction = 0.0
    
    def process(self, retinal_input: np.ndarray, motion_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Обработать ретинальный ввод с информацией о движении
        retinal_input: выход сетчатки (100 нейронов)
        motion_vectors: векторы движения для каждого объекта
        """
        self.tectal_output = np.zeros(self.num_columns)
        
        # Агрегировать зрительный ввод
        mean_visual_input = np.mean(retinal_input) if len(retinal_input) > 0 else 0.0
        
        # Агрегировать вектор движения
        if len(motion_vectors) > 0:
            aggregated_motion = np.mean(motion_vectors, axis=0)
        else:
            aggregated_motion = np.zeros(2)
        
        # Обработать каждую колонку
        for i, column in enumerate(self.columns):
            output = column.process_visual_input(mean_visual_input, aggregated_motion)
            self.tectal_output[i] = output
        
        # Вычислить доминирующее направление
        if np.sum(self.tectal_output) > 0:
            dominant_idx = np.argmax(self.tectal_output)
            self.dominant_direction = (dominant_idx / self.num_columns) * 2 * np.pi
        
        return self.tectal_output
    
    def get_movement_command(self) -> Tuple[float, float]:
        """Получить команду движения на основе активности тектума"""
        if np.sum(self.tectal_output) == 0:
            return (0.0, 0.0)
        
        # Декодировать направление из активности популяции
        direction_angle = self.dominant_direction
        magnitude = np.sum(self.tectal_output) / self.num_columns
        
        return (magnitude * np.cos(direction_angle), magnitude * np.sin(direction_angle))
    
    def reset(self):
        """Сброс тектума"""
        for column in self.columns:
            column.reset()
        self.tectal_output = np.zeros(self.num_columns)

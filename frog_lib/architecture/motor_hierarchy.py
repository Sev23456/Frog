# -*- coding: utf-8 -*-
"""
Моторная иерархия для управления движениями

Включает:
- MotorHierarchy: иерархическое управление движениями лягушки
"""

import numpy as np
from typing import Tuple, Dict
from ..core.biological_neuron import PyramidalNeuron, FastSpikingInterneuron


class MotorHierarchy:
    """Иерархическая система управления движениями"""
    
    def __init__(self):
        # Слой управления (высокий уровень)
        self.executive_neurons = [PyramidalNeuron() for _ in range(4)]  # Четыре основных движения
        
        # Слой координации (средний уровень)
        self.coordination_interneurons = [FastSpikingInterneuron() for _ in range(8)]
        
        # Слой мышечного контроля
        self.motor_neurons = [PyramidalNeuron() for _ in range(12)]  # Управление разными мышцами
        
        # Проприоцептивная обратная связь
        self.proprioceptive_input = np.zeros(12)
        
        # Состояния движения
        self.movement_state = "idle"
        self.current_movement_command = np.zeros(2)
        self.muscle_activation = np.zeros(12)
    
    def execute_movement_command(self, command: Tuple[float, float], 
                                proprioceptive_feedback: np.ndarray) -> np.ndarray:
        """Выполнить команду движения с учётом проприоцепции"""
        # Преобразовать команду в активность executive нейронов
        magnitude = np.linalg.norm(command)
        if magnitude > 0.01:
            direction = command / magnitude
            # Кодирование направления (четыре нейрона для четырёх направлений)
            angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            input_angle = np.arctan2(command[1], command[0])
            
            for i, neuron in enumerate(self.executive_neurons):
                angle_diff = np.abs(input_angle - angles[i])
                angle_diff = np.min([angle_diff, 2 * np.pi - angle_diff])
                selectivity = np.exp(-(angle_diff ** 2) / (2 * 0.5 ** 2))
                input_current = magnitude * selectivity * 15.0
                neuron.integrate(0.01, input_current)
        else:
            for neuron in self.executive_neurons:
                neuron.integrate(0.01, 0.0)
        
        # Интернейроны скоординируют активность
        exec_activity = np.mean([n.spike_output for n in self.executive_neurons])
        for inter in self.coordination_interneurons:
            inter.integrate(0.01, exec_activity * 10.0)
        
        # Моторные нейроны генерируют мышечные команды
        inter_activity = np.mean([inter.spike_output for inter in self.coordination_interneurons])
        
        for i, motor_neuron in enumerate(self.motor_neurons):
            # Возбуждающий вход от интернейронов
            exc_input = inter_activity * 10.0
            
            # Проприоцептивная обратная связь (обратное торможение)
            inh_feedback = proprioceptive_feedback[i] * 5.0 if i < len(proprioceptive_feedback) else 0.0
            
            motor_neuron.integrate(0.01, exc_input - inh_feedback)
            self.muscle_activation[i] = motor_neuron.spike_output
        
        self.current_movement_command = np.array(command)
        
        return self.muscle_activation
    
    def process_tongue_action(self, target_position: np.ndarray, 
                             current_position: np.ndarray) -> bool:
        """Обработать команду на вытягивание языка"""
        if target_position is None:
            return False
        
        distance = np.linalg.norm(target_position - current_position)
        if distance < 150.0 and distance > 10.0:
            # Активировать язык
            return True
        return False
    
    def reset(self):
        """Сброс моторной системы"""
        for neuron in self.executive_neurons:
            neuron.reset()
        for neuron in self.coordination_interneurons:
            neuron.reset()
        for neuron in self.motor_neurons:
            neuron.reset()
        self.muscle_activation = np.zeros(12)

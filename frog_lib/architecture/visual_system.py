# -*- coding: utf-8 -*-
"""
Визуальная система с рецептивными полями

Включает:
- RetinalProcessing: обработка визуальной информации с ON/OFF клетками
- CenterSurroundFilter: центр-периферический фильтр
"""

import numpy as np
from typing import List, Tuple, Optional
from ..core.biological_neuron import LIFNeuron


class CenterSurroundFilter:
    """Рецептивное поле с центр-периферической организацией"""
    
    def __init__(self, position: Tuple[float, float], center_size: float = 10.0,
                 surround_size: float = 30.0, filter_type: str = "on_center"):
        self.position = np.array(position, dtype=float)
        self.center_size = center_size
        self.surround_size = surround_size
        self.filter_type = filter_type  # "on_center" или "off_center"
        
        self.center_neuron = LIFNeuron()
        self.surround_neuron = LIFNeuron()
        self.output = 0.0
    
    def process(self, stimulus: np.ndarray) -> float:
        """Обработать визуальный стимул"""
        # Вычислить отклик центра и периферии
        center_distance = np.linalg.norm(stimulus - self.position)
        
        center_response = np.exp(-(center_distance ** 2) / (2 * self.center_size ** 2))
        surround_response = np.exp(-(center_distance ** 2) / (2 * self.surround_size ** 2))
        
        if self.filter_type == "on_center":
            # ON-центр: включение на свет в центре
            input_current = center_response - 0.3 * surround_response
        else:
            # OFF-центр: включение на темноту в центре
            input_current = surround_response - 0.3 * center_response
        
        input_current = np.clip(input_current, 0.0, 1.0) * 20.0
        
        self.center_neuron.integrate(0.01, input_current)
        self.output = self.center_neuron.spike_output
        
        return self.output


class RetinalProcessing:
    """Обработка визуальной информации в сетчатке"""
    
    def __init__(self, visual_field_size: Tuple[float, float] = (400.0, 400.0),
                 num_filters_per_side: int = 10):
        self.visual_field_size = visual_field_size
        self.num_filters_per_side = num_filters_per_side
        
        # Создаём сетку рецептивных полей
        self.filters: List[CenterSurroundFilter] = []
        
        for i in range(num_filters_per_side):
            for j in range(num_filters_per_side):
                x = (i + 0.5) * visual_field_size[0] / num_filters_per_side
                y = (j + 0.5) * visual_field_size[1] / num_filters_per_side
                
                # Чередуем ON и OFF фильтры
                filter_type = "on_center" if (i + j) % 2 == 0 else "off_center"
                filter_cell = CenterSurroundFilter((x, y), filter_type=filter_type)
                self.filters.append(filter_cell)
        
        self.retinal_output = np.zeros(len(self.filters))
        self.processed_image = None
    
    def process_visual_input(self, visual_scene: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Обработать визуальный сценарий
        visual_scene: список (x, y, brightness) для каждого объекта
        """
        self.retinal_output = np.zeros(len(self.filters))
        
        # Обработать каждый фильтр
        for idx, filter_cell in enumerate(self.filters):
            total_response = 0.0
            
            for obj_x, obj_y, brightness in visual_scene:
                stimulus = np.array([obj_x, obj_y])
                response = filter_cell.process(stimulus)
                total_response += response * brightness
            
            self.retinal_output[idx] = total_response
        
        self.processed_image = self.retinal_output.reshape(
            (self.num_filters_per_side, self.num_filters_per_side)
        )
        
        return self.retinal_output
    
    def get_spatial_attention_map(self) -> np.ndarray:
        """Получить карту пространственного внимания"""
        if self.processed_image is None:
            return np.zeros((self.num_filters_per_side, self.num_filters_per_side))
        
        # Нормализовать и размыть для получения карты внимания
        attention_map = np.clip(self.processed_image, 0.0, 1.0)
        
        # Простой Гауссов размыв
        for _ in range(2):
            attention_map = np.convolve(attention_map.flatten(), 
                                       np.array([0.25, 0.5, 0.25]), 
                                       mode='same').reshape(self.processed_image.shape)
        
        return attention_map
    
    def reset(self):
        """Сброс визуальной системы"""
        for filter_cell in self.filters:
            filter_cell.center_neuron.reset()
            filter_cell.surround_neuron.reset()
        self.retinal_output = np.zeros(len(self.filters))
        self.processed_image = None

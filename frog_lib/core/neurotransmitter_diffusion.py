# -*- coding: utf-8 -*-
"""
Модели диффузии нейромодуляторов

Включает:
- NeurotransmitterDiffusion: диффузионная модель нейромодуляторов в мозге
"""

import numpy as np
from typing import Tuple, Dict, Optional


class NeurotransmitterDiffusion:
    """Модель пространственной диффузии нейромодуляторов"""
    
    def __init__(self, space_size: Tuple[int, int] = (400, 400), 
                 diffusion_rate: float = 0.1, grid_resolution: int = 20):
        self.space_size = space_size
        self.diffusion_rate = diffusion_rate
        self.grid_resolution = grid_resolution
        
        # Сетка для диффузии (низкое разрешение для оптимизации)
        self.grid_size = int(np.sqrt(space_size[0] * space_size[1] / (grid_resolution ** 2)))
        
        # Концентрационные карты для разных нейромодуляторов
        self.dopamine_map = np.zeros((self.grid_size, self.grid_size))
        self.serotonin_map = np.zeros((self.grid_size, self.grid_size))
        self.acetylcholine_map = np.zeros((self.grid_size, self.grid_size))
        
        # Базовые уровни восстановления
        self.dopamine_baseline = 0.3
        self.serotonin_baseline = 0.3
        self.acetylcholine_baseline = 0.2
        
        # Точки выброса
        self.dopamine_sources = []
        self.serotonin_sources = []
        self.acetylcholine_sources = []
        
        # Параметры диффузии
        self.decay_tau = 500.0  # мс
        self.diffusion_kernel = self._create_diffusion_kernel()
    
    def _create_diffusion_kernel(self) -> np.ndarray:
        """Создать ядро диффузии (гауссиан)"""
        size = 3
        kernel = np.zeros((size, size))
        sigma = 1.0
        for i in range(size):
            for j in range(size):
                x = i - size // 2
                y = j - size // 2
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel
    
    def _position_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Преобразовать позицию в координаты сетки"""
        x = int(position[0] / self.space_size[0] * self.grid_size)
        y = int(position[1] / self.space_size[1] * self.grid_size)
        return (np.clip(x, 0, self.grid_size - 1), np.clip(y, 0, self.grid_size - 1))
    
    def release(self, position: Tuple[float, float], amount: float, 
                transmitter_type: str = "dopamine"):
        """Добавить выброс нейромодулятора в точку"""
        grid_pos = self._position_to_grid(position)
        
        if transmitter_type == "dopamine":
            self.dopamine_map[grid_pos] = min(1.0, self.dopamine_map[grid_pos] + amount)
        elif transmitter_type == "serotonin":
            self.serotonin_map[grid_pos] = min(1.0, self.serotonin_map[grid_pos] + amount)
        elif transmitter_type == "acetylcholine":
            self.acetylcholine_map[grid_pos] = min(1.0, self.acetylcholine_map[grid_pos] + amount)
    
    def _apply_diffusion(self, concentration_map: np.ndarray, dt: float) -> np.ndarray:
        """Применить диффузию и затухание"""
        # Диффузия через свёртку
        from scipy import signal
        try:
            diffused = signal.convolve2d(concentration_map, self.diffusion_kernel, 
                                        mode='same', boundary='fill', fillvalue=0.0)
        except:
            # Fallback если scipy недоступен
            diffused = concentration_map.copy()
        
        # Затухание
        decay = np.exp(-dt / self.decay_tau)
        result = self.dopamine_baseline + (diffused - self.dopamine_baseline) * decay
        
        return np.clip(result, 0.0, 1.0)
    
    def diffuse(self, dt: float):
        """Обновить диффузию и затухание для всех нейромодуляторов"""
        self.dopamine_map = self._apply_diffusion(self.dopamine_map, dt)
        self.serotonin_map = self._apply_diffusion(self.serotonin_map, dt)
        self.acetylcholine_map = self._apply_diffusion(self.acetylcholine_map, dt)
    
    def get_concentration(self, position: Tuple[float, float], 
                         transmitter_type: str = "dopamine") -> float:
        """Получить концентрацию нейромодулятора в точке"""
        grid_pos = self._position_to_grid(position)
        
        if transmitter_type == "dopamine":
            return float(self.dopamine_map[grid_pos])
        elif transmitter_type == "serotonin":
            return float(self.serotonin_map[grid_pos])
        elif transmitter_type == "acetylcholine":
            return float(self.acetylcholine_map[grid_pos])
        
        return 0.0
    
    def get_concentration_vector(self, position: Tuple[float, float]) -> Dict[str, float]:
        """Получить вектор всех концентраций"""
        return {
            'dopamine': self.get_concentration(position, 'dopamine'),
            'serotonin': self.get_concentration(position, 'serotonin'),
            'acetylcholine': self.get_concentration(position, 'acetylcholine'),
        }
    
    def reset(self):
        """Сброс всех карт концентрации"""
        self.dopamine_map.fill(self.dopamine_baseline)
        self.serotonin_map.fill(self.serotonin_baseline)
        self.acetylcholine_map.fill(self.acetylcholine_baseline)


class MultiNeurotransmitterSystem:
    """Система управления несколькими нейромодуляторами"""
    
    def __init__(self, space_size: Tuple[int, int] = (400, 400)):
        self.diffusion = NeurotransmitterDiffusion(space_size)
        self.modulator_levels = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'acetylcholine': 0.3,
            'gaba': 0.4,
            'glutamate': 0.5
        }
    
    def update(self, dt: float):
        """Обновить систему нейромодуляторов"""
        self.diffusion.diffuse(dt)
        
        # Восстановление базовых уровней
        for key in self.modulator_levels:
            baseline = 0.3 if key != 'acetylcholine' else 0.2
            self.modulator_levels[key] += (baseline - self.modulator_levels[key]) * 0.01
    
    def get_system_state(self) -> Dict[str, float]:
        """Получить текущее состояние системы"""
        return self.modulator_levels.copy()

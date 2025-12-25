# -*- coding: utf-8 -*-
"""
Модели энергетики нейронов и системного метаболизма

Включает:
- NeuronMetabolism: модель энергопотребления отдельного нейрона
- SystemicMetabolism: системный метаболизм организма
"""

import numpy as np
from typing import Dict


class NeuronMetabolism:
    """Модель энергопотребления и метаболизма отдельного нейрона"""
    
    def __init__(self, baseline_atp: float = 1.0, max_atp: float = 2.0):
        self.atp_level = baseline_atp
        self.max_atp = max_atp
        self.baseline_atp = baseline_atp
        
        # Энергетические константы
        self.atp_cost_spike = 0.1  # АТФ на один спайк
        self.atp_cost_rest = 0.01  # АТФ в покое (утечка)
        self.atp_recovery_rate = 0.02  # Восстановление АТФ в секунду
        
        # Влияние энергии на возбудимость
        self.excitability_modifier = 1.0
        self.sodium_potassium_pump_efficiency = 1.0
    
    def consume_energy(self, spiked: bool, firing_rate: float, dt: float):
        """Расчёт энергопотребления"""
        # Стоимость спайков (основной энергопотребитель)
        if spiked:
            self.atp_level -= self.atp_cost_spike
        
        # Базовая стоимость (натриевый насос, синтез белков и т.д.)
        baseline_cost = self.atp_cost_rest * dt
        self.atp_level -= baseline_cost
        
        # Дополнительный расход при частой стрельбе
        frequency_cost = firing_rate * 0.01 * dt
        self.atp_level -= frequency_cost
        
        # Ограничение уровня АТФ
        self.atp_level = np.clip(self.atp_level, 0.0, self.max_atp)
    
    def recover_energy(self, dt: float, oxygen_level: float = 1.0, glucose_level: float = 1.0):
        """Восстановление АТФ в зависимости от доступности ресурсов"""
        # Скорость восстановления зависит от кислорода и глюкозы
        recovery_factor = oxygen_level * glucose_level * self.sodium_potassium_pump_efficiency
        recovery = self.atp_recovery_rate * recovery_factor * dt
        
        self.atp_level = np.clip(self.atp_level + recovery, 0.0, self.max_atp)
    
    def affects_excitability(self) -> float:
        """Вычислить, как энергия влияет на возбудимость"""
        # Низкая энергия снижает возбудимость
        if self.atp_level < self.baseline_atp * 0.5:
            # Резкое падение возбудимости при утомлении
            excitability = 0.3 + 0.7 * (self.atp_level / (self.baseline_atp * 0.5))
        else:
            # Нормальная или повышенная возбудимость
            excitability = 1.0 + 0.5 * ((self.atp_level - self.baseline_atp) / self.baseline_atp)
        
        self.excitability_modifier = np.clip(excitability, 0.1, 2.0)
        return self.excitability_modifier
    
    def get_energy_state(self) -> Dict[str, float]:
        """Получить состояние энергетики"""
        return {
            'atp_level': self.atp_level,
            'max_atp': self.max_atp,
            'atp_ratio': self.atp_level / self.max_atp,
            'excitability': self.excitability_modifier
        }


class SystemicMetabolism:
    """Системный метаболизм организма"""
    
    def __init__(self):
        # Системные уровни метаболитов
        self.glucose_level = 1.0
        self.oxygen_level = 1.0
        self.lactate_level = 0.1
        
        # Параметры метаболизма
        self.glucose_consumption_rate = 0.001  # в секунду
        self.oxygen_consumption_rate = 0.0015  # в секунду
        self.glucose_recovery_rate = 0.0005
        self.oxygen_recovery_rate = 0.001
        
        # Влияние активности на метаболизм
        self.neural_activity_level = 0.0
        self.movement_activity_level = 0.0
        
        # Циркадный ритм (усталость)
        self.circadian_phase = 0.0
        self.fatigue_level = 0.0
    
    def update(self, dt: float, movement_intensity: float = 0.0, 
               neural_activity: float = 0.0) -> Dict[str, float]:
        """Обновить системный метаболизм"""
        self.neural_activity_level = 0.9 * self.neural_activity_level + 0.1 * neural_activity
        self.movement_activity_level = 0.9 * self.movement_activity_level + 0.1 * movement_intensity
        
        # Консумпция глюкозы зависит от активности
        total_activity = self.neural_activity_level + self.movement_activity_level
        glucose_consumption = self.glucose_consumption_rate * (1.0 + total_activity) * dt
        self.glucose_level -= glucose_consumption
        
        # Консумпция кислорода
        oxygen_consumption = self.oxygen_consumption_rate * (1.0 + total_activity) * dt
        self.oxygen_level -= oxygen_consumption
        
        # Восстановление (может быть замедлено усталостью)
        recovery_factor = 1.0 - 0.5 * self.fatigue_level
        self.glucose_level += self.glucose_recovery_rate * recovery_factor * dt
        self.oxygen_level += self.oxygen_recovery_rate * recovery_factor * dt
        
        # Лактат накапливается при интенсивной активности
        lactate_production = total_activity * 0.01 * dt
        self.lactate_level += lactate_production
        self.lactate_level *= np.exp(-dt / 100.0)  # Разложение лактата
        
        # Обновить циркадный ритм (простой синусоидальный цикл)
        self.circadian_phase += dt / 10000.0  # Полный цикл за 10000 секунд (~2.8 часа)
        self.circadian_phase = self.circadian_phase % (2 * np.pi)
        
        # Усталость зависит от циркадного ритма и активности
        time_of_day_fatigue = 0.3 * (1.0 - np.cos(self.circadian_phase)) / 2.0
        activity_fatigue = 0.2 * (self.neural_activity_level + self.movement_activity_level)
        resource_fatigue = 0.3 * (1.0 - (self.glucose_level + self.oxygen_level) / 2.0)
        
        self.fatigue_level = np.clip(time_of_day_fatigue + activity_fatigue + resource_fatigue, 0.0, 1.0)
        
        # Ограничить уровни
        self.glucose_level = np.clip(self.glucose_level, 0.0, 1.5)
        self.oxygen_level = np.clip(self.oxygen_level, 0.0, 1.5)
        
        return self.get_metabolic_state()
    
    def get_metabolic_state(self) -> Dict[str, float]:
        """Получить текущее состояние метаболизма"""
        return {
            'glucose': self.glucose_level,
            'oxygen': self.oxygen_level,
            'lactate': self.lactate_level,
            'fatigue': self.fatigue_level,
            'neural_activity': self.neural_activity_level,
            'movement_activity': self.movement_activity_level,
        }
    
    def reset(self):
        """Сброс системы"""
        self.glucose_level = 1.0
        self.oxygen_level = 1.0
        self.lactate_level = 0.1
        self.fatigue_level = 0.0
        self.circadian_phase = 0.0

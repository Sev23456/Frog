# -*- coding: utf-8 -*-
"""
Биологические модели синапсов с пластичностью

Включает:
- BiologicalSynapse: синапс с STDP, STP и нейромодуляцией
- DynamicSynapse: синапс с коротко-срочной пластичностью
"""

import numpy as np
from typing import Optional


class BiologicalSynapse:
    """Синапс с полной биологической пластичностью"""
    
    def __init__(self, max_weight: float = 1.0, initial_weight: Optional[float] = None,
                 min_weight: float = 0.0, stdp_window: float = 50.0, stdp_amplitude: float = 0.01):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight = initial_weight if initial_weight else np.random.uniform(0.2, 0.8)
        
        # STDP параметры
        self.stdp_window = stdp_window  # мс
        self.stdp_amplitude = stdp_amplitude
        
        # Короткая пластичность
        self.facilitation_state = 0.0
        self.depression_state = 0.0
        self.facilitation_tau = 50.0  # мс
        self.depression_tau = 200.0  # мс
        self.current_efficacy = 1.0
        
        # Нейромодуляция
        self.dopamine_modulation = 0.5
        self.serotonin_modulation = 0.5
        self.acetylcholine_level = 0.3
        
        # История для обучения
        self.presynaptic_spike_time = None
        self.postsynaptic_spike_time = None
    
    def apply_stdp(self, presynaptic_spike_time: Optional[float], 
                   postsynaptic_spike_time: Optional[float]):
        """Применить STDP-обучение (Spike-Timing-Dependent Plasticity)"""
        if presynaptic_spike_time is None or postsynaptic_spike_time is None:
            return
        
        delta_t = postsynaptic_spike_time - presynaptic_spike_time
        
        # Окно STDP: LTP если постсинаптический спайк после пресинаптического
        if abs(delta_t) < self.stdp_window:
            if delta_t > 0:  # LTP (Long-Term Potentiation)
                weight_change = self.stdp_amplitude * np.exp(-delta_t / self.stdp_window)
            else:  # LTD (Long-Term Depression)
                weight_change = -self.stdp_amplitude * np.exp(delta_t / self.stdp_window)
            
            # Модулирование нейротрансмиттерами
            modulation = 0.7 * self.dopamine_modulation + 0.3 * self.serotonin_modulation
            weight_change *= modulation
            
            self.weight = np.clip(self.weight + weight_change, self.min_weight, self.max_weight)
    
    def apply_short_term_plasticity(self, dt: float, spike: bool):
        """Краткосрочная пластичность (Facilitation/Depression)"""
        # Facilitation - усиление при повторных спайках
        if spike:
            self.facilitation_state = 1.0
        else:
            self.facilitation_state *= np.exp(-dt / self.facilitation_tau)
        
        # Depression - ослабление при повторных спайках
        if spike:
            self.depression_state = 1.0
        else:
            self.depression_state *= np.exp(-dt / self.depression_tau)
        
        # Результирующая эффективность
        self.current_efficacy = 1.0 + 0.3 * self.facilitation_state - 0.5 * self.depression_state
        self.current_efficacy = np.clip(self.current_efficacy, 0.1, 2.0)
    
    def transmit(self, presynaptic_output: float) -> float:
        """Передача сигнала через синапс"""
        return presynaptic_output * self.weight * self.current_efficacy
    
    def update_modulators(self, dopamine: float, serotonin: float, acetylcholine: float):
        """Обновить уровни нейромодуляторов"""
        self.dopamine_modulation = dopamine
        self.serotonin_modulation = serotonin
        self.acetylcholine_level = acetylcholine


class DynamicSynapse(BiologicalSynapse):
    """Синапс с усиленной короткой пластичностью для depressing/facilitating синапсов"""
    
    def __init__(self, synapse_type: str = "depressing", **kwargs):
        super().__init__(**kwargs)
        self.synapse_type = synapse_type  # "depressing" или "facilitating"
        
        if synapse_type == "depressing":
            self.facilitation_tau = 30.0
            self.depression_tau = 800.0
            self.initial_release = 0.9
        else:  # facilitating
            self.facilitation_tau = 200.0
            self.depression_tau = 100.0
            self.initial_release = 0.3
    
    def transmit_dynamic(self, presynaptic_output: float, available_resources: float) -> float:
        """Передача с учетом истощения/восстановления ресурсов"""
        # Модель истощения ресурсов (для depressing синапсов)
        utilization = presynaptic_output * self.current_efficacy
        transmission = utilization * available_resources
        
        return transmission

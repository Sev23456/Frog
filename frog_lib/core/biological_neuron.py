# -*- coding: utf-8 -*-
"""
Toy bio-inspired neuron models used by the frog agent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _ms_to_seconds(value_ms: float) -> float:
    return max(float(value_ms) / 1000.0, 1e-6)


class LIFNeuron:
    """Leaky integrate-and-fire neuron with second-based integration."""

    def __init__(
        self,
        rest_potential: float = -70.0,
        threshold: float = -40.0,
        tau_membrane: float = 20.0,
        tau_refractory: float = 2.0,
        max_firing_rate: float = 200.0,
    ):
        self.rest_potential = rest_potential
        self.threshold = threshold
        self.membrane_potential = rest_potential
        self.tau_membrane = tau_membrane  # milliseconds
        self.tau_refractory = tau_refractory  # milliseconds
        self.max_firing_rate = max_firing_rate

        self.spike_output = 0.0
        self.refractory_counter = 0.0
        self.last_spike_time = -1000.0
        self.membrane_potential_history = []

    def integrate(self, dt: float, input_current: float):
        tau_membrane = _ms_to_seconds(self.tau_membrane)
        tau_refractory = _ms_to_seconds(self.tau_refractory)

        if self.refractory_counter > 0:
            self.refractory_counter = max(0.0, self.refractory_counter - dt)
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        decay = np.exp(-dt / tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += input_current * dt / tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = tau_refractory
            self.membrane_potential = self.rest_potential
            self.last_spike_time = 0.0
        else:
            self.spike_output = 0.0
            self.last_spike_time += dt

        self.membrane_potential_history.append(self.membrane_potential)
        if len(self.membrane_potential_history) > 1000:
            self.membrane_potential_history = self.membrane_potential_history[-1000:]

    def reset(self):
        self.membrane_potential = self.rest_potential
        self.spike_output = 0.0
        self.refractory_counter = 0.0
        self.last_spike_time = -1000.0
        self.membrane_potential_history = []


class PyramidalNeuron(LIFNeuron):
    """Pyramidal neuron with simple basal/apical integration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apical_dendrite_input = 0.0
        self.basal_dendrite_input = 0.0
        self.dendritic_plateau_potential = 0.0

    def integrate(self, dt: float, basal_input: float, apical_input: Optional[float] = None):
        if apical_input is None:
            apical_input = 0.0

        tau_membrane = _ms_to_seconds(self.tau_membrane)
        tau_refractory = _ms_to_seconds(self.tau_refractory)

        if self.refractory_counter > 0:
            self.refractory_counter = max(0.0, self.refractory_counter - dt)
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        self.basal_dendrite_input = 0.85 * self.basal_dendrite_input + 0.15 * basal_input
        self.apical_dendrite_input = 0.85 * self.apical_dendrite_input + 0.15 * apical_input

        if apical_input > 0.25 and self.basal_dendrite_input > 0.15:
            self.dendritic_plateau_potential = 1.0
        else:
            self.dendritic_plateau_potential *= np.exp(-dt / _ms_to_seconds(50.0))

        soma_input = basal_input + 0.45 * apical_input * self.dendritic_plateau_potential
        decay = np.exp(-dt / tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += soma_input * dt / tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = tau_refractory
            self.membrane_potential = self.rest_potential
            self.last_spike_time = 0.0
        else:
            self.spike_output = 0.0
            self.last_spike_time += dt

        self.membrane_potential_history.append(self.membrane_potential)
        if len(self.membrane_potential_history) > 1000:
            self.membrane_potential_history = self.membrane_potential_history[-1000:]


class FastSpikingInterneuron(LIFNeuron):
    """Fast-spiking inhibitory neuron with a lightweight adaptation current."""

    def __init__(self, **kwargs):
        super().__init__(tau_membrane=5.0, tau_refractory=1.0, **kwargs)
        self.adaptation_current = 0.0
        self.adaptation_tau = 100.0  # milliseconds
        self.spike_count = 0

    def integrate(self, dt: float, input_current: float):
        tau_membrane = _ms_to_seconds(self.tau_membrane)
        tau_refractory = _ms_to_seconds(self.tau_refractory)
        adaptation_tau = _ms_to_seconds(self.adaptation_tau)

        if self.refractory_counter > 0:
            self.refractory_counter = max(0.0, self.refractory_counter - dt)
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        self.adaptation_current *= np.exp(-dt / adaptation_tau)
        decay = np.exp(-dt / tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += (input_current - 0.5 * self.adaptation_current) * dt / tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = tau_refractory
            self.membrane_potential = self.rest_potential
            self.last_spike_time = 0.0
            self.adaptation_current += 4.0
            self.spike_count += 1
        else:
            self.spike_output = 0.0
            self.last_spike_time += dt

        self.membrane_potential_history.append(self.membrane_potential)
        if len(self.membrane_potential_history) > 1000:
            self.membrane_potential_history = self.membrane_potential_history[-1000:]

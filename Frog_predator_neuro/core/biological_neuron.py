# -*- coding: utf-8 -*-
"""
Biologically informed neuron models (LIF, pyramidal, fast-spiking interneuron).

Adapted from Frog-main to provide consistent, low-level neural dynamics
for the Frog_predator_neuro agents.
"""

import numpy as np
from typing import Optional


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron."""

    def __init__(self, rest_potential: float = -70.0, threshold: float = -40.0,
                 tau_membrane: float = 20.0, tau_refractory: float = 2.0,
                 max_firing_rate: float = 200.0):
        self.rest_potential = rest_potential
        self.threshold = threshold
        self.membrane_potential = rest_potential
        self.tau_membrane = tau_membrane  # ms-like units
        self.tau_refractory = tau_refractory  # ms-like units
        self.max_firing_rate = max_firing_rate

        self.spike_output = 0.0
        self.refractory_counter = 0.0
        self.last_spike_time = -1000.0
        self.membrane_potential_history = []

    def integrate(self, dt: float, input_current: float):
        """Integrate membrane potential for a timestep dt.

        dt is expected to be a small timestep (e.g. 0.01).
        """
        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        decay = np.exp(-dt / self.tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += input_current * dt / self.tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = self.tau_refractory
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

    def activity_level(self):
        """Normalized activity level for compatibility with existing code.

        Returns a value in [0, 1] representing recent activity or depolarization.
        """
        span = max(1e-6, self.threshold - self.rest_potential)
        depolarization = (self.membrane_potential - self.rest_potential) / span
        return float(np.clip(max(self.spike_output, depolarization), 0.0, 1.0))


class PyramidalNeuron(LIFNeuron):
    """Pyramidal neuron with simple apical/basal dendritic integration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apical_dendrite_input = 0.0
        self.basal_dendrite_input = 0.0
        self.soma_input = 0.0
        self.dendritic_plateau_potential = 0.0
        self.dendritic_spike_threshold = -30.0

    def integrate(self, dt: float, basal_input: float, apical_input: Optional[float] = None):
        if apical_input is None:
            apical_input = 0.0

        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        self.basal_dendrite_input = 0.9 * self.basal_dendrite_input + 0.1 * basal_input
        self.apical_dendrite_input = 0.9 * self.apical_dendrite_input + 0.1 * apical_input

        if apical_input > 0.5 and self.basal_dendrite_input > 0.3:
            self.dendritic_plateau_potential = 1.0
        else:
            self.dendritic_plateau_potential *= np.exp(-dt / 50.0)

        soma_input = basal_input + 0.3 * apical_input * self.dendritic_plateau_potential

        decay = np.exp(-dt / self.tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay
        self.membrane_potential += soma_input * dt / self.tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = self.tau_refractory
            self.last_spike_time = 0.0
        else:
            self.spike_output = 0.0

        self.last_spike_time += dt
        self.membrane_potential_history.append(self.membrane_potential)
        if len(self.membrane_potential_history) > 1000:
            self.membrane_potential_history = self.membrane_potential_history[-1000:]


class FastSpikingInterneuron(LIFNeuron):
    """Fast-spiking interneuron with an adaptation current."""

    def __init__(self, **kwargs):
        super().__init__(tau_membrane=5.0, tau_refractory=1.0, **kwargs)
        self.adaptation_current = 0.0
        self.adaptation_tau = 100.0
        self.spike_count = 0

    def integrate(self, dt: float, input_current: float):
        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            self.spike_output = 0.0
            self.membrane_potential = self.rest_potential
            return

        decay = np.exp(-dt / self.adaptation_tau)
        self.adaptation_current *= decay

        decay_soma = np.exp(-dt / self.tau_membrane)
        self.membrane_potential = self.rest_potential + (self.membrane_potential - self.rest_potential) * decay_soma

        net_input = input_current - 0.5 * self.adaptation_current
        self.membrane_potential += net_input * dt / self.tau_membrane

        if self.membrane_potential >= self.threshold:
            self.spike_output = 1.0
            self.refractory_counter = self.tau_refractory
            self.last_spike_time = 0.0
            self.adaptation_current += 5.0
            self.spike_count += 1
        else:
            self.spike_output = 0.0

        self.last_spike_time += dt
        self.membrane_potential_history.append(self.membrane_potential)
        if len(self.membrane_potential_history) > 1000:
            self.membrane_potential_history = self.membrane_potential_history[-1000:]

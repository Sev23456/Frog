# -*- coding: utf-8 -*-
"""
Functional and structural plasticity helpers for the toy bio-inspired agent.
"""

from typing import List, Optional

import numpy as np


class StructuralPlasticityManager:
    """Tracks lightweight weight growth and elimination dynamics."""

    def __init__(self):
        self.synapse_creation_threshold = 0.1
        self.synapse_elimination_threshold = 0.01
        self.synapse_creation_rate = 0.0001
        self.synapse_elimination_rate = 0.00005

        self.created_synapses_count = 0
        self.eliminated_synapses_count = 0

    def update_structure(self, synapses: List, neural_activity: np.ndarray, dt: float):
        """Update synapse structure based on current activity."""
        for synapse in synapses:
            if synapse.weight > self.synapse_creation_threshold:
                growth = self.synapse_creation_rate * synapse.weight * dt
                synapse.weight = min(synapse.weight + growth, synapse.max_weight)
                self.created_synapses_count += 1
            elif synapse.weight < self.synapse_elimination_threshold:
                elimination_prob = self.synapse_elimination_rate * dt
                if np.random.random() < elimination_prob:
                    synapse.weight = 0.0
                    self.eliminated_synapses_count += 1


class FunctionalPlasticityManager:
    """Homeostatic scaling and intrinsic threshold adaptation."""

    def __init__(self):
        self.learning_rate = 0.01
        self.homeostatic_target = 0.3
        self.homeostatic_learning_rate = 0.0001

        self.synaptic_scaling_enabled = True
        self.intrinsic_plasticity_enabled = True

    def update(self, neurons: Optional[List] = None, dt: float = 0.01):
        if not neurons:
            return
        self.apply_homeostatic_scaling(neurons, dt)
        self.apply_intrinsic_plasticity(neurons, dt)

    def apply_homeostatic_scaling(self, neurons: List, dt: float):
        """Adjust membrane time constants toward a target activity band."""
        if not self.synaptic_scaling_enabled:
            return

        for neuron in neurons:
            if len(neuron.membrane_potential_history) > 0:
                recent_activity = np.mean(
                    [1 if value > neuron.threshold else 0 for value in neuron.membrane_potential_history[-100:]]
                )
            else:
                recent_activity = 0.0

            if recent_activity > self.homeostatic_target * 1.5:
                scaling_factor = 1.0 - self.homeostatic_learning_rate * dt
            elif recent_activity < self.homeostatic_target * 0.5:
                scaling_factor = 1.0 + self.homeostatic_learning_rate * dt
            else:
                scaling_factor = 1.0

            neuron.tau_membrane = float(np.clip(neuron.tau_membrane / scaling_factor, 5.0, 30.0))

    def apply_intrinsic_plasticity(self, neurons: List, dt: float):
        """Adapt spike thresholds toward a target activity level."""
        if not self.intrinsic_plasticity_enabled:
            return

        for neuron in neurons:
            if len(neuron.membrane_potential_history) > 0:
                recent_activity = np.mean(
                    [1 if value > neuron.threshold else 0 for value in neuron.membrane_potential_history[-100:]]
                )

                if recent_activity > self.homeostatic_target * 1.5:
                    neuron.threshold += self.homeostatic_learning_rate * dt
                elif recent_activity < self.homeostatic_target * 0.5:
                    neuron.threshold -= self.homeostatic_learning_rate * dt

                neuron.threshold = float(np.clip(neuron.threshold, -60.0, -35.0))

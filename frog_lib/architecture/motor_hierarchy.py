# -*- coding: utf-8 -*-
"""
Toy bio-inspired motor hierarchy for coarse frog locomotion.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.biological_neuron import FastSpikingInterneuron, PyramidalNeuron


class MotorHierarchy:
    def __init__(self):
        self.executive_neurons = [PyramidalNeuron(threshold=-50.0) for _ in range(4)]
        self.coordination_interneurons = [FastSpikingInterneuron(threshold=-52.0) for _ in range(8)]
        self.motor_neurons = [PyramidalNeuron(threshold=-51.0) for _ in range(12)]

        self.proprioceptive_input = np.zeros(12, dtype=float)
        self.current_movement_command = np.zeros(2, dtype=float)
        self.muscle_activation = np.zeros(12, dtype=float)

        self.executive_angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
        self.motor_preferences = [
            np.array([1.0, 0.0], dtype=float),
            np.array([1.0, 0.0], dtype=float),
            np.array([1.0, 0.0], dtype=float),
            np.array([-1.0, 0.0], dtype=float),
            np.array([-1.0, 0.0], dtype=float),
            np.array([-1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0, -1.0], dtype=float),
            np.array([0.0, -1.0], dtype=float),
            np.array([0.0, -1.0], dtype=float),
        ]

    def execute_movement_command(self, command: Tuple[float, float], proprioceptive_feedback: np.ndarray) -> np.ndarray:
        command_vec = np.array(command, dtype=float)
        magnitude = float(np.linalg.norm(command_vec))
        direction = command_vec / magnitude if magnitude > 1e-6 else np.zeros(2, dtype=float)

        if magnitude > 1e-6:
            input_angle = np.arctan2(direction[1], direction[0])
            for angle, neuron in zip(self.executive_angles, self.executive_neurons):
                angle_diff = np.abs(input_angle - angle)
                angle_diff = min(angle_diff, 2.0 * np.pi - angle_diff)
                selectivity = np.exp(-(angle_diff**2) / (2 * 0.45**2))
                neuron.integrate(0.01, magnitude * selectivity * 18.0)
        else:
            for neuron in self.executive_neurons:
                neuron.integrate(0.01, 0.0)

        exec_activity = float(np.mean([n.spike_output for n in self.executive_neurons]))
        for idx, inter in enumerate(self.coordination_interneurons):
            modulation = 1.0 + 0.12 * idx
            inter.integrate(0.01, exec_activity * 8.0 * modulation)

        inter_activity = float(np.mean([inter.spike_output for inter in self.coordination_interneurons]))
        for idx, motor_neuron in enumerate(self.motor_neurons):
            directional_drive = max(0.0, float(np.dot(direction, self.motor_preferences[idx]))) if magnitude > 0 else 0.0
            excitatory_drive = (inter_activity * 7.0) + (directional_drive * magnitude * 18.0)
            inhibitory_feedback = proprioceptive_feedback[idx] * 4.0 if idx < len(proprioceptive_feedback) else 0.0
            motor_neuron.integrate(0.01, excitatory_drive - inhibitory_feedback)
            self.muscle_activation[idx] = motor_neuron.spike_output

        self.current_movement_command = command_vec
        return self.muscle_activation.copy()

    def decode_velocity(self, muscle_activation: np.ndarray) -> np.ndarray:
        activation = np.array(muscle_activation, dtype=float)
        right = activation[0:3].mean()
        left = activation[3:6].mean()
        up = activation[6:9].mean()
        down = activation[9:12].mean()
        velocity = np.array([right - left, up - down], dtype=float)
        norm = np.linalg.norm(velocity)
        if norm > 1.0:
            velocity = velocity / norm
        return velocity

    def process_tongue_action(self, target_position: np.ndarray, current_position: np.ndarray) -> bool:
        if target_position is None:
            return False
        distance = np.linalg.norm(target_position - current_position)
        return 10.0 < distance < 150.0

    def reset(self):
        for neuron in self.executive_neurons:
            neuron.reset()
        for neuron in self.coordination_interneurons:
            neuron.reset()
        for neuron in self.motor_neurons:
            neuron.reset()
        self.muscle_activation = np.zeros(12, dtype=float)

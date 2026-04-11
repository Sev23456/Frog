# -*- coding: utf-8 -*-
"""
Toy tectum-like motion selection layer.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.biological_neuron import FastSpikingInterneuron, PyramidalNeuron


class TectalColumn:
    def __init__(self, position: Tuple[float, float], preferred_direction: float):
        self.position = np.array(position, dtype=float)
        self.preferred_direction = preferred_direction

        self.pyramidal_neurons = [PyramidalNeuron(threshold=-53.0) for _ in range(6)]
        self.interneurons = [FastSpikingInterneuron(threshold=-54.0) for _ in range(3)]
        self.output_neurons = [PyramidalNeuron(threshold=-52.0) for _ in range(2)]

        self.output = 0.0
        self.direction_selectivity = 1.0

    def process_visual_input(self, visual_input: float, motion_vector: np.ndarray) -> float:
        if np.linalg.norm(motion_vector) > 0.01:
            motion_direction = np.arctan2(motion_vector[1], motion_vector[0])
            direction_difference = abs(motion_direction - self.preferred_direction)
            direction_difference = min(direction_difference, 2.0 * np.pi - direction_difference)
            self.direction_selectivity = float(np.exp(-(direction_difference**2) / (2 * 0.55**2)))
        else:
            self.direction_selectivity = 0.25

        salience = float(np.clip(visual_input, 0.0, 2.0))
        pyramidal_drive = salience * (16.0 + 18.0 * self.direction_selectivity)
        for neuron in self.pyramidal_neurons:
            neuron.integrate(0.01, pyramidal_drive, apical_input=0.25 + self.direction_selectivity)

        inter_input = np.mean([pyr.spike_output for pyr in self.pyramidal_neurons]) * 8.0 + salience * 2.0
        for neuron in self.interneurons:
            neuron.integrate(0.01, inter_input)

        exc_input = np.mean([pyr.spike_output for pyr in self.pyramidal_neurons]) * 18.0 + salience * 6.0 * self.direction_selectivity
        inh_input = np.mean([inter.spike_output for inter in self.interneurons]) * 3.0
        for neuron in self.output_neurons:
            neuron.integrate(0.01, exc_input - inh_input, apical_input=self.direction_selectivity)

        spike_component = np.mean([out.spike_output for out in self.output_neurons])
        self.output = float(np.clip(0.75 * spike_component + 0.55 * self.direction_selectivity * salience, 0.0, 2.5))
        return self.output

    def reset(self):
        for neuron in self.pyramidal_neurons:
            neuron.reset()
        for neuron in self.interneurons:
            neuron.reset()
        for neuron in self.output_neurons:
            neuron.reset()


class Tectum:
    def __init__(self, columns: int = 16, neurons_per_column: int = 12):
        self.num_columns = columns
        self.neurons_per_column = neurons_per_column
        self.columns: List[TectalColumn] = []
        for index in range(columns):
            x = (index % 4) * 100.0
            y = (index // 4) * 100.0
            preferred_dir = (index / columns) * 2.0 * np.pi
            self.columns.append(TectalColumn((x, y), preferred_dir))

        self.tectal_output = np.zeros(columns, dtype=float)
        self.dominant_direction = 0.0
        self.direction_vector = np.zeros(2, dtype=float)
        self.preferred_vectors = np.array(
            [[np.cos(column.preferred_direction), np.sin(column.preferred_direction)] for column in self.columns],
            dtype=float,
        )

    def process(self, retinal_input: np.ndarray, motion_vectors: List[np.ndarray]) -> np.ndarray:
        self.tectal_output = np.zeros(self.num_columns, dtype=float)

        if len(retinal_input) == 0:
            salient_visual_input = 0.0
        else:
            top_k = min(10, len(retinal_input))
            sorted_activity = np.sort(retinal_input)
            salient_visual_input = float(0.4 * np.mean(sorted_activity[-top_k:]) + 0.6 * sorted_activity[-1])

        if motion_vectors:
            normalized_motion = []
            for vector in motion_vectors:
                motion = np.array(vector, dtype=float)
                norm = np.linalg.norm(motion)
                if norm > 1e-6:
                    normalized_motion.append(motion / norm)
            aggregated_motion = np.mean(normalized_motion, axis=0) if normalized_motion else np.zeros(2, dtype=float)
        else:
            aggregated_motion = np.zeros(2, dtype=float)

        for idx, column in enumerate(self.columns):
            self.tectal_output[idx] = column.process_visual_input(salient_visual_input, aggregated_motion)

        if np.sum(self.tectal_output) > 0:
            dominant_idx = int(np.argmax(self.tectal_output))
            self.dominant_direction = (dominant_idx / self.num_columns) * 2.0 * np.pi
            weighted_direction = self.tectal_output @ self.preferred_vectors
            direction_norm = np.linalg.norm(weighted_direction)
            if direction_norm > 1e-6:
                self.direction_vector = weighted_direction / direction_norm
            else:
                self.direction_vector = np.array(
                    [np.cos(self.dominant_direction), np.sin(self.dominant_direction)],
                    dtype=float,
                )
        else:
            self.direction_vector = np.zeros(2, dtype=float)

        return self.tectal_output

    def get_movement_command(self) -> Tuple[float, float]:
        total_activity = float(np.sum(self.tectal_output))
        if total_activity <= 1e-6:
            return (0.0, 0.0)

        magnitude = float(
            np.clip(
                0.45 * np.max(self.tectal_output) + total_activity / (self.num_columns * 1.6),
                0.0,
                1.0,
            )
        )
        return (magnitude * self.direction_vector[0], magnitude * self.direction_vector[1])

    def reset(self):
        for column in self.columns:
            column.reset()
        self.tectal_output = np.zeros(self.num_columns, dtype=float)
        self.direction_vector = np.zeros(2, dtype=float)

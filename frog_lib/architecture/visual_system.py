# -*- coding: utf-8 -*-
"""
Toy bio-inspired visual preprocessing with center-surround filters.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.biological_neuron import LIFNeuron


class CenterSurroundFilter:
    def __init__(
        self,
        position: Tuple[float, float],
        center_size: float = 10.0,
        surround_size: float = 30.0,
        filter_type: str = "on_center",
    ):
        self.position = np.array(position, dtype=float)
        self.center_size = center_size
        self.surround_size = surround_size
        self.filter_type = filter_type

        self.center_neuron = LIFNeuron(threshold=-54.0)
        self.surround_neuron = LIFNeuron(threshold=-55.0)
        self.output = 0.0

    def process(self, stimulus: np.ndarray, brightness: float = 1.0) -> float:
        center_distance = np.linalg.norm(stimulus - self.position)

        center_response = np.exp(-(center_distance**2) / (2 * self.center_size**2))
        surround_response = np.exp(-(center_distance**2) / (2 * self.surround_size**2))

        center_current = np.clip(center_response * brightness, 0.0, 1.0) * (26.0 + 54.0 * brightness)
        surround_current = np.clip(surround_response * brightness, 0.0, 1.0) * (12.0 + 26.0 * brightness)

        self.center_neuron.integrate(0.01, center_current)
        self.surround_neuron.integrate(0.01, surround_current)

        center_signal = 0.65 * self.center_neuron.spike_output + 0.35 * np.clip(center_current / 55.0, 0.0, 1.0)
        surround_signal = 0.55 * self.surround_neuron.spike_output + 0.45 * np.clip(surround_current / 40.0, 0.0, 1.0)

        if self.filter_type == "on_center":
            drive = center_signal - 0.30 * surround_signal
        else:
            drive = surround_signal - 0.22 * center_signal

        self.output = float(np.clip(1.25 * drive, 0.0, 1.8))
        return self.output


class RetinalProcessing:
    def __init__(self, visual_field_size: Tuple[float, float] = (400.0, 400.0), num_filters_per_side: int = 10):
        self.visual_field_size = visual_field_size
        self.num_filters_per_side = num_filters_per_side
        self.filters: List[CenterSurroundFilter] = []

        for i in range(num_filters_per_side):
            for j in range(num_filters_per_side):
                x = (i + 0.5) * visual_field_size[0] / num_filters_per_side
                y = (j + 0.5) * visual_field_size[1] / num_filters_per_side
                filter_type = "on_center" if (i + j) % 2 == 0 else "off_center"
                self.filters.append(CenterSurroundFilter((x, y), filter_type=filter_type))

        self.retinal_output = np.zeros(len(self.filters), dtype=float)
        self.processed_image = None

    def process_visual_input(self, visual_scene: List[Tuple[float, float, float]]) -> np.ndarray:
        self.retinal_output = np.zeros(len(self.filters), dtype=float)

        for idx, filter_cell in enumerate(self.filters):
            total_response = 0.0
            for obj_x, obj_y, brightness in visual_scene:
                stimulus = np.array([obj_x, obj_y], dtype=float)
                total_response += filter_cell.process(stimulus, brightness=float(brightness))
            self.retinal_output[idx] = total_response

        self.processed_image = self.retinal_output.reshape((self.num_filters_per_side, self.num_filters_per_side))
        return self.retinal_output

    def get_spatial_attention_map(self) -> np.ndarray:
        if self.processed_image is None:
            return np.zeros((self.num_filters_per_side, self.num_filters_per_side), dtype=float)

        attention_map = np.clip(self.processed_image, 0.0, 1.0)
        kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        for _ in range(2):
            attention_map = np.convolve(attention_map.flatten(), kernel, mode="same").reshape(self.processed_image.shape)
        return attention_map

    def reset(self):
        for filter_cell in self.filters:
            filter_cell.center_neuron.reset()
            filter_cell.surround_neuron.reset()
        self.retinal_output = np.zeros(len(self.filters), dtype=float)
        self.processed_image = None

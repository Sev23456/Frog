#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymunk


def to_pymunk_vec(value: np.ndarray | List[float] | Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]), float(value[1])
    return 0.0, 0.0


class SNNBrain:
    """Sparse event-driven controller with LIF neurons and reward-modulated readout."""

    def __init__(self, neurons_per_feature: int = 8, lr: float = 0.02):
        self.input_dim = 6
        self.neurons_per_feature = neurons_per_feature
        self.n_neurons = self.input_dim * neurons_per_feature
        self.lr = lr

        self.w_in = np.zeros((self.n_neurons, self.input_dim), dtype=float)
        self.w_out = np.zeros((2, self.n_neurons), dtype=float)

        for feature in range(self.input_dim):
            start = feature * neurons_per_feature
            stop = start + neurons_per_feature
            self.w_in[start:stop, feature] = np.random.uniform(0.65, 1.15, neurons_per_feature)
            noise = np.random.uniform(-0.04, 0.04, (neurons_per_feature, self.input_dim))
            self.w_in[start:stop] += noise

        # Direction-selective readout: positive-x, negative-x, positive-y, negative-y.
        self.w_out[0, 0:neurons_per_feature] = 0.45
        self.w_out[0, neurons_per_feature : 2 * neurons_per_feature] = -0.45
        self.w_out[1, 2 * neurons_per_feature : 3 * neurons_per_feature] = 0.45
        self.w_out[1, 3 * neurons_per_feature : 4 * neurons_per_feature] = -0.45
        self.w_out[:, 4 * neurons_per_feature : 5 * neurons_per_feature] += 0.18

        self.bias = np.random.uniform(0.02, 0.05, self.n_neurons)
        self.v = np.zeros((self.n_neurons,), dtype=float)
        self.threshold = 0.24
        self.tau = 0.05
        self.refractory_time = 0.02
        self.refractory = np.zeros((self.n_neurons,), dtype=float)

        self.pre_trace = np.zeros((self.n_neurons,), dtype=float)
        self.e_out = np.zeros_like(self.w_out)
        self.tau_pre = 0.08
        self.tau_e = 0.35

        self.is_juvenile = True
        self.juvenile_age = 0
        self.juvenile_duration = 5000

    def encode(self, target_vec: np.ndarray, visual_range: float, energy_ratio: float) -> np.ndarray:
        distance = max(1.0, float(np.linalg.norm(target_vec)))
        direction = target_vec / distance
        proximity = max(0.0, 1.0 - distance / visual_range)
        return np.array(
            [
                max(direction[0], 0.0),
                max(-direction[0], 0.0),
                max(direction[1], 0.0),
                max(-direction[1], 0.0),
                proximity,
                energy_ratio,
            ],
            dtype=float,
        )

    def forward(self, x: np.ndarray, dt: float = 0.01):
        decay = np.exp(-dt / self.tau)
        pre_decay = np.exp(-dt / self.tau_pre)
        e_decay = np.exp(-dt / self.tau_e)

        active = self.refractory <= 0.0
        self.refractory = np.maximum(0.0, self.refractory - dt)

        input_current = self.w_in.dot(x) + self.bias
        self.v[active] = self.v[active] * decay + input_current[active] * (1.0 - decay)
        self.v[~active] = 0.0

        spikes = (self.v >= self.threshold).astype(float)
        if spikes.any():
            spike_mask = spikes.astype(bool)
            self.v[spike_mask] = 0.0
            self.refractory[spike_mask] = self.refractory_time

        out_raw = self.w_out.dot(spikes)
        out = np.tanh(out_raw)

        self.pre_trace = self.pre_trace * pre_decay + spikes
        self.e_out = self.e_out * e_decay + np.outer(out, self.pre_trace)

        return out, float(spikes.mean()), int(spikes.sum()), spikes

    def reward_update(self, reward: float):
        if reward == 0.0:
            return
        self.w_out += self.lr * reward * self.e_out
        self.w_out = np.clip(self.w_out, -1.5, 1.5)


class FrogSNNAgent:
    """Standalone SNN frog with sparse spike-driven motor commands."""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        training_mode: bool = False,
    ):
        self.space = space
        self.position = np.array(position, dtype=float)
        self.training_mode = training_mode

        # Parameters aligned with BioFrog for fair comparison
        self.radius = 9.0
        self.max_speed = 65.0
        self.visual_range = 40.0  # Same as BioFrog VISION_RADIUS_PX
        
        moment = pymunk.moment_for_circle(1.0, 0, self.radius)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = tuple(self.position)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)

        self.brain = SNNBrain()

        self.max_energy = 30.0
        self.energy = float(self.max_energy)
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.catch_cooldown = 18

        self.hit_radius = 9.0 if training_mode else 12.0
        self.success_prob = 0.72 if training_mode else 0.9

        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target: Optional[np.ndarray] = None
        self.attached_fly = None

        self._visible_targets: List[Any] = []
        self.last_velocity = np.zeros(2, dtype=float)
        self.last_spike_count = 0

    def remove(self):
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass

    def detect_flies(self, flies: List[Any]):
        visual_scene = []
        motion_vectors = []
        self._visible_targets = []
        for fly in flies:
            if hasattr(fly, "alive") and not fly.alive:
                continue
            fly_pos = fly.body.position if hasattr(fly, "body") else fly
            fly_vec = np.array((float(fly_pos[0]), float(fly_pos[1])), dtype=float)
            distance = float(np.linalg.norm(fly_vec - self.position))
            if distance < self.visual_range:
                brightness = max(0.0, 1.0 - (distance / self.visual_range))
                visual_scene.append((float(fly_vec[0]), float(fly_vec[1]), brightness))
                motion_vectors.append((float(fly_vec[0] - self.position[0]), float(fly_vec[1] - self.position[1])))
                self._visible_targets.append(fly)
        return visual_scene, motion_vectors

    def extend_tongue(self, target_position: np.ndarray):
        if not self.tongue_extended:
            self.tongue_extended = True
            self.tongue_target = np.array(target_position, dtype=float)
            self.tongue_length = 0.0

    def retract_tongue(self):
        self.tongue_extended = False
        self.tongue_length = 0.0
        self.attached_fly = None
        self.tongue_target = None

    def update(self, dt: float, flies: List[Any]) -> Dict[str, Any]:
        self.steps += 1
        self.position = np.array(self.body.position, dtype=float)
        self.brain.juvenile_age += 1
        if self.brain.juvenile_age >= self.brain.juvenile_duration:
            self.brain.is_juvenile = False

        _, motion_vectors = self.detect_flies(flies)

        reward = 0.0
        if self.attached_fly is not None:
            reward = 1.0
            self.energy = min(self.max_energy, self.energy + 5.0)
            self.attached_fly = None

        self.energy = max(0.0, self.energy - (0.08 + 0.025 * np.linalg.norm(self.last_velocity)) * dt)
        energy_ratio = self.energy / self.max_energy

        target_distance = None
        neural_activity = 0.0
        spike_count = 0
        alignment = 0.0

        if motion_vectors:
            distances = [float(np.linalg.norm(np.array(vec, dtype=float))) for vec in motion_vectors]
            nearest_idx = int(np.argmin(distances))
            nearest_vec = np.array(motion_vectors[nearest_idx], dtype=float)
            target_distance = distances[nearest_idx]
            encoded = self.brain.encode(nearest_vec, self.visual_range, energy_ratio)
            out, neural_activity, spike_count, _ = self.brain.forward(encoded, dt=dt)

            target_dir = nearest_vec / max(1.0, target_distance)
            burst_gain = min(1.0, spike_count / max(1, self.brain.neurons_per_feature))
            velocity = np.tanh(0.9 * out + 0.45 * burst_gain * target_dir)
            if spike_count == 0:
                velocity *= 0.25

            if self.training_mode:
                self.brain.reward_update(reward)

            if np.linalg.norm(velocity) > 1e-6:
                alignment = float(np.dot(velocity / np.linalg.norm(velocity), target_dir))
        else:
            drift = np.random.uniform(-0.08, 0.08, size=2)
            velocity = 0.7 * self.last_velocity + 0.3 * drift

        velocity = 0.55 * self.last_velocity + 0.45 * velocity
        if np.linalg.norm(velocity) > 1.0:
            velocity = velocity / np.linalg.norm(velocity)

        # Use max_speed instead of hardcoded 120.0 for fair comparison
        self.body.velocity = to_pymunk_vec(velocity * self.max_speed)
        self.last_velocity = velocity
        self.last_spike_count = spike_count

        if not self.tongue_extended and motion_vectors:
            distances = [float(np.linalg.norm(np.array(vec, dtype=float))) for vec in motion_vectors]
            nearest_idx = int(np.argmin(distances))
            if (
                distances[nearest_idx] < 125.0
                and spike_count >= 2
                and (self.steps - self.last_catch_time) > self.catch_cooldown
            ):
                target = self._visible_targets[nearest_idx]
                self.extend_tongue(np.array(target.body.position, dtype=float))

        if self.tongue_extended and self.tongue_target is not None:
            direction = self.tongue_target - self.position
            distance = float(np.linalg.norm(direction))
            if distance > 0:
                direction = direction / distance
            self.tongue_length += 320.0 * dt
            if self.tongue_length >= 150.0 or distance < self.tongue_length:
                self.retract_tongue()
            if self.attached_fly is None and (self.steps - self.last_catch_time) > self.catch_cooldown:
                tongue_end = self.position + direction * self.tongue_length
                for fly in self._visible_targets:
                    fly_pos = np.array(fly.body.position, dtype=float)
                    if np.linalg.norm(fly_pos - tongue_end) < self.hit_radius and random.random() < self.success_prob:
                        self.attached_fly = fly
                        self.caught_flies += 1
                        self.last_catch_time = self.steps
                        break

        return {
            "position": self.position,
            "velocity": velocity,
            "energy": self.energy,
            "caught_flies": self.caught_flies,
            "fatigue": 1.0 - (self.energy / self.max_energy),
            "is_juvenile": self.brain.is_juvenile,
            "juvenile_progress": min(1.0, self.brain.juvenile_age / self.brain.juvenile_duration),
            "tongue_extended": self.tongue_extended,
            "tongue_length": self.tongue_length,
            "controller_signal": neural_activity,
            "neural_activity": neural_activity,
            "spike_count": spike_count,
            "reward": reward,
            "caught_fly": self.attached_fly,
            "target_distance": target_distance,
            "alignment": alignment,
            "eligibility_norm": float(np.linalg.norm(self.brain.e_out)),
        }

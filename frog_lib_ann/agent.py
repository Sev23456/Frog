#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymunk

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def to_pymunk_vec(value: np.ndarray | List[float] | Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]), float(value[1])
    return 0.0, 0.0


if TORCH_AVAILABLE:
    class TorchPolicyNet(nn.Module):
        def __init__(self, input_dim: int = 4, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
                nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)


    class TorchBrain:
        def __init__(self, lr: float = 1e-3, device: Optional[str] = None):
            self.device = torch.device(device or "cpu")
            self.net = TorchPolicyNet().to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        def act(self, obs: np.ndarray) -> np.ndarray:
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                return self.net(x).cpu().numpy().reshape(2,)

        def learn(self, obs: np.ndarray, target: np.ndarray):
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            t = torch.tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.opt.zero_grad()
            pred = self.net(x)
            loss = nn.functional.mse_loss(pred, t)
            loss.backward()
            self.opt.step()


class NumpyPolicy:
    def __init__(self, input_dim: int = 4, hidden: int = 64, lr: float = 2e-3):
        self.w1 = np.random.randn(hidden, input_dim) * 0.15
        self.b1 = np.zeros((hidden,), dtype=float)
        self.w2 = np.random.randn(2, hidden) * 0.15
        self.b2 = np.zeros((2,), dtype=float)
        self.lr = lr

    def act(self, obs: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, self.w1.dot(obs) + self.b1)
        return np.tanh(self.w2.dot(hidden) + self.b2)

    def learn(self, obs: np.ndarray, target: np.ndarray):
        hidden = np.maximum(0.0, self.w1.dot(obs) + self.b1)
        pred_raw = self.w2.dot(hidden) + self.b2
        pred = np.tanh(pred_raw)
        err = pred - target
        tanh_grad = 1.0 - pred**2
        delta2 = err * tanh_grad
        dw2 = np.outer(delta2, hidden)
        db2 = delta2
        delta1 = (self.w2.T.dot(delta2)) * (hidden > 0.0)
        dw1 = np.outer(delta1, obs)
        db1 = delta1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1


class FrogANNAgent:
    """Conventional ANN frog controller with smooth continuous actions."""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        training_mode: bool = False,
        use_torch: Optional[bool] = None,
    ):
        self.space = space
        self.position = np.array(position, dtype=float)
        self.training_mode = training_mode

        moment = pymunk.moment_for_circle(1.0, 0, 30.0)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = tuple(self.position)
        self.shape = pymunk.Circle(self.body, 30.0)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)

        if use_torch is None:
            use_torch = TORCH_AVAILABLE
        self.brain = TorchBrain() if use_torch and TORCH_AVAILABLE else NumpyPolicy()

        self.max_energy = 30.0
        self.energy = float(self.max_energy)
        self.visual_range = 220.0
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.catch_cooldown = 20

        self.hit_radius = 24.0 if training_mode else 34.0
        self.success_prob = 0.75 if training_mode else 0.92

        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target: Optional[np.ndarray] = None
        self.attached_fly = None

        self._visible_targets: List[Any] = []
        self.last_velocity = np.zeros(2, dtype=float)

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

    def _observation(self, target_vec: np.ndarray) -> np.ndarray:
        distance = max(1.0, float(np.linalg.norm(target_vec)))
        direction = target_vec / distance
        proximity = max(0.0, 1.0 - (distance / self.visual_range))
        energy_ratio = self.energy / self.max_energy
        return np.array([direction[0], direction[1], proximity, energy_ratio], dtype=float)

    def update(self, dt: float, flies: List[Any]) -> Dict[str, Any]:
        self.steps += 1
        self.position = np.array(self.body.position, dtype=float)

        _, motion_vectors = self.detect_flies(flies)

        reward = 0.0
        if self.attached_fly is not None:
            reward = 1.0
            self.energy = min(self.max_energy, self.energy + 5.0)
            self.attached_fly = None

        self.energy = max(0.0, self.energy - (0.08 + 0.02 * np.linalg.norm(self.last_velocity)) * dt)

        controller_signal = 0.0
        target_distance = None
        alignment = 0.0

        if motion_vectors:
            distances = [float(np.linalg.norm(np.array(vec, dtype=float))) for vec in motion_vectors]
            nearest_idx = int(np.argmin(distances))
            nearest_vec = np.array(motion_vectors[nearest_idx], dtype=float)
            target_distance = distances[nearest_idx]
            obs = self._observation(nearest_vec)
            pursuit_dir = nearest_vec / max(1.0, target_distance)

            policy_velocity = self.brain.act(obs)
            velocity = np.tanh(0.65 * policy_velocity + 0.55 * pursuit_dir)
            controller_signal = float(np.linalg.norm(policy_velocity))
            alignment = float(np.dot(velocity / max(np.linalg.norm(velocity), 1e-6), pursuit_dir))

            if self.training_mode:
                self.brain.learn(obs, pursuit_dir)
        else:
            angle = random.uniform(0.0, 2.0 * math.pi)
            drift = np.array([math.cos(angle), math.sin(angle)], dtype=float) * 0.15
            velocity = np.tanh(0.8 * self.last_velocity + 0.2 * drift)

        velocity = 0.78 * self.last_velocity + 0.22 * velocity
        speed = float(np.linalg.norm(velocity))
        if speed > 1.0:
            velocity = velocity / speed

        self.body.velocity = to_pymunk_vec(velocity * 115.0)
        self.last_velocity = velocity

        if not self.tongue_extended and motion_vectors:
            distances = [float(np.linalg.norm(np.array(vec, dtype=float))) for vec in motion_vectors]
            nearest_idx = int(np.argmin(distances))
            if (
                distances[nearest_idx] < 125.0
                and alignment > 0.2
                and controller_signal > 0.08
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
            "tongue_extended": self.tongue_extended,
            "tongue_length": self.tongue_length,
            "controller_signal": controller_signal,
            "neural_activity": controller_signal,
            "reward": reward,
            "caught_fly": self.attached_fly,
            "target_distance": target_distance,
            "alignment": alignment,
        }

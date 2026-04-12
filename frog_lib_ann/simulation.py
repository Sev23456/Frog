#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pymunk

from .agent import FrogANNAgent


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def to_pymunk_vec(value) -> Tuple[float, float]:
    if isinstance(value, np.ndarray):
        return float(value[0]), float(value[1])
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def to_pygame_vec(value) -> Tuple[int, int]:
    if isinstance(value, np.ndarray):
        return int(value[0]), int(value[1])
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return 0, 0


class Fly:
    def __init__(self, space: pymunk.Space, position: Tuple[float, float]):
        self.space = space
        self.alive = True
        moment = pymunk.moment_for_circle(0.1, 0, 5.0)
        self.body = pymunk.Body(0.1, moment)
        self.body.position = to_pymunk_vec(position)
        self.body.velocity = (random.uniform(-45, 45), random.uniform(-45, 45))
        self.shape = pymunk.Circle(self.body, 5.0)
        self.shape.elasticity = 0.9
        self.shape.friction = 0.5
        self.space.add(self.body, self.shape)

    def update(self, dt: float, width: int, height: int):
        if not self.alive:
            return
        pos = self.body.position
        vel = self.body.velocity
        if pos.x < 10 or pos.x > width - 10:
            vel = (-vel[0], vel[1])
        if pos.y < 10 or pos.y > height - 10:
            vel = (vel[0], -vel[1])
        if random.random() < 0.08:
            vel = (random.uniform(-55, 55), random.uniform(-55, 55))
        self.body.velocity = vel

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        pos = to_pygame_vec(self.body.position)
        pygame.draw.circle(surface, (150, 150, 255), pos, 9)
        pygame.draw.circle(surface, (70, 70, 170), pos, 5)

    def remove(self):
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass
        self.alive = False


class ANNFlyCatchingSimulation:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        num_flies: int = 15,
        headless: bool = False,
        training_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.num_flies = num_flies
        self.headless = headless
        self.training_mode = training_mode

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.dt = 0.01
        self.physics_dt = 0.01

        self.frog = FrogANNAgent(self.space, position=(width // 2, height // 2), training_mode=training_mode)
        self.flies: List[Fly] = []
        self.spawn_flies(num_flies)

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("ANN Frog - smooth policy pursuit")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 20)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.step_count = 0
        self.caught_count = 0
        self.energy_history = deque(maxlen=2000)
        self.catch_history = deque(maxlen=2000)
        self.distance_history = deque(maxlen=2000)
        self.alignment_history = deque(maxlen=2000)
        self.speed_history = deque(maxlen=2000)
        self.smoothness_history = deque(maxlen=2000)
        self.controller_history = deque(maxlen=2000)
        self._last_velocity = np.zeros(2, dtype=float)

    def _spawn_position(self) -> Tuple[float, float]:
        margin_x = min(100, max(20, self.width // 5))
        margin_y = min(100, max(20, self.height // 5))
        return (
            random.uniform(margin_x, max(margin_x, self.width - margin_x)),
            random.uniform(margin_y, max(margin_y, self.height - margin_y)),
        )

    def spawn_flies(self, count: int):
        for _ in range(count):
            self.flies.append(Fly(self.space, self._spawn_position()))

    def respawn_flies(self):
        self.flies = [fly for fly in self.flies if fly.alive]
        missing = self.num_flies - len(self.flies)
        if missing > 0:
            self.spawn_flies(missing)

    def _nearest_live_fly_distance(self) -> Optional[float]:
        alive = [fly for fly in self.flies if fly.alive]
        if not alive:
            return None
        frog_pos = np.array(self.frog.body.position, dtype=float)
        return min(float(np.linalg.norm(np.array(fly.body.position, dtype=float) - frog_pos)) for fly in alive)

    def step(self) -> Dict[str, Any]:
        for fly in self.flies:
            fly.update(self.dt, self.width, self.height)

        agent_state = self.frog.update(self.dt, self.flies)
        self.space.step(self.dt)

        if agent_state["caught_fly"] is not None and getattr(agent_state["caught_fly"], "alive", False):
            agent_state["caught_fly"].remove()
            self.caught_count += 1
            self.catch_history.append(1)
        else:
            self.catch_history.append(0)

        velocity = np.array(agent_state["velocity"], dtype=float)
        speed = float(np.linalg.norm(velocity))
        previous_speed = float(np.linalg.norm(self._last_velocity))
        smoothness = 1.0
        if speed > 1e-6 and previous_speed > 1e-6:
            smoothness = float(np.dot(velocity, self._last_velocity) / (speed * previous_speed))

        self.energy_history.append(float(agent_state["energy"]))
        self.distance_history.append(agent_state["target_distance"] if agent_state["target_distance"] is not None else float("nan"))
        self.alignment_history.append(float(agent_state["alignment"]))
        self.speed_history.append(speed)
        self.smoothness_history.append(smoothness)
        self.controller_history.append(float(agent_state["controller_signal"]))
        self._last_velocity = velocity

        self.respawn_flies()
        self.step_count += 1
        return agent_state

    def draw(self):
        if self.headless or self.screen is None:
            return
        self.screen.fill((238, 242, 220))
        for fly in self.flies:
            fly.draw(self.screen)

        frog_pos = to_pygame_vec(self.frog.body.position)
        pygame.draw.circle(self.screen, (20, 135, 40), frog_pos, 30)
        pygame.draw.circle(self.screen, (40, 185, 60), frog_pos, 24)

        if self.frog.tongue_extended and self.frog.tongue_target is not None:
            pygame.draw.line(self.screen, (240, 90, 90), frog_pos, to_pygame_vec(self.frog.tongue_target), 3)

        stats = [
            f"Steps: {self.step_count}",
            f"Catches: {self.frog.caught_flies}",
            f"Energy: {self.frog.energy:.1f}",
            f"Avg alignment: {np.nanmean(self.alignment_history) if self.alignment_history else 0.0:.2f}",
            f"Controller: {self.controller_history[-1] if self.controller_history else 0.0:.2f}",
        ]
        for idx, text in enumerate(stats):
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, 10 + idx * 22))

        pygame.display.flip()

    def run_simulation(self, max_steps: int = 2000):
        print(f"ANN simulation start: steps={max_steps}, headless={self.headless}, training={self.training_mode}")
        for _ in range(max_steps):
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
            self.step()
            if not self.headless:
                self.draw()
                self.clock.tick(60)

        stats = self.get_statistics()
        print(
            "ANN summary | catches={caught_flies} success={success_rate:.3f} "
            "alignment={avg_alignment:.3f} smoothness={movement_smoothness:.3f}".format(**stats)
        )

    def get_statistics(self) -> Dict[str, Any]:
        valid_distances = [value for value in self.distance_history if not np.isnan(value)]
        return {
            "total_steps": self.step_count,
            "caught_flies": self.frog.caught_flies,
            "success_rate": (sum(self.catch_history) / len(self.catch_history)) if self.catch_history else 0.0,
            "final_energy": float(self.frog.energy),
            "avg_alignment": float(np.mean(self.alignment_history)) if self.alignment_history else 0.0,
            "avg_distance_to_target": float(np.mean(valid_distances)) if valid_distances else float("nan"),
            "avg_speed": float(np.mean(self.speed_history)) if self.speed_history else 0.0,
            "movement_smoothness": float(np.mean(self.smoothness_history)) if self.smoothness_history else 1.0,
            "avg_controller_signal": float(np.mean(self.controller_history)) if self.controller_history else 0.0,
            "architecture_signature": "smooth continuous pursuit",
        }

    def save_state(self, filename: str = "ann_frog_state.json"):
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.get_statistics(), handle, indent=2, ensure_ascii=False)

    def plot_results(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plotting is unavailable: {exc}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle("ANN frog behavior summary")
        axes[0, 0].plot(list(self.energy_history))
        axes[0, 0].set_title("Energy")
        axes[0, 1].plot(list(self.alignment_history))
        axes[0, 1].set_title("Pursuit alignment")
        axes[1, 0].plot(list(self.speed_history))
        axes[1, 0].set_title("Speed")
        axes[1, 1].plot(list(self.controller_history))
        axes[1, 1].set_title("Controller signal")
        for row in axes:
            for axis in row:
                axis.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.screen is not None:
            pygame.quit()
        self.frog.remove()
        for fly in self.flies:
            fly.remove()

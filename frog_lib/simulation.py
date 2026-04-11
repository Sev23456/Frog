"""
BioFrog v2.0 - simulation and visualization for the toy bio-inspired agent.
"""

from __future__ import annotations

import json
import random
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pymunk

from .bio_frog_agent import BioFrogAgent


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
        self.body.velocity = (random.uniform(-50, 50), random.uniform(-50, 50))
        self.shape = pymunk.Circle(self.body, 5.0)
        self.shape.elasticity = 0.9
        self.shape.friction = 0.5
        self.space.add(self.body, self.shape)

    def update(self, dt: float, width: int = 800, height: int = 600):
        if not self.alive:
            return
        pos = self.body.position
        vel = self.body.velocity
        if pos.x < 10 or pos.x > width - 10:
            vel = (-vel[0], vel[1])
        if pos.y < 10 or pos.y > height - 10:
            vel = (vel[0], -vel[1])
        if random.random() < 0.1:
            vel = (random.uniform(-55, 55), random.uniform(-55, 55))
        self.body.velocity = vel

    def draw(self, surface: pygame.Surface):
        if not self.alive:
            return
        pos = to_pygame_vec(self.body.position)
        pygame.draw.circle(surface, (150, 150, 255), pos, 10)
        pygame.draw.circle(surface, (80, 80, 180), pos, 6)
        pygame.draw.circle(surface, (255, 255, 255), (pos[0] - 3, pos[1] - 2), 2)
        pygame.draw.circle(surface, (255, 255, 255), (pos[0] + 3, pos[1] - 2), 2)

    def remove(self):
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass
        self.alive = False


class BioFlyCatchingSimulation:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        skip_training: bool = False,
        bio_mode: bool = True,
        juvenile_mode: bool = True,
        num_flies: int = 15,
        headless: bool = False,
    ):
        self.width = width
        self.height = height
        self.bio_mode = bio_mode
        self.juvenile_mode = juvenile_mode
        self.num_flies = num_flies
        self.headless = headless

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.dt = 0.01

        self.frog = BioFrogAgent(
            self.space,
            position=(width // 2, height // 2),
            space_size=(width, height),
            bio_mode=bio_mode,
            juvenile_mode=juvenile_mode,
            training_mode=not skip_training,
            instinct_mode=skip_training,
        )

        self.flies: List[Fly] = []
        self.spawn_flies(num_flies)

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("BioFrog - toy bio-inspired pursuit")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 20)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.step_count = 0
        self.caught_count = 0
        self.energy_history = deque(maxlen=2000)
        self.dopamine_history = deque(maxlen=2000)
        self.catch_history = deque(maxlen=2000)
        self.juvenile_history = deque(maxlen=2000)
        self.neural_activity_history = deque(maxlen=2000)
        self.fatigue_history = deque(maxlen=2000)
        self.distance_history = deque(maxlen=2000)
        self.alignment_history = deque(maxlen=2000)
        self.speed_history = deque(maxlen=2000)
        self.exploration_history = deque(maxlen=2000)

        print(
            "BioFrog simulation start | size={}x{} flies={} juvenile={} headless={}".format(
                width,
                height,
                num_flies,
                juvenile_mode,
                headless,
            )
        )

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

    def respawn_dead_flies(self):
        self.flies = [fly for fly in self.flies if fly.alive]
        missing = self.num_flies - len(self.flies)
        if missing > 0:
            self.spawn_flies(missing)

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
        exploring = 1.0 if agent_state["target_distance"] is None and speed > 0.05 else 0.0

        self.energy_history.append(float(agent_state.get("energy", 0.0)))
        self.dopamine_history.append(float(agent_state.get("dopamine", 0.0)))
        self.juvenile_history.append(1.0 if agent_state.get("is_juvenile") else 0.0)
        self.neural_activity_history.append(float(agent_state.get("neural_activity", 0.0)))
        self.fatigue_history.append(float(agent_state.get("fatigue", 0.0)))
        self.distance_history.append(
            float(agent_state["target_distance"]) if agent_state["target_distance"] is not None else float("nan")
        )
        self.alignment_history.append(float(agent_state.get("alignment", 0.0)))
        self.speed_history.append(speed)
        self.exploration_history.append(exploring)

        self.step_count += 1
        self.respawn_dead_flies()
        return agent_state

    def draw(self):
        if self.screen is None or self.headless:
            return

        self.screen.fill((230, 240, 180))
        for y in range(0, self.height, 40):
            color = (100 + random.randint(0, 50), 150 + random.randint(0, 50), 50 + random.randint(0, 30))
            pygame.draw.rect(self.screen, color, (0, y, self.width, 20))

        for fly in self.flies:
            fly.draw(self.screen)

        position = to_pygame_vec(self.frog.position)
        pygame.draw.circle(self.screen, (0, 100, 0), position, 30)
        pygame.draw.circle(self.screen, (0, 150, 0), position, 25)
        pygame.draw.circle(self.screen, (0, 180, 0), to_pygame_vec(np.array(position) + np.array([0, -15])), 15)

        for offset in [np.array([-8, -20]), np.array([8, -20])]:
            eye_pos = to_pygame_vec(np.array(position) + offset)
            pygame.draw.circle(self.screen, (255, 255, 255), eye_pos, 8)
            pygame.draw.circle(self.screen, (0, 0, 0), eye_pos, 4)

        if self.frog.tongue_extended and self.frog.tongue_target is not None:
            pygame.draw.line(self.screen, (255, 100, 100), position, to_pygame_vec(self.frog.tongue_target), 3)

        self.draw_stats()
        pygame.display.flip()

    def draw_stats(self):
        stats = [
            f"Steps: {self.step_count}",
            f"Flies caught: {self.frog.caught_flies}",
            f"Energy: {self.frog.energy:.1f}",
            f"Dopamine: {self.frog.brain.dopamine_level:.2f}",
            f"Neural activity: {self.neural_activity_history[-1] if self.neural_activity_history else 0.0:.2f}",
        ]
        for idx, text in enumerate(stats):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + idx * 24))

    def run_simulation(self, max_steps: int = 20000):
        print(f"Running BioFrog for {max_steps} steps...")
        try:
            for step in range(max_steps):
                if not self.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return

                agent_state = self.step()

                if not self.headless:
                    self.draw()
                    self.clock.tick(60)

                if step % 1000 == 0:
                    print(
                        "  step={:5d} catches={} energy={:.2f} dopamine={:.2f} neural={:.2f}".format(
                            step,
                            self.frog.caught_flies,
                            agent_state.get("energy", 0.0),
                            agent_state.get("dopamine", 0.0),
                            agent_state.get("neural_activity", 0.0),
                        )
                    )
        except KeyboardInterrupt:
            print("Simulation interrupted.")

        stats = self.get_statistics()
        print(
            "BioFrog summary | catches={caught_flies} success={success_rate:.3f} "
            "dopamine={avg_dopamine:.3f} exploration={exploration_ratio:.3f}".format(**stats)
        )

    def plot_results(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plotting is unavailable: {exc}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("BioFrog toy bio-inspired behavior")

        axes[0, 0].plot(list(self.energy_history), label="Energy", color="green")
        axes[0, 0].plot(list(self.fatigue_history), label="Fatigue", color="red", alpha=0.7)
        axes[0, 0].set_title("Body state")
        axes[0, 0].legend()

        axes[0, 1].plot(list(self.dopamine_history), label="Dopamine", color="orange")
        axes[0, 1].plot(list(self.neural_activity_history), label="Neural activity", color="blue")
        axes[0, 1].set_title("Modulation and activity")
        axes[0, 1].legend()

        catch_values = list(self.catch_history)
        if len(catch_values) >= 20:
            catch_smooth = np.convolve(catch_values, np.ones(20) / 20, mode="valid")
        else:
            catch_smooth = catch_values
        axes[1, 0].plot(catch_smooth, label="Catch rate", color="purple")
        axes[1, 0].set_title("Hunting success")
        axes[1, 0].legend()

        axes[1, 1].plot(list(self.exploration_history), label="Exploration", color="brown")
        axes[1, 1].plot(list(self.juvenile_history), label="Juvenile mode", color="teal", alpha=0.7)
        axes[1, 1].set_title("Exploration and developmental state")
        axes[1, 1].legend()

        for row in axes:
            for axis in row:
                axis.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("biofrog_results.png", dpi=150, bbox_inches="tight")
        plt.show()

    def reset_simulation(self):
        self.frog.remove()
        for fly in self.flies:
            fly.remove()
        self.flies.clear()

        self.frog = BioFrogAgent(
            self.space,
            position=(self.width // 2, self.height // 2),
            space_size=(self.width, self.height),
            bio_mode=self.bio_mode,
            juvenile_mode=self.juvenile_mode,
        )
        self.spawn_flies(self.num_flies)

        self.step_count = 0
        self.caught_count = 0
        self.energy_history.clear()
        self.dopamine_history.clear()
        self.catch_history.clear()
        self.juvenile_history.clear()
        self.neural_activity_history.clear()
        self.fatigue_history.clear()
        self.distance_history.clear()
        self.alignment_history.clear()
        self.speed_history.clear()
        self.exploration_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        success_rate = (sum(self.catch_history) / len(self.catch_history)) if self.catch_history else 0.0
        valid_distances = [value for value in self.distance_history if not np.isnan(value)]
        return {
            "total_steps": self.step_count,
            "caught_flies": self.frog.caught_flies,
            "success_rate": success_rate,
            "final_energy": self.frog.energy,
            "avg_dopamine": float(np.mean(self.dopamine_history)) if self.dopamine_history else 0.0,
            "avg_neural_activity": float(np.mean(self.neural_activity_history)) if self.neural_activity_history else 0.0,
            "avg_fatigue": float(np.mean(self.fatigue_history)) if self.fatigue_history else 0.0,
            "avg_alignment": float(np.mean(self.alignment_history)) if self.alignment_history else 0.0,
            "avg_distance_to_target": float(np.mean(valid_distances)) if valid_distances else float("nan"),
            "avg_speed": float(np.mean(self.speed_history)) if self.speed_history else 0.0,
            "exploration_ratio": float(np.mean(self.exploration_history)) if self.exploration_history else 0.0,
            "is_juvenile": self.frog.brain.is_juvenile,
            "juvenile_age": self.frog.brain.juvenile_age,
            "juvenile_progress": min(1.0, self.frog.brain.juvenile_age / self.frog.brain.juvenile_duration),
            "architecture_signature": "stateful neuromodulated exploratory pursuit",
        }

    def save_state(self, filename: str = "biofrog_state.json"):
        state = {
            "frog": self.get_statistics(),
            "step_count": self.step_count,
            "caught_count": self.caught_count,
            "timestamp": str(np.datetime64("now")),
        }
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)

    def close(self):
        if self.screen is not None:
            pygame.quit()
        self.frog.remove()
        for fly in self.flies:
            fly.remove()

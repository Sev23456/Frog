"""
Biorealistic Frog Predator Neuro - Fly Catching Simulation
Adapted from frog_lib_ann with BioFrogBrain neural architecture
"""

import json
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pymunk

from Frog_predator_neuro.brain import BioFrogBrain
from Frog_predator_neuro.config import (
    AGENT_BODY_ENERGY_MAX,
    AGENT_BODY_ENERGY_START,
    AGENT_MAX_SPEED,
    AGENT_IDLE_DRAIN,
    AGENT_MOVE_DRAIN,
)

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
    """Simple fly physics object"""
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


class BioFrogPredatorAgent:
    """Biorealistic frog with BioFrogBrain for fly catching"""
    
    def __init__(
        self,
        agent_id: int = 1,
        space: pymunk.Space = None,
        position: Tuple[float, float] = (400, 300),
        training_mode: bool = True,
    ):
        self.agent_id = agent_id
        self.space = space or pymunk.Space()
        self.training_mode = training_mode
        
        # Physics body
        moment = pymunk.moment_for_circle(1.0, 0, 30.0)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = to_pymunk_vec(position)
        self.body.velocity = (0, 0)
        self.shape = pymunk.Circle(self.body, 30.0)
        self.shape.elasticity = 0.5
        self.shape.friction = 0.8
        self.space.add(self.body, self.shape)
        
        # Brain
        self.brain = BioFrogBrain()
        
        # Energy and state
        self.energy = AGENT_BODY_ENERGY_START
        self.caught_flies = 0
        self.alive = True
        
        # Catch parameters (consistent with frog_lib_ann and frog_lib_snn)
        self.hit_radius = 24.0 if training_mode else 34.0
        self.success_prob = 0.75 if training_mode else 0.92
        self.tongue_reach = 60.0  # Virtual tongue reach for simplified model
        
        # History
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.last_velocity = np.array([0.0, 0.0], dtype=float)
        self.tongue_extended = False
        self.tongue_target = None
        self.color = (80, 200, 100)
        
    def _get_fly_observations(self, flies: List[Fly], width: int, height: int) -> Tuple[np.ndarray, Optional[Fly], float]:
        """Compute observations of flies around frog. Returns (obs, nearest_fly, distance)"""
        if not flies:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float), None, 0.0
        
        frog_pos = np.array(self.body.position, dtype=float)
        alive_flies = [f for f in flies if f.alive]
        
        if not alive_flies:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float), None, 0.0
        
        # Find nearest fly
        distances = [np.linalg.norm(np.array(f.body.position, dtype=float) - frog_pos) for f in alive_flies]
        nearest_idx = np.argmin(distances)
        nearest_fly = alive_flies[nearest_idx]
        nearest_pos = np.array(nearest_fly.body.position, dtype=float)
        
        # Compute relative position and velocity
        rel_pos = nearest_pos - frog_pos
        rel_dist = np.linalg.norm(rel_pos)
        
        # Normalization
        if rel_dist < 1e-6:
            rel_pos_norm = np.array([0.0, 0.0], dtype=float)
            alignment = 0.0
        else:
            rel_pos_norm = rel_pos / (rel_dist + 1e-6)
            alignment = float(np.dot(self.velocity, rel_pos_norm) / (np.linalg.norm(self.velocity) + 1e-6))
        
        # Obs: [fly_dir_x, fly_dir_y, distance_norm, alignment]
        obs = np.array([
            float(rel_pos_norm[0]),
            float(rel_pos_norm[1]),
            min(1.0, rel_dist / 200.0),
            max(-1.0, min(1.0, alignment))
        ], dtype=float)
        
        return obs, nearest_fly, rel_dist
    
    def update(self, dt: float, flies: List[Fly], width: int, height: int) -> Dict[str, Any]:
        """Update frog brain & physics"""
        obs, target_fly, actual_distance = self._get_fly_observations(flies, width, height)
        
        # Update brain with observation
        visual_input = obs[:2]  # fly direction
        self.brain.visual_system.inputs = visual_input
        # Note: In fly-catching mode, we skip full affective state update
        # Full update requires maze perception which we don't have here
        
        # Compute motor output (simplified)
        fly_dir = obs[:2]
        alignment = obs[3]
        
        # Simple pursuit: move towards nearest fly
        if np.linalg.norm(fly_dir) > 1e-6:
            drive_force = fly_dir * AGENT_MAX_SPEED * 0.8
        else:
            drive_force = np.array([0.0, 0.0], dtype=float)
        
        # Apply velocity limit
        self.velocity = np.array(self.body.velocity, dtype=float)
        target_vel = drive_force
        self.body.velocity = to_pymunk_vec(target_vel)
        
        # Check tongue/catch
        caught_fly = None
        self.tongue_extended = False
        # Catch range: frog_radius + tongue_reach + hit_radius
        catch_distance = self.shape.radius + self.tongue_reach + self.hit_radius
        if target_fly is not None and actual_distance < catch_distance:
            self.tongue_extended = True
            self.tongue_target = to_pygame_vec(target_fly.body.position)
            if random.random() < self.success_prob:
                caught_fly = target_fly
                self.caught_flies += 1
                # Energy reward: match frog_lib_ann (5.0 per fly)
                self.energy = min(AGENT_BODY_ENERGY_MAX, self.energy + 5.0)
        
        # Energy drain (match frog_lib_ann/snn)
        speed = np.linalg.norm(self.velocity)
        energy_drain = AGENT_IDLE_DRAIN + AGENT_MOVE_DRAIN * speed
        self.energy = max(0.0, self.energy - energy_drain * dt)
        
        if self.energy <= 0:
            self.alive = False
        
        self.last_velocity = self.velocity.copy()
        
        return {
            "velocity": self.velocity,
            "alignment": alignment,
            "target_distance": actual_distance if target_fly else None,
            "caught_fly": caught_fly,
            "energy": self.energy,
            "controller_signal": float(np.linalg.norm(drive_force) / (AGENT_MAX_SPEED + 1e-6)),
        }
    
    def draw(self, surface: pygame.Surface):
        """Draw frog and tongue"""
        frog_pos = to_pygame_vec(self.body.position)
        pygame.draw.circle(surface, self.color, frog_pos, 30)
        pygame.draw.circle(surface, (100, 220, 130), frog_pos, 24)
        
        if self.tongue_extended and self.tongue_target is not None:
            pygame.draw.line(surface, (240, 90, 90), frog_pos, self.tongue_target, 2)
    
    def remove(self):
        """Clean up physics body"""
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass
        self.alive = False


class BiorealisticFlyCatchingSimulation:
    """Frog predator neuro - biorealistic fly catching simulation"""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        num_flies: int = 15,
        headless: bool = False,
        training_mode: bool = True,
        agent_count: int = 1,
    ):
        self.width = width
        self.height = height
        self.num_flies = num_flies
        self.headless = headless
        self.training_mode = training_mode
        self.agent_count = agent_count

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.dt = 0.01

        # Create single agent
        self.frogs = [
            BioFrogPredatorAgent(
                agent_id=1,
                space=self.space,
                position=(width // 2, height // 2),
                training_mode=training_mode,
            )
        ]
        
        self.flies: List[Fly] = []
        self.spawn_flies(num_flies)

        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Biorealistic Frog Predator Neuro")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 18)
        else:
            # Headless mode: dummy display
            self.screen = None
            self.clock = None
            self.font = None

        self.step_count = 0
        self.running = True
        self.paused = False

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

    def handle_events(self):
        """Process input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def step(self) -> Dict[str, Any]:
        """Update one time step"""
        for fly in self.flies:
            fly.update(self.dt, self.width, self.height)

        agents_state = []
        for frog in self.frogs:
            if not frog.alive:
                continue
            state = frog.update(self.dt, self.flies, self.width, self.height)
            agents_state.append(state)
            
            if state["caught_fly"] is not None and getattr(state["caught_fly"], "alive", False):
                state["caught_fly"].remove()

        self.space.step(self.dt)
        self.respawn_flies()
        self.step_count += 1
        
        return {"agents": agents_state, "step": self.step_count}

    def update(self, dt: float = None):
        """Update simulation"""
        if not self.paused:
            self.step()

    def draw(self):
        """Render frame"""
        if self.headless or self.screen is None:
            return
        
        self.screen.fill((238, 242, 220))
        
        for fly in self.flies:
            fly.draw(self.screen)
        
        for frog in self.frogs:
            if frog.alive:
                frog.draw(self.screen)

        # HUD
        y_pos = 10
        for frog in self.frogs:
            text = f"Frog {frog.agent_id} | Catches: {frog.caught_flies} | Energy: {frog.energy:.2f}"
            color = (0, 0, 0) if frog.alive else (150, 150, 150)
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (10, y_pos))
            y_pos += 20

        stats_text = f"Step: {self.step_count} | Flies: {len([f for f in self.flies if f.alive])}"
        surface = self.font.render(stats_text, True, (0, 0, 0))
        self.screen.blit(surface, (10, y_pos))

        pygame.display.flip()

    def run(self, max_steps: int = 6000):
        """Main game loop"""
        print(f"BioFrog simulation starting | steps={max_steps} | headless={self.headless}")
        
        while self.running and self.step_count < max_steps:
            dt = 1.0 / 60.0
            
            if self.headless:
                pygame.event.pump()
            else:
                self.handle_events()
            
            if not self.paused:
                self.update(dt)
            
            if not self.headless:
                self.draw()
                if self.clock:
                    self.clock.tick(60)

        stats = self.get_statistics()
        print(f"BioFrog summary | frogs_alive={stats['frogs_alive']} | total_catches={stats['total_catches']}")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Collect final statistics"""
        return {
            "total_steps": self.step_count,
            "frogs_alive": len([f for f in self.frogs if f.alive]),
            "total_catches": sum(f.caught_flies for f in self.frogs),
            "avg_energy": float(np.mean([f.energy for f in self.frogs]) if self.frogs else 0.0),
            "architecture": "biorealistic neural with fly catching",
        }

    def save_state(self, filename: str = "biofrog_state.json"):
        """Save statistics"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.get_statistics(), f, indent=2, ensure_ascii=False)

    def close(self):
        """Cleanup"""
        if self.screen is not None:
            pygame.quit()
        for frog in self.frogs:
            frog.remove()
        for fly in self.flies:
            fly.remove()


# Aliases for compatibility
Simulation = BiorealisticFlyCatchingSimulation

"""
BioFrog v2.0 - toy bio-inspired frog agent.

This module intentionally stays bio-inspired rather than biologically exact:
the goal is a stateful, neuromodulated, spiking-flavored controller whose
behavior differs from ANN and SNN baselines in a readable way.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pymunk

from .architecture.motor_hierarchy import MotorHierarchy
from .architecture.tectum import Tectum
from .architecture.visual_system import RetinalProcessing
from .core.biological_neuron import FastSpikingInterneuron, LIFNeuron, PyramidalNeuron
from .core.glial_cells import GlialNetwork
from .core.neurotransmitter_diffusion import NeurotransmitterDiffusion
from .core.synapse_models import BiologicalSynapse, DynamicSynapse
from .metabolism.systemic_metabolism import NeuronMetabolism, SystemicMetabolism
from .plasticity.functional_plasticity import FunctionalPlasticityManager
from .plasticity.structural_plasticity import StructuralPlasticityManager


def to_pymunk_vec(value: Union[np.ndarray, list, tuple]) -> Tuple[float, float]:
    if isinstance(value, np.ndarray):
        return float(value[0]), float(value[1])
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return 0.0, 0.0


class BioFrogBrain:
    """Toy bio-inspired controller with retinal, tectal, motor and modulatory state."""

    def __init__(self, space_size: Tuple[int, int] = (800, 600), juvenile_mode: bool = True, dt: float = 0.01):
        self.space_size = space_size
        self.dt = dt

        self.is_juvenile = juvenile_mode
        self.juvenile_duration = 5000
        self.juvenile_age = 0

        self.visual_system = RetinalProcessing(visual_field_size=space_size)
        self.neurotransmitter_system = NeurotransmitterDiffusion(space_size=space_size)
        self.tectum = Tectum(columns=16)
        self.motor_hierarchy = MotorHierarchy()
        self.glial_network = GlialNetwork(num_astrocytes=24)

        self.metabolism = SystemicMetabolism()
        self.neuron_metabolism = NeuronMetabolism()
        self.functional_plasticity = FunctionalPlasticityManager()
        self.structural_plasticity = StructuralPlasticityManager()

        self.dopamine_level = 0.8 if juvenile_mode else 0.45
        self.serotonin_level = 0.7 if juvenile_mode else 0.5

        self.synapses: List[BiologicalSynapse] = [BiologicalSynapse() for _ in range(12)]
        self.modulatory_synapses: List[DynamicSynapse] = [DynamicSynapse() for _ in range(4)]

        self.steps = 0
        self.activity_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.dopamine_history = deque(maxlen=1000)
        self.exploration_bonus = 0.0

        self.plastic_neurons = [flt.center_neuron for flt in self.visual_system.filters]
        for column in self.tectum.columns:
            self.plastic_neurons.extend(column.pyramidal_neurons)
            self.plastic_neurons.extend(column.output_neurons)
        self.plastic_neurons.extend(self.motor_hierarchy.executive_neurons)
        self.plastic_neurons.extend(self.motor_hierarchy.motor_neurons)

    def process_sensory_input(self, visual_scene: List[Tuple[float, float, float]]) -> Dict[str, np.ndarray]:
        retinal_output = self.visual_system.process_visual_input(visual_scene)
        attention_map = self.visual_system.get_spatial_attention_map()
        return {
            "retinal_output": retinal_output,
            "attention_map": attention_map,
            "attention_peak": float(np.max(attention_map)) if attention_map.size else 0.0,
        }

    def process_motion(self, retinal_input: np.ndarray, motion_vectors: List[Tuple[float, float]]) -> Dict[str, Any]:
        tectal_output = self.tectum.process(retinal_input, motion_vectors)
        movement_cmd = np.array(self.tectum.get_movement_command(), dtype=float)
        return {
            "tectal_output": tectal_output,
            "movement_command": movement_cmd,
            "tectal_activity": float(np.mean(tectal_output)) if len(tectal_output) > 0 else 0.0,
        }

    def generate_motor_output(
        self,
        movement_command: np.ndarray,
        proprioceptive_feedback: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        feedback = proprioceptive_feedback if proprioceptive_feedback is not None else np.zeros(12, dtype=float)
        muscle_activation = self.motor_hierarchy.execute_movement_command(tuple(movement_command), feedback)
        decoded_velocity = self.motor_hierarchy.decode_velocity(muscle_activation)
        blended_velocity = 0.35 * decoded_velocity + 0.65 * movement_command
        speed = np.linalg.norm(blended_velocity)
        if speed < 0.08 and np.linalg.norm(movement_command) > 1e-6:
            command_direction = movement_command / np.linalg.norm(movement_command)
            blended_velocity = command_direction * max(0.08, np.linalg.norm(movement_command))
            speed = np.linalg.norm(blended_velocity)
        if speed > 1.0:
            blended_velocity = blended_velocity / speed
        return {
            "muscle_activation": muscle_activation,
            "velocity": blended_velocity,
            "motor_activity": float(np.mean(muscle_activation)),
        }

    def update_neuromodulation(self, reward: float, controller_activity: float, dt: float):
        reward_component = 0.28 * reward
        exploration_component = 0.18 * self.exploration_bonus
        baseline_dopamine = 0.78 if self.is_juvenile else 0.45
        baseline_serotonin = 0.68 if self.is_juvenile else 0.5

        self.dopamine_level = np.clip(baseline_dopamine + reward_component + exploration_component, 0.0, 1.0)
        energy_component = self.metabolism.glucose_level / 2.0
        calm_component = 0.08 * (1.0 - min(1.0, controller_activity))
        self.serotonin_level = np.clip(baseline_serotonin + 0.1 * energy_component + calm_component, 0.0, 1.0)

        for synapse in self.synapses:
            synapse.update_modulators(self.dopamine_level, self.serotonin_level, 0.35 + 0.1 * controller_activity)

        if reward > 0:
            self.neurotransmitter_system.release((self.space_size[0] / 2, self.space_size[1] / 2), reward, "dopamine")
        self.neurotransmitter_system.diffuse(dt)

    def apply_plasticity(self, controller_activity: float):
        self.functional_plasticity.update(self.plastic_neurons, self.dt)
        self.structural_plasticity.update_structure(self.synapses, np.array([controller_activity], dtype=float), self.dt)

    def update_metabolism(self, movement_intensity: float, controller_activity: float, dt: float):
        self.metabolism.update(dt, movement_intensity, controller_activity)

    def _exploratory_command(self) -> np.ndarray:
        angle = random.uniform(0.0, 2.0 * np.pi)
        amplitude = 0.35 if self.is_juvenile else 0.18
        return np.array([np.cos(angle), np.sin(angle)], dtype=float) * amplitude

    def update(
        self,
        visual_scene: List[Tuple[float, float, float]],
        motion_vectors: List[Tuple[float, float]],
        reward: float = 0.0,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        if dt is None:
            dt = self.dt

        self.steps += 1
        self.juvenile_age += 1
        if self.is_juvenile and self.juvenile_age >= self.juvenile_duration:
            self.is_juvenile = False

        sensory_data = self.process_sensory_input(visual_scene)
        retinal_output = sensory_data["retinal_output"]

        motion_data = self.process_motion(retinal_output, motion_vectors)
        movement_command = np.array(motion_data["movement_command"], dtype=float)
        if motion_vectors:
            averaged_motion = np.mean(np.array(motion_vectors, dtype=float), axis=0)
            motion_norm = np.linalg.norm(averaged_motion)
            if motion_norm > 1e-6:
                target_pull = averaged_motion / motion_norm
                salience = float(np.clip(0.2 + sensory_data["attention_peak"] + motion_data["tectal_activity"], 0.0, 1.0))
                movement_command = 0.7 * movement_command + 0.3 * target_pull * salience
        if len(motion_vectors) == 0:
            self.exploration_bonus = 0.25 if self.is_juvenile else 0.08
            movement_command = 0.6 * movement_command + 0.4 * self._exploratory_command()
        else:
            self.exploration_bonus = 0.0

        motor_data = self.generate_motor_output(movement_command)
        velocity = np.array(motor_data["velocity"], dtype=float)

        retinal_activity = float(np.mean(np.abs(retinal_output))) if len(retinal_output) > 0 else 0.0
        tectal_activity = float(motion_data["tectal_activity"])
        motor_activity = float(motor_data["motor_activity"])
        controller_activity = (
            0.30 * retinal_activity
            + 0.35 * tectal_activity
            + 0.20 * motor_activity
            + 0.15 * sensory_data["attention_peak"]
        )

        self.update_neuromodulation(reward, controller_activity, dt)

        movement_intensity = float(np.linalg.norm(velocity))
        self.update_metabolism(movement_intensity, controller_activity, dt)
        self.apply_plasticity(controller_activity)

        neural_activity_map = np.array([controller_activity], dtype=float)
        neural_positions = np.array([[self.space_size[0] / 2.0, self.space_size[1] / 2.0]], dtype=float)
        self.glial_network.update(neural_activity_map, neural_positions, dt)

        self.activity_history.append(controller_activity)
        self.dopamine_history.append(self.dopamine_level)
        if reward > 0:
            self.reward_history.append(reward)

        return {
            "position": (float(self.space_size[0] / 2.0), float(self.space_size[1] / 2.0)),
            "velocity": velocity,
            "visual_output": retinal_output,
            "motor_output": velocity,
            "dopamine": self.dopamine_level,
            "serotonin": self.serotonin_level,
            "energy": self.metabolism.glucose_level,
            "fatigue": self.metabolism.fatigue_level,
            "neural_activity": controller_activity,
            "retinal_activity": retinal_activity,
            "tectal_activity": tectal_activity,
            "motor_activity": motor_activity,
            "attention_peak": sensory_data["attention_peak"],
            "controller_signal": controller_activity,
            "is_juvenile": self.is_juvenile,
            "juvenile_progress": min(1.0, self.juvenile_age / self.juvenile_duration),
            "reward": reward,
        }


class BioFrogAgent:
    """Toy bio-inspired frog agent for comparison against ANN and SNN baselines."""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        space_size: Tuple[int, int] = (800, 600),
        bio_mode: bool = True,
        juvenile_mode: bool = True,
        training_mode: bool = False,
        instinct_mode: bool = False,
    ):
        self.space = space
        self.position = np.array(position, dtype=float)
        self.bio_mode = bio_mode
        self.training_mode = training_mode
        self.instinct_mode = instinct_mode

        moment = pymunk.moment_for_circle(1.0, 0, 30.0)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = to_pymunk_vec(self.position)
        self.shape = pymunk.Circle(self.body, 30.0)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)

        self.space_size = tuple(space_size)
        self.brain = BioFrogBrain(space_size=self.space_size, juvenile_mode=juvenile_mode, dt=0.01)

        self.max_energy = 30.0
        self.energy = float(self.max_energy)
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.visual_range = max(140.0, 0.55 * max(self.space_size))
        self.last_velocity = np.zeros(2, dtype=float)

        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target: Optional[np.ndarray] = None
        self.attached_fly = None
        self._visible_targets: List[Any] = []

        if training_mode or instinct_mode:
            self.hit_radius = 24.0
            self.success_prob = 0.72
        else:
            self.hit_radius = 34.0
            self.success_prob = 0.92
        self.catch_cooldown = 20

    def detect_flies(self, flies: List[Any]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
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

    def extend_tongue(self, target_position: Optional[np.ndarray] = None):
        if not self.tongue_extended and target_position is not None:
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

        visual_scene, motion_vectors = self.detect_flies(flies)

        reward = 0.0
        if self.attached_fly is not None:
            reward = 1.0
            self.attached_fly = None

        brain_output = self.brain.update(visual_scene, motion_vectors, reward=reward, dt=dt)
        metab_energy = float(brain_output.get("energy", 1.0))
        self.energy = min(self.max_energy, metab_energy * self.max_energy)

        velocity = np.array(brain_output["velocity"], dtype=float)
        fatigue_scale = max(0.25, 1.0 - float(brain_output["fatigue"]))
        velocity = 0.25 * self.last_velocity + 0.75 * velocity
        velocity *= fatigue_scale
        if np.linalg.norm(velocity) > 1.0:
            velocity = velocity / np.linalg.norm(velocity)
        self.body.velocity = to_pymunk_vec(velocity * 135.0)
        self.last_velocity = velocity

        target_distance = None
        alignment = 0.0
        if motion_vectors:
            distances = [float(np.linalg.norm(np.array(vec, dtype=float))) for vec in motion_vectors]
            nearest_idx = int(np.argmin(distances))
            target_distance = distances[nearest_idx]
            target_direction = np.array(motion_vectors[nearest_idx], dtype=float)
            if np.linalg.norm(target_direction) > 1e-6 and np.linalg.norm(velocity) > 1e-6:
                alignment = float(
                    np.dot(
                        velocity / np.linalg.norm(velocity),
                        target_direction / np.linalg.norm(target_direction),
                    )
                )

            if (
                not self.tongue_extended
                and target_distance < 125.0
                and brain_output["controller_signal"] > 0.02
                and brain_output["tectal_activity"] > 0.02
                and (self.steps - self.last_catch_time) > self.catch_cooldown
            ):
                target = self._visible_targets[nearest_idx]
                self.extend_tongue(np.array(target.body.position, dtype=float))

        if self.tongue_extended and self.tongue_target is not None:
            direction = self.tongue_target - self.position
            distance = float(np.linalg.norm(direction))
            if distance > 0:
                direction = direction / distance
            self.tongue_length += 300.0 * dt
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
            "dopamine": brain_output["dopamine"],
            "serotonin": brain_output["serotonin"],
            "fatigue": brain_output["fatigue"],
            "is_juvenile": brain_output["is_juvenile"],
            "juvenile_progress": brain_output["juvenile_progress"],
            "tongue_extended": self.tongue_extended,
            "tongue_length": self.tongue_length,
            "neural_activity": brain_output["neural_activity"],
            "controller_signal": brain_output["controller_signal"],
            "tectal_activity": brain_output["tectal_activity"],
            "attention_peak": brain_output["attention_peak"],
            "reward": brain_output["reward"],
            "caught_fly": self.attached_fly,
            "target_distance": target_distance,
            "alignment": alignment,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "energy": self.energy,
            "caught_flies": self.caught_flies,
            "is_juvenile": self.brain.is_juvenile,
            "juvenile_age": self.brain.juvenile_age,
            "dopamine": self.brain.dopamine_level,
            "serotonin": self.brain.serotonin_level,
            "steps": self.steps,
        }

    def remove(self):
        try:
            if self.shape in self.space.shapes and self.body in self.space.bodies:
                self.space.remove(self.body, self.shape)
        except Exception:
            pass

"""Biological frog brain with distributed spatial memory and predatory hunting."""

import copy
import math
import random
from dataclasses import dataclass

import numpy as np

from Frog_predator_neuro.architecture import MotorHierarchy, RetinalProcessing, SpatialMemory, Tectum
from Frog_predator_neuro.config import (
    AGENT_CRITICAL_THRESHOLD,
    JUVENILE_STEPS,
    RETINA_FILTERS_PER_SIDE,
    TECTUM_COLUMNS,
    TILE_SIZE,
)
from Frog_predator_neuro.core import NeurotransmitterDiffusion, PyramidalNeuron
from Frog_predator_neuro.metabolism.systemic_metabolism import NeuronMetabolism, SystemicMetabolism
from Frog_predator_neuro.plasticity.functional_plasticity import FunctionalPlasticityManager, StructuralPlasticityManager
from Frog_predator_neuro.utils import add_vectors, clamp, normalize_vector, scale_vector, tile_center


@dataclass
class BrainProfile:
    exploration_gain: float = 1.0
    memory_gain: float = 1.0
    food_signal_gain: float = 1.0
    metabolism_efficiency: float = 1.0
    persistence_gain: float = 1.0


@dataclass
class AffectiveState:
    arousal: float = 0.32
    curiosity: float = 0.55
    frustration: float = 0.08
    reward_confidence: float = 0.28
    restlessness: float = 0.34


class BioFrogBrain:
    def __init__(self, juvenile_mode=True, rng=None):
        self.dt = 0.01
        self.rng = rng or random.Random()
        self.is_juvenile = juvenile_mode
        self.juvenile_duration = JUVENILE_STEPS
        self.juvenile_age = 0
        self.maturity_readiness = 0.08 if juvenile_mode else 1.0
        self.maturity_stability = 0.02 if juvenile_mode else 1.0
        self.self_selected_maturity_age = None

        self.visual_system = RetinalProcessing(visual_field_size=(5.0, 5.0), num_filters_per_side=RETINA_FILTERS_PER_SIDE)
        self.tectum = Tectum(columns=TECTUM_COLUMNS)
        self.motor_hierarchy = MotorHierarchy()
        self.spatial_memory = SpatialMemory(rng=self.rng)

        self.neurotransmitter_system = NeurotransmitterDiffusion(space_size=(360, 360), grid_resolution=28)
        self.metabolism = SystemicMetabolism()
        self.neuron_metabolism = NeuronMetabolism()
        self.functional_plasticity = FunctionalPlasticityManager()
        self.structural_plasticity = StructuralPlasticityManager()

        self.num_astrocytes = 12
        self.profile = BrainProfile()
        self.affect = AffectiveState(
            arousal=0.40 if juvenile_mode else 0.32,
            curiosity=0.68 if juvenile_mode else 0.52,
            frustration=0.04 if juvenile_mode else 0.08,
            reward_confidence=0.24,
            restlessness=0.46 if juvenile_mode else 0.34,
        )

        self.dopamine_level = 0.85 if juvenile_mode else 0.5
        self.serotonin_level = 0.75 if juvenile_mode else 0.5
        self.acetylcholine_level = 0.3
        self.steps = 0
        self.last_reward = 0.0
        self.last_memory_score = 0.0
        self.last_velocity = (0.0, 0.0)
        self.last_memory_vector = (0.0, 0.0)

        self._neuron_cache = []
        self._synapse_cache = []
        self._neural_positions = np.zeros((0, 2), dtype=float)
        self.refresh_caches()

    def refresh_caches(self):
        neurons = []
        neurons.extend(filter_cell.center_neuron for filter_cell in self.visual_system.filters)
        for column in self.tectum.columns:
            neurons.extend(column.pyramidal_neurons)
            neurons.extend(column.interneurons)
            neurons.extend(column.output_neurons)
        neurons.extend(self.motor_hierarchy.all_neurons())
        neurons.extend(self.spatial_memory.all_neurons())
        self._neuron_cache = list(neurons)

        synapses = []
        for column in self.tectum.columns:
            synapses.extend(column.synapses)
        synapses.extend(self.spatial_memory.all_synapses())
        self._synapse_cache = synapses

        grid_width = int(math.ceil(math.sqrt(max(1, len(self._neuron_cache)))))
        self._neural_positions = np.array(
            [[(index % grid_width) * 12.0, (index // grid_width) * 12.0] for index in range(len(self._neuron_cache))],
            dtype=float,
        )

    def clone(self, heterogeneous=False, rng=None):
        rng = rng or random.Random()
        clone = copy.deepcopy(self)
        if heterogeneous:
            clone.apply_variation(rng)
        return clone

    def apply_variation(self, rng):
        self.profile.exploration_gain *= rng.uniform(0.82, 1.18)
        self.profile.memory_gain *= rng.uniform(0.82, 1.18)
        self.profile.food_signal_gain *= rng.uniform(0.82, 1.20)
        self.profile.metabolism_efficiency *= rng.uniform(0.85, 1.15)
        self.profile.persistence_gain *= rng.uniform(0.82, 1.18)

        self.metabolism.glucose_consumption_rate *= rng.uniform(0.9, 1.1)
        self.metabolism.oxygen_consumption_rate *= rng.uniform(0.9, 1.1)
        self.functional_plasticity.homeostatic_learning_rate *= rng.uniform(0.85, 1.20)

        for neuron in self.all_neurons():
            neuron.threshold += rng.uniform(-2.2, 2.2)
            neuron.tau_membrane *= rng.uniform(0.92, 1.08)

    def update_affective_state(
        self,
        perception,
        body_energy,
        reward,
        observational_reward,
        memory_score,
        exploration_bonus,
        neural_activity,
    ):
        visible_food_count = len(perception["visible_food_tiles"])
        fatigue = self.metabolism.fatigue_level
        expectation_gap = clamp(max(0.0, memory_score) * 0.38 + self.spatial_memory.food_expectation * 0.05, 0.0, 1.2)
        unrewarded_probe = 1.0 if reward <= 0.0 and visible_food_count == 0 and memory_score > 0.18 else 0.0
        novelty_pressure = clamp(1.0 - self.spatial_memory.external_drive + exploration_bonus * 2.5, 0.0, 1.4)
        food_salience = clamp(visible_food_count * 0.45, 0.0, 1.0)
        normalized_activity = clamp(neural_activity, 0.0, 1.0)
        hunger_pressure = clamp(max(0.0, 0.68 - body_energy) / 0.68, 0.0, 1.0)
        energy_surplus = clamp((body_energy - 0.72) / 0.20, 0.0, 1.0)
        desperation = clamp((0.60 - body_energy) / 0.30, 0.0, 1.0)
        recent_progress = clamp(perception.get("recent_progress", 0.0), 0.0, 1.0)

        target_frustration = clamp(
            0.03
            + unrewarded_probe * (0.18 + expectation_gap * 0.18)
            + fatigue * 0.08
            - reward * 0.40
            - observational_reward * 0.26
            - food_salience * 0.12,
            0.0,
            1.0,
        )
        target_reward_confidence = clamp(
            0.16
            + reward * 0.90
            + observational_reward * 0.34
            + food_salience * 0.32
            + max(0.0, self.spatial_memory.food_expectation) * 0.08
            - target_frustration * 0.10
            - hunger_pressure * 0.08,
            0.0,
            1.0,
        )
        target_curiosity = clamp(
            0.18
            + novelty_pressure * 0.35
            + target_reward_confidence * 0.12
            + energy_surplus * 0.28
            + desperation * 0.10
            + observational_reward * 0.10
            - target_frustration * 0.25
            - fatigue * 0.18,
            0.0,
            1.0,
        )
        target_arousal = clamp(
            0.18
            + normalized_activity * 0.38
            + food_salience * 0.24
            + reward * 0.20
            + observational_reward * 0.08
            + energy_surplus * 0.06
            + desperation * 0.26
            - fatigue * 0.14,
            0.0,
            1.0,
        )
        target_restlessness = clamp(
            0.14
            + max(0.0, 0.55 - recent_progress) * 0.58
            + novelty_pressure * 0.12
            + max(0.0, 0.24 - reward) * 0.04
            + observational_reward * 0.06
            + energy_surplus * 0.28
            + desperation * 0.24
            - food_salience * 0.42
            - fatigue * 0.18,
            0.0,
            1.0,
        )

        # Increased update rates for faster emotional response
        self.affect.frustration = clamp(self.affect.frustration * 0.920 + target_frustration * 0.080, 0.0, 1.0)
        self.affect.reward_confidence = clamp(self.affect.reward_confidence * 0.920 + target_reward_confidence * 0.080, 0.08, 0.92)
        self.affect.curiosity = clamp(self.affect.curiosity * 0.920 + target_curiosity * 0.080, 0.10, 0.92)
        self.affect.arousal = clamp(self.affect.arousal * 0.920 + target_arousal * 0.080, 0.10, 0.84)
        self.affect.restlessness = clamp(self.affect.restlessness * 0.920 + target_restlessness * 0.080, 0.06, 0.88)

    def all_neurons(self):
        return self._neuron_cache

    def all_synapses(self):
        return self._synapse_cache

    def local_affordance(self, direction, open_vectors):
        if not open_vectors:
            return 0.0
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return 0.0
        unit = direction / norm
        best = 0.0
        for vector_x, vector_y, weight in open_vectors:
            dot_product = max(0.0, vector_x * unit[0] + vector_y * unit[1])
            best = max(best, dot_product * clamp(weight, 0.0, 1.0))
        return clamp(best, 0.0, 1.0)

    def build_visual_scene(self, perception):
        current_tile = perception["current_tile"]
        scene = []
        for tile in perception["visible_walls"]:
            rel = (tile[0] - current_tile[0] + 2.5, tile[1] - current_tile[1] + 2.5, 0.45)
            scene.append(rel)
        for tile in perception["visible_food_tiles"]:
            rel = (tile[0] - current_tile[0] + 2.5, tile[1] - current_tile[1] + 2.5, 1.0)
            scene.append(rel)
        return scene

    def compute_open_drive(self, perception, memory_vector):
        open_vectors = perception.get("open_vectors", [])
        if not open_vectors:
            return (0.0, 0.0)

        heading_angle = perception["heading_angle"]
        heading_vector = np.array([math.cos(heading_angle), math.sin(heading_angle)], dtype=float)
        wander_vector = np.array(
            [math.cos(self.spatial_memory.wander_angle), math.sin(self.spatial_memory.wander_angle)],
            dtype=float,
        )
        memory_np = np.array(memory_vector, dtype=float) if memory_vector != (0.0, 0.0) else np.zeros(2, dtype=float)

        combined = np.zeros(2, dtype=float)
        for open_x, open_y, weight in open_vectors:
            unit = np.array([open_x, open_y], dtype=float)
            heading_align = max(0.0, float(np.dot(unit, heading_vector)))
            wander_align = max(0.0, float(np.dot(unit, wander_vector)))
            memory_align = max(0.0, float(np.dot(unit, memory_np))) if np.linalg.norm(memory_np) > 1e-6 else 0.0
            score = weight * (0.34 + heading_align * 0.22 + wander_align * 0.42 + memory_align * 0.35)
            combined += unit * score

        norm = np.linalg.norm(combined)
        if norm <= 1e-6:
            strongest = max(open_vectors, key=lambda item: item[2])
            return normalize_vector((float(strongest[0]), float(strongest[1])))
        return tuple((combined / norm).tolist())

    def build_salience_vectors(self, perception, body_energy):
        current_px = np.array(perception["position"], dtype=float)
        vectors = []
        arousal_gain = 0.85 + self.affect.arousal * 0.45
        open_vectors = perception.get("open_vectors", [])
        hunger_signal = clamp(1.0 - body_energy, 0.0, 1.0)

        for tile in perception["visible_food_tiles"]:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = target_px - current_px
            distance = np.linalg.norm(direction)
            if distance > 0:
                affordance = self.local_affordance(direction, open_vectors)
                if affordance < 0.12 and distance > TILE_SIZE * 0.85:
                    continue
                motor_affordance = affordance ** 1.45
                local_capture = clamp(1.0 - distance / (1.05 * TILE_SIZE), 0.0, 1.0)
                gain = 0.20 + motor_affordance * 1.32 + local_capture * 1.10
                food_drive = 0.18 + hunger_signal * hunger_signal * 2.15 + local_capture * 0.42
                vectors.append((direction / distance) * food_drive * arousal_gain * gain)

        for tile in perception["visible_walls"]:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = current_px - target_px
            distance = np.linalg.norm(direction)
            if distance > 0:
                vectors.append((direction / distance) * max(0.0, 1.15 - distance / (2.25 * TILE_SIZE)) * (0.9 + self.affect.arousal * 0.2))

        if not vectors:
            vectors.append(np.zeros(2))
        return vectors

    def compute_food_capture_vector(self, perception, body_energy):
        if not perception["visible_food_tiles"]:
            return (0.0, 0.0), 0.0

        current_px = np.array(perception["position"], dtype=float)
        open_vectors = perception.get("open_vectors", [])
        hunger_signal = clamp(1.0 - body_energy, 0.0, 1.0)
        best_direction = np.zeros(2, dtype=float)
        best_score = 0.0

        for tile in perception["visible_food_tiles"]:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = target_px - current_px
            distance = np.linalg.norm(direction)
            if distance <= 1e-6:
                continue
            affordance = self.local_affordance(direction, open_vectors)
            proximity = clamp(1.0 - distance / (1.40 * TILE_SIZE), 0.0, 1.0)
            score = affordance * 0.82 + proximity * 0.96 + hunger_signal * 0.24
            if score > best_score:
                best_score = score
                best_direction = direction / distance

        if best_score <= 0.0:
            return (0.0, 0.0), 0.0
        return (float(best_direction[0]), float(best_direction[1])), clamp(best_score, 0.0, 1.4)

    def compute_developmental_novelty_vector(self, perception, body_energy):
        if perception["visible_food_tiles"]:
            return (0.0, 0.0), 0.0

        current_px = np.array(perception["position"], dtype=float)
        open_vectors = perception.get("open_vectors", [])
        visible_unvisited = perception.get("visible_unvisited_tiles", set())
        time_since_reward = float(perception.get("time_since_reward", 999.0))
        food_collected = perception.get("food_collected", 0)
        novelty_pressure = clamp(
            (0.70 if self.is_juvenile else 0.22)
            + clamp((time_since_reward - 4.0) / 18.0, 0.0, 1.0) * (0.62 if self.is_juvenile else 0.18)
            + max(0.0, 0.78 - body_energy) * 0.12
            + max(0.0, 1 - food_collected) * (0.26 if self.is_juvenile else 0.08)
            + self.affect.curiosity * 0.26
            + self.affect.restlessness * 0.20,
            0.0,
            1.6,
        )
        if novelty_pressure <= 0.02:
            return (0.0, 0.0), 0.0

        best_direction = np.zeros(2, dtype=float)
        best_score = 0.0
        weighted = np.zeros(2, dtype=float)

        for tile in visible_unvisited:
            target_px = np.array(tile_center(tile), dtype=float)
            direction = target_px - current_px
            distance = np.linalg.norm(direction)
            if distance <= 1e-6:
                continue
            affordance = self.local_affordance(direction, open_vectors)
            if affordance <= 0.04:
                continue
            normalized_distance = clamp(distance / (2.4 * TILE_SIZE), 0.18, 1.0)
            heading_alignment = max(
                0.0,
                float(
                    np.dot(
                        direction / distance,
                        np.array([math.cos(perception["heading_angle"]), math.sin(perception["heading_angle"])], dtype=float),
                    )
                ),
            )
            score = affordance * (0.44 + normalized_distance * 0.46 + heading_alignment * 0.10)
            weighted += (direction / distance) * score
            if score > best_score:
                best_score = score
                best_direction = direction / distance

        if np.linalg.norm(weighted) > 1e-6:
            direction = normalize_vector((float(weighted[0]), float(weighted[1])))
        elif best_score > 0.0:
            direction = (float(best_direction[0]), float(best_direction[1]))
        elif open_vectors:
            breakout = max(open_vectors, key=lambda item: item[2] * (1.0 - abs(item[0] * self.last_velocity[0] + item[1] * self.last_velocity[1])))
            direction = normalize_vector((float(breakout[0]), float(breakout[1])))
            best_score = breakout[2]
        else:
            return (0.0, 0.0), 0.0

        return direction, clamp(best_score * novelty_pressure, 0.0, 1.3)

    def update_spatial_memory(self, perception, reward, body_energy):
        current_position = np.array(perception["position"], dtype=float)
        open_vectors = perception.get("open_vectors", [])

        visible_food_sources = []
        for tile in perception["visible_food_tiles"]:
            target_position = tile_center(tile)
            direction = np.array(target_position, dtype=float) - current_position
            distance = np.linalg.norm(direction)
            affordance = self.local_affordance(direction, open_vectors)
            if distance <= TILE_SIZE * 0.75:
                strength = 1.0
            else:
                strength = clamp((0.08 + affordance * 0.92) ** 1.65, 0.03, 1.0)
                if affordance < 0.12 and distance > TILE_SIZE * 0.85:
                    strength *= 0.12
            visible_food_sources.append((target_position, strength))

        hunger_drive = clamp(
            (1.0 - body_energy) * self.profile.memory_gain
            + self.affect.frustration * 0.06
            + (1.0 - self.affect.reward_confidence) * 0.08,
            0.05,
            1.6,
        )
        memory_vector, memory_score = self.spatial_memory.memory_vector(
            position=perception["position"],
            heading_angle=perception["heading_angle"],
            open_vectors=perception["open_vectors"],
            hunger_drive=hunger_drive,
            social_need=0.0,
            distress_bias=0.0,
            curiosity_drive=self.affect.curiosity,
            frustration=self.affect.frustration,
            social_comfort=0.0,
            reward_confidence=self.affect.reward_confidence,
            restlessness=self.affect.restlessness,
            stall_pressure=clamp(perception.get("stall_time", 0.0) / 3.2, 0.0, 1.2),
        )
        self.spatial_memory.observe(
            position=perception["position"],
            heading_angle=perception["heading_angle"],
            reward=reward,
            visible_food_sources=visible_food_sources,
            visible_peer_sources=[],
            heard_signals=[],
            fatigue=self.metabolism.fatigue_level,
            social_need=0.0,
        )
        self.last_memory_score = memory_score
        self.last_memory_vector = memory_vector
        return memory_vector, memory_score

    def update_neuromodulation(self, reward, observational_reward, exploration_bonus, neural_activity):
        dopamine_base = 0.85 if self.is_juvenile else 0.5
        serotonin_base = 0.75 if self.is_juvenile else 0.5
        target_dopamine = np.clip(
            dopamine_base
            + 0.32 * reward
            + 0.14 * observational_reward
            + 0.12 * exploration_bonus
            + 0.08 * self.affect.curiosity
            - 0.08 * self.affect.frustration
            - 0.1 * self.metabolism.fatigue_level,
            0.22 if self.is_juvenile else 0.20,
            0.86 if self.is_juvenile else 0.80,
        )
        self.dopamine_level = float(np.clip(self.dopamine_level * 0.97 + target_dopamine * 0.03, 0.20, 0.86))
        target_serotonin = np.clip(
            serotonin_base
            + 0.08 * self.metabolism.glucose_level
            + 0.03 * observational_reward
            + 0.05 * self.affect.reward_confidence
            - 0.12 * self.metabolism.fatigue_level,
            0.24,
            0.84,
        )
        self.serotonin_level = float(np.clip(self.serotonin_level * 0.97 + target_serotonin * 0.03, 0.24, 0.84))
        target_acetylcholine = np.clip(
            0.30
            + neural_activity * 0.08
            + self.spatial_memory.replay_gate * 0.06
            + self.affect.arousal * 0.06
            + self.affect.curiosity * 0.04,
            0.10,
            0.68,
        )
        self.acetylcholine_level = float(np.clip(self.acetylcholine_level * 0.96 + target_acetylcholine * 0.04, 0.10, 0.68))
        if reward > 0:
            self.neurotransmitter_system.release((180.0, 180.0), reward, "dopamine")
        self.neurotransmitter_system.diffuse(self.dt)

    def update_maturation_state(self, perception, body_energy, reward, observational_reward, memory_score):
        age_progress = clamp(self.juvenile_age / max(1.0, float(self.juvenile_duration)), 0.0, 1.0)
        exploration_competence = clamp((perception.get("visited_count", 1) - 1) / 10.0, 0.0, 1.0)
        foraging_competence = clamp(perception.get("food_collected", 0) / 3.0, 0.0, 1.0)
        energy_stability = clamp((body_energy - (AGENT_CRITICAL_THRESHOLD + 0.08)) / 0.55, 0.0, 1.0)
        memory_competence = clamp(
            max(0.0, memory_score) * 0.85
            + max(0.0, self.spatial_memory.food_expectation) * 0.10,
            0.0,
            1.0,
        )
        progress_competence = clamp(perception.get("recent_progress", 0.0), 0.0, 1.0)
        reward_signal = clamp(reward * 1.4 + observational_reward * 0.55 + self.affect.reward_confidence * 0.7, 0.0, 1.0)
        stall_penalty = clamp(perception.get("stall_time", 0.0) / 4.0, 0.0, 1.0)
        fatigue_penalty = clamp(self.metabolism.fatigue_level, 0.0, 1.0)
        instability_penalty = clamp(
            self.affect.frustration * 0.58 + fatigue_penalty * 0.12,
            0.0,
            1.0,
        )

        readiness_target = clamp(
            0.03
            + age_progress * 0.18
            + exploration_competence * 0.16
            + foraging_competence * 0.24
            + energy_stability * 0.16
            + memory_competence * 0.10
            + reward_signal * 0.08
            + progress_competence * 0.06
            - stall_penalty * 0.12
            - instability_penalty * 0.16,
            0.0,
            1.0,
        )
        stability_target = clamp(
            readiness_target * 0.62
            + energy_stability * 0.16
            + progress_competence * 0.10
            + reward_signal * 0.10
            - stall_penalty * 0.18
            - instability_penalty * 0.16,
            0.0,
            1.0,
        ) * (0.10 + age_progress * 0.90)

        self.maturity_readiness = clamp(self.maturity_readiness * 0.992 + readiness_target * 0.008, 0.0, 1.0)
        self.maturity_stability = clamp(self.maturity_stability * 0.994 + stability_target * 0.006, 0.0, 1.0)
        maturity_signal = clamp(self.maturity_readiness * 0.58 + self.maturity_stability * 0.42, 0.0, 1.0)

        if (
            self.is_juvenile
            and maturity_signal > 0.70
            and self.maturity_stability > 0.56
            and foraging_competence > 0.15
            and energy_stability > 0.35
        ):
            self.is_juvenile = False
            self.self_selected_maturity_age = self.juvenile_age

        return maturity_signal

    def update(self, perception, body_energy, reward, time_s):
        self.steps += 1
        self.juvenile_age += 1

        visual_scene = self.build_visual_scene(perception)
        retinal_output = self.visual_system.process_visual_input(visual_scene)
        salience_vectors = self.build_salience_vectors(perception, body_energy)
        self.tectum.process(retinal_output, salience_vectors)
        tectum_vector = self.tectum.get_movement_command()

        memory_vector, memory_score = self.update_spatial_memory(
            perception,
            reward,
            body_energy,
        )
        policy_metrics = self.spatial_memory.last_policy_metrics
        action_vigor = float(policy_metrics.get("vigor", 0.35))
        search_burst = float(policy_metrics.get("search_burst", 0.18))
        policy_confidence = float(policy_metrics.get("confidence", 0.22))
        policy_conflict = float(policy_metrics.get("conflict", 0.35))
        habit_pressure = float(policy_metrics.get("habit_pressure", 0.0))
        loop_pressure = float(policy_metrics.get("loop_pressure", 0.0))
        reorientation_drive = float(policy_metrics.get("reorientation_drive", 0.0))
        reorientation_vector = tuple(self.spatial_memory.reorientation_vector.tolist()) if np.linalg.norm(self.spatial_memory.reorientation_vector) > 1e-6 else (0.0, 0.0)
        food_capture_vector, food_capture_strength = self.compute_food_capture_vector(perception, body_energy)
        developmental_novelty_vector, developmental_novelty_strength = self.compute_developmental_novelty_vector(perception, body_energy)
        food_directness = clamp(food_capture_strength, 0.0, 1.0)
        exploration_bonus = max(0.0, memory_score) * 0.1 + max(0.0, 1.0 - self.spatial_memory.external_drive) * 0.06
        open_drive = self.compute_open_drive(perception, memory_vector)
        memory_affordance = self.local_affordance(memory_vector, perception.get("open_vectors", []))
        gated_memory_vector = scale_vector(memory_vector, 0.06 + 0.94 * (memory_affordance ** 1.35))

        exploratory_gain = self.profile.memory_gain * (0.84 + self.affect.curiosity * 0.52)
        persistence = self.profile.persistence_gain * (
            0.06
            + self.affect.reward_confidence * 0.06
            + policy_confidence * 0.12
        )
        persistence *= max(0.28, 1.0 - reorientation_drive * 0.72)
        desired_vector = add_vectors(
            scale_vector(tectum_vector, 0.82 + (1.0 - body_energy) * 0.24 + self.affect.arousal * 0.16 + food_directness * 0.90),
            scale_vector(food_capture_vector, 0.36 + food_directness * 1.85),
            scale_vector(developmental_novelty_vector, developmental_novelty_strength * (0.58 + (0.42 if self.is_juvenile else 0.0))),
            scale_vector(gated_memory_vector, (0.86 + exploratory_gain + action_vigor * 0.48) * (1.0 - food_directness * 0.40)),
            scale_vector(reorientation_vector, 0.12 + reorientation_drive * 1.05),
            scale_vector(open_drive, 0.10 + self.affect.restlessness * 0.34 + search_burst * 0.70),
            scale_vector(self.last_velocity, persistence),
        )
        desired_vector = normalize_vector(desired_vector)

        proprio_feedback = np.zeros(12)
        locomotor_tone = clamp(0.18 + action_vigor * 0.70 + search_burst * 0.22 + food_directness * 0.25, 0.12, 1.0)
        locomotor_tone = clamp(locomotor_tone + reorientation_drive * 0.12, 0.12, 1.0)
        self.motor_hierarchy.execute_movement_command(desired_vector, proprio_feedback, tonic_drive=locomotor_tone)
        velocity = scale_vector(
            self.motor_hierarchy.get_velocity_vector(desired_vector),
            1.15 + action_vigor * 1.85 + search_burst * 0.42 + food_directness * 0.65,
        )
        neural_activity = float(np.mean(np.abs(retinal_output))) if len(retinal_output) > 0 else 0.0
        neural_activity += float(np.mean(self.tectum.tectal_output)) * 0.35
        neural_activity += float(np.mean(self.motor_hierarchy.muscle_activation)) * 0.18
        neural_activity += float(np.mean(self.spatial_memory.last_activations)) * 0.45

        movement_intensity = float(np.linalg.norm(velocity))
        self.metabolism.update(self.dt, movement_intensity, neural_activity)
        self.neuron_metabolism.consume_energy(False, neural_activity * 10.0, self.dt)
        self.neuron_metabolism.recover_energy(self.dt, self.metabolism.oxygen_level, self.metabolism.glucose_level)
        excitability = self.neuron_metabolism.affects_excitability()

        neural_map = np.array([neuron.activity_level() for neuron in self._neuron_cache], dtype=float)
        self.update_affective_state(
            perception,
            body_energy,
            reward,
            0.0,
            memory_score,
            exploration_bonus,
            neural_activity,
        )
        maturity_signal = self.update_maturation_state(perception, body_energy, reward, 0.0, memory_score)
        self.update_neuromodulation(
            reward,
            0.0,
            exploration_bonus,
            neural_activity,
        )
        if self.steps % 4 == 0:
            self.functional_plasticity.apply_homeostatic_scaling(self._neuron_cache, self.dt * 4.0)
            self.functional_plasticity.apply_intrinsic_plasticity(self._neuron_cache, self.dt * 4.0)
        if self.steps % 6 == 0:
            self.structural_plasticity.update_structure(self._synapse_cache, neural_map, self.dt * 6.0)
        for synapse in self._synapse_cache:
            synapse.update_modulators(self.dopamine_level, self.serotonin_level, self.acetylcholine_level)
            synapse.apply_short_term_plasticity(self.dt, reward > 0)

        self.last_reward = reward
        # Minimal velocity smoothing: 50% old + 50% new
        smoothed_velocity_x = velocity[0] * 0.50 + self.last_velocity[0] * 0.50
        smoothed_velocity_y = velocity[1] * 0.50 + self.last_velocity[1] * 0.50
        self.last_velocity = (smoothed_velocity_x, smoothed_velocity_y)
        energy_surplus = clamp((body_energy - 0.72) / 0.20, 0.0, 1.0)
        desperation = clamp((0.60 - body_energy) / 0.30, 0.0, 1.0)

        return {
            "velocity": self.last_velocity,
            "memory_vector": gated_memory_vector,
            "memory_score": memory_score,
            "dopamine": self.dopamine_level,
            "serotonin": self.serotonin_level,
            "acetylcholine": self.acetylcholine_level,
            "fatigue": self.metabolism.fatigue_level,
            "glucose": self.metabolism.glucose_level,
            "neural_activity": neural_activity,
            "replay_gate": self.spatial_memory.replay_gate,
            "arousal": float(self.affect.arousal),
            "curiosity": float(self.affect.curiosity),
            "frustration": float(self.affect.frustration),
            "reward_confidence": float(self.affect.reward_confidence),
            "restlessness": float(self.affect.restlessness),
            "energy_surplus": float(energy_surplus),
            "desperation": float(desperation),
            "action_vigor": action_vigor,
            "search_burst": search_burst,
            "policy_confidence": policy_confidence,
            "policy_conflict": policy_conflict,
            "habit_pressure": habit_pressure,
            "loop_pressure": loop_pressure,
            "reorientation_drive": reorientation_drive,
            "food_directness": food_directness,
            "developmental_novelty": float(developmental_novelty_strength),
            "observational_reward": 0.0,
            "is_juvenile": self.is_juvenile,
            "juvenile_progress": 1.0 if not self.is_juvenile else float(maturity_signal),
            "maturity_readiness": float(self.maturity_readiness),
            "maturity_stability": float(self.maturity_stability),
            "self_selected_maturity_age": self.self_selected_maturity_age,
            "reward": reward,
        }

    def state_vector(self):
        return {
            "is_juvenile": self.is_juvenile,
            "juvenile_age": self.juvenile_age,
            "dopamine": self.dopamine_level,
            "serotonin": self.serotonin_level,
            "acetylcholine": self.acetylcholine_level,
            "glucose": self.metabolism.glucose_level,
            "fatigue": self.metabolism.fatigue_level,
            "arousal": float(self.affect.arousal),
            "curiosity": float(self.affect.curiosity),
            "frustration": float(self.affect.frustration),
            "reward_confidence": float(self.affect.reward_confidence),
            "restlessness": float(self.affect.restlessness),
            "maturity_readiness": float(self.maturity_readiness),
            "maturity_stability": float(self.maturity_stability),
        }

import csv
import importlib
import math
import os
import random
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymunk

try:
    import torch
except Exception:
    torch = None

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def import_fresh(module_name):
    stale = [name for name in sys.modules if name == module_name or name.startswith(module_name + ".")]
    for name in stale:
        sys.modules.pop(name, None)
    return importlib.import_module(module_name)


class FrogDiagnosticsTests(unittest.TestCase):
    def test_frog_visual_filter_habituates_static_stimulus(self):
        from Frog_predator_neuro.architecture.visual_system import CenterSurroundFilter

        filter_cell = CenterSurroundFilter((2.5, 2.5))
        drives = []
        for _ in range(8):
            filter_cell.process(np.array([2.5, 2.5], dtype=float))
            drives.append(filter_cell.current_drive)

        self.assertGreater(drives[0], drives[-1])

    def test_frog_brain_update_returns_expected_fields(self):
        from Frog_predator_neuro.brain import BioFrogBrain

        brain = BioFrogBrain(rng=random.Random(0))
        perception = {
            "position": (100.0, 100.0),
            "current_tile": (2, 2),
            "heading_angle": 0.0,
            "visible_passable_tiles": {(2, 2), (3, 2)},
            "visible_walls": set(),
            "visible_food_tiles": set(),
            "visible_unvisited_tiles": {(3, 2)},
            "open_vectors": [(1.0, 0.0, 0.62)],
            "wall_vectors": [],
            "recent_progress": 0.0,
            "visited_count": 1,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 10.0,
        }

        output = brain.update(perception, body_energy=0.6, reward=0.0, time_s=0.0)

        self.assertIn("velocity", output)
        self.assertIn("memory_vector", output)
        self.assertIn("dopamine", output)
        self.assertIn("memory_score", output)
        self.assertIn("motivation_context", output)
        self.assertIn("food_prediction_error", output)
        self.assertIn("bg_gating_signal", output)
        self.assertIn("hunger_bias", output["motivation_context"])
        self.assertEqual(len(output["velocity"]), 2)
        self.assertTrue(all(math.isfinite(float(v)) for v in output["velocity"]))
        self.assertGreaterEqual(float(output["dopamine"]), 0.0)
        self.assertLessEqual(float(output["dopamine"]), 1.0)
        self.assertGreaterEqual(float(output["bg_gating_signal"]), 0.0)
        self.assertLessEqual(float(output["bg_gating_signal"]), 1.0)

    def test_frog_hunger_bias_modulates_early_food_path(self):
        from Frog_predator_neuro.brain import BioFrogBrain

        brain = BioFrogBrain(rng=random.Random(0))
        perception = {
            "position": (100.0, 100.0),
            "current_tile": (2, 2),
            "heading_angle": 0.0,
            "visible_passable_tiles": {(2, 2), (3, 2)},
            "visible_walls": set(),
            "visible_food_tiles": {(3, 2)},
            "visible_unvisited_tiles": {(3, 2)},
            "open_vectors": [(1.0, 0.0, 0.72)],
            "wall_vectors": [],
            "recent_progress": 0.0,
            "visited_count": 1,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 10.0,
        }

        low_energy_context = brain.build_motivation_context(perception, body_energy=8.0)
        _, low_food_strength = brain.compute_food_capture_vector(
            perception,
            body_energy=8.0,
            motivation_context=low_energy_context,
        )
        high_energy_context = brain.build_motivation_context(perception, body_energy=28.0)
        _, high_food_strength = brain.compute_food_capture_vector(
            perception,
            body_energy=28.0,
            motivation_context=high_energy_context,
        )

        self.assertGreater(low_energy_context["hunger_bias"], high_energy_context["hunger_bias"])
        self.assertGreater(low_food_strength, high_food_strength)

    def test_frog_compare_mode_adds_task_set_without_overwriting_hunger(self):
        from Frog_predator_neuro.brain import BioFrogBrain
        from Frog_predator_neuro_compare.brain import BioFrogBrain as CompareBioFrogBrain

        perception = {
            "position": (100.0, 100.0),
            "current_tile": (2, 2),
            "heading_angle": 0.0,
            "visible_passable_tiles": {(2, 2), (3, 2)},
            "visible_walls": set(),
            "visible_food_tiles": {(3, 2)},
            "visible_unvisited_tiles": {(3, 2)},
            "open_vectors": [(1.0, 0.0, 0.72)],
            "wall_vectors": [],
            "recent_progress": 0.0,
            "visited_count": 1,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 10.0,
        }

        baseline = BioFrogBrain(rng=random.Random(0))
        comparable = CompareBioFrogBrain(rng=random.Random(0), comparable_mode=True)

        baseline_context = baseline.build_motivation_context(perception, body_energy=28.0)
        comparable_context = comparable.build_motivation_context(perception, body_energy=28.0)

        self.assertAlmostEqual(
            float(baseline_context["hunger_bias"]),
            float(comparable_context["hunger_bias"]),
            places=6,
        )
        self.assertGreater(float(comparable_context["task_set_bias"]), 0.0)
        self.assertGreater(
            float(comparable_context["reward_seek_bias"]),
            float(baseline_context["reward_seek_bias"]),
        )
        self.assertGreater(
            float(comparable_context["predation_bias"]),
            float(baseline_context["predation_bias"]),
        )

    def test_frog_food_prediction_error_tracks_expectation_gap(self):
        from Frog_predator_neuro.architecture.spatial_memory import SpatialMemory

        memory = SpatialMemory(rng=random.Random(0))
        kwargs = {
            "heading_angle": 0.0,
            "visible_peer_sources": [],
            "heard_signals": [],
            "fatigue": 0.2,
            "social_need": 0.0,
        }

        memory.observe(
            position=(100.0, 100.0),
            reward=1.0,
            visible_food_sources=[((100.0, 100.0), 1.0)],
            **kwargs,
        )
        positive_error = memory.food_prediction_error

        memory.food_expectation = 0.9
        memory.observe(
            position=(100.0, 100.0),
            reward=0.0,
            visible_food_sources=[],
            **kwargs,
        )

        self.assertGreater(positive_error, 0.0)
        self.assertLess(memory.food_prediction_error, positive_error)

    def test_frog_predator_simulation_runs_headless(self):
        from Frog_predator_neuro.simulation import Simulation

        random.seed(0)
        sim = Simulation(headless=True, num_flies=8)
        try:
            for _ in range(10):
                sim.update(1 / 60)
            frog = sim.frogs[0]
            self.assertEqual(sim.step_count, 10)
            self.assertTrue(frog.alive)
            self.assertTrue(math.isfinite(float(frog.energy)))
        finally:
            sim.close()

    def test_frog_predator_simulation_adds_world_boundaries_for_frog(self):
        from Frog_predator_neuro.simulation import Simulation

        sim = Simulation(width=260, height=220, num_flies=0, headless=True)
        try:
            frog = sim.frogs[0]
            frog.body.position = (190.0, 110.0)
            frog.body.velocity = (500.0, 0.0)
            for _ in range(40):
                sim.space.step(sim.dt)
            self.assertLessEqual(float(frog.body.position[0]), 259.0)
        finally:
            sim.close()

    def test_frog_predator_agent_exposes_neuro_panel_sections(self):
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent

        agent = BioFrogPredatorAgent(space=pymunk.Space())
        agent._visible_targets = [{"distance": 72.0}]
        agent.strike_drive = 0.55
        agent.last_brain_output = {
            "dopamine": 0.61,
            "serotonin": 0.47,
            "acetylcholine": 0.31,
            "glucose": 0.92,
            "fatigue": 0.18,
            "excitability": 0.88,
            "neural_activity": 0.44,
            "replay_gate": 0.12,
            "arousal": 0.57,
            "curiosity": 0.42,
            "frustration": 0.11,
            "reward_confidence": 0.38,
            "restlessness": 0.29,
            "food_prediction_error": 0.21,
            "memory_score": 0.35,
            "food_directness": 0.49,
            "policy_confidence": 0.54,
            "policy_conflict": 0.18,
            "bg_gating_signal": 0.26,
            "bg_confidence": 0.41,
            "effective_motor_gate": 0.52,
            "goal_commitment": 0.48,
            "search_pressure": 0.17,
            "command_strength": 0.39,
            "strike_readiness": 0.63,
            "action_vigor": 0.58,
            "search_burst": 0.14,
            "developmental_novelty": 0.22,
            "focus_distance": 72.0,
            "is_juvenile": True,
            "juvenile_progress": 0.64,
            "maturity_readiness": 0.57,
            "maturity_stability": 0.42,
            "self_selected_maturity_age": None,
            "motivation_context": {
                "hunger_bias": 0.44,
                "predation_bias": 0.58,
                "reward_seek_bias": 0.37,
                "vigilance_bias": 0.25,
                "exploration_bias": 0.33,
                "locomotor_bias": 0.29,
            },
        }

        sections = agent.neuro_panel_sections()
        titles = [title for title, _ in sections]
        flattened = "\n".join(line for _, lines in sections for line in lines)

        self.assertIn("Neurochemistry", titles)
        self.assertIn("Development", titles)
        self.assertIn("Motivation Context", titles)
        self.assertIn("Action Selection", titles)
        self.assertIn("food PE", flattened)
        self.assertIn("BG gate", flattened)
        self.assertIn("Readiness", flattened)
        self.assertIn("Visible flies 1", flattened)

    def test_frog_variants_share_common_baseline_conditions(self):
        from Frog_predator_neuro.config import AGENT_BODY_ENERGY_START, AGENT_MAX_SPEED, PREDATOR_VISUAL_RANGE_PX
        from Frog_predator_neuro.simulation import FROG_INNER_COLOR as BIO_INNER_COLOR
        from Frog_predator_neuro.simulation import FROG_INNER_SCALE as BIO_INNER_SCALE
        from Frog_predator_neuro.simulation import FROG_OUTER_COLOR as BIO_OUTER_COLOR
        from Frog_predator_neuro.simulation import FROG_RENDER_RADIUS_MIN as BIO_RENDER_RADIUS_MIN
        from Frog_predator_neuro.simulation import FROG_RENDER_SCALE as BIO_RENDER_SCALE
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent
        from frog_lib_ann.simulation import FROG_INNER_COLOR as ANN_INNER_COLOR
        from frog_lib_ann.simulation import FROG_INNER_SCALE as ANN_INNER_SCALE
        from frog_lib_ann.simulation import FROG_OUTER_COLOR as ANN_OUTER_COLOR
        from frog_lib_ann.simulation import FROG_RENDER_RADIUS_MIN as ANN_RENDER_RADIUS_MIN
        from frog_lib_ann.simulation import FROG_RENDER_SCALE as ANN_RENDER_SCALE
        from frog_lib_ann.agent import FrogANNAgent
        from frog_lib_snn.simulation import FROG_INNER_COLOR as SNN_INNER_COLOR
        from frog_lib_snn.simulation import FROG_INNER_SCALE as SNN_INNER_SCALE
        from frog_lib_snn.simulation import FROG_OUTER_COLOR as SNN_OUTER_COLOR
        from frog_lib_snn.simulation import FROG_RENDER_RADIUS_MIN as SNN_RENDER_RADIUS_MIN
        from frog_lib_snn.simulation import FROG_RENDER_SCALE as SNN_RENDER_SCALE
        from frog_lib_snn.agent import FrogSNNAgent

        ann = FrogANNAgent(pymunk.Space(), position=(100.0, 100.0), training_mode=False)
        snn = FrogSNNAgent(pymunk.Space(), position=(100.0, 100.0), training_mode=False)
        bio = BioFrogPredatorAgent(space=pymunk.Space(), position=(100.0, 100.0), training_mode=False)
        try:
            self.assertAlmostEqual(float(ann.shape.radius), float(snn.shape.radius), places=6)
            self.assertAlmostEqual(float(ann.shape.radius), float(bio.shape.radius), places=6)
            self.assertAlmostEqual(float(ann.shape.elasticity), float(snn.shape.elasticity), places=6)
            self.assertAlmostEqual(float(ann.shape.elasticity), float(bio.shape.elasticity), places=6)
            self.assertAlmostEqual(float(ann.shape.friction), float(snn.shape.friction), places=6)
            self.assertAlmostEqual(float(ann.shape.friction), float(bio.shape.friction), places=6)
            self.assertEqual(float(ann.max_speed), float(snn.max_speed))
            self.assertEqual(float(ann.max_speed), float(AGENT_MAX_SPEED))
            self.assertEqual(float(ann.visual_range), float(snn.visual_range))
            self.assertEqual(float(ann.visual_range), float(PREDATOR_VISUAL_RANGE_PX))
            self.assertEqual(float(bio.visual_range_px), float(PREDATOR_VISUAL_RANGE_PX))
            self.assertEqual(float(ann.energy), float(snn.energy))
            self.assertEqual(float(ann.energy), float(bio.energy))
            self.assertEqual(float(ann.energy), float(AGENT_BODY_ENERGY_START))
            self.assertEqual(float(ann.hit_radius), float(snn.hit_radius))
            self.assertEqual(float(ann.hit_radius), float(bio.hit_radius))
            self.assertEqual(float(ann.success_prob), float(snn.success_prob))
            self.assertEqual(float(ann.success_prob), float(bio.success_prob))
            self.assertEqual(float(ann.tongue_reach), float(snn.tongue_reach))
            self.assertEqual(float(ann.tongue_reach), float(bio.tongue_reach))
            self.assertEqual(float(ANN_RENDER_SCALE), float(SNN_RENDER_SCALE))
            self.assertEqual(float(ANN_RENDER_SCALE), float(BIO_RENDER_SCALE))
            self.assertEqual(int(ANN_RENDER_RADIUS_MIN), int(SNN_RENDER_RADIUS_MIN))
            self.assertEqual(int(ANN_RENDER_RADIUS_MIN), int(BIO_RENDER_RADIUS_MIN))
            self.assertGreaterEqual(int(ANN_RENDER_RADIUS_MIN), 24)
            self.assertEqual(float(ANN_INNER_SCALE), float(SNN_INNER_SCALE))
            self.assertEqual(float(ANN_INNER_SCALE), float(BIO_INNER_SCALE))
            self.assertEqual(tuple(ANN_OUTER_COLOR), tuple(SNN_OUTER_COLOR))
            self.assertEqual(tuple(ANN_OUTER_COLOR), tuple(BIO_OUTER_COLOR))
            self.assertEqual(tuple(ANN_INNER_COLOR), tuple(SNN_INNER_COLOR))
            self.assertEqual(tuple(ANN_INNER_COLOR), tuple(BIO_INNER_COLOR))
            self.assertEqual(int(bio.render_outer_radius()), int(max(BIO_RENDER_RADIUS_MIN, round(float(bio.shape.radius) * BIO_RENDER_SCALE))))
            self.assertFalse(bool(snn.brain.is_juvenile))
            self.assertFalse(bool(bio.brain.is_juvenile))
        finally:
            ann.remove()
            snn.remove()
            bio.remove()

    def test_frog_predator_brain_commits_motion_when_prey_visible(self):
        from Frog_predator_neuro.brain import BioFrogBrain

        brain = BioFrogBrain(rng=random.Random(0))
        output = brain.update_predator_mode(
            position=(400.0, 300.0),
            heading_angle=0.0,
            visible_targets=[
                {
                    "vector": (80.0, 0.0),
                    "brightness": 0.85,
                    "motion": 0.90,
                    "facing": 1.0,
                    "position": (480.0, 300.0),
                }
            ],
            body_energy=30.0,
            reward=0.0,
            visual_range_px=220.0,
        )

        self.assertGreater(float(output["effective_motor_gate"]), 0.15)
        self.assertGreater(float(np.linalg.norm(np.array(output["velocity"], dtype=float))), 1e-3)
        self.assertGreater(float(output["strike_readiness"]), 0.28)
        self.assertIn("is_juvenile", output)
        self.assertIn("juvenile_progress", output)
        self.assertIn("maturity_readiness", output)
        self.assertIn("maturity_stability", output)

    def test_frog_predator_brain_can_self_select_maturity_in_open_field(self):
        from Frog_predator_neuro.brain import BioFrogBrain

        brain = BioFrogBrain(rng=random.Random(0))
        brain.maturity_readiness = 0.92
        brain.maturity_stability = 0.82
        brain.spatial_memory.food_expectation = 0.65
        brain.affect.reward_confidence = 0.76

        output = brain.update_predator_mode(
            position=(400.0, 300.0),
            heading_angle=0.0,
            visible_targets=[
                {
                    "vector": (46.0, 8.0),
                    "brightness": 0.92,
                    "motion": 0.88,
                    "facing": 0.94,
                    "position": (446.0, 308.0),
                }
            ],
            body_energy=28.0,
            reward=1.0,
            visual_range_px=220.0,
            food_collected=1,
            stall_time=0.0,
            time_since_reward=0.0,
            visited_count=8,
        )

        self.assertFalse(output["is_juvenile"])
        self.assertEqual(output["juvenile_progress"], 1.0)
        self.assertIsNotNone(output["self_selected_maturity_age"])

    def test_frog_predator_brain_does_not_mature_from_age_alone(self):
        from Frog_predator_neuro.brain import BioFrogBrain
        from Frog_predator_neuro.config import JUVENILE_STEPS

        brain = BioFrogBrain(rng=random.Random(0))
        brain.juvenile_age = JUVENILE_STEPS * 3
        brain.maturity_readiness = 0.10
        brain.maturity_stability = 0.06
        brain.affect.reward_confidence = 0.12
        brain.spatial_memory.food_expectation = 0.0

        output = brain.update_predator_mode(
            position=(400.0, 300.0),
            heading_angle=0.0,
            visible_targets=[],
            body_energy=28.0,
            reward=0.0,
            visual_range_px=220.0,
            food_collected=0,
            stall_time=1.2,
            time_since_reward=120.0,
            visited_count=1,
        )

        self.assertTrue(output["is_juvenile"])
        self.assertLess(float(output["juvenile_progress"]), 0.40)
        self.assertIsNone(output["self_selected_maturity_age"])

    def test_frog_predator_agent_routes_control_through_brain(self):
        from Frog_predator_neuro.config import AGENT_MAX_SPEED
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent, Fly

        agent = BioFrogPredatorAgent(space=pymunk.Space())
        fly = Fly(agent.space, (450, 300))
        called = {"value": False}

        def fake_update(*args, **kwargs):
            called["value"] = True
            return {
                "velocity": (0.25, 0.0),
                "memory_vector": (1.0, 0.0),
                "memory_score": 0.4,
                "dopamine": 0.5,
                "serotonin": 0.5,
                "acetylcholine": 0.4,
                "fatigue": 0.1,
                "glucose": 1.0,
                "neural_activity": 0.2,
                "arousal": 0.3,
                "curiosity": 0.4,
                "frustration": 0.1,
                "reward_confidence": 0.3,
                "action_vigor": 0.2,
                "search_burst": 0.1,
                "food_directness": 0.5,
                "focus_vector": (1.0, 0.0),
                "focus_distance": 150.0,
                "focus_position": (450.0, 300.0),
                "strike_readiness": 0.0,
                "reward": 0.0,
                "excitability": 0.9,
            }

        agent.brain.update_predator_mode = fake_update
        output = agent.update(0.01, [fly], 800, 600)

        self.assertTrue(called["value"])
        self.assertEqual(output["dopamine"], 0.5)
        self.assertAlmostEqual(float(agent.body.velocity[0]), AGENT_MAX_SPEED * 0.25, places=6)
        self.assertAlmostEqual(float(output["excitability"]), 0.9, places=6)

    def test_frog_agent_updates_capture_profile_after_maturation(self):
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent

        agent = BioFrogPredatorAgent(space=pymunk.Space(), training_mode=True)
        self.assertTrue(agent.brain.is_juvenile)
        self.assertLess(agent.hit_radius, agent.adult_hit_radius)
        self.assertLess(agent.success_prob, agent.adult_success_prob)

        agent.brain.is_juvenile = False
        agent._apply_developmental_stage()

        self.assertEqual(agent.hit_radius, agent.adult_hit_radius)
        self.assertEqual(agent.success_prob, agent.adult_success_prob)
        self.assertEqual(agent.tongue_reach, agent.adult_tongue_reach)

    def test_frog_agent_development_badge_text_reflects_stage(self):
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent

        agent = BioFrogPredatorAgent(space=pymunk.Space(), training_mode=True)
        agent.last_brain_output["juvenile_progress"] = 0.42
        self.assertEqual(agent.development_badge_text(), "JUV 42%")

        agent.brain.is_juvenile = False
        self.assertEqual(agent.development_badge_text(), "ADULT")

    def test_frog_runtime_filters_out_distant_flies(self):
        from Frog_predator_neuro.config import PREDATOR_VISUAL_RANGE_PX
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent, Fly

        agent = BioFrogPredatorAgent(space=pymunk.Space())
        near_distance = max(40.0, PREDATOR_VISUAL_RANGE_PX * 0.5)
        near_fly = Fly(agent.space, (float(agent.body.position[0]) + near_distance, float(agent.body.position[1])))
        far_fly = Fly(agent.space, (float(agent.body.position[0]) + PREDATOR_VISUAL_RANGE_PX + 40.0, float(agent.body.position[1])))
        near_fly.body.velocity = (45.0, 0.0)
        far_fly.body.velocity = (45.0, 0.0)
        captured_targets = {"value": None}

        def fake_update(**kwargs):
            captured_targets["value"] = kwargs["visible_targets"]
            return {
                "velocity": (0.0, 0.0),
                "memory_vector": (0.0, 0.0),
                "memory_score": 0.0,
                "dopamine": 0.0,
                "serotonin": 0.0,
                "acetylcholine": 0.0,
                "fatigue": 0.0,
                "glucose": 1.0,
                "neural_activity": 0.0,
                "arousal": 0.0,
                "curiosity": 0.0,
                "frustration": 0.0,
                "reward_confidence": 0.0,
                "action_vigor": 0.0,
                "search_burst": 0.0,
                "food_directness": 0.0,
                "focus_vector": (1.0, 0.0),
                "focus_distance": 0.0,
                "focus_position": None,
                "strike_readiness": 0.0,
                "reward": 0.0,
                "excitability": 1.0,
            }

        agent.brain.update_predator_mode = fake_update
        agent.update(0.01, [near_fly, far_fly], 800, 600)

        self.assertIsNotNone(captured_targets["value"])
        self.assertEqual(len(captured_targets["value"]), 1)
        self.assertAlmostEqual(float(captured_targets["value"][0]["distance"]), near_distance, delta=2.5)

    def test_frog_strike_accumulator_requires_multiple_frames(self):
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent, Fly

        agent = BioFrogPredatorAgent(space=pymunk.Space())
        agent.success_prob = 1.0
        fly = Fly(agent.space, (float(agent.body.position[0]) + 40.0, float(agent.body.position[1])))
        agent.brain.update_predator_mode = lambda **kwargs: {
            "velocity": (0.0, 0.0),
            "memory_vector": (0.0, 0.0),
            "memory_score": 0.0,
            "dopamine": 0.0,
            "serotonin": 0.0,
            "acetylcholine": 0.0,
            "fatigue": 0.0,
            "glucose": 1.0,
            "neural_activity": 0.0,
            "arousal": 0.0,
            "curiosity": 0.0,
            "frustration": 0.0,
            "reward_confidence": 0.0,
            "action_vigor": 0.0,
            "search_burst": 0.0,
            "food_directness": 0.0,
            "focus_vector": (1.0, 0.0),
            "focus_distance": 40.0,
            "focus_position": tuple(fly.body.position),
            "strike_readiness": 1.0,
            "reward": 0.0,
            "excitability": 1.0,
        }

        first = agent.update(0.01, [fly], 800, 600)
        self.assertFalse(first["caught_fly"])
        self.assertFalse(agent.tongue_extended)

        caught = None
        for _ in range(4):
            caught = agent.update(0.01, [fly], 800, 600)
            if caught["caught_fly"] is not None:
                break

        self.assertIsNotNone(caught["caught_fly"])
        self.assertTrue(agent.tongue_extended)

    def test_juvenile_frog_runtime_accepts_moderate_strike_readiness(self):
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent, Fly

        agent = BioFrogPredatorAgent(space=pymunk.Space(), training_mode=True)
        agent.success_prob = 1.0
        fly = Fly(agent.space, (float(agent.body.position[0]) + 42.0, float(agent.body.position[1])))

        agent.brain.update_predator_mode = lambda **kwargs: {
            "velocity": (0.0, 0.0),
            "memory_vector": (0.0, 0.0),
            "memory_score": 0.0,
            "dopamine": 0.0,
            "serotonin": 0.0,
            "acetylcholine": 0.0,
            "fatigue": 0.0,
            "glucose": 1.0,
            "neural_activity": 0.0,
            "arousal": 0.0,
            "curiosity": 0.0,
            "frustration": 0.0,
            "reward_confidence": 0.0,
            "action_vigor": 0.0,
            "search_burst": 0.0,
            "food_directness": 0.0,
            "focus_vector": (1.0, 0.0),
            "focus_distance": 42.0,
            "focus_position": tuple(fly.body.position),
            "strike_readiness": 0.23,
            "reward": 0.0,
            "excitability": 1.0,
            "is_juvenile": True,
        }

        caught = None
        for _ in range(4):
            caught = agent.update(0.01, [fly], 800, 600)
            if caught["caught_fly"] is not None:
                break

        self.assertIsNotNone(caught["caught_fly"])
        self.assertGreater(agent.caught_flies, 0)

    def test_frog_strike_uses_brain_selected_focus_target(self):
        from Frog_predator_neuro.simulation import BioFrogPredatorAgent, Fly

        agent = BioFrogPredatorAgent(space=pymunk.Space())
        agent.success_prob = 1.0
        lead_fly = Fly(agent.space, (float(agent.body.position[0]), float(agent.body.position[1]) - 42.0))
        chosen_fly = Fly(agent.space, (float(agent.body.position[0]) + 54.0, float(agent.body.position[1]) + 4.0))

        def fake_update(**kwargs):
            target_vector = np.array(kwargs["visible_targets"][1]["vector"], dtype=float)
            target_unit = target_vector / max(1e-6, float(np.linalg.norm(target_vector)))
            return {
                "velocity": (0.0, 0.0),
                "memory_vector": (0.0, 0.0),
                "memory_score": 0.0,
                "dopamine": 0.0,
                "serotonin": 0.0,
                "acetylcholine": 0.0,
                "fatigue": 0.0,
                "glucose": 1.0,
                "neural_activity": 0.0,
                "arousal": 0.0,
                "curiosity": 0.0,
                "frustration": 0.0,
                "reward_confidence": 0.0,
                "action_vigor": 0.0,
                "search_burst": 0.0,
                "food_directness": 0.0,
                "focus_vector": tuple(target_unit.tolist()),
                "focus_distance": kwargs["visible_targets"][1]["distance"],
                "focus_position": kwargs["visible_targets"][1]["position"],
                "strike_readiness": 1.0,
                "reward": 0.0,
                "excitability": 1.0,
            }

        agent.brain.update_predator_mode = fake_update

        caught = None
        for _ in range(4):
            caught = agent.update(0.01, [lead_fly, chosen_fly], 800, 600)
            if caught["caught_fly"] is not None:
                break

        self.assertIsNotNone(caught["caught_fly"])
        self.assertIs(caught["caught_fly"], chosen_fly)

    def test_frog_legacy_quick_test_contract_is_supported(self):
        from Frog_predator_neuro.simulation import Simulation

        sim = Simulation(headless=True)
        try:
            self.assertTrue(hasattr(sim, "agents"))
            self.assertIs(sim.agents, sim.frogs)
            agent = sim.agents[0]
            self.assertTrue(hasattr(agent, "body_energy"))
            self.assertTrue(hasattr(agent, "food_collected"))
            self.assertTrue(hasattr(agent, "x"))
            self.assertTrue(hasattr(agent, "y"))
        finally:
            sim.close()

    def test_frog_basal_ganglia_component_returns_action_selection(self):
        from Frog_predator_neuro.architecture.basal_ganglia import BasalGanglia

        bg = BasalGanglia()
        output = bg.select_action(np.ones(8) * 0.5, 0.5)

        self.assertIn("gating_signal", output)
        self.assertIn("confidence", output)
        self.assertGreaterEqual(float(output["gating_signal"]), 0.0)
        self.assertLessEqual(float(output["gating_signal"]), 1.0)


class WolfDiagnosticsTests(unittest.TestCase):
    def test_wolf_realistic_brain_update_returns_social_outputs(self):
        from Wolf_for_laba_v3_realistic_neuro.brain import BioWolfBrain

        brain = BioWolfBrain(rng=random.Random(0))
        perception = {
            "position": (100.0, 100.0),
            "current_tile": (2, 2),
            "heading_angle": 0.0,
            "visible_passable_tiles": {(2, 2), (3, 2)},
            "visible_walls": set(),
            "visible_food_tiles": set(),
            "visible_unvisited_tiles": {(3, 2)},
            "visible_peers": [],
            "heard_signals": [],
            "open_vectors": [(1.0, 0.0, 0.62)],
            "wall_vectors": [],
            "recent_progress": 0.0,
            "visited_count": 1,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 10.0,
        }

        output = brain.update(perception, body_energy=0.6, reward=0.0, time_s=0.0)

        self.assertIn("social_need", output)
        self.assertIn("signal_outputs", output)
        self.assertEqual(set(output["signal_outputs"].keys()), {"contact_howl", "food_bark", "distress_whine", "rally_yip"})
        self.assertIn("bg_gating_signal", output)
        self.assertIn("rumination", output)
        self.assertIn("social_tone", output)
        self.assertIn("food_prediction_error", output)
        self.assertIn("motivation_context", output)
        self.assertIn("hunger_bias", output["motivation_context"])

    def test_wolf_hunger_bias_modulates_early_food_path(self):
        from Wolf_for_laba_v3_realistic_neuro.brain import BioWolfBrain

        brain = BioWolfBrain(rng=random.Random(0))
        perception = {
            "position": (100.0, 100.0),
            "current_tile": (2, 2),
            "heading_angle": 0.0,
            "visible_passable_tiles": {(2, 2), (3, 2)},
            "visible_walls": set(),
            "visible_food_tiles": {(3, 2)},
            "visible_unvisited_tiles": {(3, 2)},
            "visible_peers": [],
            "heard_signals": [],
            "open_vectors": [(1.0, 0.0, 0.72)],
            "wall_vectors": [],
            "recent_progress": 0.0,
            "visited_count": 1,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 10.0,
        }

        low_energy_context = brain.build_motivation_context(perception, body_energy=0.30)
        _, low_food_strength = brain.compute_food_capture_vector(
            perception,
            body_energy=0.30,
            motivation_context=low_energy_context,
        )
        high_energy_context = brain.build_motivation_context(perception, body_energy=0.90)
        _, high_food_strength = brain.compute_food_capture_vector(
            perception,
            body_energy=0.90,
            motivation_context=high_energy_context,
        )

        self.assertGreater(low_energy_context["hunger_bias"], high_energy_context["hunger_bias"])
        self.assertGreater(low_food_strength, high_food_strength)

    def test_wolf_food_imprint_raises_reward_confidence_and_food_expectation(self):
        from Wolf_for_laba_v3_realistic_neuro.brain import BioWolfBrain

        brain = BioWolfBrain(rng=random.Random(0))
        baseline_confidence = brain.affect.reward_confidence
        baseline_expectation = brain.spatial_memory.food_expectation

        brain.register_food_experience((100.0, 100.0), reward_strength=0.65, social_context=True)

        self.assertGreater(brain.affect.reward_confidence, baseline_confidence)
        self.assertGreater(brain.spatial_memory.food_expectation, baseline_expectation)
        self.assertIsNotNone(brain.social_learning_target)

    def test_wolf_spatial_memory_uses_heterogeneous_delays(self):
        from Wolf_for_laba_v3_realistic_neuro.architecture.spatial_memory import SpatialMemory

        memory = SpatialMemory(rng=random.Random(0))
        delays = {round(link.synapse.delay, 2) for link in memory.recurrent_links}

        self.assertGreater(len(delays), 10)

    def test_wolf_replay_gate_prefers_idle_periods(self):
        from Wolf_for_laba_v3_realistic_neuro.architecture.spatial_memory import SpatialMemory

        memory = SpatialMemory(rng=random.Random(0))
        kwargs = {
            "heading_angle": 0.0,
            "reward": 0.0,
            "visible_food_sources": [],
            "visible_peer_sources": [],
            "heard_signals": [],
            "fatigue": 0.6,
            "social_need": 0.2,
            "energy_level": 1.0,
        }

        memory.observe(position=(100.0, 100.0), **kwargs)
        memory.observe(position=(100.0, 100.0), **kwargs)
        idle_gate = memory.replay_gate
        memory.observe(position=(200.0, 100.0), **kwargs)
        moving_gate = memory.replay_gate

        self.assertGreater(idle_gate, moving_gate)

    def test_wolf_agent_applies_body_inertia_to_sharp_turns(self):
        from Wolf_for_laba_v3_realistic_neuro.agent import BioWolfAgent

        class DummyWorld:
            def try_offset(self, agent, dx, dy):
                agent.x += dx
                agent.y += dy

        agent = BioWolfAgent(1, (2, 2), (255, 0, 0), rng=random.Random(0))
        world = DummyWorld()
        perception = {
            "wall_vectors": [],
            "visible_food_tiles": set(),
            "open_vectors": [(1.0, 0.0, 0.62), (-1.0, 0.0, 0.62)],
        }
        agent.last_output.update(
            {
                "energy_surplus": 0.0,
                "desperation": 0.0,
                "action_vigor": 0.45,
                "search_burst": 0.0,
                "policy_confidence": 0.3,
                "memory_vector": (0.0, 0.0),
                "reorientation_drive": 0.0,
                "loop_pressure": 0.0,
            }
        )

        dt = 1.0 / 60.0
        agent.apply_motion(world, (1.0, 0.0), perception, dt)
        forward_velocity = agent.body_velocity
        agent.apply_motion(world, (-1.0, 0.0), perception, dt)

        self.assertGreater(forward_velocity[0], 0.0)
        self.assertGreater(agent.body_velocity[0], -agent.max_speed * 0.25)

    def test_wolf_affect_tracks_social_tone_and_rumination(self):
        from Wolf_for_laba_v3_realistic_neuro.brain import BioWolfBrain

        distress_brain = BioWolfBrain(rng=random.Random(0))
        rally_brain = BioWolfBrain(rng=random.Random(0))
        base = {
            "position": (100.0, 100.0),
            "current_tile": (2, 2),
            "heading_angle": 0.0,
            "visible_passable_tiles": {(2, 2), (3, 2)},
            "visible_walls": set(),
            "visible_food_tiles": set(),
            "visible_unvisited_tiles": {(3, 2)},
            "visible_peers": [],
            "open_vectors": [(1.0, 0.0, 0.62)],
            "wall_vectors": [],
            "recent_progress": 0.0,
            "visited_count": 1,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 10.0,
        }

        distress = distress_brain.update(
            {**base, "heard_signals": [SimpleNamespace(tile=(2, 2), channel="distress_whine", strength=1.0)]},
            body_energy=0.55,
            reward=0.0,
            time_s=0.0,
        )
        rally = rally_brain.update(
            {**base, "heard_signals": [SimpleNamespace(tile=(2, 2), channel="rally_yip", strength=1.0)]},
            body_energy=0.55,
            reward=0.0,
            time_s=0.0,
        )

        self.assertGreater(distress["rumination"], 0.0)
        self.assertLess(distress["social_tone"], rally["social_tone"])

    def test_wolf_realistic_simulation_runs_headless(self):
        from Wolf_for_laba_v3_realistic_neuro.simulation import Simulation

        sim = Simulation(headless=True, seed=0)
        try:
            for _ in range(10):
                sim.update(1 / 60)
            self.assertEqual(len(sim.agents), 2)
            self.assertTrue(all(agent.alive for agent in sim.agents))
            self.assertIn("social_need", sim.agents[0].last_output)
            self.assertEqual(type(sim.agents[0].brain).__module__, "Wolf_for_laba_v3_realistic_neuro.brain")
        finally:
            sim.close()

    def test_wolf_training_bootstrap_seeds_food_learning_without_fake_food_count(self):
        from Wolf_for_laba_v3_realistic_neuro.simulation import Simulation

        sim = Simulation(headless=True, seed=0)
        try:
            self.assertTrue(all(agent.nurture_feedings == 1 for agent in sim.agents))
            self.assertTrue(all(agent.food_collected == 0 for agent in sim.agents))
            self.assertTrue(all(agent.pending_reward > 0.0 for agent in sim.agents))
            self.assertTrue(all(agent.brain.affect.reward_confidence > 0.35 for agent in sim.agents))
        finally:
            sim.close()

    def test_wolf_realistic_simulation_maintains_activity(self):
        from Wolf_for_laba_v3_realistic_neuro.simulation import Simulation

        sim = Simulation(headless=True, seed=0)
        try:
            for _ in range(240):
                sim.update(1 / 60)
            distances = [agent.total_distance for agent in sim.agents]
            self.assertTrue(all(agent.alive for agent in sim.agents))
            self.assertGreater(max(distances), 25.0)
        finally:
            sim.close()

    def test_wolf_training_bootstrap_leads_to_real_food_collection(self):
        from Wolf_for_laba_v3_realistic_neuro.simulation import Simulation

        sim = Simulation(headless=True, seed=0)
        try:
            for _ in range(360):
                sim.update(1 / 60)
            total_food = sum(agent.food_collected for agent in sim.agents)
            self.assertGreaterEqual(total_food, 1)
        finally:
            sim.close()

    def test_wolf_food_visibility_matches_local_navigation_radius(self):
        from Wolf_for_laba_v3_realistic_neuro.agent import BioWolfAgent
        from Wolf_for_laba_v3_realistic_neuro.models import Food

        class DummyWorld:
            def __init__(self):
                self.maze = [[0 for _ in range(10)] for _ in range(10)]
                self.foods = [Food(tile=(5, 2), room_id=-1), Food(tile=(6, 2), room_id=-1)]
                self.time_s = 0.0

            def active_agents(self):
                return []

            def heard_signals_for(self, _agent):
                return []

        agent = BioWolfAgent(1, (2, 2), (255, 0, 0), rng=random.Random(0))
        perception = agent.collect_perception(DummyWorld())

        self.assertIn((5, 2), perception["visible_food_tiles"])
        self.assertIn((5, 2), perception["visible_passable_tiles"])
        self.assertNotIn((6, 2), perception["visible_food_tiles"])

    def test_wolf_signal_hearing_extends_beyond_previous_radius(self):
        from Wolf_for_laba_v3_realistic_neuro.models import SocialSignal
        from Wolf_for_laba_v3_realistic_neuro.simulation import Simulation
        from Wolf_for_laba_v3_realistic_neuro.utils import tile_center

        sim = Simulation(headless=True, seed=0)
        try:
            agent = sim.agents[0]
            signal_tile = (agent.tile[0] + 5, agent.tile[1])
            signal = SocialSignal(
                time_s=0.0,
                sender_id=999,
                channel="food_bark",
                utterance="food-bark",
                summary="range check",
                tile=signal_tile,
                strength=1.0,
            )
            sim.active_signals = [signal]
            sim.time_s = 0.1
            heard = sim.heard_signals_for(agent)

            self.assertGreater(
                ((tile_center(signal_tile)[0] - agent.x) ** 2 + (tile_center(signal_tile)[1] - agent.y) ** 2) ** 0.5,
                160.0,
            )
            self.assertTrue(any(item.channel == "food_bark" and item.tile == signal_tile for item in heard))
        finally:
            sim.close()

    def test_wolf_realistic_agent_wraps_realistic_brain(self):
        from Wolf_for_laba_v3_realistic_neuro.agent import BioWolfAgent

        agent = BioWolfAgent(1, (2, 2), (255, 0, 0), rng=random.Random(0))
        self.assertEqual(type(agent.brain).__module__, "Wolf_for_laba_v3_realistic_neuro.brain")
        self.assertEqual(type(agent.brain).__name__, "BioWolfBrain")

    def test_wolf_realistic_brain_import_succeeds(self):
        module = import_fresh("Wolf_for_laba_v3_realistic_neuro.brain")
        self.assertTrue(hasattr(module, "BioWolfBrain"))

    def test_wolf_realistic_architecture_import_succeeds(self):
        module = import_fresh("Wolf_for_laba_v3_realistic_neuro.architecture")
        self.assertTrue(hasattr(module, "BasalGanglia"))
        self.assertTrue(hasattr(module, "SpatialMemory"))


class FrogSnnDiagnosticsTests(unittest.TestCase):
    def test_snn_brain_produces_spikes_for_salient_target(self):
        from frog_lib_snn.agent import SNNBrain

        np.random.seed(0)
        brain = SNNBrain()
        encoded = brain.encode(np.array([8.0, 0.0], dtype=float), visual_range=50.0, energy_ratio=1.0)
        spike_counts = []
        for _ in range(5):
            _, _, spike_count, _ = brain.forward(encoded, dt=0.01)
            spike_counts.append(spike_count)

        self.assertGreater(max(spike_counts), 0)

    def test_snn_frog_moves_toward_nearby_fly(self):
        from frog_lib_snn.agent import FrogSNNAgent

        random.seed(0)
        np.random.seed(0)
        space = pymunk.Space()
        frog = FrogSNNAgent(space, position=(100.0, 100.0), training_mode=False)
        fly_body = SimpleNamespace(position=(118.0, 100.0))
        fly = SimpleNamespace(alive=True, body=fly_body)
        try:
            states = [frog.update(0.01, [fly]) for _ in range(6)]
            spike_counts = [state["spike_count"] for state in states]
            speeds = [float(np.linalg.norm(np.array(state["velocity"], dtype=float))) for state in states]

            self.assertGreater(max(spike_counts), 0)
            self.assertGreater(max(speeds), 0.15)
        finally:
            frog.remove()

    def test_snn_simulation_adds_world_boundaries_for_frog(self):
        from frog_lib_snn.simulation import SNNFlyCatchingSimulation

        sim = SNNFlyCatchingSimulation(width=260, height=220, num_flies=0, headless=True)
        try:
            sim.frog.body.position = (210.0, 110.0)
            sim.frog.body.velocity = (500.0, 0.0)
            for _ in range(40):
                sim.space.step(sim.dt)
            self.assertLessEqual(float(sim.frog.body.position[0]), 259.0)
        finally:
            sim.close()

    def test_snn_simulation_exposes_detailed_neuro_panel(self):
        from frog_lib_snn.simulation import SNNFlyCatchingSimulation

        sim = SNNFlyCatchingSimulation(width=260, height=220, num_flies=0, headless=True)
        try:
            sim.last_agent_state = {
                "energy": 21.4,
                "fatigue": 0.29,
                "target_distance": 44.0,
                "alignment": 0.71,
                "tongue_extended": False,
                "tongue_length": 0.0,
                "spike_count": 5,
                "neural_activity": 0.083,
                "avg_membrane": 0.142,
                "refractory_fraction": 0.18,
                "eligibility_norm": 0.64,
                "controller_signal": 0.57,
                "reward": 1.0,
                "learning_reward": 0.22,
                "juvenile_progress": 0.48,
                "visibility": 0.88,
                "visual_range": 180.0,
                "focus_brightness": 0.67,
                "focus_motion": 0.74,
                "focus_facing": 0.52,
                "strike_drive": 0.61,
                "strike_intent": 0.34,
                "strike_commitment": 2,
                "strike_cooldown": 0.05,
            }

            sections = sim._neuro_panel_sections()
            titles = [title for title, _ in sections]
            flattened = "\n".join(line for _, lines in sections for line in lines)

            self.assertIn("Spike Dynamics", titles)
            self.assertIn("Learning", titles)
            self.assertIn("Sensory Focus", titles)
            self.assertIn("Eligibility", flattened)
            self.assertIn("Facing", flattened)
            self.assertIn("Commitment", flattened)
        finally:
            sim.close()


class FrogAnnDiagnosticsTests(unittest.TestCase):
    def test_ann_frog_moves_toward_nearby_fly(self):
        from frog_lib_ann.agent import FrogANNAgent

        random.seed(0)
        np.random.seed(0)
        space = pymunk.Space()
        frog = FrogANNAgent(space, position=(100.0, 100.0), training_mode=False)
        fly_body = SimpleNamespace(position=(130.0, 100.0), velocity=(0.0, 0.0))
        fly = SimpleNamespace(alive=True, body=fly_body)
        try:
            states = [frog.update(0.01, [fly]) for _ in range(8)]
            speeds = [float(np.linalg.norm(np.array(state["velocity"], dtype=float))) for state in states]
            distances = [state["target_distance"] for state in states if state["target_distance"] is not None]

            self.assertTrue(distances)
            self.assertGreater(max(speeds), 0.10)
        finally:
            frog.remove()

    def test_ann_agent_exposes_rl_signals(self):
        from frog_lib_ann.agent import FrogANNAgent

        random.seed(1)
        np.random.seed(1)
        space = pymunk.Space()
        frog = FrogANNAgent(space, position=(100.0, 100.0), training_mode=True)
        fly_body = SimpleNamespace(position=(126.0, 104.0), velocity=(0.0, 0.0))
        fly = SimpleNamespace(alive=True, body=fly_body)
        try:
            states = [frog.update(0.01, [fly]) for _ in range(6)]
            final = states[-1]

            self.assertIn("learning_reward", final)
            self.assertIn("actor_advantage", final)
            self.assertIn("strike_intent", final)
            self.assertTrue(math.isfinite(float(final["learning_reward"])))
            self.assertTrue(math.isfinite(float(final["actor_advantage"])))
        finally:
            frog.remove()

    def test_ann_simulation_adds_world_boundaries_for_frog(self):
        from frog_lib_ann.simulation import ANNFlyCatchingSimulation

        sim = ANNFlyCatchingSimulation(width=260, height=220, num_flies=0, headless=True)
        try:
            sim.frog.body.position = (210.0, 110.0)
            sim.frog.body.velocity = (500.0, 0.0)
            for _ in range(40):
                sim.space.step(sim.dt)
            self.assertLessEqual(float(sim.frog.body.position[0]), 259.0)
        finally:
            sim.close()

    def test_ann_frozen_agent_keeps_weights_constant(self):
        from frog_lib_ann_frozen.agent import FrogANNAgent

        random.seed(2)
        np.random.seed(2)
        space = pymunk.Space()
        frog = FrogANNAgent(space, position=(100.0, 100.0), training_mode=True)
        fly_body = SimpleNamespace(position=(128.0, 102.0), velocity=(0.0, 0.0))
        fly = SimpleNamespace(alive=True, body=fly_body)
        try:
            if hasattr(frog.brain, "net"):
                before = {key: value.detach().clone() for key, value in frog.brain.net.state_dict().items()}
                for _ in range(6):
                    frog.update(0.01, [fly])
                after = frog.brain.net.state_dict()
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, after[key]))
            else:
                before_actor = np.array(frog.brain.w_actor, copy=True)
                before_value = np.array(frog.brain.w_value, copy=True)
                for _ in range(6):
                    frog.update(0.01, [fly])
                self.assertTrue(np.array_equal(before_actor, frog.brain.w_actor))
                self.assertTrue(np.array_equal(before_value, frog.brain.w_value))
        finally:
            frog.remove()

    def test_ann_simulation_exposes_detailed_neuro_panel(self):
        from frog_lib_ann.simulation import ANNFlyCatchingSimulation

        sim = ANNFlyCatchingSimulation(width=260, height=220, num_flies=0, headless=True)
        try:
            sim.last_agent_state = {
                "energy": 24.1,
                "fatigue": 0.18,
                "target_distance": 38.0,
                "alignment": 0.82,
                "tongue_extended": True,
                "tongue_length": 19.5,
                "value_estimate": 0.47,
                "actor_advantage": 0.16,
                "learning_reward": 0.21,
                "reward": 1.0,
                "exploration_scale": 0.34,
                "controller_signal": 0.66,
                "sampled_action_x": 0.31,
                "sampled_action_y": -0.12,
                "sampled_action_strike": 0.58,
                "mean_action_x": 0.22,
                "mean_action_y": -0.08,
                "mean_action_strike": 0.44,
                "visibility": 0.94,
                "visual_range": 180.0,
                "focus_brightness": 0.71,
                "focus_motion": 0.63,
                "strike_drive": 0.69,
                "strike_intent": 0.54,
                "strike_commitment": 3,
                "strike_cooldown": 0.03,
            }

            sections = sim._neuro_panel_sections()
            titles = [title for title, _ in sections]
            flattened = "\n".join(line for _, lines in sections for line in lines)

            self.assertIn("Actor-Critic", titles)
            self.assertIn("Policy Output", titles)
            self.assertIn("Sensory Focus", titles)
            self.assertIn("Value", flattened)
            self.assertIn("Sample strike", flattened)
            self.assertIn("Commitment", flattened)
        finally:
            sim.close()

    def test_ann_sated_agent_blocks_strike_when_satiated_and_resumes_when_hungry(self):
        from frog_lib_ann_sated.agent import FrogANNAgent

        class DummyBrain:
            def act(self, obs, exploration_scale=1.0):
                del obs
                return {
                    "action": np.array([1.0, 0.0, 1.0], dtype=float),
                    "mean_action": np.array([1.0, 0.0, 1.0], dtype=float),
                    "value_estimate": 0.0,
                    "exploration_scale": float(exploration_scale),
                    "strike_intent": 1.0,
                }

            def learn(self, transition, reward, next_obs, done=False):
                del transition, reward, next_obs, done
                return 0.0

        random.seed(5)
        np.random.seed(5)
        space = pymunk.Space()
        frog = FrogANNAgent(space, position=(100.0, 100.0), training_mode=False)
        fly_body = SimpleNamespace(position=(116.0, 100.0), velocity=(0.0, 0.0))
        fly = SimpleNamespace(alive=True, body=fly_body)
        frog.brain = DummyBrain()
        try:
            frog.energy = frog.max_energy
            blocked_states = [frog.update(0.01, [fly]) for _ in range(4)]
            self.assertTrue(all(not state["hunting_enabled"] for state in blocked_states))
            self.assertEqual(frog.caught_flies, 0)
            self.assertFalse(any(state["tongue_extended"] for state in blocked_states))

            frog.energy = frog.max_energy * 0.50
            frog.last_catch_time = -100
            resumed_states = [frog.update(0.01, [fly]) for _ in range(6)]
            self.assertTrue(any(state["hunting_enabled"] for state in resumed_states))
            self.assertTrue(any(state["tongue_extended"] for state in resumed_states))
        finally:
            frog.remove()


class FrogBenchmarkScriptTests(unittest.TestCase):
    def test_benchmark_runner_and_report_generate_outputs(self):
        from benchmark_frogs import run_benchmark_suite
        from benchmark_frogs_report import generate_report

        temp_dir = Path(os.getcwd()) / f"benchmark_smoke_{os.getpid()}_{random.randint(0, 1_000_000)}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            output_dir = run_benchmark_suite(
                steps=120,
                spawn_seeds=(0,),
                repeats=1,
                modes=("adult",),
                architectures=("ANN", "SNN", "BIO"),
                sample_interval=20,
                competence_catches=2,
                output_dir=Path(temp_dir),
                workers=1,
            )
            report_path = generate_report(output_dir)

            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "run_metrics.csv").exists())
            self.assertTrue((output_dir / "aggregate_metrics.csv").exists())
            self.assertTrue((output_dir / "time_series.csv.gz").exists())
            self.assertTrue((output_dir / "event_log.csv.gz").exists())
            self.assertTrue(report_path.exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_benchmark_runner_supports_variant_architectures(self):
        from benchmark_frogs import run_benchmark_suite

        temp_dir = Path(os.getcwd()) / f"benchmark_variants_{os.getpid()}_{random.randint(0, 1_000_000)}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            output_dir = run_benchmark_suite(
                steps=40,
                spawn_seeds=(0,),
                repeats=1,
                modes=("adult",),
                architectures=(
                    "ANN",
                    "ANN_FROZEN",
                    "ANN_SATED",
                    "ANN_FROZEN_SATED",
                    "SNN",
                    "SNN_FROZEN",
                    "BIO",
                    "BIO_COMPARE",
                    "BIO_DUAL",
                    "BIO_DUAL_COMPARE",
                    "BIO_FAST",
                    "BIO_FAST_COMPARE",
                    "BIO_DUAL_FAST",
                    "BIO_DUAL_FAST_COMPARE",
                ),
                sample_interval=20,
                competence_catches=2,
                output_dir=Path(temp_dir),
                workers=1,
            )
            with (output_dir / "run_metrics.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 14)
            self.assertEqual(
                {row["arch"] for row in rows},
                {
                    "ANN",
                    "ANN_FROZEN",
                    "ANN_SATED",
                    "ANN_FROZEN_SATED",
                    "SNN",
                    "SNN_FROZEN",
                    "BIO",
                    "BIO_COMPARE",
                    "BIO_DUAL",
                    "BIO_DUAL_COMPARE",
                    "BIO_FAST",
                    "BIO_FAST_COMPARE",
                    "BIO_DUAL_FAST",
                    "BIO_DUAL_FAST_COMPARE",
                },
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_benchmark_runner_preserves_explicit_repeat_indices(self):
        from benchmark_frogs import run_benchmark_suite

        temp_dir = Path(os.getcwd()) / f"benchmark_repeats_{os.getpid()}_{random.randint(0, 1_000_000)}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            output_dir = run_benchmark_suite(
                steps=10,
                spawn_seeds=(2,),
                repeats=5,
                repeat_indices=(1, 4),
                modes=("adult",),
                architectures=("ANN",),
                sample_interval=10,
                competence_catches=2,
                output_dir=Path(temp_dir),
                workers=1,
            )
            with (output_dir / "run_metrics.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["repeat"] for row in rows], ["1", "4"])
            self.assertEqual({row["spawn_seed"] for row in rows}, {"2"})
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_subprocess_study_splits_seed_and_repeat_batches(self):
        from run_frog_study_subprocess import build_specs

        args = SimpleNamespace(
            steps=20,
            seeds=3,
            repeats=2,
            seed_batch_size=1,
            repeat_batch_size=1,
            modes=("adult",),
            architectures=("BIO",),
            sample_interval=10,
            competence_catches=2,
        )

        specs = build_specs(args, Path("dummy_study"))
        self.assertEqual(len(specs), 6)
        self.assertEqual(specs[0].spawn_seeds, (0,))
        self.assertEqual(specs[0].repeat_indices, (0,))
        self.assertEqual(specs[-1].spawn_seeds, (2,))
        self.assertEqual(specs[-1].repeat_indices, (1,))
        self.assertIn("s2_r1", specs[-1].shard_name)


class FrogVariantDiagnosticsTests(unittest.TestCase):
    def test_ann_sated_variants_import_and_step(self):
        from frog_lib_ann_sated.simulation import ANNFlyCatchingSimulation as ANNSatedSimulation
        from frog_lib_ann_frozen_sated.simulation import ANNFlyCatchingSimulation as ANNFrozenSatedSimulation

        for simulation_cls in (ANNSatedSimulation, ANNFrozenSatedSimulation):
            sim = simulation_cls(width=260, height=220, num_flies=2, headless=True, training_mode=False)
            try:
                result = sim.step()
                self.assertIn("energy", result)
                self.assertIn("hunting_enabled", result)
                self.assertIn("satiety_gate", result)
            finally:
                sim.close()

    def test_bio_fast_variants_import_and_step(self):
        from Frog_predator_neuro_fast.simulation import Simulation as BioFastSimulation
        from Frog_predator_neuro_dual_fast.simulation import Simulation as BioDualFastSimulation

        for simulation_cls in (BioFastSimulation, BioDualFastSimulation):
            sim = simulation_cls(width=260, height=220, num_flies=2, headless=True, training_mode=False, brain_seed=4)
            try:
                result = sim.step()
                self.assertIn("agents", result)
                self.assertGreaterEqual(len(result["agents"]), 1)
                self.assertIn("energy", result["agents"][0])
            finally:
                sim.close()

    def test_bio_compare_variants_import_and_step(self):
        from Frog_predator_neuro_compare.simulation import Simulation as BioSimulation
        from Frog_predator_neuro_dual_compare.simulation import Simulation as BioDualSimulation
        from Frog_predator_neuro_fast_compare.simulation import Simulation as BioFastSimulation
        from Frog_predator_neuro_dual_fast_compare.simulation import Simulation as BioDualFastSimulation

        for simulation_cls in (BioSimulation, BioDualSimulation, BioFastSimulation, BioDualFastSimulation):
            sim = simulation_cls(
                width=260,
                height=220,
                num_flies=2,
                headless=True,
                training_mode=False,
                brain_seed=4,
            )
            try:
                result = sim.step()
                self.assertIn("agents", result)
                self.assertTrue(bool(getattr(sim.frogs[0].brain, "comparable_mode", False)))
                self.assertIn("energy", result["agents"][0])
            finally:
                sim.close()

    def test_snn_frozen_agent_keeps_readout_constant(self):
        from frog_lib_snn_frozen.agent import FrogSNNAgent

        random.seed(3)
        np.random.seed(3)
        space = pymunk.Space()
        frog = FrogSNNAgent(space, position=(100.0, 100.0), training_mode=True)
        fly_body = SimpleNamespace(position=(122.0, 100.0), velocity=(0.0, 0.0))
        fly = SimpleNamespace(alive=True, body=fly_body)
        try:
            before = np.array(frog.brain.w_out, copy=True)
            for _ in range(8):
                frog.update(0.01, [fly])
            self.assertTrue(np.array_equal(before, frog.brain.w_out))
        finally:
            frog.remove()

    def test_bio_dual_brain_exposes_fast_loop_fields(self):
        from Frog_predator_neuro_dual.brain import BioFrogBrain

        brain = BioFrogBrain(rng=random.Random(0))
        output = brain.update_predator_mode(
            position=(100.0, 100.0),
            heading_angle=0.0,
            visible_targets=[
                {
                    "vector": (28.0, 0.0),
                    "brightness": 0.78,
                    "motion": 0.64,
                    "facing": 1.0,
                    "position": (128.0, 100.0),
                }
            ],
            body_energy=18.0,
            reward=0.0,
            visual_range_px=180.0,
            food_collected=0,
            stall_time=0.0,
            time_since_reward=4.0,
            visited_count=1,
        )

        self.assertIn("prey_permission", output)
        self.assertIn("fast_target_lock", output)
        self.assertIn("fast_loop_gate", output)
        self.assertIn("fast_strike_drive", output)
        self.assertGreaterEqual(float(output["prey_permission"]), 0.0)

    def test_bio_dual_compare_mode_lifts_prey_permission_for_satiated_visible_prey(self):
        from Frog_predator_neuro_dual.brain import BioFrogBrain
        from Frog_predator_neuro_dual_compare.brain import BioFrogBrain as CompareBioFrogBrain

        kwargs = {
            "position": (100.0, 100.0),
            "heading_angle": 0.0,
            "visible_targets": [
                {
                    "vector": (28.0, 0.0),
                    "brightness": 0.82,
                    "motion": 0.68,
                    "facing": 1.0,
                    "position": (128.0, 100.0),
                }
            ],
            "body_energy": 28.0,
            "reward": 0.0,
            "visual_range_px": 180.0,
            "food_collected": 0,
            "stall_time": 0.0,
            "time_since_reward": 4.0,
            "visited_count": 1,
        }

        baseline = BioFrogBrain(rng=random.Random(0))
        comparable = CompareBioFrogBrain(rng=random.Random(0), comparable_mode=True)

        baseline_output = baseline.update_predator_mode(**kwargs)
        comparable_output = comparable.update_predator_mode(**kwargs)

        self.assertGreater(
            float(comparable_output["motivation_context"]["task_set_bias"]),
            0.0,
        )
        self.assertGreater(
            float(comparable_output["prey_permission"]),
            float(baseline_output["prey_permission"]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest

from biotorch_proto import BrainGraph, CustomInstinctRule, InstinctBundle, RegionSpec
from biotorch_proto.core import ProjectionSpec
from biotorch_proto.instincts import InstinctBuffers
from biotorch_proto.models import build_frog_model, build_wolf_model
from biotorch_proto.neurochemistry import ChemistryProfile, TransmitterSpec
from biotorch_proto.registries import neuron_registry, transmitter_registry


class TerritorialPulseRule(CustomInstinctRule):
    def __init__(self):
        super().__init__("territorial_pulse")

    def step(self, context, state, buffers: InstinctBuffers, dt: float) -> None:
        strength = context.get("audio.social", 0.0) * 0.5 + context.get("vision.threat", 0.0) * 0.5
        buffers.add_vote("vocalization", strength)
        state["territorial_alert"] = strength
        context["territorial_alert"] = strength


class BioTorchPrototypeTests(unittest.TestCase):
    def test_region_spec_is_parameterized(self):
        region = RegionSpec("tectum").geometry(columns=64, layers=3, microcolumns=4, maps=["prey", "threat"])
        region.add_cell_type("projection", "AdaptiveLIF", fraction=0.55)
        region.add_microcircuit("lateral_inhibition", connectivity="topographic", inhibition_strength=0.4)

        self.assertEqual(region.columns, 64)
        self.assertEqual(region.layer_count, 3)
        self.assertEqual(region.microcolumns, 4)
        self.assertIn("prey", region.maps)
        self.assertEqual(region.cell_types[0].neuron_type, "AdaptiveLIF")

    def test_registries_expose_extended_neurons_and_catecholamines(self):
        self.assertIn("BurstingNeuron", neuron_registry)
        self.assertIn("D1MSN", neuron_registry)
        self.assertIn("norepinephrine", transmitter_registry)
        self.assertIn("epinephrine", transmitter_registry)

    def test_custom_transmitter_can_be_registered(self):
        transmitter_registry.register(
            "territorial_peptide_x",
            TransmitterSpec("territorial_peptide_x", "peptide", "diffuse", "slow", 0.05),
        )
        self.assertIn("territorial_peptide_x", transmitter_registry)

    def test_frog_model_compiles_and_runs(self):
        brain = build_frog_model(tectum_columns=20, hippocampus_columns=36)
        compiled = brain.compile(device="cpu", mode="mixed")

        step = compiled.step({"vision.prey": 0.92, "body.energy": 0.35}, dt=1.0)

        self.assertEqual(step["device"], "cpu")
        self.assertIn("locomotion", step["effectors"])
        self.assertIn("tongue", step["effectors"])
        self.assertGreater(step["effectors"]["locomotion"], 0.0)
        self.assertGreaterEqual(step["effectors"]["tongue"], 0.0)
        self.assertIn("hunger", step["instinct_state"])
        self.assertIn("epinephrine", step["chemistry"])

    def test_wolf_model_compiles_and_runs(self):
        brain = build_wolf_model(tectum_columns=28, hippocampus_columns=72)
        compiled = brain.compile(device="cpu", mode="mixed")

        step = compiled.step(
            {
                "vision.food": 0.65,
                "vision.threat": 0.15,
                "audio.social": 0.72,
                "body.energy": 0.48,
                "body.social_contact": 0.22,
            },
            dt=1.0,
        )

        self.assertIn("locomotion", step["effectors"])
        self.assertIn("vocalization", step["effectors"])
        self.assertGreater(step["effectors"]["locomotion"], 0.0)
        self.assertGreater(step["effectors"]["vocalization"], 0.0)
        self.assertIn("norepinephrine", step["chemistry"])
        self.assertIn("epinephrine", step["chemistry"])
        self.assertIn("attachment_drive", step["instinct_state"])

    def test_custom_instinct_rule_works_without_core_changes(self):
        brain = BrainGraph("custom_proto")
        brain.add_sensor("audio.social", channel="audio.social")
        brain.add_sensor("vision.threat", channel="vision.threat")
        brain.add_effector("effector.vocalization", channel="vocalization")
        brain.use_chemistry(ChemistryProfile().include("dopamine"))
        brain.add_instinct(InstinctBundle("territorial_bundle", [TerritorialPulseRule()]))
        compiled = brain.compile()

        step = compiled.step({"audio.social": 0.8, "vision.threat": 0.6})

        self.assertGreater(step["effectors"]["vocalization"], 0.0)
        self.assertIn("territorial_alert", step["instinct_state"])

    def test_projections_support_declared_bridge_types(self):
        brain = BrainGraph("bridge_proto")
        brain.add_sensor("vision.signal", channel="vision.signal")
        brain.add_population("salience", "SpikingPopulation", size=8)
        brain.add_effector("move", channel="locomotion")
        projection = brain.connect("vision.signal", "salience", bridge="tensor_to_spikes", weight=1.1)
        brain.connect("salience", "move", bridge="spikes_to_rate", weight=1.0)
        compiled = brain.compile()

        self.assertIsInstance(projection, ProjectionSpec)
        step = compiled.step({"vision.signal": 0.9})
        self.assertGreater(step["effectors"]["locomotion"], 0.0)


if __name__ == "__main__":
    unittest.main()

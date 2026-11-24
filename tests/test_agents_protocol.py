import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from driveaudit.agents.predict import ego_state, selected_rgb


class PolicyInputTests(unittest.TestCase):
    def test_generated_branch_rejects_organizer_access(self):
        args = SimpleNamespace(source="generated", rgb_scenes=Path("generated"),
                               real_mapping=Path("private"), cases=["scene"])
        with self.assertRaises(ValueError):
            selected_rgb(args)
        args.real_mapping = None
        self.assertEqual(selected_rgb(args)["scene"], Path("generated/scene/future/rgb"))

    def test_real_branch_uses_exact_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "mapping.json"
            mapping.write_text(json.dumps([{"public_scene": "a", "source_path": "/real/other"}]))
            args = SimpleNamespace(source="real", rgb_scenes=None, real_mapping=mapping, cases=["a"])
            self.assertEqual(selected_rgb(args)["a"], Path("/real/other/future/rgb"))
            args.cases = ["b"]
            with self.assertRaises(KeyError):
                selected_rgb(args)

    def test_ego_velocity_uses_only_three_past_measurements(self):
        # Constant 2 m/s along world Y, facing world Y. Future data is absent.
        states = {i: (0.0, y, math.pi/2) for i, y in enumerate([0, 0.2, 0.6])}
        clock = {0: 0, 1: 100_000, 2: 300_000}
        state = ego_state(2, clock, states)
        self.assertAlmostEqual(state["velocity"][0], 2.0)
        self.assertAlmostEqual(state["velocity"][1], 0.0)
        self.assertAlmostEqual(state["acceleration"][0], 0.0)
        self.assertEqual(state["timestamp"], clock[2])

    def test_legacy_policy_coordinates_and_scene_seed(self):
        from driveaudit.agents.native.camera_only_temporal import stable_request_seed
        from driveaudit.agents.temporal import planning_to_unified_ego

        self.assertEqual(planning_to_unified_ego([[1, 2], [3, 4]]).tolist(), [[2, -1], [4, -3]])
        self.assertNotEqual(stable_request_seed("a", 12), stable_request_seed("b", 12))

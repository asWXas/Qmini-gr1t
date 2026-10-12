"""Standalone curve tests. Run python scripts/test_curves.py -v; no Isaac Sim."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ENV_DIR = Path(__file__).resolve().parents[1] / "source/Qmini/Qmini/tasks/manager_based/umv/env"
sys.path.insert(0, str(ENV_DIR))

import curve_utils as curves
import point1
import point2


class CurveTests(unittest.TestCase):
    def generate(self, builder, count=25):
        return curves.generate_piecewise_guideline(builder(), count, preserve_junctions=True)

    def test_origin_and_critical_nodes(self):
        for builder in (point1.build_three_point_turn_segments, point2.build_backflip_segments):
            segments = builder()
            guideline, dense, arcs = self.generate(builder)
            points = np.asarray([wp["p"] for wp in guideline])
            self.assertEqual(points.shape, (25, 3))
            np.testing.assert_array_equal(points[0], [0, 0, 0])
            np.testing.assert_array_equal(dense[0], [0, 0, 0])
            np.testing.assert_array_equal(points[-1], segments[-1].x1)
            self.assertTrue(np.isfinite(points).all())
            self.assertTrue((points[:, 2] >= -1e-8).all())
            self.assertTrue((np.diff([wp["d"] for wp in guideline]) > 0).all())
            self.assertAlmostEqual(guideline[-1]["d"], arcs[-1])
            for segment in segments:
                self.assertTrue(np.isclose(points, segment.x1, atol=1e-8).all(axis=1).any())

    def test_three_point_turn_shape(self):
        _, dense, _ = self.generate(point1.build_three_point_turn_segments)
        self.assertAlmostEqual(dense[:, 1].max(), 1.4)
        self.assertAlmostEqual(dense[:, 2].max(), 0.45)
        self.assertTrue((np.diff(dense[:, 0]) < 0).any())
        self.assertTrue((np.diff(dense[:, 0]) > 0).any())

    def test_backflip_loop_order_and_relative_height(self):
        guideline, _, _ = self.generate(point2.build_backflip_segments)
        points = np.asarray([wp["p"] for wp in guideline])
        np.testing.assert_array_equal(points[:, 1], 0)
        self.assertAlmostEqual(points[:, 2].max(), 1.1)
        indices = np.flatnonzero(np.isclose(points, [1.95, 0, 0.15]).all(axis=1))
        self.assertEqual(len(indices), 2)
        peak = int(points[:, 2].argmax())
        self.assertLess(indices[0], peak)
        self.assertGreater(indices[1], peak)
        self.assertGreater(guideline[indices[1]]["d"], guideline[indices[0]]["d"])

    def test_json_assets_match_generators(self):
        for filename, builder in (
            ("three_point_turn.json", point1.build_three_point_turn_segments),
            ("backflip.json", point2.build_backflip_segments),
        ):
            payload = json.loads((ENV_DIR / filename).read_text())
            guideline, _, _ = self.generate(builder)
            np.testing.assert_array_equal(payload["waypoints"][0], [0, 0, 0])
            np.testing.assert_array_equal(payload["waypoints"], np.round([wp["p"] for wp in guideline], 4))
            self.assertEqual(payload["frame"], "stunt_start")
            self.assertTrue(payload["position_only"])

    def test_original_jump_unchanged(self):
        guideline, _, _ = curves.generate_piecewise_guideline(point2.build_jump_segments(), 12)
        payload = json.loads((ENV_DIR / "mini_hop.json").read_text())
        np.testing.assert_array_equal(payload["waypoints"], np.round([wp["p"] for wp in guideline], 4))

    def test_invalid_input_and_nonzero_origin_rejected(self):
        with self.assertRaises(ValueError):
            curves.generate_piecewise_guideline([], 25)
        with self.assertRaises(ValueError):
            self.generate(point2.build_backflip_segments, 8)
        segment = curves.HermiteConfig(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            curves.generate_piecewise_guideline([segment])
        segments = point1.build_three_point_turn_segments()
        segments[1].x0 = np.array([10, 0, 0])
        with self.assertRaises(ValueError):
            curves.generate_piecewise_guideline(segments)
        guideline, _, _ = self.generate(point1.build_three_point_turn_segments)
        guideline[0]["p"] = np.array([0, 0, 0.3])
        with self.assertRaises(ValueError):
            curves.guideline_payload(guideline, "bad", "")

    def run_cli(self, script, *args):
        return subprocess.run([sys.executable, str(ENV_DIR / script), *args], capture_output=True, text=True)

    def test_cli_and_overwrite_protection(self):
        for script, name in (("point1.py", "three_point_turn"), ("point2.py", "backflip")):
            result = self.run_cli(script, "--output", "-")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["name"], name)
            self.assertEqual(payload["waypoints"][0], [0, 0, 0])
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / "curve.json")
            result = self.run_cli("point1.py", "--output", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = Path(output).read_bytes()
            result = self.run_cli("point2.py", "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(Path(output).read_bytes(), original)
            result = self.run_cli("point2.py", "--output", output, "--force")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(Path(output).read_text())["name"], "backflip")
            result = self.run_cli("point1.py", "--output", output, "--plot", output, "--force")
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

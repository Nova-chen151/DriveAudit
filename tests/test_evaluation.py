"""Fast checks for the paired demonstration, independent of any private scenes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from driveaudit.evaluation.score import paired_summary, read_records, validate_pair


class PairedEvaluationTest(unittest.TestCase):
    def test_missing_attribution_is_kept_in_support_denominator(self):
        def branch(scores):
            rows = {i: {"rpdms": v} for i, v in enumerate(scores)}
            ep = dict(route_completion_ratio=.5, route_completion_status="valid",
                      route_projection_coverage=1., termination_reason="scene_end")
            return dict(rows=rows, episode={"episode": ep}, unscored_tail=[4, 5])
        result = paired_summary("case", dict(real=branch([.8, .7]), generated=branch([None, .4])))
        self.assertEqual(result["paired_windows"], 1)
        self.assertEqual(result["legal_windows"], 2)
        self.assertEqual(result["real"]["rpdms"], .7)
        self.assertIsNone(result["single_policy_TE"])

    def test_rgb_provenance_is_bound_to_side_and_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {s: root/s/"case" for s in ("generated", "real")}
            pairs = {}
            for side, scene in paths.items():
                scene.mkdir(parents=True)
                (scene/"timeline.json").write_text(json.dumps(dict(frames=[
                    dict(frame_index=3, reference_time_us=0), dict(frame_index=4, reference_time_us=4)])))
                (scene/"generation_input.json").write_text('{"reference_frame":3}')
                for name in ("history/ego.csv", "future/ego_trajectory.csv", "map/map_meta.json",
                             "map/semantic_map.npz", "calibration/reference_ego_poses.json"):
                    path = scene/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(name)
                image = scene/"future/rgb/CAM_FRONT/000004.jpg"
                image.parent.mkdir(parents=True)
                image.write_bytes(side.encode())
                row = dict(policy="test", checkpoint_sha256="a"*64, config_sha256="b"*64,
                           backbone_sha256="c"*64, rgb_sha256={"CAM_FRONT": hashlib.sha256(image.read_bytes()).hexdigest()},
                           request_seed=0, reference_time_us=4, frame_id=0, frame_index=4,
                           case_id="case", source=side, coordinate_system="ego_x_forward_y_left",
                           dt_s=.5, trajectory_xy=[[0, 0]]*6)
                records = root/(side+".jsonl")
                records.write_text(json.dumps(row)+"\n")
                pairs[side] = read_records(records, side)["case"]
            validate_pair("case", paths, pairs)
            (paths["real"]/"future/rgb/CAM_FRONT/000004.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "sampled image hash mismatch"):
                validate_pair("case", paths, pairs)


class FullPanelSupportTest(unittest.TestCase):
    def test_common_window_mask_is_shared_by_all_twelve_branches(self):
        from driveaudit.evaluation.panel import POLICIES, common_frames
        scores = {p: {side: {i: {"rpdms": 1.} for i in range(15)}
                       for side in ("real", "generated")} for p in POLICIES}
        scores["drivor"]["real"][0]["rpdms"] = None
        scores["vad"]["generated"][1]["rpdms"] = None
        ids, denominator = common_frames(scores)
        self.assertEqual(ids, list(range(2, 15)))
        self.assertEqual(denominator, 15)
        scores["uniad"]["generated"].pop(14)
        with self.assertRaisesRegex(ValueError, "same native decision set"):
            common_frames(scores)

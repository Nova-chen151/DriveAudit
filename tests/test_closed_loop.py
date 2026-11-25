import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from driveaudit.closed_loop.core import execute_prefix
class ExecutionTests(unittest.TestCase):
    def test_rotated_translation(self):
        result=execute_prefix([10,20,1,math.pi/2],[[2,0,0]]*8,np.arange(1,9)*.5)
        np.testing.assert_allclose(result[-1],[10,22,1,math.pi/2])
        np.testing.assert_allclose(result[0,:2],[10,20.4])
    def test_feedback_not_reset(self):
        a=execute_prefix([0,0,0,0],[[1,0,0]]*8,np.arange(1,9)*.5)
        b=execute_prefix(a[-1],[[1,0,0]]*8,np.arange(1,9)*.5)
        self.assertEqual(b[-1,0],2)
    def test_bad_output_rejected(self):
        with self.assertRaises(ValueError):execute_prefix([0,0,0,0],[[float('nan'),0,0]]*8,np.arange(1,9)*.5)
    def test_future_access_rejected(self):
        from driveaudit.closed_loop.worker import deny_future
        with self.assertRaises(RuntimeError):deny_future('open',('/data/scene/future/rgb/CAM_FRONT/000010.jpg',))
        with self.assertRaises(RuntimeError):deny_future('open',('/data/scene/future/ego_trajectory.csv',))
        deny_future('open',('/data/scene/history/ego.csv',))
    def test_heading_shortest_arc(self):
        result=execute_prefix([0,0,0,0],[[0,0,2*math.pi-.1]]*8,np.arange(1,9)*.5)
        self.assertAlmostEqual(result[-1,3],-.1)

class WorkerSnapshotTests(unittest.TestCase):
    def test_seven_profiles_snapshot_their_model_sources(self):
        from driveaudit.closed_loop.run_goal import snapshot_sources
        from driveaudit.generation import MODEL_DIRS
        from driveaudit.resources import resource_root

        models = resource_root() / "models"
        with tempfile.TemporaryDirectory() as directory:
            for model, image_source in MODEL_DIRS.items():
                with self.subTest(model=model):
                    run = Path(directory) / model
                    run.mkdir()
                    snapshot_sources(run, model)
                    manifest = json.loads((run / "code_manifest.json").read_text())
                    sources = {
                        "catk_traffic.py": models / "generation/trajectory/CAT-K/closed_loop.py",
                        "drivor_policy.py": models / "agents/DrivoR/closed_loop.py",
                        f"render_{model}.py": image_source / "closed_loop.py",
                    }
                    for name, source in sources.items():
                        recorded = (run / "code_snapshot" / name).read_bytes()
                        self.assertEqual(recorded, source.read_bytes())
                        self.assertEqual(manifest[name], hashlib.sha256(recorded).hexdigest())


if __name__=='__main__':unittest.main()

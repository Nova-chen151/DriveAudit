import json
from pathlib import Path
import tempfile
import unittest

from driveaudit.pipeline import Pipeline, resolve


class PipelineTest(unittest.TestCase):
    def test_virtualenv_interpreter_uses_its_own_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            interpreter = base / "venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to("/usr/bin/python3")
            self.assertEqual(resolve(base/"profile.json", "venv/bin/python"), interpreter)

    def test_participant_stage_has_no_organizer_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"generation.json").write_text(json.dumps(dict(archive="input.tar.gz", work_dir="generation",
                                                               scenes=["case_000001"])))
            profile = root/"config.json"
            profile.write_text(json.dumps(dict(renderer="opendwm", generation="generation.json", output="out")))
            pipe = Pipeline(profile)
            for action in ("traffic", "stage", "render"):
                args = [str(arg) for arg in pipe.commands(action)[0]]
                self.assertNotIn("--mapping", args)
                self.assertIn(str((root/"generation.json").resolve()), args)

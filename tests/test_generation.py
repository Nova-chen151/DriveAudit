import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from driveaudit.generation import MODELS, VENDOR
from driveaudit.generation.__main__ import load_config, main, scene_ids, traffic
from driveaudit.generation.renderer import launch_plan, overlay, runtime_for
from driveaudit.generation.staging import relative_scene, stage, verify_stage, write


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.case = "case_000017"
        self.archive = self.root / "participant.tar.gz"
        self.archive.write_bytes(b"archive fixture")
        self.cfg = dict(scenes=[self.case], archive=str(self.archive), work_dir=str(self.root / "work"),
                        catk_run=str(self.root / "previous"), engine_code_root=str(self.root / "code"),
                        models_json=str(self.root / "models.json"), v2_python="/env/mddit/bin/python",
                        catk_python="/env/catk/bin/python")

    def make_bridge(self):
        previous = Path(self.cfg["catk_run"])
        ref = relative_scene(self.case)
        future = previous / "render_inputs" / self.case / ref / "future"
        future.mkdir(parents=True)
        (future / "future_trajectories.csv").write_bytes(b"actors")
        (future / "ego_trajectory.csv").write_bytes(b"public motion")
        public = previous / "inputs/participant/scenes" / self.case / "conditioning/ego_trajectory.csv"
        public.parent.mkdir(parents=True)
        public.write_bytes(b"public motion")
        sha = lambda value: hashlib.sha256(value).hexdigest()
        write(previous / "COMPLETED.json", dict(status="PASS", future_actor_gt_used=False, future_rgb_used=False,
              input=dict(scene_ids=[self.case], archive_sha256=sha(self.archive.read_bytes())),
              renderer_loader_checks=[dict(scene_id=self.case, status="PASS", public_ego_unchanged=True,
                                      generated_actor_csv_sha256=sha(b"actors"))]))
        return future

    def test_stages_public_ego_and_generated_actors_then_detects_tamper(self):
        self.make_bridge()
        out = stage(self.cfg)
        self.assertEqual(out, verify_stage(self.cfg))
        with self.assertRaises(FileExistsError):
            stage(self.cfg)
        (out / relative_scene(self.case) / "future/future_trajectories.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Invalid staged conditions"):
            verify_stage(self.cfg)

    def test_detects_mutated_calibration_before_native_render(self):
        self.make_bridge()
        out = stage(self.cfg)
        (out / relative_scene(self.case) / "participant_render_frame.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Invalid staged conditions"):
            verify_stage(self.cfg)

    def test_private_rgb_rejected_without_partial_final(self):
        (self.make_bridge() / "rgb").mkdir()
        with self.assertRaisesRegex(ValueError, "future RGB"):
            stage(self.cfg)
        self.assertFalse((Path(self.cfg["work_dir"]) / "conditions").exists())

    def test_reject_wrong_archive(self):
        self.make_bridge()
        self.archive.write_bytes(b"other archive")
        with self.assertRaisesRegex(ValueError, "Archive differs"):
            stage(self.cfg)

    def test_config_and_real_traffic_entry(self):
        path = self.root / "run.json"
        main(["init", "--archive", str(self.archive), "--work-dir", self.cfg["work_dir"],
              "--engine-code-root", self.cfg["engine_code_root"], "--conda-envs", str(self.root),
              "--scene", self.case, "--output", str(path)])
        cfg = load_config(path)
        with patch("driveaudit.generation.__main__.subprocess.run") as run:
            traffic(cfg)
        argv = run.call_args.args[0]
        self.assertTrue(Path(argv[2]).is_file())
        self.assertIn(self.case, argv)
        self.assertIn(str(self.archive), argv)
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        with self.assertRaises(ValueError):
            scene_ids(["../../bad"])

    def test_clean_profile_needs_no_previous_experiment(self):
        profile = self.root / "clean.json"
        main(["init", "--archive", str(self.archive), "--work-dir", self.cfg["work_dir"],
              "--engine-code-root", str(self.root / "engines"), "--conda-envs", str(self.root / "envs"),
              "--weights-root", str(self.root / "weights"), "--scene", self.case, "--output", str(profile)])
        config = load_config(profile)
        self.assertNotIn("models_json", config)
        self.assertEqual(set(config["models"]), set(MODELS) - {"magicdrivev2"})
        self.assertEqual(config["models"]["opendwm"]["python"]["renderer"], str(self.root / "envs/opendwm/bin/python"))
        self.assertTrue(config["models"]["opendwm"]["render_checkpoint"].startswith(str(self.root / "weights/")))

    def test_native_adapter_commands_and_companion_libraries(self):
        shell = dict(PATH="/bin", SLURM_JOB_ID="123", CUDA_VISIBLE_DEVICES="3,5",
                     GENBENCH_STATIC_ASSET_ROOT="/other/dataset", GENBENCH_CATK_AGENT_ONLY="1")
        cfg = dict(self.cfg, scenes=[self.case, "case_000020"])
        weights = dict(render_checkpoint="/weights/model", sd15="/weights/sd", single_checkpoint="/weights/single",
                       clip_checkpoint="/weights/clip", base_model="/weights/base", wan_checkpoint="/weights/wan",
                       python={"renderer": "/env/bin/python"})
        write(Path(cfg["models_json"]), {name: weights for name in MODELS})
        root = self.root / "conditions"
        for name in MODELS:
            with self.subTest(model=name):
                plans = launch_plan(cfg, name, VENDOR, root, self.root, shell)
                cmd, _, env = plans[0]
                self.assertEqual(env["GENBENCH_USE_CATK_EGO_PREDICTION"], "0")
                self.assertEqual(env["GENBENCH_SCENE_INDICES"], "0,1")
                self.assertNotIn("GENBENCH_STATIC_ASSET_ROOT", env)
                self.assertNotIn("GENBENCH_CATK_AGENT_ONLY", env)
                if name == "panacea":
                    self.assertIn("--ordered-targets", cmd)
                    self.assertIn("autoregressive", cmd)
                if name == "magicdrivev2":
                    self.assertIn("full89", cmd)
                if name in ("dreamforge", "basedreamer", "magicdrive"):
                    self.assertEqual([p[2]["CUDA_VISIBLE_DEVICES"] for p in plans], ["3", "5"])
                else:
                    self.assertEqual(len(plans), 1)
                if name != "horizondrive":
                    entry = next(p for p in cmd if p.endswith(".py"))
                    self.assertTrue(Path(entry).is_file(), entry)

    def test_retries_keep_native_config_and_code_paths(self):
        native = self.root / "native"
        (native / "gen-benchmark").mkdir(parents=True)
        logs = self.root / "logs"
        logs.mkdir()
        first = runtime_for(native, "opendwm", logs)
        second = runtime_for(native, "opendwm", logs)
        self.assertEqual(first, second)
        self.assertTrue((first / "gen-benchmark/OpenDWM-main/infer_genbench_gt2.py").is_file())

    def test_overlay_shares_weights_but_uses_copied_adapters(self):
        native, patches, dst = [self.root / n for n in ("native", "override", "runtime")]
        for parent in (native, patches):
            (parent / "project").mkdir(parents=True)
        (native / "project/weights").write_bytes(b"external")
        (native / "project/main.py").write_text("old")
        (patches / "project/main.py").write_text("new")
        overlay(native, patches, dst)
        self.assertEqual((dst / "project/weights").resolve(), native / "project/weights")
        self.assertEqual((dst / "project/main.py").read_text(), "new")
        self.assertEqual((native / "project/main.py").read_text(), "old")
        self.assertFalse((dst / "project/main.py").is_symlink())

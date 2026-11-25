import json
import unittest


from driveaudit.generation import MODELS, native


class NativeDistributionTests(unittest.TestCase):
    def test_bundle_matches_manifest_and_preserves_official_catk_checksums(self):
        source = native.resource_root() / "models/generation"
        manifest = json.loads((native.BUNDLES / "native-sources.tar.json").read_text())
        self.assertEqual({p.name for p in source.iterdir() if p.is_dir()}, {"image", "trajectory"})
        self.assertEqual({p.name.lower().replace("-", "") for p in (source / "image").iterdir()
                          if p.is_dir()}, set(MODELS))
        self.assertEqual({p.name for p in (source / "trajectory").iterdir() if p.is_dir()}, {"CAT-K"})
        paths = {name: native.generation_source(name) for name in manifest["files"]}
        self.assertEqual(len(set(paths.values())), len(paths))
        found = {name: native.sha(path) for name, path in paths.items()}
        self.assertEqual(found, manifest["files"])
        for line in (source / "trajectory/CAT-K/official_source.sha256").read_text().splitlines():
            digest, _, name = line.partition("  ")
            if name.startswith("dependencies/catk-official/src/") and name.endswith(".py"):
                installed = name.replace("dependencies/", "catk/third_party/", 1)
                self.assertEqual(found[installed], digest)

    def test_native_opendwm_config_has_the_documented_image_condition(self):
        name = "gen-benchmark/OpenDWM-main/examples/ctsd_unimlvg_6views_video_generation.json"
        pipeline = json.loads(native.generation_source(name).read_text())["pipeline"]
        self.assertTrue(pipeline["common_config"]["condition_on_all_frames"])
        self.assertTrue(pipeline["common_config"]["explicit_view_modeling"])
        self.assertFalse(pipeline["common_config"]["cond_with_action"])
        self.assertNotIn("added_time_ids", pipeline["common_config"])
        self.assertEqual(pipeline["model"]["condition_image_adapter_config"]["in_channels"], 6)

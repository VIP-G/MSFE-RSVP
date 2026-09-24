from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RepositorySkeletonTest(unittest.TestCase):
    def test_final_config_and_metadata_exist(self):
        config_path = ROOT / "configs/rec/uyghur/msfe_rsvp.yml"
        self.assertTrue(config_path.is_file())
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(
            config["Architecture"]["Encoder"]["name"],
            "ViTLearnableFusionRSVP",
        )
        self.assertEqual(
            config["Architecture"]["Encoder"]["interaction_mode"],
            "intermediate_rsvp",
        )
        self.assertEqual(
            config["Architecture"]["Encoder"]["vertical_pooling"],
            "rsvp",
        )
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertTrue((ROOT / "requirements.txt").is_file())
        self.assertTrue((ROOT / "project.md").is_file())

    def test_only_final_runtime_directories_are_present(self):
        for directory in ("opendet", "analysis_results", "prediction_results"):
            self.assertFalse((ROOT / directory).exists())
        configs = sorted((ROOT / "configs/rec").rglob("*.yml"))
        self.assertEqual(configs, [ROOT / "configs/rec/uyghur/msfe_rsvp.yml"])

from pathlib import Path
import tempfile
import unittest

import yaml

from openrec.losses import build_loss
from openrec.metrics import build_metric
from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess import create_operators


ROOT = Path(__file__).resolve().parents[1]


class ComponentRegistryTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (ROOT / "configs/rec/uyghur/msfe_rsvp.yml").read_text()
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dict_path = Path(self.temp_dir.name) / "dict.txt"
        self.dict_path.write_text("a\nb\nc\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_final_components_build(self):
        global_config = self.config["Global"]
        global_config["character_dict_path"] = str(self.dict_path)
        post_config = dict(self.config["PostProcess"])
        post_config["character_dict_path"] = str(self.dict_path)
        post_process = build_post_process(post_config, global_config)
        architecture = self.config["Architecture"]
        architecture["Decoder"]["out_channels"] = (
            post_process.get_character_num()
        )
        model = build_model(architecture)
        self.assertEqual(model.decoder.out_channels, 5)
        build_loss(self.config["Loss"])
        build_metric(self.config["Metric"])
        operators = create_operators(
            self.config["Eval"]["dataset"]["transforms"], global_config
        )
        self.assertEqual(
            [type(operator).__name__ for operator in operators],
            ["DecodeImagePIL", "CTCLabelEncode", "RecTVResize", "KeepKeys"],
        )

    def test_removed_registry_name_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model({"Encoder": {"name": "RDBAFormer"}})


if __name__ == "__main__":
    unittest.main()

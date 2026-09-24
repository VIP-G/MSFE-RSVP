from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from openrec.modeling import build_model
from openrec.postprocess import build_post_process


ROOT = Path(__file__).resolve().parents[1]


class ModelForwardTest(unittest.TestCase):
    def test_cpu_forward_has_expected_sequence_length(self):
        config = yaml.safe_load(
            (ROOT / "configs/rec/uyghur/msfe_rsvp.yml").read_text()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            dictionary = Path(temp_dir) / "dict.txt"
            dictionary.write_text("a\nb\nc\n", encoding="utf-8")
            config["Global"]["character_dict_path"] = str(dictionary)
            config["PostProcess"]["character_dict_path"] = str(dictionary)
            post_process = build_post_process(
                config["PostProcess"], config["Global"]
            )
            architecture = deepcopy(config["Architecture"])
            architecture["Decoder"]["out_channels"] = (
                post_process.get_character_num()
            )
            model = build_model(architecture).eval()
            output = model(torch.randn(2, 3, 32, 128))
            self.assertEqual(tuple(output.shape), (2, 32, 5))
            self.assertTrue(torch.allclose(output.sum(dim=-1), torch.ones(2, 32), atol=1e-5))


if __name__ == "__main__":
    unittest.main()

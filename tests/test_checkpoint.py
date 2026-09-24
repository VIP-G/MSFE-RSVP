from copy import deepcopy
import os
from pathlib import Path
import unittest

import torch
import yaml

from openrec.modeling import build_model
from openrec.postprocess import build_post_process


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    os.environ.get("MSFE_RSVP_CHECKPOINT", "output/msfe_rsvp/best.pth")
)
DICTIONARY = Path(
    os.environ.get("MSFE_RSVP_DICTIONARY", "data/uyghur_dict.txt")
)


@unittest.skipUnless(
    CHECKPOINT.is_file() and DICTIONARY.is_file(),
    "the local final checkpoint and dictionary are not available",
)
class CheckpointCompatibilityTest(unittest.TestCase):
    def test_final_checkpoint_loads_strictly(self):
        config = yaml.safe_load(
            (ROOT / "configs/rec/uyghur/msfe_rsvp.yml").read_text()
        )
        config["Global"]["character_dict_path"] = str(DICTIONARY)
        config["PostProcess"]["character_dict_path"] = str(DICTIONARY)
        post_process = build_post_process(
            config["PostProcess"], config["Global"]
        )
        architecture = deepcopy(config["Architecture"])
        architecture["Decoder"]["out_channels"] = (
            post_process.get_character_num()
        )
        model = build_model(architecture)
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        incompatible = model.load_state_dict(
            checkpoint["state_dict"], strict=True
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()

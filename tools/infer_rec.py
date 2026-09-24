"""Run single-image inference with the final MSFE-RSVP recognizer."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess.resize import RecTVResize
from tools.engine.config import Config


def parse_args() -> argparse.Namespace:
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, nargs="+")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    return parser.parse_args()


def load_state_dict(checkpoint_path: str) -> dict:
    """Read a model state dict from a full checkpoint or a raw state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state_dict mapping")
    return checkpoint


def main() -> None:
    """Build the model, decode images and print recognized text."""
    args = parse_args()
    config = Config(args.config)
    global_config = config.cfg["Global"]
    post_process = build_post_process(
        config.cfg["PostProcess"], global_config
    )
    architecture = config.cfg["Architecture"]
    architecture["Decoder"]["out_channels"] = post_process.get_character_num()
    model = build_model(architecture)
    model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    device = torch.device(
        "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()

    resize = RecTVResize(
        image_shape=(32, 128),
        padding=False,
    )
    with torch.inference_mode():
        for image_path in args.image:
            data = {"image": Image.open(image_path).convert("RGB")}
            tensor = resize(data)["image"].unsqueeze(0).to(device)
            predictions = model(tensor)
            result = post_process(predictions)
            print(f"{image_path}\t{result[0][0]}")


if __name__ == "__main__":
    main()

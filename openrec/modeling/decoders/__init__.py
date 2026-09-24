"""Decoder registry for the final CTC recognizer."""

from .ctc_decoder import CTCDecoder


def build_decoder(config):
    """Build the final CTC decoder."""
    config = dict(config)
    module_name = config.pop("name")
    if module_name != "CTCDecoder":
        raise ValueError(f"Only CTCDecoder is supported, got {module_name!r}")
    return CTCDecoder(**config)

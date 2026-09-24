"""Encoder registry for the final MSFE-RSVP model."""

from importlib import import_module


def build_encoder(config):
    """Build the only encoder supported by this repository."""
    config = dict(config)
    module_name = config.pop("name")
    if module_name != "ViTLearnableFusionRSVP":
        raise ValueError(
            "Only ViTLearnableFusionRSVP is supported, "
            f"got {module_name!r}"
        )
    module = import_module(
        ".vit_learnable_fusion_rsvp_clean", package=__package__
    )
    return module.ViTLearnableFusionRSVP(**config)

"""Minimal CTC decoder used by the final MSFE-RSVP recognizer."""

import torch
from torch import nn


class CTCDecoder(nn.Module):
    """Project encoder sequences to CTC vocabulary logits.

    The final configuration uses an encoder output shaped ``[B, T, C]`` and
    returns ``[B, T, V]`` during training or normalized probabilities during
    evaluation. The ``fc`` name is intentionally preserved for checkpoint
    compatibility with the source OpenOCR implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 6625,
        mid_channels: int | None = None,
        return_feats: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if mid_channels is not None:
            raise ValueError(
                "The final MSFE-RSVP decoder uses a single CTC projection; "
                "mid_channels must be None."
            )
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        self.return_feats = bool(return_feats)

    def forward(
        self, x: torch.Tensor, data: object | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return CTC logits or evaluation probabilities."""
        predicts = self.fc(x)
        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            return torch.softmax(predicts, dim=2)
        return result

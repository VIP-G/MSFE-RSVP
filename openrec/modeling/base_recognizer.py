import torch
from torch import nn

from openrec.modeling.decoders import build_decoder
from openrec.modeling.encoders import build_encoder
from openrec.modeling.transforms import build_transform

__all__ = ['BaseRecognizer']


class BaseRecognizer(nn.Module):

    def __init__(self, config):
        """the module for OCR.

        args:
            config (dict): the super parameters for module.
        """
        super(BaseRecognizer, self).__init__()
        in_channels = config.get('in_channels', 3)
        self.use_wd = config.get('use_wd', True)
        # build transfrom,
        # for rec, transfrom can be TPS,None
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone
        if 'Encoder' not in config or config['Encoder'] is None:
            self.use_encoder = False
        else:
            self.use_encoder = True
            config['Encoder']['in_channels'] = in_channels
            self.encoder = build_encoder(config['Encoder'])
            in_channels = self.encoder.out_channels

        # build decoder
        if 'Decoder' not in config or config['Decoder'] is None:
            self.use_decoder = False
        else:
            self.use_decoder = True
            config['Decoder']['in_channels'] = in_channels
            self.decoder = build_decoder(config['Decoder'])

        self.freeze_initial_recognizer = bool(
            self.use_decoder
            and getattr(self.decoder, 'freeze_initial_branch', False)
        )
        if self.freeze_initial_recognizer:
            if not self.use_encoder:
                raise ValueError(
                    'freeze_initial_branch requires an Encoder to freeze'
                )
            self._freeze_initial_recognizer_parameters()
            self._keep_initial_recognizer_in_eval()

    def _initial_recognizer_modules(self) -> tuple[nn.Module, ...]:
        """Return modules that form the fixed Clean initial recognizer."""
        modules = []
        if self.use_transform:
            modules.append(self.transform)
        if self.use_encoder:
            modules.append(self.encoder)
        if self.use_decoder and self.decoder.svtr_encoder is not None:
            modules.append(self.decoder.svtr_encoder)
        if self.use_decoder and hasattr(self.decoder, 'fc'):
            modules.append(self.decoder.fc)
        return tuple(modules)

    def _freeze_initial_recognizer_parameters(self) -> None:
        """Disable gradients for every parameter in the Clean path."""
        for module in self._initial_recognizer_modules():
            module.requires_grad_(False)

    def _keep_initial_recognizer_in_eval(self) -> None:
        """Keep frozen BN, Dropout, and DropPath behavior deterministic."""
        for module in self._initial_recognizer_modules():
            module.eval()

    def train(self, mode: bool = True) -> 'BaseRecognizer':
        """Set mode while protecting the config-gated frozen Clean path."""
        super().train(mode)
        if self.freeze_initial_recognizer:
            self._keep_initial_recognizer_in_eval()
        return self

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.use_wd:
            if hasattr(self.encoder, 'no_weight_decay'):
                no_weight_decay = self.encoder.no_weight_decay()
            else:
                no_weight_decay = {}
            if hasattr(self.decoder, 'no_weight_decay'):
                no_weight_decay.update(self.decoder.no_weight_decay())
            return no_weight_decay
        else:
            return {}

    def forward(self, x, data=None):
        if self.use_transform:
            x = self.transform(x)
        if self.use_encoder:
            x = self.encoder(x)
        if self.use_decoder:
            x = self.decoder(x, data=data)
        return x

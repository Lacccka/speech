"""Model architectures for speech recognition."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from .config import ModelConfig


class CRNN(nn.Module):
    """A simple convolutional recurrent network for speech recognition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.cnn = nn.Sequential(
            nn.Conv2d(1, config.cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.cnn_channels, config.cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        freq_dim = math.ceil(config.in_features / 2)
        rnn_input_size = config.cnn_channels * freq_dim

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            dropout=config.dropout if config.rnn_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.rnn_hidden_size * 2, config.rnn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rnn_hidden_size, config.num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, time, features)``.
        lengths:
            Length of each sequence before padding.
        """

        batch, time, feat = x.shape
        if feat != self.config.in_features:
            raise ValueError(
                f"Expected feature dimension {self.config.in_features}, got {feat}"
            )

        x = x.transpose(1, 2).unsqueeze(1)  # (batch, 1, features, time)
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # lengths unchanged because pooling only affected frequency axis
        outputs, _ = self.rnn(x)
        logits = self.classifier(outputs)
        return logits, lengths


class SpeechTransformer(nn.Module):
    """Transformer encoder based acoustic model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.in_features, config.rnn_hidden_size),
            nn.LayerNorm(config.rnn_hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.rnn_hidden_size,
            nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_num_layers
        )
        self.classifier = nn.Linear(config.rnn_hidden_size, config.num_classes)

    @staticmethod
    def _generate_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idxs = torch.arange(max_len, device=lengths.device)
        return idxs.unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != self.config.in_features:
            raise ValueError(
                f"Expected feature dimension {self.config.in_features}, got {x.size(-1)}"
            )

        x = self.input_projection(x)
        mask = self._generate_padding_mask(lengths, x.size(1))
        encoded = self.encoder(x, src_key_padding_mask=mask)
        logits = self.classifier(encoded)
        return logits, lengths


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate a model based on the provided configuration."""

    architecture = config.architecture.lower()
    if architecture == "crnn":
        return CRNN(config)
    if architecture == "transformer":
        return SpeechTransformer(config)
    raise ValueError(f"Unsupported architecture '{config.architecture}'")


def load_pretrained_weights(
    model: nn.Module,
    weight_path: Path,
    *,
    map_location: Optional[str | torch.device] = None,
    strict: bool = True,
) -> None:
    """Load pretrained weights into the model."""

    if not weight_path.exists():
        raise FileNotFoundError(f"Pretrained weight file '{weight_path}' not found")
    checkpoint = torch.load(weight_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)


def model_from_config(
    config: ModelConfig,
    *,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """Create a model and optionally load pretrained weights."""

    model = build_model(config)
    if device is not None:
        model = model.to(device)

    if config.pretrained_path is not None:
        load_pretrained_weights(
            model,
            config.pretrained_path.expanduser(),
            map_location=device or "cpu",
            strict=strict,
        )
    return model


__all__ = [
    "CRNN",
    "SpeechTransformer",
    "build_model",
    "load_pretrained_weights",
    "model_from_config",
]


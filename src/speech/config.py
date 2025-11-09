"""Utilities for loading application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class BotConfig:
    """Configuration options for the Telegram bot."""

    token: str


@dataclass(slots=True)
class FeatureExtractionConfig:
    """Configuration for acoustic feature extraction."""

    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: Optional[int] = None
    hop_length: int = 160
    n_mels: int = 80
    n_mfcc: int = 13
    f_min: float = 0.0
    f_max: Optional[float] = None
    log_mel: bool = True
    power: float = 2.0
    center: bool = True


@dataclass(slots=True)
class ModelConfig:
    """Configuration describing the acoustic model architecture."""

    architecture: str = "crnn"
    num_classes: int = 29
    in_features: int = 80
    cnn_channels: int = 64
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 3
    dropout: float = 0.1
    transformer_num_layers: int = 6
    transformer_nhead: int = 8
    transformer_dim_feedforward: int = 1024
    pretrained_path: Optional[Path] = None


@dataclass(slots=True)
class TrainingConfig:
    """Configuration options for model training."""

    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 16
    log_interval: int = 50
    grad_clip: Optional[float] = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_best_only: bool = True
    metric_name: str = "loss"
    mixed_precision: bool = False
    mixed_precision_dtype: str = "float16"


@dataclass(slots=True)
class InferenceConfig:
    """Configuration options for inference/decoding."""

    beam_width: int = 10
    lm_weight: float = 0.3
    blank_index: int = 0
    max_symbol_per_step: int = 30
    streaming_chunk_seconds: float = 2.0


@dataclass(slots=True)
class TTSConfig:
    """Configuration options for the XTTS synthesizer."""

    language: str = "ru"
    gpt_conditioning_length: Optional[int] = None
    reference_duration: Optional[float] = None


@dataclass(slots=True)
class AppConfig:
    """Container for application configuration."""

    bot: BotConfig
    project_root: Path
    features: FeatureExtractionConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    tts: TTSConfig


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_env_int(name: str, default: int) -> int:
    value = _get_env(name)
    if value is None:
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = _get_env(name)
    if value is None:
        return default
    return float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = _get_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_env_path(name: str) -> Optional[Path]:
    value = _get_env(name)
    return Path(value) if value else None


def load_config(env_file: Optional[os.PathLike[str] | str] = None) -> AppConfig:
    """Load configuration from environment variables.

    Parameters
    ----------
    env_file:
        Optional path to a ``.env`` file. When provided, the file is loaded
        before reading environment variables.

    Returns
    -------
    AppConfig
        Populated configuration dataclass.

    Raises
    ------
    ValueError
        If required configuration values are missing.
    """

    if env_file is not None:
        load_dotenv(env_file)
    else:
        load_dotenv()

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN environment variable is not set")

    win_length_env = _get_env("FEATURE_WIN_LENGTH")
    f_max_env = _get_env("FEATURE_F_MAX")

    feature_config = FeatureExtractionConfig(
        sample_rate=_get_env_int("FEATURE_SAMPLE_RATE", 16_000),
        n_fft=_get_env_int("FEATURE_N_FFT", 400),
        win_length=int(win_length_env) if win_length_env else None,
        hop_length=_get_env_int("FEATURE_HOP_LENGTH", 160),
        n_mels=_get_env_int("FEATURE_N_MELS", 80),
        n_mfcc=_get_env_int("FEATURE_N_MFCC", 13),
        f_min=_get_env_float("FEATURE_F_MIN", 0.0),
        f_max=float(f_max_env) if f_max_env else None,
        log_mel=_get_env_bool("FEATURE_LOG_MEL", True),
        power=_get_env_float("FEATURE_POWER", 2.0),
        center=_get_env_bool("FEATURE_CENTER", True),
    )

    model_config = ModelConfig(
        architecture=_get_env("MODEL_ARCH", "crnn") or "crnn",
        num_classes=_get_env_int("MODEL_NUM_CLASSES", 29),
        in_features=_get_env_int("MODEL_IN_FEATURES", feature_config.n_mels),
        cnn_channels=_get_env_int("MODEL_CNN_CHANNELS", 64),
        rnn_hidden_size=_get_env_int("MODEL_RNN_HIDDEN", 256),
        rnn_num_layers=_get_env_int("MODEL_RNN_LAYERS", 3),
        dropout=_get_env_float("MODEL_DROPOUT", 0.1),
        transformer_num_layers=_get_env_int("MODEL_TRANSFORMER_LAYERS", 6),
        transformer_nhead=_get_env_int("MODEL_TRANSFORMER_NHEAD", 8),
        transformer_dim_feedforward=_get_env_int(
            "MODEL_TRANSFORMER_FF", 1024
        ),
        pretrained_path=_get_env_path("MODEL_PRETRAINED_PATH"),
    )

    training_config = TrainingConfig(
        epochs=_get_env_int("TRAIN_EPOCHS", 50),
        learning_rate=_get_env_float("TRAIN_LR", 1e-3),
        batch_size=_get_env_int("TRAIN_BATCH_SIZE", 16),
        log_interval=_get_env_int("TRAIN_LOG_INTERVAL", 50),
        grad_clip=(lambda v: float(v) if v else None)(_get_env("TRAIN_GRAD_CLIP")),
        checkpoint_dir=Path(_get_env("TRAIN_CHECKPOINT_DIR", "checkpoints")),
        save_best_only=_get_env_bool("TRAIN_SAVE_BEST_ONLY", True),
        metric_name=_get_env("TRAIN_METRIC_NAME", "loss") or "loss",
        mixed_precision=_get_env_bool("TRAIN_MIXED_PRECISION", False),
        mixed_precision_dtype=_get_env("TRAIN_MP_DTYPE", "float16") or "float16",
    )

    inference_config = InferenceConfig(
        beam_width=_get_env_int("INFER_BEAM_WIDTH", 10),
        lm_weight=_get_env_float("INFER_LM_WEIGHT", 0.3),
        blank_index=_get_env_int("INFER_BLANK_INDEX", 0),
        max_symbol_per_step=_get_env_int("INFER_MAX_SYMBOL_PER_STEP", 30),
        streaming_chunk_seconds=_get_env_float("INFER_STREAM_CHUNK_SEC", 2.0),
    )

    tts_config = TTSConfig(
        language=_get_env("TTS_LANGUAGE", "ru") or "ru",
        gpt_conditioning_length=(
            lambda value: int(value) if value else None
        )(_get_env("TTS_GPT_CONDITION_LENGTH")),
        reference_duration=(
            lambda value: float(value) if value else None
        )(_get_env("TTS_REFERENCE_DURATION")),
    )

    return AppConfig(
        bot=BotConfig(token=token),
        project_root=Path.cwd(),
        features=feature_config,
        model=model_config,
        training=training_config,
        inference=inference_config,
        tts=tts_config,
    )


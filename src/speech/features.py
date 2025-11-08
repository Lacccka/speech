"""Feature extraction utilities for speech processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch

try:
    import torchaudio  # type: ignore

    _HAS_TORCHAUDIO = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torchaudio = None  # type: ignore
    _HAS_TORCHAUDIO = False

from .config import FeatureExtractionConfig


_EPS = torch.finfo(torch.float32).eps


@dataclass(slots=True)
class CTCInput:
    """Container holding padded features and their lengths."""

    features: torch.Tensor
    lengths: torch.Tensor


def _mel_filterbank(
    config: FeatureExtractionConfig,
    device: Optional[torch.device],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a mel filterbank matrix compatible with ``torch.stft`` output."""

    if _HAS_TORCHAUDIO:  # pragma: no branch - prefer torchaudio implementation
        fb = torchaudio.functional.create_fb_matrix(  # type: ignore[attr-defined]
            n_freqs=config.n_fft // 2 + 1,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            dtype=dtype,
            device=device,
        )
        return fb

    # Manual implementation using librosa-style mel filter construction.
    # Reference: https://github.com/librosa/librosa
    def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(torch.tensor(1.0, dtype=freq.dtype, device=freq.device) + freq / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    n_freqs = config.n_fft // 2 + 1
    m_min = hz_to_mel(torch.tensor(config.f_min, dtype=dtype, device=device))
    m_max = hz_to_mel(
        torch.tensor(
            config.f_max if config.f_max is not None else config.sample_rate / 2,
            dtype=dtype,
            device=device,
        )
    )
    m_points = torch.linspace(m_min, m_max, config.n_mels + 2, device=device, dtype=dtype)
    f_points = mel_to_hz(m_points)

    fft_freqs = torch.linspace(
        0.0,
        config.sample_rate / 2,
        n_freqs,
        device=device,
        dtype=dtype,
    )

    filterbank = torch.zeros(config.n_mels, n_freqs, dtype=dtype, device=device)
    for m in range(1, config.n_mels + 1):
        f_m_left, f_m_center, f_m_right = f_points[m - 1 : m + 2]
        left_slope = (fft_freqs - f_m_left) / (f_m_center - f_m_left)
        right_slope = (f_m_right - fft_freqs) / (f_m_right - f_m_center)
        filterbank[m - 1] = torch.clamp(torch.min(left_slope, right_slope), min=0.0)
    return filterbank


def _apply_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=_EPS))


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    config: FeatureExtractionConfig,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute a mel spectrogram for a single waveform.

    Parameters
    ----------
    waveform:
        Tensor of shape ``(channels, samples)`` or ``(samples,)`` containing the
        audio waveform. Values should be in the range ``[-1, 1]``.
    config:
        Feature extraction configuration controlling STFT and mel parameters.
    normalize:
        If ``True`` the mel spectrogram is mean-variance normalised across the
        time dimension.
    """

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    dtype = waveform.dtype
    device = waveform.device

    if _HAS_TORCHAUDIO:
        transform = torchaudio.transforms.MelSpectrogram(  # type: ignore[attr-defined]
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
            center=config.center,
        )
        transform = transform.to(device=device, dtype=dtype)
        mel_spec = transform(waveform)
    else:
        stft = torch.stft(
            waveform,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=torch.hann_window(
                config.win_length or config.n_fft, device=device, dtype=dtype
            ),
            center=config.center,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitude = stft.abs() ** config.power
        fb = _mel_filterbank(config, device, dtype)
        mel_spec = torch.einsum("mf,cft->cmt", fb, magnitude)

    if config.log_mel:
        mel_spec = _apply_log(mel_spec)

    if normalize:
        mean = mel_spec.mean(dim=-1, keepdim=True)
        std = mel_spec.std(dim=-1, keepdim=True).clamp_min(_EPS)
        mel_spec = (mel_spec - mean) / std

    return mel_spec


def _create_dct(n_mfcc: int, n_mels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a DCT (type-II) matrix for MFCC computation."""

    n = torch.arange(n_mels, device=device, dtype=dtype)
    k = torch.arange(n_mfcc, device=device, dtype=dtype).unsqueeze(1)
    dct = torch.cos(torch.pi / n_mels * (n + 0.5) * k)
    dct[0] *= 1 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    dct *= torch.sqrt(torch.tensor(2.0 / n_mels, device=device, dtype=dtype))
    return dct


def compute_mfcc(
    waveform: torch.Tensor,
    config: FeatureExtractionConfig,
    *,
    lifter: Optional[int] = 22,
) -> torch.Tensor:
    """Compute MFCC features from a waveform."""

    mel_spec = compute_mel_spectrogram(waveform, config, normalize=False)
    log_mel = _apply_log(mel_spec) if not config.log_mel else mel_spec
    dct = _create_dct(config.n_mfcc, config.n_mels, log_mel.device, log_mel.dtype)
    mfcc = torch.matmul(dct, log_mel)

    if lifter is not None and lifter > 0:
        n = torch.arange(config.n_mfcc, device=mfcc.device, dtype=mfcc.dtype)
        lift = 1 + lifter / 2.0 * torch.sin(torch.pi * (n + 1) / lifter)
        mfcc = mfcc * lift.unsqueeze(-1)

    return mfcc


def prepare_ctc_inputs(
    features: Sequence[torch.Tensor],
    *,
    lengths: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    padding_value: float = 0.0,
    normalize: bool = True,
) -> CTCInput:
    """Pad a list of feature tensors for CTC models.

    Parameters
    ----------
    features:
        Iterable of tensors with shape ``(feature_dim, frames)``.
    lengths:
        Optional explicit lengths. If omitted the frame dimension of each tensor
        is used.
    device:
        Target device for the padded batch.
    padding_value:
        Value used to pad shorter sequences.
    normalize:
        If ``True`` each feature tensor is z-normalised prior to padding.
    """

    processed: List[torch.Tensor] = []
    frame_lengths: List[int] = []
    for idx, feat in enumerate(features):
        if feat.dim() != 2:
            raise ValueError(
                f"Feature tensor at index {idx} must have shape (feature_dim, frames), "
                f"but got {tuple(feat.shape)}"
            )
        if normalize:
            mean = feat.mean(dim=-1, keepdim=True)
            std = feat.std(dim=-1, keepdim=True).clamp_min(_EPS)
            feat = (feat - mean) / std
        processed.append(feat.transpose(0, 1))  # (frames, feature_dim)
        frame_lengths.append(feat.shape[-1])

    if lengths is not None:
        frame_lengths = list(lengths)

    padded = torch.nn.utils.rnn.pad_sequence(
        processed,
        batch_first=True,
        padding_value=padding_value,
    )

    if device is not None:
        padded = padded.to(device)

    length_tensor = torch.tensor(frame_lengths, dtype=torch.long, device=padded.device)
    return CTCInput(features=padded, lengths=length_tensor)


def batch_compute_mel_spectrogram(
    waveforms: Iterable[torch.Tensor],
    config: FeatureExtractionConfig,
    *,
    normalize: bool = True,
) -> List[torch.Tensor]:
    """Compute mel spectrograms for a batch of waveforms."""

    return [compute_mel_spectrogram(wf, config, normalize=normalize) for wf in waveforms]


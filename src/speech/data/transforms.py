"""Audio preprocessing pipeline utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:  # pragma: no cover - optional dependency
    from torchaudio import functional as F
    from torchaudio import transforms as T
except Exception as exc:  # pragma: no cover - ensure early feedback
    raise RuntimeError(
        "torchaudio is required for speech.data.transforms"
    ) from exc

_EPS = 1e-8


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Normalise the audio waveform to unit peak amplitude."""

    peak = waveform.abs().max().clamp(min=_EPS)
    return waveform / peak


def match_target_loudness(waveform: torch.Tensor, target_dbfs: float = -23.0) -> torch.Tensor:
    """Scale the waveform to the requested loudness."""

    rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=_EPS)
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    return waveform * gain


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Remove leading/trailing silence using energy-based VAD."""

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    frames = F.amplitude_to_DB(
        torch.stft(
            waveform,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            window=torch.hann_window(frame_length, device=waveform.device),
            return_complex=True,
        ).abs() ** 2,
        multiplier=10.0,
        amin=_EPS,
        db_multiplier=0.0,
    )
    energy = frames.mean(dim=-2).mean(dim=0)
    mask = energy > -threshold

    if not torch.any(mask):
        return waveform.squeeze(0)

    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    start = indices[0].item() * hop_length
    end = min((indices[-1].item() + 1) * hop_length, waveform.size(-1))
    return waveform[..., start:end].squeeze(0)


@dataclass
class SpeechTransformPipeline:
    """Composes common audio preprocessing steps."""

    sample_rate: int
    target_dbfs: float = -23.0
    spectrogram_dir: Optional[Path] = None
    n_mels: int = 80

    def __post_init__(self) -> None:
        if self.spectrogram_dir is not None:
            self.spectrogram_dir = Path(self.spectrogram_dir)
            self.spectrogram_dir.mkdir(parents=True, exist_ok=True)
        self._mel_transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=self.n_mels)

    def process(self, waveform: torch.Tensor, utterance_id: Optional[str] = None) -> torch.Tensor:
        """Apply the pipeline to a waveform."""

        waveform = normalize_waveform(waveform)
        waveform = match_target_loudness(waveform, self.target_dbfs)
        waveform = trim_silence(waveform, sample_rate=self.sample_rate)

        if self.spectrogram_dir is not None and utterance_id is not None:
            save_path = self.spectrogram_dir / f"{utterance_id}.pt"
            save_spectrogram(waveform, self.sample_rate, save_path, self._mel_transform)

        return waveform


def save_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    output_path: Path,
    mel_transform: Optional[T.MelSpectrogram] = None,
    power: float = 1.0,
) -> Path:
    """Compute and persist a log-mel spectrogram to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mel_transform is None:
        mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=80)

    mel = mel_transform(waveform)
    mel = mel.clamp(min=_EPS).pow(power)
    log_mel = torch.log(mel)
    torch.save(log_mel.cpu(), output_path)
    return output_path

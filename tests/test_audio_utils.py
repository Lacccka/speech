from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

pydub = pytest.importorskip("pydub")
generators = pytest.importorskip("pydub.generators")
pytest.importorskip("pyloudnorm")

AudioSegment = pydub.AudioSegment
Sine = generators.Sine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_utils

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which(AudioSegment.converter),
    reason="ffmpeg binary is required for export",
)


def _build_tone(duration_ms: int, gain_db: float) -> AudioSegment:
    return Sine(440).to_audio_segment(duration=duration_ms).apply_gain(gain_db)


def test_apply_loudness_normalization_reaches_target():
    segment = _build_tone(1000, -30.0).set_channels(1).set_frame_rate(24_000)

    normalized = audio_utils.apply_loudness_normalization(segment, target_lufs=-16.0)
    loudness = audio_utils.measure_loudness_lufs(normalized)

    assert loudness == pytest.approx(-16.0, abs=1.0)


@requires_ffmpeg
def test_wav_to_ogg_opus_applies_loudness(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.ogg"

    raw = _build_tone(1500, -25.0).set_channels(1).set_frame_rate(24_000)
    raw.export(input_path, format="wav")

    audio_utils.wav_to_ogg_opus(str(input_path), str(output_path), target_lufs=-18.0)

    processed = AudioSegment.from_file(output_path, format="ogg")
    loudness = audio_utils.measure_loudness_lufs(processed)

    assert loudness == pytest.approx(-18.0, abs=1.5)

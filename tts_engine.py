# tts_engine.py
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List

from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_tts = None


def _split_text_with_overlap(
    text: str,
    *,
    min_length: int = 200,
    max_length: int = 250,
    overlap: int = 50,
) -> List[str]:
    """Split ``text`` into chunks between ``min_length`` and ``max_length`` characters.

    The chunks are created with a fixed overlap to keep transitions smooth. Whenever
    possible the splitter prefers whitespace boundaries so that sentences are not
    torn apart.
    """

    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        tentative_end = min(start + max_length, text_length)
        end = tentative_end

        if end < text_length:
            # Try to split at whitespace between ``min_length`` and ``max_length``.
            split_window_start = min(start + min_length, end)
            split_window_end = end
            split_at = text.rfind(" ", split_window_start, split_window_end)

            if split_at == -1 or split_at <= start:
                # Fall back to the next whitespace after ``max_length`` characters.
                split_at = text.find(" ", end)
                if split_at != -1:
                    end = split_at
                else:
                    end = text_length
            else:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, 0)
        if start >= text_length:
            break

    return chunks


def _concatenate_audio(segments: Iterable[AudioSegment]) -> AudioSegment:
    """Concatenate the provided audio segments preserving their order."""

    iterator = iter(segments)
    first = next(iterator, None)
    if first is None:
        raise ValueError("No audio segments provided for concatenation")

    result = first
    for segment in iterator:
        result += segment
    return result


def get_tts():
    global _tts
    if _tts is None:
        # загрузится 1 раз на всё время работы бота
        _tts = TTS(MODEL_NAME)
    return _tts


def synthesize_ru(text: str, profile_wav: str, out_wav: str):
    tts = get_tts()
    if len(text) <= 250:
        tts.tts_to_file(
            text=text,
            file_path=out_wav,
            speaker_wav=profile_wav,
            language="ru",
            enable_text_splitting=False,
        )
        audio = AudioSegment.from_file(out_wav)
        audio = normalize(audio)
        audio.export(out_wav, format="wav")
        return

    text_chunks = _split_text_with_overlap(text)
    audio_segments: List[AudioSegment] = []

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, chunk in enumerate(text_chunks):
            chunk_path = tmp_path / f"chunk_{idx}.wav"
            tts.tts_to_file(
                text=chunk,
                file_path=str(chunk_path),
                speaker_wav=profile_wav,
                language="ru",
                enable_text_splitting=True,
            )
            audio_segments.append(AudioSegment.from_file(chunk_path))

    combined = _concatenate_audio(audio_segments)
    combined = normalize(combined)
    combined.export(out_wav, format="wav")

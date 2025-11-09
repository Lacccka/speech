# tts_engine.py
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional

from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_tts: Optional[TTS] = None
_synthesizer: Any | None = None


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


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        # загрузится 1 раз на всё время работы бота
        _tts = TTS(MODEL_NAME)
    return _tts


def _build_synthesizer(tts: TTS) -> Any:
    synthesizer = getattr(tts, "synthesizer", None)
    if synthesizer is not None:
        return synthesizer

    load_model = getattr(tts, "load_model", None)
    init_synthesizer = getattr(tts, "init_synthesizer", None)
    if callable(load_model) and callable(init_synthesizer):
        model_name = getattr(tts, "model_name", MODEL_NAME)
        try:
            components = load_model(model_name)
        except TypeError:
            components = load_model()

        try:
            if isinstance(components, dict):
                synthesizer = init_synthesizer(**components)
            elif isinstance(components, (list, tuple)):
                synthesizer = init_synthesizer(*components)
            else:
                synthesizer = init_synthesizer(components)
        except Exception:
            synthesizer = getattr(tts, "synthesizer", None)

    if synthesizer is None and hasattr(tts, "load_tts_model_by_name"):
        tts.load_tts_model_by_name(getattr(tts, "model_name", MODEL_NAME))
        synthesizer = getattr(tts, "synthesizer", None)

    if synthesizer is None:
        raise RuntimeError("Не удалось инициализировать синтезатор XTTS")

    return synthesizer


def _get_synthesizer() -> Any:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = _build_synthesizer(get_tts())
    return _synthesizer


def _synthesize_to_file(
    text: str,
    profile_wav: str,
    out_wav: str,
    *,
    language: Optional[str],
    gpt_cond_len: Optional[int],
    reference_duration: Optional[float],
    extra_options: Dict[str, Any],
) -> AudioSegment:
    synthesizer = _get_synthesizer()
    synthesis_args: Dict[str, Any] = {
        "text": text,
        "speaker_wav": profile_wav,
        "split_sentences": False,
    }

    if language:
        synthesis_args["language_name"] = language
    if gpt_cond_len is not None:
        synthesis_args["gpt_cond_len"] = gpt_cond_len
    if reference_duration is not None:
        synthesis_args["reference_duration"] = reference_duration

    synthesis_args.update(extra_options)

    wav = synthesizer.tts(**synthesis_args)
    synthesizer.save_wav(wav=wav, path=out_wav)

    audio = AudioSegment.from_file(out_wav)
    audio = normalize(audio)
    audio.export(out_wav, format="wav")
    return audio


def synthesize_ru(
    text: str,
    profile_wav: str,
    out_wav: str,
    *,
    language: Optional[str] = "ru",
    gpt_cond_len: Optional[int] = None,
    reference_duration: Optional[float] = None,
    **extra_options: Any,
) -> None:
    if len(text) <= 250:
        _synthesize_to_file(
            text,
            profile_wav,
            out_wav,
            language=language,
            gpt_cond_len=gpt_cond_len,
            reference_duration=reference_duration,
            extra_options=extra_options,
        )
        return

    text_chunks = _split_text_with_overlap(text)
    audio_segments: List[AudioSegment] = []

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, chunk in enumerate(text_chunks):
            chunk_path = tmp_path / f"chunk_{idx}.wav"
            audio_segments.append(
                _synthesize_to_file(
                    chunk,
                    profile_wav,
                    str(chunk_path),
                    language=language,
                    gpt_cond_len=gpt_cond_len,
                    reference_duration=reference_duration,
                    extra_options=extra_options,
                )
            )

    combined = _concatenate_audio(audio_segments)
    combined = normalize(combined)
    combined.export(out_wav, format="wav")

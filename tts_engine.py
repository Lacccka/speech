# tts_engine.py
from __future__ import annotations

import math
import inspect
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Sequence

from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from speech.config import TTSConfig, load_config

_CONFIG_LOADER: Callable[[], TTSConfig] = lambda: load_config().tts
_cached_tts_config: Optional[TTSConfig] = None
_tts: Optional[TTS] = None
_synthesizer: Any | None = None

BackendFactory = Callable[[TTSConfig], Any]
_BACKEND_FACTORIES: Dict[str, BackendFactory] = {}


def register_tts_backend(name: str, factory: BackendFactory) -> None:
    """Register a callable ``factory`` that builds a TTS engine for ``name``."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Backend name must be a non-empty string")
    _BACKEND_FACTORIES[normalized] = factory


def _get_tts_config() -> TTSConfig:
    global _cached_tts_config
    if _cached_tts_config is None:
        _cached_tts_config = _CONFIG_LOADER()
    return _cached_tts_config


def _coqui_backend_factory(config: TTSConfig) -> TTS:
    kwargs: Dict[str, Any] = {}
    if config.model_path is not None:
        kwargs["model_path"] = str(config.model_path)
    return TTS(config.model_name, **kwargs)


register_tts_backend("coqui", _coqui_backend_factory)


def _invoke_factory(factory: Callable[..., Any], config: TTSConfig) -> Any:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(config)

    params = signature.parameters
    if not params:
        return factory()

    positional_args: List[Any] = []
    keyword_args: Dict[str, Any] = {}

    def _as_path(value: Optional[Path]) -> Optional[str]:
        return str(value) if value is not None else None

    for name, param in params.items():
        if name == "self":
            continue

        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # Assume the factory can figure out advanced signatures itself.
            continue

        wants_argument = param.default is inspect._empty

        if name in {"config", "cfg", "tts_config"} or param.annotation is TTSConfig:
            target = config
        elif name in {"model_name", "model", "model_id"}:
            target = config.model_name
        elif name in {"model_path", "checkpoint_path"}:
            target = _as_path(config.model_path)
            if target is None and wants_argument:
                raise ValueError(
                    "TTS backend requires a model path but none was configured"
                )
        else:
            if wants_argument:
                raise TypeError(
                    f"Cannot determine value for required parameter '{name}' when "
                    f"instantiating backend '{factory!r}'"
                )
            continue

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_args.append(target)
        else:
            keyword_args[name] = target

    return factory(*positional_args, **keyword_args)


def _resolve_backend_factory(backend_name: str) -> BackendFactory:
    normalized = backend_name.strip().lower()
    if not normalized:
        raise ValueError("TTS backend name must not be empty")

    if normalized in _BACKEND_FACTORIES:
        return _BACKEND_FACTORIES[normalized]

    module_name, separator, attr_name = backend_name.rpartition(".")
    if not separator:
        raise ValueError(
            f"Unknown TTS backend '{backend_name}'. Provide a registered name or"
            " a dotted import path"
        )

    module = import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(
            f"Backend '{backend_name}' resolved to non-callable object {factory!r}"
        )

    return lambda cfg: _invoke_factory(factory, cfg)


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


def trim_silence(
    segment: AudioSegment,
    *,
    silence_thresh: int = -50,
    chunk_size: int = 10,
) -> AudioSegment:
    """Remove leading and trailing silence from ``segment``.

    Parameters
    ----------
    segment:
        The audio segment to trim.
    silence_thresh:
        Amplitude threshold in dBFS below which audio is treated as silence.
    chunk_size:
        Resolution of the silence detector in milliseconds.
    """

    if len(segment) == 0:
        return segment

    nonsilent_ranges = detect_nonsilent(
        segment,
        min_silence_len=chunk_size,
        silence_thresh=silence_thresh,
    )

    if not nonsilent_ranges:
        return segment

    start, end = nonsilent_ranges[0][0], nonsilent_ranges[-1][1]
    return segment[start:end]


def apply_deesser(
    segment: AudioSegment,
    *,
    frequency: int = 6000,
    reduction_db: float = 12.0,
) -> AudioSegment:
    """Apply a light-weight de-esser approximation to ``segment``.

    The implementation relies on inverting a high-pass filtered copy of the
    signal to attenuate overly sharp sibilants.
    """

    if len(segment) == 0:
        return segment

    high_band = segment.high_pass_filter(frequency)
    inverted = high_band.invert_phase().apply_gain(reduction_db)
    return segment.overlay(inverted)


def normalize_to_target(segment: AudioSegment, *, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize ``segment`` so that its loudness matches ``target_dbfs``."""

    if len(segment) == 0:
        return segment

    current_dbfs = segment.dBFS
    if math.isinf(current_dbfs):
        return segment

    gain = target_dbfs - current_dbfs
    return segment.apply_gain(gain)


def assemble_segments(
    segments: Sequence[AudioSegment],
    *,
    crossfade_ms: int = 75,
) -> AudioSegment:
    """Join audio ``segments`` using crossfades to ensure smooth transitions."""

    if not segments:
        raise ValueError("No audio segments provided for assembly")

    combined = segments[0]
    if len(segments) == 1:
        return combined

    for segment in segments[1:]:
        effective_crossfade = max(
            0,
            min(crossfade_ms, len(combined), len(segment)),
        )
        combined = combined.append(segment, crossfade=effective_crossfade)
    return combined


def get_tts() -> Any:
    global _tts
    if _tts is None:
        config = _get_tts_config()
        factory = _resolve_backend_factory(config.backend)
        _tts = factory(config)
    return _tts


def _build_synthesizer(tts: Any) -> Any:
    synthesizer = getattr(tts, "synthesizer", None)
    if synthesizer is not None:
        return synthesizer

    load_model = getattr(tts, "load_model", None)
    init_synthesizer = getattr(tts, "init_synthesizer", None)
    if callable(load_model) and callable(init_synthesizer):
        model_name = getattr(tts, "model_name", _get_tts_config().model_name)
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
        tts.load_tts_model_by_name(
            getattr(tts, "model_name", _get_tts_config().model_name)
        )
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
    speaker_wavs: Sequence[str],
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
        "speaker_wavs": list(speaker_wavs),
        "split_sentences": False,
    }

    speaker_identifier: Optional[str] = None
    if speaker_wavs:
        # Некоторые версии XTTS по-прежнему ожидают ``speaker_wav`` с одиночным путём.
        synthesis_args.setdefault("speaker_wav", speaker_wavs[0])

        first_reference = Path(speaker_wavs[0])
        speaker_identifier = first_reference.stem or first_reference.name or "speaker"

    if language:
        synthesis_args["language_name"] = language
    if gpt_cond_len is not None:
        synthesis_args["gpt_cond_len"] = gpt_cond_len
    if reference_duration is not None:
        synthesis_args["reference_duration"] = reference_duration

    synthesis_args.update(extra_options)

    if speaker_identifier:
        for key in ("speaker", "speaker_id"):
            if key not in synthesis_args:
                synthesis_args[key] = speaker_identifier
                continue

            current_value = synthesis_args[key]
            if current_value is None:
                synthesis_args[key] = speaker_identifier
            elif isinstance(current_value, str) and not current_value.strip():
                synthesis_args[key] = speaker_identifier

    wav = synthesizer.tts(**synthesis_args)
    synthesizer.save_wav(wav=wav, path=out_wav)

    return AudioSegment.from_file(out_wav)


def synthesize_ru(
    text: str,
    speaker_wavs: Sequence[str] | str,
    out_wav: str,
    *,
    language: Optional[str] = "ru",
    gpt_cond_len: Optional[int] = None,
    reference_duration: Optional[float] = None,
    crossfade_ms: int = 75,
    target_dbfs: Optional[float] = -20.0,
    silence_threshold: int = -50,
    silence_chunk_len: int = 10,
    deesser_frequency: int = 6000,
    deesser_reduction_db: float = 12.0,
    enable_post_processing: bool = True,
    **extra_options: Any,
) -> None:
    speaker_list: List[str]
    if isinstance(speaker_wavs, str):
        speaker_list = [speaker_wavs]
    else:
        speaker_list = list(speaker_wavs)

    if not speaker_list:
        raise ValueError("speaker_wavs must contain at least one reference")

    def _process_segment(segment: AudioSegment) -> AudioSegment:
        if not enable_post_processing:
            return segment

        trimmed = trim_silence(
            segment,
            silence_thresh=silence_threshold,
            chunk_size=silence_chunk_len,
        )
        deessed = apply_deesser(
            trimmed,
            frequency=deesser_frequency,
            reduction_db=deesser_reduction_db,
        )
        if target_dbfs is None:
            return deessed
        return normalize_to_target(deessed, target_dbfs=target_dbfs)

    if len(text) <= 250:
        segment = _synthesize_to_file(
            text,
            speaker_list,
            out_wav,
            language=language,
            gpt_cond_len=gpt_cond_len,
            reference_duration=reference_duration,
            extra_options=extra_options,
        )
        processed = _process_segment(segment)
        processed.export(out_wav, format="wav")
        return

    text_chunks = _split_text_with_overlap(text)
    processed_segments: List[AudioSegment] = []

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, chunk in enumerate(text_chunks):
            chunk_path = tmp_path / f"chunk_{idx}.wav"
            raw_segment = _synthesize_to_file(
                chunk,
                speaker_list,
                str(chunk_path),
                language=language,
                gpt_cond_len=gpt_cond_len,
                reference_duration=reference_duration,
                extra_options=extra_options,
            )
            processed_segments.append(_process_segment(raw_segment))

    combined = assemble_segments(processed_segments, crossfade_ms=crossfade_ms)
    if enable_post_processing and target_dbfs is not None:
        combined = normalize_to_target(combined, target_dbfs=target_dbfs)
    combined.export(out_wav, format="wav")

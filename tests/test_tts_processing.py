from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pydub = pytest.importorskip("pydub")
generators = pytest.importorskip("pydub.generators")

AudioSegment = pydub.AudioSegment
Sine = generators.Sine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types

fake_tts = types.ModuleType("TTS")
fake_tts_api = types.ModuleType("TTS.api")


class _DummyTTS:  # pragma: no cover - stub for imports only
    pass


fake_tts_api.TTS = _DummyTTS
fake_tts.api = fake_tts_api
sys.modules.setdefault("TTS", fake_tts)
sys.modules.setdefault("TTS.api", fake_tts_api)

fake_aiogram = types.ModuleType("aiogram")


class _DummyBot:
    def __init__(self, token: str) -> None:  # pragma: no cover - stub
        self.token = token


class _DummyDispatcher:  # pragma: no cover - stub
    def __init__(self) -> None:
        self.handlers: list[object] = []

    def message(self, *args: object, **kwargs: object):
        def decorator(func):
            self.handlers.append(func)
            return func

        return decorator


class _DummyFilter:  # pragma: no cover - stub
    def __getattr__(self, name: str) -> "_DummyFilter":
        return self

    def __call__(self, *args: object, **kwargs: object) -> "_DummyFilter":
        return self

    def __eq__(self, other: object) -> "_DummyFilter":
        return self


fake_aiogram.Bot = _DummyBot
fake_aiogram.Dispatcher = _DummyDispatcher
fake_aiogram.F = _DummyFilter()

fake_aiogram_exceptions = types.ModuleType("aiogram.exceptions")


class _DummyTelegramBadRequest(Exception):
    pass


fake_aiogram_exceptions.TelegramBadRequest = _DummyTelegramBadRequest

fake_aiogram_filters = types.ModuleType("aiogram.filters")


class _DummyCommandStart:
    def __call__(self, *args: object, **kwargs: object) -> "_DummyCommandStart":
        return self


fake_aiogram_filters.CommandStart = _DummyCommandStart

fake_aiogram_types = types.ModuleType("aiogram.types")


class _DummyMessage:
    async def answer(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - stub
        return None


class _DummyFSInputFile:
    def __init__(self, path: str) -> None:
        self.path = path


fake_aiogram_types.Message = _DummyMessage
fake_aiogram_types.FSInputFile = _DummyFSInputFile

sys.modules.setdefault("aiogram", fake_aiogram)
sys.modules.setdefault("aiogram.exceptions", fake_aiogram_exceptions)
sys.modules.setdefault("aiogram.filters", fake_aiogram_filters)
sys.modules.setdefault("aiogram.types", fake_aiogram_types)

fake_db = types.ModuleType("db")


def _noop(*args: object, **kwargs: object) -> None:  # pragma: no cover - stub
    return None


fake_db.init_db = _noop
fake_db.get_user = _noop
fake_db.set_state = _noop
fake_db.set_profile = _noop
fake_db.set_speaker_references = _noop
fake_db.start_user_session = _noop
fake_db.add_sample = _noop
fake_db.get_latest_samples = lambda *args, **kwargs: []
fake_db.delete_user_samples = _noop
fake_db.set_pending_tts_text = _noop
fake_db.get_pending_tts_text = lambda *args, **kwargs: None
fake_db.get_speaker_references = lambda *args, **kwargs: []

sys.modules.setdefault("db", fake_db)

fake_keyboards = types.ModuleType("keyboards")
fake_keyboards.main_kb = lambda: None
fake_keyboards.generation_mode_kb = lambda: None
sys.modules.setdefault("keyboards", fake_keyboards)

fake_audio_utils = types.ModuleType("audio_utils")
fake_audio_utils.user_voice_dir = lambda user_id: Path("/tmp") / str(user_id)
fake_audio_utils.convert_to_wav = _noop
fake_audio_utils.clear_user_voices = _noop
fake_audio_utils.user_output_path = lambda user_id: Path("/tmp") / f"{user_id}.wav"
fake_audio_utils.user_output_ogg_path = lambda user_id: Path("/tmp") / f"{user_id}.ogg"
fake_audio_utils.wav_to_ogg_opus = _noop
sys.modules.setdefault("audio_utils", fake_audio_utils)

fake_training = types.ModuleType("training")
fake_training.continue_training = lambda *args, **kwargs: ["ref.wav"]
fake_training.train_new_voice = lambda *args, **kwargs: ["ref.wav"]
sys.modules.setdefault("training", fake_training)

fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = _noop
sys.modules.setdefault("dotenv", fake_dotenv)

import tts_engine
from speech.config import TTSConfig


def _load_main(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOT_TOKEN", "token")
    if "main" in sys.modules:
        return importlib.reload(sys.modules["main"])
    return importlib.import_module("main")


def test_split_text_prefers_sentence_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    main_module = _load_main(monkeypatch)
    sentences = [
        "This is the first sentence that includes several descriptive clauses to create a comfortable length for testing purposes.",
        "Meanwhile, the second sentence elaborates on the scenario with enough wording to approach our target boundary.",
        "Finally, the third sentence wraps things up by providing a short conclusion that should stay grouped with the second when limits are respected.",
    ]
    text = " ".join(sentences)

    chunks = main_module.split_text_for_tts(text, max_chars=240)

    assert chunks == [f"{sentences[0]} {sentences[1]}", sentences[2]]
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert len(chunks[0]) >= 200


def test_split_text_handles_abbreviations(monkeypatch: pytest.MonkeyPatch) -> None:
    main_module = _load_main(monkeypatch)
    text = (
        "Dr. Smith met with Sen. Johnson at 5 p.m. to discuss the U.S. launch schedule. "
        "The meeting was productive and ended on time."
    )

    chunks = main_module.split_text_for_tts(text, max_chars=240)

    assert chunks == [text]


def test_split_text_uses_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    main_module = _load_main(monkeypatch)
    text = " ".join(["segment"] * 80)

    chunks = main_module.split_text_for_tts(text, max_chars=240)

    assert len(chunks) > 1
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert all(chunk == chunk.strip() for chunk in chunks)

    original_tokens = text.split()
    reconstructed_tokens: list[str] = []
    for chunk in chunks:
        reconstructed_tokens.extend(chunk.split())

    assert reconstructed_tokens == original_tokens


def _dummy_backend_factory(tts_config: TTSConfig) -> object:
    class _DummyBackend:
        def __init__(self, name: str, model_path: str | None) -> None:
            self.name = name
            self.model_path = model_path

    return _DummyBackend(
        tts_config.model_name,
        str(tts_config.model_path) if tts_config.model_path is not None else None,
    )


def _build_segment(
    duration_ms: int,
    *,
    silence_before: int,
    silence_after: int,
    gain_db: float,
) -> AudioSegment:
    tone = Sine(440).to_audio_segment(duration=duration_ms).apply_gain(gain_db)
    return AudioSegment.silent(duration=silence_before) + tone + AudioSegment.silent(
        duration=silence_after
    )


def test_synthesize_to_file_fills_missing_speaker_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_calls: list[dict[str, object]] = []

    class _DummySynthesizer:
        def __init__(self) -> None:
            self._wav_token = object()

        def tts(self, **kwargs: object) -> object:
            captured_calls.append(dict(kwargs))
            return self._wav_token

        def save_wav(self, wav: object, path: str) -> None:  # pragma: no cover - helper stub
            assert wav is self._wav_token
            AudioSegment.silent(duration=50).export(path, format="wav")

    synthesizer = _DummySynthesizer()
    monkeypatch.setattr(tts_engine, "_synthesizer", synthesizer)

    speaker_reference = tmp_path / "user_profile.wav"
    result = tts_engine._synthesize_to_file(
        "hello",
        [str(speaker_reference)],
        str(tmp_path / "out.wav"),
        language=None,
        gpt_cond_len=None,
        reference_duration=None,
        extra_options={},
    )

    assert isinstance(result, AudioSegment)
    assert captured_calls, "Synthesizer.tts was not invoked"

    call_kwargs = captured_calls[0]
    expected_identifier = speaker_reference.stem

    speaker_value = call_kwargs.get("speaker")
    speaker_id_value = call_kwargs.get("speaker_id")

    assert speaker_value == expected_identifier or speaker_id_value == expected_identifier
    if speaker_value is not None:
        assert str(speaker_value).strip()
    if speaker_id_value is not None:
        assert str(speaker_id_value).strip()


def test_synthesize_ru_applies_processing_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_segments = [
        _build_segment(200, silence_before=40, silence_after=40, gain_db=-6.0),
        _build_segment(200, silence_before=60, silence_after=60, gain_db=-9.0),
    ]

    chunks = ["chunk-1", "chunk-2"]

    def fake_split(_: str) -> list[str]:
        return chunks

    synth_calls = {"index": 0}

    def fake_synthesize_to_file(*args, **kwargs) -> AudioSegment:  # type: ignore[override]
        idx = synth_calls["index"]
        synth_calls["index"] += 1
        segment = raw_segments[idx]
        out_path = Path(args[2])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        segment.export(out_path, format="wav")
        return segment

    monkeypatch.setattr(tts_engine, "_split_text_with_overlap", fake_split)
    monkeypatch.setattr(tts_engine, "_synthesize_to_file", fake_synthesize_to_file)

    output_path = tmp_path / "combined.wav"

    tts_engine.synthesize_ru(
        "x" * 400,
        ["profile.wav"],
        str(output_path),
        crossfade_ms=80,
        target_dbfs=-14.0,
        silence_threshold=-40,
        silence_chunk_len=20,
    )

    combined = AudioSegment.from_file(output_path)

    processed_segments = [
        tts_engine.normalize_to_target(
            tts_engine.apply_deesser(
                tts_engine.trim_silence(
                    segment,
                    silence_thresh=-40,
                    chunk_size=20,
                ),
                frequency=6000,
                reduction_db=12.0,
            ),
            target_dbfs=-14.0,
        )
        for segment in raw_segments
    ]

    expected_length = sum(len(segment) for segment in processed_segments) - 80
    assert len(combined) == pytest.approx(expected_length, abs=4)
    assert len(combined) < sum(len(segment) for segment in raw_segments)
    assert combined.dBFS == pytest.approx(-14.0, abs=1.5)

    direct_concat = processed_segments[0] + processed_segments[1]
    assert len(combined) == len(direct_concat) - 80


def test_assemble_segments_crossfade_leniency():
    seg1 = Sine(440).to_audio_segment(duration=300).apply_gain(-6)
    seg2 = Sine(880).to_audio_segment(duration=300).apply_gain(-12)

    combined = tts_engine.assemble_segments([seg1, seg2], crossfade_ms=100)

    assert len(combined) == len(seg1) + len(seg2) - 100
    assert combined.dBFS < seg1.dBFS + 1


def test_main_synthesize_with_splitting_crossfade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOT_TOKEN", "token")

    import importlib
    import sys

    if "main" in sys.modules:
        main_module = sys.modules["main"]
        main_module = importlib.reload(main_module)
    else:
        main_module = importlib.import_module("main")

    segments = [
        Sine(330).to_audio_segment(duration=220).apply_gain(-12),
        Sine(660).to_audio_segment(duration=220).apply_gain(-12),
    ]

    monkeypatch.setattr(main_module, "split_text_for_tts", lambda text, max_chars=180: ["a", "b"])

    call_index = {"value": 0}

    def fake_synthesize_ru(
        text: str, speaker_wavs: object, out_path: str, **kwargs: object
    ) -> None:
        segment = segments[call_index["value"]]
        call_index["value"] += 1
        segment.export(out_path, format="wav")

    monkeypatch.setattr(main_module, "synthesize_ru", fake_synthesize_ru)
    monkeypatch.setattr(main_module.config.tts, "chunk_crossfade_ms", 60)
    monkeypatch.setattr(main_module.config.tts, "chunk_target_dbfs", -16.0)

    output_path = tmp_path / "result.wav"
    main_module.synthesize_with_splitting(
        "hello world" * 30,
        ["profile.wav"],
        output_path,
        mode="quality",
    )

    combined = AudioSegment.from_file(output_path)

    assert call_index["value"] == 2
    expected_length = sum(len(segment) for segment in segments) - 60
    assert len(combined) == pytest.approx(expected_length, abs=3)
    assert combined.dBFS == pytest.approx(-16.0, abs=1.5)


def test_main_synthesize_with_splitting_fast_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BOT_TOKEN", "token")

    import importlib
    import sys

    if "main" in sys.modules:
        main_module = sys.modules["main"]
        main_module = importlib.reload(main_module)
    else:
        main_module = importlib.import_module("main")

    segments = [
        Sine(220).to_audio_segment(duration=200).apply_gain(-10),
        Sine(440).to_audio_segment(duration=200).apply_gain(-12),
    ]

    monkeypatch.setattr(main_module, "split_text_for_tts", lambda text, max_chars=180: ["a", "b"])

    call_index = {"value": 0}

    def fake_synthesize_ru(
        text: str, speaker_wavs: object, out_path: str, **kwargs: object
    ) -> None:
        segment = segments[call_index["value"]]
        call_index["value"] += 1
        segment.export(out_path, format="wav")

    monkeypatch.setattr(main_module, "synthesize_ru", fake_synthesize_ru)
    monkeypatch.setattr(main_module.config.tts, "chunk_crossfade_ms", 50)
    monkeypatch.setattr(main_module.config.tts, "chunk_target_dbfs", -18.0)

    output_path = tmp_path / "result_fast.wav"
    main_module.synthesize_with_splitting(
        "hello world" * 30,
        ["profile.wav"],
        output_path,
        mode="fast",
    )

    combined = AudioSegment.from_file(output_path)

    assert call_index["value"] == 2
    expected_length = sum(len(segment) for segment in segments)
    assert len(combined) == pytest.approx(expected_length, abs=3)
    assert abs(combined.dBFS - (-18.0)) > 1.5


def test_get_tts_uses_model_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = TTSConfig(
        backend="coqui",
        model_name="custom/model",
        model_path=tmp_path / "checkpoint.pth",
    )

    loader_calls = {"value": 0}

    def fake_loader() -> TTSConfig:
        loader_calls["value"] += 1
        return config

    class DummyEngine:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            self.model_name = model_name
            self.kwargs = kwargs

    monkeypatch.setattr(tts_engine, "_CONFIG_LOADER", fake_loader)
    monkeypatch.setattr(tts_engine, "_cached_tts_config", None)
    monkeypatch.setattr(tts_engine, "_tts", None)
    monkeypatch.setattr(tts_engine, "_synthesizer", None)
    monkeypatch.setattr(tts_engine, "TTS", DummyEngine)

    engine = tts_engine.get_tts()

    assert loader_calls["value"] == 1
    assert isinstance(engine, DummyEngine)
    assert engine.model_name == "custom/model"
    assert engine.kwargs == {"model_path": str(config.model_path)}

    assert tts_engine.get_tts() is engine
    assert loader_calls["value"] == 1


def test_get_tts_supports_custom_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = TTSConfig(
        backend="tests.test_tts_processing._dummy_backend_factory",
        model_name="alt/model",
        model_path=tmp_path / "backend.pth",
    )

    def fake_loader() -> TTSConfig:
        return config

    monkeypatch.setattr(tts_engine, "_CONFIG_LOADER", fake_loader)
    monkeypatch.setattr(tts_engine, "_cached_tts_config", None)
    monkeypatch.setattr(tts_engine, "_tts", None)
    monkeypatch.setattr(tts_engine, "_synthesizer", None)

    engine = tts_engine.get_tts()

    assert engine.name == "alt/model"
    assert engine.model_path == str(config.model_path)

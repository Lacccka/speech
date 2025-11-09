# tts_engine.py
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_tts = None


def get_tts():
    global _tts
    if _tts is None:
        # загрузится 1 раз на всё время работы бота
        _tts = TTS(MODEL_NAME)
    return _tts


def synthesize_ru(text: str, profile_wav: str, out_wav: str):
    tts = get_tts()
    tts.tts_to_file(
        text=text,
        file_path=out_wav,
        speaker_wav=profile_wav,
        language="ru",
        enable_text_splitting=True,
    )
    # лёгкая постобработка
    audio = AudioSegment.from_file(out_wav)
    audio = normalize(audio)
    audio.export(out_wav, format="wav")

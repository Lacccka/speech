# audio_utils.py
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize

DATA_DIR = Path("data")
VOICES_DIR = DATA_DIR / "voices"
PROFILES_DIR = DATA_DIR / "profiles"
OUTPUTS_DIR = DATA_DIR / "outputs"

for p in [VOICES_DIR, PROFILES_DIR, OUTPUTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def user_voice_dir(user_id: int) -> Path:
    """куда временно складываем голосовые пользователя"""
    d = VOICES_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_profile_dir(user_id: int) -> Path:
    """папка с профилем именно этого пользователя"""
    d = PROFILES_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_profile_path(user_id: int) -> Path:
    """итоговый склеенный профиль"""
    return user_profile_dir(user_id) / "profile.wav"


def user_output_path(user_id: int) -> Path:
    d = OUTPUTS_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / "out.wav"


def convert_to_wav(src_path: str, dst_path: str):
    """конвертация ogg/opus → wav"""
    audio = AudioSegment.from_file(src_path)
    audio.export(dst_path, format="wav")


def merge_user_voices(user_id: int, profile_path: Path):
    """
    Склеиваем все wav пользователя в один и сохраняем как профиль.
    """
    voice_dir = user_voice_dir(user_id)
    parts = sorted(voice_dir.glob("*.wav"))
    if not parts:
        return None

    combined = AudioSegment.empty()
    for part in parts:
        audio = AudioSegment.from_wav(part)
        combined += audio

    combined = normalize(combined)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(profile_path, format="wav")
    return str(profile_path)


def clear_user_voices(user_id: int):
    """удаляем временные куски после склейки"""
    voice_dir = user_voice_dir(user_id)
    for f in voice_dir.glob("*"):
        f.unlink(missing_ok=True)

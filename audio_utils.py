# audio_utils.py
from pathlib import Path
from typing import List, Optional, Sequence

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


def user_output_ogg_path(user_id: int) -> Path:
    d = OUTPUTS_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / "out.ogg"


def convert_to_wav(src_path: str, dst_path: str):
    """конвертация ogg/opus → wav"""
    audio = AudioSegment.from_file(src_path)
    audio.export(dst_path, format="wav")


def wav_to_ogg_opus(src: str, dst: str):
    """конвертация wav → ogg/opus с нормализацией громкости"""
    audio = AudioSegment.from_wav(src)
    normalized = normalize(audio)
    normalized = normalized.set_channels(1).set_frame_rate(24_000)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.export(
        dst_path,
        format="ogg",
        codec="libopus",
        bitrate="64k",
    )


def merge_user_voices(
    user_id: int,
    profile_dir: Path,
    *,
    selected_clips: Optional[Sequence[Path | str]] = None,
    max_references: int = 3,
) -> List[str]:
    """Подбирает до ``max_references`` лучших клипов и сохраняет их как эталон."""

    voice_dir = user_voice_dir(user_id)
    available_parts = sorted(voice_dir.glob("*.wav"))
    if not available_parts:
        return []

    if selected_clips:
        normalised: List[Path] = []
        for clip in selected_clips:
            clip_path = Path(clip)
            if not clip_path.is_absolute():
                clip_path = voice_dir / clip_path
            if clip_path.exists():
                normalised.append(clip_path)
        parts = normalised[:max_references]
    else:
        scored: List[tuple[Path, int]] = []
        for part in available_parts:
            try:
                audio = AudioSegment.from_wav(part)
            except Exception:
                continue
            scored.append((part, len(audio)))

        if not scored:
            return []

        scored.sort(key=lambda item: item[1], reverse=True)
        parts = [path for path, _ in scored[:max_references]]

    if not parts:
        return []

    profile_dir.mkdir(parents=True, exist_ok=True)
    curated_paths: List[str] = []
    for idx, part in enumerate(parts, start=1):
        try:
            audio = AudioSegment.from_wav(part)
        except Exception:
            continue
        processed = normalize(audio)
        target_path = profile_dir / f"reference_{idx}.wav"
        processed.export(target_path, format="wav")
        curated_paths.append(str(target_path))

    return curated_paths


def clear_user_voices(user_id: int):
    """удаляем временные куски после склейки"""
    voice_dir = user_voice_dir(user_id)
    for f in voice_dir.glob("*"):
        f.unlink(missing_ok=True)

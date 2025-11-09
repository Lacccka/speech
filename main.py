# main.py
import asyncio
import sys
from pathlib import Path
from typing import Any, List, Optional

from pydub import AudioSegment

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import Message, FSInputFile

from db import (
    init_db,
    get_user,
    set_state,
    set_profile,
    start_user_session,
    add_sample,
    get_latest_samples,
    delete_user_samples,
)
from keyboards import main_kb
from audio_utils import (
    user_voice_dir,
    convert_to_wav,
    clear_user_voices,
    user_output_path,
    user_output_ogg_path,
    wav_to_ogg_opus,
)
from training import continue_training, train_new_voice
from tts_engine import assemble_segments, normalize_to_target, synthesize_ru

from speech.config import load_config
from speech.logging import configure_logging, get_logger

SAFE_TEXT_LENGTH = 250
DEFAULT_CHUNK_LENGTH = 180

TRAINING_STATE_NEW = "training_new"
TRAINING_STATE_CONTINUE = "training_continue"
TRAINING_STATE_SELECT = "training_select"

TRAINING_MODE_NEW_COMMANDS = {"новое обучение", "начать заново", "новое"}
TRAINING_MODE_CONTINUE_COMMANDS = {"продолжить обучение", "дообучить", "продолжить"}


configure_logging()
logger = get_logger(__name__)
config = load_config()

bot = Bot(token=config.bot.token)
dp = Dispatcher()


def split_text_for_tts(text: str, max_chars: int = DEFAULT_CHUNK_LENGTH) -> List[str]:
    """
    Делит текст на части, стараясь не превышать max_chars и не разбивать слова.
    """

    chunks: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            combined = " ".join(buffer).strip()
            if combined:
                chunks.append(combined)
            buffer.clear()

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            flush_buffer()
            continue

        words = paragraph.split()
        for word in words:
            if not buffer:
                buffer.append(word)
                continue

            prospective = f"{' '.join(buffer)} {word}"
            if len(prospective) <= max_chars:
                buffer.append(word)
            else:
                flush_buffer()
                buffer.append(word)

    flush_buffer()

    if not chunks:
        stripped = text.strip()
        if stripped:
            return [stripped[:max_chars]]
        return []

    return chunks


def synthesize_with_splitting(
    text: str,
    profile_path: str,
    out_path: Path,
    *,
    language: Optional[str] = None,
    gpt_condition_length: Optional[int] = None,
    reference_duration: Optional[float] = None,
    **synthesis_kwargs: Any,
) -> None:
    """
    Вызывает синтез с дроблением длинного текста и объединяет WAV-файлы.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = split_text_for_tts(text)
    if not chunks:
        raise ValueError("Передан пустой текст для синтеза")

    effective_language = language if language is not None else config.tts.language
    effective_gpt_condition_length = (
        gpt_condition_length
        if gpt_condition_length is not None
        else config.tts.gpt_conditioning_length
    )
    effective_reference_duration = (
        reference_duration
        if reference_duration is not None
        else config.tts.reference_duration
    )

    chunk_kwargs = dict(synthesis_kwargs)
    chunk_kwargs.setdefault("crossfade_ms", config.tts.chunk_crossfade_ms)
    chunk_kwargs.setdefault("target_dbfs", config.tts.chunk_target_dbfs)
    chunk_kwargs.setdefault("silence_threshold", config.tts.silence_threshold)
    chunk_kwargs.setdefault("silence_chunk_len", config.tts.silence_chunk_len)
    chunk_kwargs.setdefault("deesser_frequency", config.tts.deesser_frequency)
    chunk_kwargs.setdefault("deesser_reduction_db", config.tts.deesser_reduction_db)

    if len(chunks) == 1:
        synthesize_ru(
            chunks[0],
            profile_path,
            str(out_path),
            language=effective_language,
            gpt_cond_len=effective_gpt_condition_length,
            reference_duration=effective_reference_duration,
            **chunk_kwargs,
        )
        return

    temp_paths: List[Path] = []
    chunk_segments: List[AudioSegment] = []

    try:
        for idx, chunk in enumerate(chunks):
            temp_path = out_path.with_name(f"{out_path.stem}_part{idx}.wav")
            synthesize_ru(
                chunk,
                profile_path,
                str(temp_path),
                language=effective_language,
                gpt_cond_len=effective_gpt_condition_length,
                reference_duration=effective_reference_duration,
                **chunk_kwargs,
            )
            temp_paths.append(temp_path)
            chunk_segments.append(AudioSegment.from_wav(temp_path))

        effective_crossfade = chunk_kwargs["crossfade_ms"]
        combined = assemble_segments(
            chunk_segments,
            crossfade_ms=effective_crossfade,
        )
        combined = normalize_to_target(
            combined,
            target_dbfs=chunk_kwargs["target_dbfs"],
        )
        combined.export(out_path, format="wav")
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await get_user(message.from_user.id)  # создаём запись в БД если нет
    await message.answer(
        "Привет! Я бот для клонирования голоса.\n\n"
        "🔹 Нажми «🎙 Начать обучение» и пришли несколько голосовых.\n"
        "🔹 Отправь 5–10 голосовых сообщений длиной 5–10 секунд, чтобы я смог собрать базовый профиль.\n"
        "🔹 Если нужно обновить голос позже — используй команду «Дообучить».\n"
        "🔹 Потом «🛑 Завершить обучение» — я соберу профиль.\n"
        "🔹 Потом «🗣 Сгенерировать» и пришли текст — я озвучу его твоим голосом.\n\n"
        f"ℹ️ Один запрос — до {SAFE_TEXT_LENGTH} символов текста.",
        reply_markup=main_kb(),
    )


async def _enter_training_mode(message: Message, user_id: int, mode: str) -> None:
    session_id: int | None = None
    if mode == TRAINING_STATE_NEW:
        await delete_user_samples(user_id)
        clear_user_voices(user_id)
        session_id = await start_user_session(user_id)
        await set_state(user_id, TRAINING_STATE_NEW)
        await message.answer(
            "Я очистил предыдущие записи. Присылай новые голосовые подряд. Для стартового обучения отправь хотя бы 5–10"
            " сообщений длиной 5–10 секунд, а для стабильного результата собери 20–60 минут чистых записей одним голосом:"
            " короткие сегменты по 2–10 секунд в одинаковых условиях (ровная тональность, без шумов, один и тот же"
            " микрофон). Записи можно накапливать и использовать позже для дообучения."
            " Когда закончишь — нажми «🛑 Завершить обучение»."
        )
    elif mode == TRAINING_STATE_CONTINUE:
        session_id = await start_user_session(user_id)
        await set_state(user_id, TRAINING_STATE_CONTINUE)
        await message.answer(
            "Принял режим дообучения. Присылай дополнительные голосовые — я добавлю их к тем, что уже сохранены."
            " Лучше всего отправить 5–10 новых сообщений по 5–10 секунд, чтобы обновление прошло заметнее."
            " Когда закончишь — нажми «🛑 Завершить обучение»."
        )
    else:
        logger.error("Unknown training mode %s for user %s", mode, user_id)
        return

    logger.info("Started training session %s for user %s", session_id, user_id)


def _has_saved_voices(user_id: int) -> bool:
    voice_dir = user_voice_dir(user_id)
    return any(voice_dir.glob("*.wav"))


@dp.message(F.text == "🎙 Начать обучение")
async def start_training(message: Message):
    user_id = message.from_user.id
    _, state, profile_path, _ = await get_user(user_id)

    if state in {TRAINING_STATE_NEW, TRAINING_STATE_CONTINUE}:
        await message.answer(
            "Я уже жду голосовые. Отправь ещё несколько или нажми «🛑 Завершить обучение»."
        )
        return

    if _has_saved_voices(user_id) or profile_path:
        await set_state(user_id, TRAINING_STATE_SELECT)
        await message.answer(
            "У тебя уже есть записи. Выбери режим: напиши «Новое обучение», чтобы начать заново (старые записи удалю)"
            " и снова собрать 5–10 стартовых голосовых, или «Дообучить»/«Продолжить обучение», чтобы добавить новые"
            " образцы к уже сохранённым."
        )
        return

    await _enter_training_mode(message, user_id, TRAINING_STATE_NEW)


@dp.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    _, state, _, current_session = await get_user(user_id)

    if state not in {TRAINING_STATE_NEW, TRAINING_STATE_CONTINUE}:
        await message.answer(
            "Сейчас ты не в режиме обучения. Нажми «🎙 Начать обучение»."
        )
        return

    # скачиваем голосовое
    voice = message.voice
    logger.info(
        f"voice.file_id={voice.file_id}, voice.file_unique_id={voice.file_unique_id}"
    )

    if message.chat.type != "private" or message.forward_date:
        await message.answer(
            "Пришли голос, записанный прямо сюда, не пересланный."
        )
        return

    user_dir = user_voice_dir(user_id)
    ogg_path = user_dir / f"{voice.file_unique_id}.ogg"
    wav_path = user_dir / f"{voice.file_unique_id}.wav"

    try:
        await bot.download(voice, destination=ogg_path)
    except TelegramBadRequest as exc:
        logger.warning("Failed to download voice message: %s", exc)
        await message.answer("Перешли голос ещё раз")
        return

    if not ogg_path.exists():
        logger.error("Downloaded voice file not found at %s", ogg_path)
        await message.answer("Перешли голос ещё раз")
        return

    convert_to_wav(str(ogg_path), str(wav_path))
    ogg_path.unlink(missing_ok=True)

    await add_sample(user_id, str(wav_path), session_id=current_session)

    await message.answer("Принял голосовое 👍")


@dp.message(F.text == "🛑 Завершить обучение")
async def finish_training(message: Message):
    user_id = message.from_user.id
    _, state, _, current_session = await get_user(user_id)

    if state not in {TRAINING_STATE_NEW, TRAINING_STATE_CONTINUE}:
        await message.answer(
            "Сейчас обучение не идёт. Сначала нажми «🎙 Начать обучение» и пришли голосовые."
        )
        return

    samples = await get_latest_samples(user_id, current_session) if current_session else []
    if not samples:
        await message.answer(
            "Я не нашёл записей. Пришли хотя бы одно голосовое в режиме обучения."
        )
        return

    if state == TRAINING_STATE_NEW:
        merged = train_new_voice(user_id)
    else:
        merged = continue_training(user_id)

    if not merged:
        await message.answer(
            "Я не нашёл записей. Пришли хотя бы одно голосовое в режиме обучения."
        )
        return

    await set_profile(user_id, merged)
    await set_state(user_id, "idle")

    await message.answer(
        "Готово! Я собрал твой голос. Теперь нажми «🗣 Сгенерировать» и пришли текст."
    )


@dp.message(F.text == "🗣 Сгенерировать")
async def ask_text(message: Message):
    user_id = message.from_user.id
    await set_state(user_id, "generate")
    await message.answer(
        f"Пришли текст, который нужно озвучить твоим голосом (до {SAFE_TEXT_LENGTH} символов за один запрос)."
        " Если нужно больше — отправляй по частям или используй пошаговую генерацию."
    )


@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    user_id, state, profile_path, _ = await get_user(user_id)

    if state == TRAINING_STATE_SELECT:
        text = (message.text or "").casefold()
        if text in TRAINING_MODE_NEW_COMMANDS:
            await _enter_training_mode(message, user_id, TRAINING_STATE_NEW)
        elif text in TRAINING_MODE_CONTINUE_COMMANDS:
            await _enter_training_mode(message, user_id, TRAINING_STATE_CONTINUE)
        else:
            await message.answer(
                "Не понял режим. Напиши «Новое обучение» или «Продолжить обучение»."
            )
        return

    # реагируем только если пользователь в режиме генерации
    if state != "generate":
        return

    if not profile_path:
        await message.answer(
            "У тебя нет профиля голоса. Сначала обучи меня голосовыми."
        )
        await set_state(user_id, "idle")
        return

    text = message.text.strip()
    if not text:
        await message.answer(
            f"Текст пустой 🤔 Пришли нормальный текст до {SAFE_TEXT_LENGTH} символов."
        )
        return

    if len(text) > SAFE_TEXT_LENGTH:
        await message.answer(
            "Сообщение слишком длинное для безопасной генерации."
            f" Пожалуйста, сократи его до {SAFE_TEXT_LENGTH} символов"
            " или запусти пошаговую генерацию, отправляя фрагменты последовательно."
        )
        return

    out_path = user_output_path(user_id)
    ogg_path = user_output_ogg_path(user_id)
    await message.answer(
        "Генерирую... Помни, что лучше держать отдельные запросы в пределах допустимой длины."
    )

    # синтез
    synthesize_with_splitting(text, profile_path, out_path)

    try:
        wav_to_ogg_opus(str(out_path), str(ogg_path))
    except Exception:
        logger.exception("Failed to convert WAV to OGG/Opus via ffmpeg")
        await message.answer("не смог перекодировать аудио, проверь ffmpeg/libopus")
        await set_state(user_id, "idle")
        return

    voice_file = FSInputFile(str(ogg_path), filename="voice.ogg")
    await message.answer_voice(voice_file)

    # вернём в обычный режим
    await set_state(user_id, "idle")


async def main():
    await init_db()
    logger.info("Starting Telegram bot polling")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

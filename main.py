# main.py
import asyncio
import sys
from pathlib import Path
from typing import List

from pydub import AudioSegment
from pydub.effects import normalize

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import Message, FSInputFile

from db import init_db, get_user, set_state, set_profile
from keyboards import main_kb
from audio_utils import (
    user_voice_dir,
    convert_to_wav,
    merge_user_voices,
    clear_user_voices,
    user_profile_path,
    user_output_path,
    user_output_ogg_path,
    wav_to_ogg_opus,
)
from tts_engine import synthesize_ru

from speech.config import load_config
from speech.logging import configure_logging, get_logger

SAFE_TEXT_LENGTH = 250
DEFAULT_CHUNK_LENGTH = 180


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


def synthesize_with_splitting(text: str, profile_path: str, out_path: Path) -> None:
    """
    Вызывает синтез с дроблением длинного текста и объединяет WAV-файлы.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = split_text_for_tts(text)
    if not chunks:
        raise ValueError("Передан пустой текст для синтеза")

    if len(chunks) == 1:
        synthesize_ru(chunks[0], profile_path, str(out_path))
        return

    temp_paths: List[Path] = []
    combined = AudioSegment.silent(duration=0)

    try:
        for idx, chunk in enumerate(chunks):
            temp_path = out_path.with_name(f"{out_path.stem}_part{idx}.wav")
            synthesize_ru(chunk, profile_path, str(temp_path))
            temp_paths.append(temp_path)
            combined += AudioSegment.from_wav(temp_path)

        combined = normalize(combined)
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
        "🔹 Потом «🛑 Завершить обучение» — я соберу профиль.\n"
        "🔹 Потом «🗣 Сгенерировать» и пришли текст — я озвучу его твоим голосом.\n\n"
        f"ℹ️ Один запрос — до {SAFE_TEXT_LENGTH} символов текста.",
        reply_markup=main_kb(),
    )


@dp.message(F.text == "🎙 Начать обучение")
async def start_training(message: Message):
    user_id = message.from_user.id
    _, state, _ = await get_user(user_id)

    if state == "training":
        await message.answer(
            "Я уже жду новые голосовые. Отправь ещё несколько или нажми «🛑 Завершить обучение»."
        )
        return

    clear_user_voices(user_id)
    await set_state(user_id, "training")
    await message.answer(
        "Ок, я в режиме обучения. Присылай голосовые подряд. Для стабильного результата собери 20–60 минут"
        " чистых записей одним голосом: короткие сегменты по 2–10 секунд в одинаковых условиях (ровная тональность,"
        " без шумов, один и тот же микрофон). Записи можно накапливать и использовать позже для дообучения."
        " Когда закончишь — нажми «🛑 Завершить обучение»."
    )


@dp.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    _, state, _ = await get_user(user_id)

    if state != "training":
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

    await message.answer("Принял голосовое 👍")


@dp.message(F.text == "🛑 Завершить обучение")
async def finish_training(message: Message):
    user_id = message.from_user.id
    _, state, _ = await get_user(user_id)

    if state != "training":
        await message.answer(
            "Сейчас обучение не идёт. Сначала нажми «🎙 Начать обучение» и пришли голосовые."
        )
        return

    profile_path = user_profile_path(user_id)

    merged = merge_user_voices(user_id, profile_path)
    if not merged:
        await message.answer(
            "Я не нашёл записей. Пришли хотя бы одно голосовое в режиме обучения."
        )
        return

    await set_profile(user_id, str(profile_path))
    await set_state(user_id, "idle")
    clear_user_voices(user_id)

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
    user_id, state, profile_path = await get_user(user_id)

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

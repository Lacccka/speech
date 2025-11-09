# main.py
import asyncio
import re
import sys
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence

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
    set_speaker_references,
    start_user_session,
    add_sample,
    get_latest_samples,
    delete_user_samples,
    set_pending_tts_text,
    get_pending_tts_text,
    get_speaker_references,
)
from keyboards import main_kb, generation_mode_kb, training_selection_kb
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
DEFAULT_CHUNK_LENGTH = 240

_SENTENCE_ENDINGS = {".", "!", "?", "…"}
_CLOSING_PUNCTUATION = set("\"'”»)]}›")
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
    "p.m.",
    "a.m.",
    "sen.",
    "rep.",
    "dept.",
    "inc.",
    "ltd.",
    "co.",
    "no.",
    "г.",
    "ул.",
    "д.",
    "т.д.",
    "т.п.",
    "рис.",
    "стр.",
}
_INITIALISM_RE = re.compile(r"^(?:[A-Za-zА-Яа-я]\.){2,}$")

GENERATION_STATE_AWAITING_TEXT = "generate"
GENERATION_STATE_AWAITING_MODE = "generate_mode"

FAST_MODE_COMMANDS = {"fast", "быстро", "⚡", "⚡ быстро", "⚡быстро"}
QUALITY_MODE_COMMANDS = {"quality", "качество", "🎧", "🎧 качество", "🎧качество"}
BACK_COMMANDS = {"назад", "⬅️", "⬅️ назад", "back"}

TRAINING_STATE_NEW = "training_new"
TRAINING_STATE_CONTINUE = "training_continue"
TRAINING_STATE_SELECT = "training_select"

TRAINING_MODE_NEW_COMMANDS = {"новое обучение", "начать заново", "новое"}
TRAINING_MODE_CONTINUE_COMMANDS = {
    "продолжить обучение",
    "дообучить",
    "продолжить",
    "дообучить/продолжить обучение",
}


configure_logging()
logger = get_logger(__name__)
config = load_config()

bot = Bot(token=config.bot.token)
dp = Dispatcher()


def split_text_for_tts(text: str, max_chars: int = DEFAULT_CHUNK_LENGTH) -> List[str]:
    """
    Делит текст на части, стараясь учитывать границы предложений и ограничение по длине.
    """

    def _normalize_token(token: str) -> str:
        return token.lower().strip("\"'“”«»()[]{}")

    def _is_probable_abbreviation(token: str) -> bool:
        normalized = _normalize_token(token)
        if not normalized:
            return False
        if normalized in _ABBREVIATIONS:
            return True
        if _INITIALISM_RE.match(normalized):
            return True
        if normalized.endswith(".") and len(normalized.rstrip(".")) == 1:
            return True
        return False

    def _split_oversized_token(token: str) -> List[str]:
        return [token[i : i + max_chars] for i in range(0, len(token), max_chars)]

    def _split_long_segment(segment: str) -> List[str]:
        stripped = segment.strip()
        if not stripped:
            return []
        tokens = stripped.split()
        if not tokens:
            return _split_oversized_token(stripped)

        parts: List[str] = []
        current = ""

        for token in tokens:
            if len(token) > max_chars:
                if current:
                    parts.append(current.strip())
                    current = ""
                oversized = _split_oversized_token(token)
                for piece in oversized[:-1]:
                    parts.append(piece)
                last_piece = oversized[-1]
                if len(last_piece) == max_chars:
                    parts.append(last_piece)
                    current = ""
                else:
                    current = last_piece
                continue

            if not current:
                current = token
                continue

            prospective = f"{current} {token}"
            if len(prospective) <= max_chars:
                current = prospective
            else:
                parts.append(current.strip())
                current = token

        if current:
            parts.append(current.strip())
        return parts

    def _split_paragraph_into_sentences(paragraph: str) -> List[str]:
        sentences: List[str] = []
        start = 0
        length = len(paragraph)
        i = 0

        while i < length:
            char = paragraph[i]
            if char in _SENTENCE_ENDINGS:
                next_char = paragraph[i + 1] if i + 1 < length else ""
                prev_char = paragraph[i - 1] if i - 1 >= 0 else ""

                if char == ".":
                    if next_char == ".":
                        i += 1
                        continue
                    if prev_char.isdigit() and next_char.isdigit():
                        i += 1
                        continue

                end = i + 1
                while end < length and paragraph[end] in _CLOSING_PUNCTUATION:
                    end += 1

                segment = paragraph[start:end].strip()
                if not segment:
                    start = end
                    i = end
                    continue

                last_token = segment.split()[-1] if segment.split() else ""
                if char == "." and _is_probable_abbreviation(last_token):
                    i += 1
                    continue

                sentences.append(segment)
                start = end
                while start < length and paragraph[start].isspace():
                    start += 1
                i = start
                continue

            i += 1

        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
        return sentences

    preferred_min = 200 if max_chars >= 200 else max_chars
    chunks: List[str] = []
    current_chunk = ""

    def _flush_current() -> None:
        nonlocal current_chunk
        chunk = current_chunk.strip()
        if chunk:
            chunks.append(chunk)
        current_chunk = ""

    for raw_paragraph in text.splitlines():
        paragraph = raw_paragraph.strip()
        if not paragraph:
            _flush_current()
            continue

        sentences = _split_paragraph_into_sentences(paragraph)
        if not sentences:
            sentences = [paragraph]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                long_parts = _split_long_segment(sentence)
                if current_chunk:
                    _flush_current()
                if not long_parts:
                    continue
                chunks.extend(part.strip() for part in long_parts[:-1])
                current_chunk = long_parts[-1].strip()
                if len(current_chunk) > max_chars:
                    for part in _split_long_segment(current_chunk):
                        chunks.append(part.strip())
                    current_chunk = ""
                continue

            if not current_chunk:
                current_chunk = sentence
                continue

            prospective = f"{current_chunk} {sentence}"
            if len(prospective) <= max_chars:
                current_chunk = prospective
                continue

            if len(current_chunk) < preferred_min and len(sentence) < preferred_min:
                merged_parts = _split_long_segment(prospective)
                if len(merged_parts) > 1:
                    chunks.extend(part.strip() for part in merged_parts[:-1])
                    current_chunk = merged_parts[-1].strip()
                    continue

            _flush_current()
            current_chunk = sentence

    _flush_current()

    if not chunks:
        stripped = text.strip()
        if stripped:
            return [stripped[:max_chars]]
        return []

    return chunks


GenerationMode = Literal["fast", "quality"]


def synthesize_with_splitting(
    text: str,
    speaker_references: Sequence[str] | str,
    out_path: Path,
    *,
    mode: GenerationMode = "fast",
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

    mode_normalized = mode.lower()
    if mode_normalized not in {"fast", "quality"}:
        raise ValueError(f"Unsupported synthesis mode: {mode}")

    effective_language = language if language is not None else config.tts.language

    if mode_normalized == "quality":
        effective_gpt_condition_length = (
            gpt_condition_length
            if gpt_condition_length is not None
            else (
                config.tts.quality_gpt_conditioning_length
                if config.tts.quality_gpt_conditioning_length is not None
                else (
                    config.tts.gpt_conditioning_length
                    if config.tts.gpt_conditioning_length is not None
                    else 12
                )
            )
        )
        effective_reference_duration = (
            reference_duration
            if reference_duration is not None
            else (
                config.tts.quality_reference_duration
                if config.tts.quality_reference_duration is not None
                else config.tts.reference_duration
            )
        )
    else:
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
    enable_post_processing = mode_normalized == "quality"
    chunk_kwargs.setdefault("enable_post_processing", enable_post_processing)

    if enable_post_processing:
        crossfade_default = (
            config.tts.quality_crossfade_ms
            if config.tts.quality_crossfade_ms is not None
            else config.tts.chunk_crossfade_ms
        )
        target_dbfs_default = (
            config.tts.quality_target_dbfs
            if config.tts.quality_target_dbfs is not None
            else config.tts.chunk_target_dbfs
        )
    else:
        crossfade_default = 0
        target_dbfs_default = None

    chunk_kwargs.setdefault("crossfade_ms", crossfade_default)
    chunk_kwargs.setdefault("target_dbfs", target_dbfs_default)
    chunk_kwargs.setdefault("silence_threshold", config.tts.silence_threshold)
    chunk_kwargs.setdefault("silence_chunk_len", config.tts.silence_chunk_len)
    chunk_kwargs.setdefault("deesser_frequency", config.tts.deesser_frequency)
    chunk_kwargs.setdefault("deesser_reduction_db", config.tts.deesser_reduction_db)

    if len(chunks) == 1:
        synthesize_ru(
            chunks[0],
            speaker_references,
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
                speaker_references,
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
        target_dbfs = chunk_kwargs["target_dbfs"]
        if target_dbfs is not None:
            combined = normalize_to_target(
                combined,
                target_dbfs=target_dbfs,
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
            " Когда закончишь — нажми «🛑 Завершить обучение».",
            reply_markup=main_kb(),
        )
    elif mode == TRAINING_STATE_CONTINUE:
        session_id = await start_user_session(user_id)
        await set_state(user_id, TRAINING_STATE_CONTINUE)
        await message.answer(
            "Принял режим дообучения. Присылай дополнительные голосовые — я добавлю их к тем, что уже сохранены."
            " Лучше всего отправить 5–10 новых сообщений по 5–10 секунд, чтобы обновление прошло заметнее."
            " Когда закончишь — нажми «🛑 Завершить обучение».",
            reply_markup=main_kb(),
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
    _, state, profile_path, _, _ = await get_user(user_id)
    existing_references = await get_speaker_references(user_id)

    if state in {TRAINING_STATE_NEW, TRAINING_STATE_CONTINUE}:
        await message.answer(
            "Я уже жду голосовые. Отправь ещё несколько или нажми «🛑 Завершить обучение»."
        )
        return

    if _has_saved_voices(user_id) or existing_references or profile_path:
        await set_state(user_id, TRAINING_STATE_SELECT)
        await message.answer(
            "У тебя уже есть записи. Выбери режим обучения с помощью кнопок ниже:",
            reply_markup=training_selection_kb(),
        )
        return

    await _enter_training_mode(message, user_id, TRAINING_STATE_NEW)


@dp.message(F.text.func(lambda text: (text or "").casefold() in TRAINING_MODE_NEW_COMMANDS))
async def select_new_training_mode(message: Message):
    user_id = message.from_user.id
    _, state, _, _, _ = await get_user(user_id)

    if state != TRAINING_STATE_SELECT:
        return

    await _enter_training_mode(message, user_id, TRAINING_STATE_NEW)


@dp.message(F.text.func(lambda text: (text or "").casefold() in TRAINING_MODE_CONTINUE_COMMANDS))
async def select_continue_training_mode(message: Message):
    user_id = message.from_user.id
    _, state, _, _, _ = await get_user(user_id)

    if state != TRAINING_STATE_SELECT:
        return

    await _enter_training_mode(message, user_id, TRAINING_STATE_CONTINUE)


@dp.message(F.text.func(lambda text: (text or "").casefold() in BACK_COMMANDS))
async def cancel_training_selection(message: Message):
    user_id = message.from_user.id
    _, state, _, _, _ = await get_user(user_id)

    if state != TRAINING_STATE_SELECT:
        return

    await set_state(user_id, "idle")
    await message.answer(
        "Хорошо, вернул в главное меню. Выбирай, что делать дальше.",
        reply_markup=main_kb(),
    )


@dp.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    _, state, _, current_session, _ = await get_user(user_id)

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
    _, state, _, current_session, _ = await get_user(user_id)

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
        references = train_new_voice(user_id)
    else:
        references = continue_training(user_id)

    if not references:
        await message.answer(
            "Я не нашёл записей. Пришли хотя бы одно голосовое в режиме обучения."
        )
        return

    await set_speaker_references(user_id, references)
    await set_state(user_id, "idle")

    await message.answer(
        "Готово! Я собрал твой голос. Теперь нажми «🗣 Сгенерировать» и пришли текст."
    )


@dp.message(F.text == "🗣 Сгенерировать")
async def ask_text(message: Message):
    user_id = message.from_user.id
    await set_pending_tts_text(user_id, None)
    await set_state(user_id, GENERATION_STATE_AWAITING_TEXT)
    await message.answer(
        f"Пришли текст, который нужно озвучить твоим голосом (до {SAFE_TEXT_LENGTH} символов за один запрос)."
        " После этого выбери режим генерации: быстрый или качественный."
        " Если нужно больше — отправляй по частям или используй пошаговую генерацию."
    )


@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    user_id, state, profile_path, _, pending_text = await get_user(user_id)
    speaker_references = await get_speaker_references(user_id)

    # реагируем только если пользователь в одном из режимов генерации
    if state not in {GENERATION_STATE_AWAITING_TEXT, GENERATION_STATE_AWAITING_MODE}:
        return

    if not speaker_references:
        await message.answer(
            "У тебя нет профиля голоса. Сначала обучи меня голосовыми."
        )
        await set_state(user_id, "idle")
        return

    raw_text = (message.text or "").strip()

    if state == GENERATION_STATE_AWAITING_MODE:
        choice = raw_text.casefold()

        if choice in BACK_COMMANDS:
            await set_state(user_id, GENERATION_STATE_AWAITING_TEXT)
            await set_pending_tts_text(user_id, None)
            await message.answer(
                f"Хорошо, пришли новый текст до {SAFE_TEXT_LENGTH} символов.",
                reply_markup=main_kb(),
            )
            return

        if choice in FAST_MODE_COMMANDS:
            mode = "fast"
        elif choice in QUALITY_MODE_COMMANDS:
            mode = "quality"
        else:
            await message.answer(
                "Выбери «⚡ Быстро» для мгновенного ответа или «🎧 Качество» для более тщательного синтеза.",
                reply_markup=generation_mode_kb(),
            )
            return

        pending_raw = pending_text or await get_pending_tts_text(user_id)
        pending = (pending_raw or "").strip()
        if not pending:
            await message.answer(
                "Не нашёл текст для синтеза. Пришли текст ещё раз.",
                reply_markup=main_kb(),
            )
            await set_state(user_id, GENERATION_STATE_AWAITING_TEXT)
            return

        await set_pending_tts_text(user_id, None)

        out_path = user_output_path(user_id)
        ogg_path = user_output_ogg_path(user_id)
        await message.answer(
            "Генерирую... Помни, что лучше держать отдельные запросы в пределах допустимой длины.",
            reply_markup=main_kb(),
        )

        synthesize_with_splitting(pending, speaker_references, out_path, mode=mode)

        try:
            wav_to_ogg_opus(str(out_path), str(ogg_path))
        except Exception:
            logger.exception("Failed to convert WAV to OGG/Opus via ffmpeg")
            await message.answer("не смог перекодировать аудио, проверь ffmpeg/libopus")
            await set_state(user_id, "idle")
            return

        voice_file = FSInputFile(str(ogg_path), filename="voice.ogg")
        await message.answer_voice(voice_file)

        await set_state(user_id, "idle")
        return

    # state == GENERATION_STATE_AWAITING_TEXT
    if not raw_text:
        await message.answer(
            f"Текст пустой 🤔 Пришли нормальный текст до {SAFE_TEXT_LENGTH} символов."
        )
        return

    if len(raw_text) > SAFE_TEXT_LENGTH:
        await message.answer(
            "Сообщение слишком длинное для безопасной генерации."
            f" Пожалуйста, сократи его до {SAFE_TEXT_LENGTH} символов"
            " или запусти пошаговую генерацию, отправляя фрагменты последовательно."
        )
        return

    await set_pending_tts_text(user_id, raw_text)
    await set_state(user_id, GENERATION_STATE_AWAITING_MODE)
    await message.answer(
        "Выбери режим генерации: ⚡ Быстро — минимальная задержка, 🎧 Качество — лучшее звучание.",
        reply_markup=generation_mode_kb(),
    )


async def main():
    await init_db()
    logger.info("Starting Telegram bot polling")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

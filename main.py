# main.py
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
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
)
from tts_engine import synthesize_ru

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await get_user(message.from_user.id)  # создаём запись в БД если нет
    await message.answer(
        "Привет! Я бот для клонирования голоса.\n\n"
        "🔹 Нажми «🎙 Начать обучение» и пришли несколько голосовых.\n"
        "🔹 Потом «🛑 Завершить обучение» — я соберу профиль.\n"
        "🔹 Потом «🗣 Сгенерировать» и пришли текст — я озвучу его твоим голосом.",
        reply_markup=main_kb(),
    )


@dp.message(F.text == "🎙 Начать обучение")
async def start_training(message: Message):
    user_id = message.from_user.id
    await set_state(user_id, "training")
    await message.answer(
        "Ок, я в режиме обучения. Присылай голосовые подряд. Когда закончишь — нажми «🛑 Завершить обучение»."
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
    file = await bot.get_file(voice.file_id)

    user_dir = user_voice_dir(user_id)
    ogg_path = user_dir / f"{voice.file_unique_id}.ogg"
    wav_path = user_dir / f"{voice.file_unique_id}.wav"

    await bot.download_file(file.file_path, destination=ogg_path)
    convert_to_wav(str(ogg_path), str(wav_path))

    await message.answer("Принял голосовое 👍")


@dp.message(F.text == "🛑 Завершить обучение")
async def finish_training(message: Message):
    user_id = message.from_user.id
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
    await message.answer("Пришли текст, который нужно озвучить твоим голосом.")


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
        return

    text = message.text.strip()
    if not text:
        await message.answer("Текст пустой 🤔 Пришли нормальный текст.")
        return

    out_path = user_output_path(user_id)
    await message.answer("Генерирую...")

    # синтез
    synthesize_ru(text, profile_path, str(out_path))

    voice_file = FSInputFile(str(out_path), filename="voice.wav")
    await message.answer_voice(voice_file)

    # вернём в обычный режим
    await set_state(user_id, "idle")


async def main():
    await init_db()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

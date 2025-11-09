# keyboards.py
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


def main_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🎙 Начать обучение")],
            [KeyboardButton(text="🛑 Завершить обучение")],
            [KeyboardButton(text="🗣 Сгенерировать")],
        ],
        resize_keyboard=True,
    )


def generation_mode_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="⚡ Быстро"), KeyboardButton(text="🎧 Качество")],
            [KeyboardButton(text="⬅️ Назад")],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

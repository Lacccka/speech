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

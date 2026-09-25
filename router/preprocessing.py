"""Нормализация текста обращений перед векторизацией."""
import re

_NON_WORD = re.compile(r"[^0-9a-zа-я]+")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return _NON_WORD.sub(" ", text).strip()

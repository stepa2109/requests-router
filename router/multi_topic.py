"""Признаки того, что в обращении несколько вопросов (в разные отделы)."""
import re

from router.preprocessing import normalize

_MARKERS = re.compile(r"\b(и еще|а еще|еще вопрос|еще один вопрос|также|кроме того)\b")


def has_extra_question(text: str) -> bool:
    return text.count("?") >= 2 or bool(_MARKERS.search(normalize(text)))

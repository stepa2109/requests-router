"""Правила определения срочности обращения по ключевым словам."""
import re

from router.config import URGENCY_HIGH, URGENCY_NORMAL
from router.preprocessing import normalize

_URGENT = re.compile(
    r"\b(срочн\w*|немедленно|как можно скорее|сегодня|завтра|дедлайн\w*|"
    r"горит|последний день|отчисл\w*|не допуст\w*)\b"
)


def detect_urgency(text: str) -> str:
    return URGENCY_HIGH if _URGENT.search(normalize(text)) else URGENCY_NORMAL

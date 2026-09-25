import pytest

from router.config import URGENCY_HIGH, URGENCY_NORMAL
from router.urgency import detect_urgency


@pytest.mark.parametrize("text", [
    "Срочно! Не могу войти в личный кабинет",
    "Завтра экзамен, а тест не открывается",
    "Сегодня последний день оплаты",
    "Меня хотят отчислить, помогите",
    "Горит дедлайн по курсовой",
    "Нужно как можно скорее получить справку",
])
def test_high(text):
    assert detect_urgency(text) == URGENCY_HIGH


@pytest.mark.parametrize("text", [
    "Где посмотреть расписание сессии?",
    "Как оформить академический отпуск",
    "Хочу вступить в студенческий совет",
])
def test_normal(text):
    assert detect_urgency(text) == URGENCY_NORMAL

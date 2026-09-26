import pytest

from router.multi_topic import has_extra_question


@pytest.mark.parametrize("text", [
    "Как оплатить обучение и еще не открывается вебинар",
    "Как оплатить обучение? И ещё: не открывается вебинар",
    "Нужна справка. Также не могу войти в личный кабинет",
    "Где расписание? Кроме того, как заселиться в общежитие",
    "Ещё вопрос: когда ярмарка вакансий",
    "Где расписание? Когда экзамен?",
])
def test_markers_detected(text):
    assert has_extra_question(text)


@pytest.mark.parametrize("text", [
    "Как оплатить обучение?",
    "Не могу войти в личный кабинет, пишет неверный пароль",
    "Пишу уже второй раз. Где расписание сессии?",
])
def test_single_question_not_flagged(text):
    assert not has_extra_question(text)

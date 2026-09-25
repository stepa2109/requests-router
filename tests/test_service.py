import pytest

from router.config import DEPARTMENTS, URGENCY_HIGH
from router.service import RequestRouter


def test_route_confident(trained_pipeline):
    result = RequestRouter(trained_pipeline).route("Не могу войти в личный кабинет, пишет неверный пароль")
    assert result.department == "IT-поддержка"
    assert result.auto_routed
    assert 0.6 <= result.confidence <= 1.0
    assert len(result.top3) == 3
    assert result.top3[0][0] == result.department


def test_urgency_passed_through(trained_pipeline):
    result = RequestRouter(trained_pipeline).route("Срочно! Завтра экзамен, где расписание?")
    assert result.urgency == URGENCY_HIGH


def test_threshold_one_forces_manual(trained_pipeline):
    result = RequestRouter(trained_pipeline, threshold=1.0).route("Как оплатить обучение?")
    assert not result.auto_routed


@pytest.mark.parametrize("text", ["", "  ", "???", "!!", "🙂🙂"])
def test_too_short_rejected(trained_pipeline, text):
    with pytest.raises(ValueError):
        RequestRouter(trained_pipeline).route(text)


@pytest.mark.parametrize("text", ["hello", "купите слона", "qwerty asdf"])
def test_off_topic_goes_to_manual(trained_pipeline, text):
    result = RequestRouter(trained_pipeline).route(text)
    assert result.department in DEPARTMENTS
    assert not result.auto_routed

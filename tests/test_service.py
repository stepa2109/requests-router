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


def fake_router(**probabilities):
    from conftest import FakePipeline
    return RequestRouter(FakePipeline(probabilities))


def test_strong_second_option_goes_to_manual():
    result = fake_router(Бухгалтерия=0.78, IT=0.20, Деканат=0.02).route("Как оплатить и не открывается вебинар")
    assert result.department == "Бухгалтерия"
    assert result.multi_topic
    assert not result.auto_routed


def test_weak_second_option_stays_auto():
    result = fake_router(Бухгалтерия=0.90, IT=0.08, Деканат=0.02).route("Как оплатить обучение")
    assert not result.multi_topic
    assert result.auto_routed


def test_second_option_rule_can_be_disabled():
    from conftest import FakePipeline
    router = RequestRouter(FakePipeline({"Бухгалтерия": 0.78, "IT": 0.20, "Деканат": 0.02}), second_threshold=1.0)
    result = router.route("Как оплатить и не открывается вебинар")
    assert result.auto_routed and not result.multi_topic


def test_second_option_above_new_threshold_goes_to_manual():
    result = fake_router(Бухгалтерия=0.86, IT=0.13, Деканат=0.01).route("Как оплатить обучение")
    assert result.multi_topic and not result.auto_routed


def test_marker_with_weak_second_option_goes_to_manual():
    result = fake_router(Бухгалтерия=0.93, IT=0.07, Деканат=0.0).route("Как оплатить? И ещё: не открывается вебинар")
    assert result.multi_topic and not result.auto_routed


def test_weak_second_option_without_marker_stays_auto():
    result = fake_router(Бухгалтерия=0.93, IT=0.07, Деканат=0.0).route("Как оплатить обучение за семестр")
    assert result.auto_routed and not result.multi_topic


def test_marker_with_negligible_second_option_stays_auto():
    result = fake_router(Бухгалтерия=0.97, IT=0.03, Деканат=0.0).route("Как оплатить? И ещё: можно ли картой?")
    assert result.auto_routed and not result.multi_topic

from datetime import datetime

import pytest

from router.service import RoutingResult
from router.storage import Storage


def result(department="Бухгалтерия", auto=True, urgency="обычная"):
    return RoutingResult(department, 0.9 if auto else 0.4, urgency, auto, [(department, 0.9)])


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "db.sqlite")


def test_empty(storage):
    assert storage.all_requests().empty
    assert storage.pending().empty
    assert storage.corrections().empty


def test_auto_request_is_processed(storage):
    storage.log("Как оплатить?", result(auto=True))
    row = storage.all_requests().iloc[0]
    assert row["final_department"] == "Бухгалтерия"
    assert not row["corrected"]
    assert storage.pending().empty


def test_manual_request_waits_then_confirmed(storage):
    request_id = storage.log("Непонятный вопрос", result(auto=False))
    assert list(storage.pending()["id"]) == [request_id]
    storage.confirm(request_id)
    assert storage.pending().empty
    assert storage.all_requests().iloc[0]["final_department"] == "Бухгалтерия"


def test_correction_recorded(storage):
    request_id = storage.log("Справка для вычета", result(auto=True))
    storage.correct(request_id, "Справки и документы")
    row = storage.all_requests().iloc[0]
    assert row["corrected"]
    assert row["final_department"] == "Справки и документы"
    assert storage.corrections().to_dict("records") == [
        {"text": "Справка для вычета", "department": "Справки и документы"}
    ]


def test_correct_to_same_department_is_confirmation(storage):
    request_id = storage.log("Как оплатить?", result(auto=False))
    storage.correct(request_id, "Бухгалтерия")
    assert not storage.all_requests().iloc[0]["corrected"]
    assert storage.corrections().empty


def test_pending_urgent_first(storage):
    storage.log("старое обычное", result(auto=False), created_at=datetime(2026, 1, 1))
    urgent_id = storage.log("новое срочное", result(auto=False, urgency="высокая"), created_at=datetime(2026, 1, 2))
    assert storage.pending().iloc[0]["id"] == urgent_id


def test_created_at_is_datetime(storage):
    storage.log("текст", result(), created_at=datetime(2026, 3, 5, 10, 0))
    assert storage.all_requests().iloc[0]["created_at"] == datetime(2026, 3, 5, 10, 0)


def test_clear(storage):
    storage.log("текст", result())
    storage.clear()
    assert storage.all_requests().empty


def test_get(storage):
    request_id = storage.log("текст", result())
    assert storage.get(request_id)["text"] == "текст"
    assert storage.get(request_id + 100) is None


def test_corrections_are_distinct(storage):
    for _ in range(2):
        request_id = storage.log("Справка для вычета", result())
        storage.correct(request_id, "Справки и документы")
    assert len(storage.corrections()) == 1


def test_training_examples_include_corrections_and_confirmed_manual(storage):
    storage.log("авто верно", result(auto=True))
    auto_fixed = storage.log("авто исправлено", result(auto=True))
    manual_ok = storage.log("ручное подтверждено", result(auto=False))
    storage.log("ручное ждёт", result(auto=False))
    storage.correct(auto_fixed, "Деканат")
    storage.confirm(manual_ok)
    examples = storage.training_examples()
    assert sorted(examples["text"]) == ["авто исправлено", "ручное подтверждено"]
    assert dict(zip(examples["text"], examples["department"])) == {
        "авто исправлено": "Деканат", "ручное подтверждено": "Бухгалтерия"}

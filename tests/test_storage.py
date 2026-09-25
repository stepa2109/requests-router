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

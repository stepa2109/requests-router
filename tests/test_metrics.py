import pandas as pd
import pytest

from router.metrics import by_department, daily, summary

COLUMNS = ["created_at", "predicted", "confidence", "auto_routed", "final_department", "corrected"]


def frame(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def test_empty():
    s = summary(frame([]))
    assert s == {"total": 0, "auto_share": 0.0, "accuracy": None, "auto_accuracy": None, "auto_errors": 0,
                 "avg_confidence": None, "pending": 0, "corrected": 0, "saved_hours": 0.0}
    assert daily(frame([])).empty
    assert by_department(frame([])).empty


def sample():
    return frame([
        ("2026-09-01 10:00", "Бухгалтерия", 0.9, True, "Бухгалтерия", False),
        ("2026-09-01 11:00", "Деканат", 0.8, True, "Учебный отдел", True),
        ("2026-09-02 09:00", "Деканат", 0.5, False, "Деканат", False),
        ("2026-09-02 12:00", "Бухгалтерия", 0.4, False, None, False),
    ])


def test_summary():
    s = summary(sample(), minutes_per_request=6)
    assert s["total"] == 4
    assert s["auto_share"] == 0.5
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["avg_confidence"] == pytest.approx(0.65)
    assert s["pending"] == 1
    assert s["corrected"] == 1
    assert s["saved_hours"] == pytest.approx(0.2)  # 2 авто × 6 мин
    assert s["auto_errors"] == 1
    assert s["auto_accuracy"] == pytest.approx(0.5)  # из 2 авто одно исправлено


def test_daily():
    d = daily(sample())
    assert list(d["total"]) == [2, 2]
    assert list(d["auto"]) == [2, 0]
    assert list(d["corrected"]) == [1, 0]


def test_by_department():
    counts = by_department(sample())
    assert counts.to_dict() == {"Бухгалтерия": 2, "Учебный отдел": 1, "Деканат": 1}
